use std::collections::HashMap;

use anyhow::Result;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::llm::{Model, Session, SessionTimers, Token};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub struct VllmToken(i64);

pub struct VllmModel {
    vllm: Py<PyModule>,
    model: Py<PyAny>,
    tokenizer: Py<PyAny>,
    dec_cache: HashMap<VllmToken, String>,
}

pub fn tok_to_str(t: VllmToken, model: &mut VllmModel) -> String {
    model
        .dec_cache
        .entry(t)
        .or_insert_with(|| {
            Python::attach(|py| {
                model
                    .tokenizer
                    .call_method1(py, intern!(py, "decode"), (vec![t.0],))
                    .unwrap()
                    .extract(py)
                    .unwrap()
            })
        })
        .clone()
}

pub fn toks_to_str(t: &[VllmToken], model: &VllmModel) -> String {
    Python::attach(|py| {
        model
            .tokenizer
            .call_method1(
                py,
                intern!(py, "decode"),
                (t.iter().map(|t| t.0).collect::<Vec<_>>(),),
            )
            .unwrap()
            .extract(py)
            .unwrap()
    })
}

pub fn str_to_tokens(s: &str, model: &VllmModel) -> Vec<VllmToken> {
    Python::attach(|py| {
        model
            .tokenizer
            .call_method1(py, intern!(py, "encode"), (s,))
            .unwrap()
            .extract::<Vec<i64>>(py)
            .unwrap()
            .into_iter()
            .map(|t| VllmToken(t))
            .collect::<Vec<_>>()
    })
}

pub fn str_to_tokens_maybe_with_prefix_space(s: &str, model: &VllmModel) -> (Vec<VllmToken>, bool) {
    let toks_without_space = str_to_tokens(s, model);

    let toks_with_space = str_to_tokens(&format!(" {s}"), model);

    if toks_without_space.len() < toks_with_space.len() {
        (toks_without_space, false)
    } else {
        (toks_with_space, true)
    }
}

impl VllmModel {
    pub fn new(model_name: &str) -> Result<VllmModel> {
        Python::attach(|py| -> Result<VllmModel> {
            let vllm = py.import("vllm")?;

            let llm_class = vllm.getattr("LLM")?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("model", model_name)?;
            kwargs.set_item("enable_prefix_caching", true)?;
            // Needs to be tweakable:
            kwargs.set_item("gpu_memory_utilization", 0.75)?;
            kwargs.set_item("max_logprobs", -1)?;
            let model = llm_class.call((), Some(&kwargs))?;

            let tokenizer = model.getattr("get_tokenizer")?.call0()?;

            Ok(VllmModel {
                vllm: vllm.unbind(),
                model: model.unbind(),
                tokenizer: tokenizer.unbind(),
                dec_cache: HashMap::new(),
            })
        })
    }
}

impl Model for VllmModel {
    fn str_to_tokens(&mut self, s: &str) -> Vec<Token> {
        str_to_tokens(s, self)
            .into_iter()
            .map(|VllmToken(t)| t as i32)
            .collect()
    }

    fn toks_to_str(&mut self, toks: &[Token]) -> String {
        let vllm_toks: Vec<VllmToken> = toks.iter().map(|&t| VllmToken(t as i64)).collect();
        toks_to_str(&vllm_toks, self)
    }

    fn tok_to_str(&mut self, tok: Token) -> String {
        tok_to_str(VllmToken(tok as i64), self)
    }

    fn new_session<'b>(&'b mut self, prompt: &str) -> Box<dyn crate::llm::Session<'b> + 'b> {
        Box::new(VllmSession::new(self, prompt))
    }
}

pub struct VllmSession<'a> {
    model: &'a VllmModel,
    boost_toks: HashMap<VllmToken, f64>,
    pub timers: SessionTimers,
    pfx: Vec<VllmToken>,
}

impl<'a> VllmSession<'a> {
    pub fn new(model: &'a VllmModel, pfx: &str) -> VllmSession<'a> {
        let pfx = str_to_tokens(pfx, model);
        VllmSession {
            model,
            boost_toks: HashMap::new(),
            timers: SessionTimers::default(),
            pfx,
        }
    }

    pub fn boost(&mut self, tok: VllmToken, boost: f64) {
        self.boost_toks.insert(tok, boost);
    }

    pub fn predict_str(&mut self, text: &str, top_p: Option<f32>) -> Vec<(VllmToken, f64)> {
        let toks = str_to_tokens(text, &self.model);
        self.predict_internal(&toks, top_p)
    }

    pub fn predict_internal(
        &mut self,
        toks: &[VllmToken],
        top_p: Option<f32>,
    ) -> Vec<(VllmToken, f64)> {
        Python::attach(|py| -> anyhow::Result<Vec<(VllmToken, f64)>> {
            let sampling_params_class = self.model.vllm.getattr(py, "SamplingParams").unwrap();

            let params_kwargs = PyDict::new(py);
            params_kwargs.set_item("detokenize", false)?;

            params_kwargs.set_item("flat_logprobs", true)?;
            if let Some(top_p) = top_p {
                params_kwargs.set_item("top_p", top_p)?;
            }
            params_kwargs.set_item("temperature", 0.0)?;
            params_kwargs.set_item("max_tokens", 1)?;
            params_kwargs.set_item("logprobs", 10_000)?;
            let sampling_params = sampling_params_class.call(py, (), Some(&params_kwargs))?;
            let input_dict = PyDict::new(py);
            input_dict.set_item(
                "prompt_token_ids",
                self.pfx
                    .iter()
                    .chain(toks.iter())
                    .map(|t| t.0)
                    .collect::<Vec<_>>(),
            )?;

            let outputs: Bound<PyList> = self
                .model
                .model
                .call_method1(py, "generate", (input_dict.clone(), sampling_params))?
                .extract(py)
                .unwrap();
            let output = outputs
                .get_item(/*batch index*/ 0)?
                .getattr("outputs")?
                .get_item(/*always 0 for us*/ 0)?;
            let cand_tokens: Bound<PyList> = output.getattr("token_ids")?.extract().unwrap();
            let cand_logprobs: Bound<PyList> = output.getattr("logprobs")?.extract().unwrap();

            let tokens: Vec<i64> = cand_tokens.extract().unwrap();
            let logprobs: Vec<f64> = cand_logprobs.extract().unwrap();

            let mut results: Vec<(VllmToken, f64)> = tokens
                .into_iter()
                .zip(logprobs.into_iter())
                .map(|(token, logprob)| (VllmToken(token), logprob.exp()))
                .collect();

            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            Ok(results)
        })
        .unwrap()
    }
}

impl<'a> Session<'a> for VllmSession<'a> {
    fn boost(&mut self, tok: Token, boost: f64) {
        self.boost_toks.insert(VllmToken(tok as i64), boost);
    }

    fn timers(&self) -> &SessionTimers {
        &self.timers
    }

    fn predict_str(&mut self, text: &str, top_p: Option<f32>) -> Vec<(Token, f64)> {
        let toks = str_to_tokens(text, self.model);
        self.predict(
            &toks
                .into_iter()
                .map(|VllmToken(t)| t as i32)
                .collect::<Vec<_>>(),
            top_p,
        )
    }

    fn predict(&mut self, toks: &[Token], top_p: Option<f32>) -> Vec<(Token, f64)> {
        let vllm_toks: Vec<VllmToken> = toks.iter().map(|&t| VllmToken(t as i64)).collect();
        self.predict_internal(&vllm_toks, top_p)
            .into_iter()
            .map(|(VllmToken(t), p)| (t as i32, p))
            .collect()
    }
}
