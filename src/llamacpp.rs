#![allow(dead_code)]
use core::f64;
use std::collections::HashMap;

use lazy_static::lazy_static;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::{params::LlamaModelParams, LlamaModel};
use llama_cpp_2::token::LlamaToken;

use crate::llm::{Model, Session, SessionTimers, Token};

lazy_static! {
    pub static ref BACKEND: LlamaBackend = LlamaBackend::init().unwrap();
}

pub fn model_from_gguf(path: impl AsRef<std::path::Path>, on_gpu: bool) -> LlamaModel {
    let gag_or = gag::Gag::stderr();
    if let Err(e) = gag_or {
        println!("Trouble silencing stderr: {e}");
    }
    let params: LlamaModelParams =
        LlamaModelParams::default().with_n_gpu_layers(if on_gpu { 1000000 } else { 0 });
    // TODO: wish we could set the defrag threshold. Manually defrag?

    LlamaModel::load_from_file(&BACKEND, path, &params).expect("loading model")
}

impl Model for LlamaModel {
    fn str_to_tokens_bos(&self, s: &str, bos: bool) -> Vec<Token> {
        str_to_tokens_bos(s, bos, self)
    }

    fn toks_to_str(&self, toks: &[Token]) -> String {
        let llama_toks: Vec<LlamaToken> = toks.iter().map(|&t| LlamaToken(t)).collect();
        toks_to_str(&llama_toks, self)
    }

    fn tok_to_str(&self, tok: Token) -> String {
        tok_to_str(LlamaToken(tok), self)
    }

    fn new_session<'b>(
        &'b self,
        prompt: &str,
        extra_toks: usize,
    ) -> Box<dyn crate::llm::Session<'b> + 'b> {
        let prompt_toks = self.str_to_tokens_bos(prompt, true);

        let mut sess = LlamaCppSession::new(self, (prompt_toks.len() + extra_toks) as u32);
        let first_candidates = sess.advance_and_predict(&prompt_toks, Some(0.9996));

        sess.first_candidates = first_candidates;

        sess.save_prompt();

        Box::new(sess)
    }
}

// TODO: inline these!
fn tok_to_str(t: LlamaToken, model: &LlamaModel) -> String {
    model
        .token_to_str(t, llama_cpp_2::model::Special::Plaintext)
        .unwrap_or("<???>".to_string())
}

fn toks_to_str(t: &[LlamaToken], model: &LlamaModel) -> String {
    model
        .tokens_to_str(t, llama_cpp_2::model::Special::Tokenize)
        .expect("stringifying tokens")
}

fn str_to_tokens(s: &str, model: &LlamaModel) -> Vec<LlamaToken> {
    model
        .str_to_token(s, llama_cpp_2::model::AddBos::Never)
        .unwrap()
}

fn str_to_tokens_bos(s: &str, bos: bool, model: &LlamaModel) -> Vec<Token> {
    model
        .str_to_token(
            s,
            if bos {
                llama_cpp_2::model::AddBos::Always
            } else {
                llama_cpp_2::model::AddBos::Never
            },
        )
        .unwrap()
        .into_iter()
        .map(|t| t.0)
        .collect()
}

pub struct LlamaCppSession<'a> {
    ctx: LlamaContext<'a>,
    prompt_toks: Option<usize>,
    empty: bool,
    toks: usize,
    pub timers: SessionTimers,
    boost_toks: HashMap<Token, f64>,
    first_candidates: Vec<(Token, f64)>,
}

static PROMPT_SEQ_ID: i32 = 0;
static OTHER_SEQ_ID: i32 = 1;

impl<'a> LlamaCppSession<'a> {
    pub fn new(model: &'a LlamaModel, toks: u32) -> LlamaCppSession<'a> {
        let params: LlamaContextParams =
            LlamaContextParams::default().with_n_ctx(std::num::NonZero::new(toks));

        let sess_ctx = {
            let _stderr_gag = gag::Gag::stderr().unwrap();
            model
                .new_context(&BACKEND, params)
                .expect("Unable to build context")
        };
        LlamaCppSession {
            ctx: sess_ctx,
            // batch: LlamaBatch::new(toks as usize, 2),
            prompt_toks: None,
            empty: true,
            toks: 0,
            timers: SessionTimers::default(),
            boost_toks: HashMap::new(),
            first_candidates: vec![], // TODO: this should be set here; pass in the prompt!
        }
    }

    pub fn boost(&mut self, tok: Token, boost: f64) {
        self.boost_toks.insert(tok, boost);
    }

    // Only mutable to update the time
    fn candidates(&mut self, top_p: Option<f32>) -> Vec<(Token, f64)> {
        let mut max_logit = f32::MIN;

        // TODO: we might be at least curious about the split between `decode()` and `candidates()`
        // ...but changing the serializable timing data invalidates save files.
        let before_predict = std::time::Instant::now();
        let raw_candidates = self.ctx.candidates();
        self.timers.predict_time += before_predict.elapsed().as_micros();

        let before_score = std::time::Instant::now();

        let mut candidates: Vec<(Token, f64)> = raw_candidates
            .map(|c| {
                max_logit = max_logit.max(c.logit());
                (c.id().0, c.logit() as f64)
            })
            .collect();

        let max_logit = max_logit as f64;

        // Reversed; we want largest-first:
        candidates.sort_by(|l, r| r.1.partial_cmp(&l.1).unwrap());

        let mut total_weight: f64 = 0.0;
        for (tok, ref mut logit) in &mut candidates {
            *logit = f64::exp(*logit - max_logit) * self.boost_toks.get(tok).unwrap_or(&1.0);
            total_weight += *logit;
        }

        // Scale so it sums to one:
        for (_, ref mut logit) in &mut candidates {
            *logit /= total_weight;
        }

        // Lazy sort doesn't seem to improve performance, but needs more investigation...
        // let sorted_candidates = candidates
        //     .iter()
        //     .sorted_by(|l, r| r.1.partial_cmp(&l.1).unwrap());

        if let Some(top_p) = top_p {
            let mut total_p = 0.0;
            let mut elts_needed = 0;
            for (i, (_, p)) in candidates.iter().enumerate() {
                total_p += p;
                if total_p > top_p as f64 {
                    elts_needed = i + 1;
                    break;
                }
            }
            if elts_needed == 0 {
                println!("Warning: Unable to find {top_p} total probability; got {total_p:.5} with {} candidates. Total weight was {total_weight:.3}", candidates.len());
                for (t, p) in candidates.iter().take(20) {
                    print!("{} => {:.2}  ", self.model().tok_to_str(*t), p * 100.0);
                }
                println!();
            } else {
                candidates.truncate(elts_needed);
            }
        }
        self.timers.score_time += before_score.elapsed().as_micros();

        candidates
    }

    pub fn advance_and_predict_str(&mut self, text: &str, top_p: Option<f32>) -> Vec<(Token, f64)> {
        let toks = str_to_tokens_bos(text, self.empty, self.model());
        self.advance_and_predict(&toks, top_p)
    }

    pub fn advance_and_predict(&mut self, toks: &[Token], top_p: Option<f32>) -> Vec<(Token, f64)> {
        if toks.is_empty() {
            panic!("No tokens!!");
        }
        // let seq_ids = if self.prompt_toks.is_none() {
        //     vec![PROMPT_SEQ_ID, OTHER_SEQ_ID]
        // } else {
        //     vec![OTHER_SEQ_ID]
        // };
        let seq_ids = vec![OTHER_SEQ_ID];

        let before_advance = std::time::Instant::now();
        let mut batch = LlamaBatch::new(toks.len() as usize, 2);

        for (i, t) in toks.iter().enumerate() {
            batch
                .add(
                    LlamaToken(*t),
                    self.toks as i32,
                    &seq_ids,
                    i + 1 == toks.len(),
                )
                .unwrap();
            self.toks += 1;
        }

        // self.batch
        //     .add_sequence(toks, seq_id, /*logits_all*/ false)
        //     .expect("advancing context");
        self.empty = false;

        self.timers.advance_time += before_advance.elapsed().as_micros();
        {
            //let _stderr_gag = gag::Gag::stderr().unwrap();
            let before_predict = std::time::Instant::now();
            self.ctx.decode(&mut batch).expect("predicting");
            self.timers.predict_time += before_predict.elapsed().as_micros();
        }

        let res = self.candidates(top_p);
        //self.batch.clear(); // Maybe we should just have `batch` be a local to this function?
        res
    }

    pub fn save_prompt(&mut self) {
        if self.prompt_toks.is_some() {
            panic!("Prompt already saved!");
        }
        self.prompt_toks = Some(self.toks);
    }

    pub fn truncate_to_prompt(&mut self) {
        // We don't reset `empty`, assuming there's a non-empty prompt.
        let before_truncate = std::time::Instant::now();
        self.ctx
            .clear_kv_cache_seq(None, Some(self.prompt_toks.unwrap() as u32), None)
            .unwrap();
        self.timers.truncate_time += before_truncate.elapsed().as_micros();

        self.toks = self.prompt_toks.expect("prompt saved");
        // self.ctx.kv_cache_defrag();
    }

    pub fn model(&self) -> &LlamaModel {
        self.ctx.model
    }
}

impl<'a> Session<'a> for LlamaCppSession<'a> {
    fn boost(&mut self, tok: Token, boost: f64) {
        self.boost_toks.insert(tok, boost);
    }

    fn timers(&self) -> &SessionTimers {
        &self.timers
    }

    fn timers_mut(&mut self) -> &mut SessionTimers {
        &mut self.timers
    }

    fn predict_str(&mut self, text: &str, top_p: Option<f32>) -> Vec<(Token, f64)> {
        self.predict(&self.model().str_to_tokens(text), top_p)
    }

    fn predict(&mut self, toks: &[Token], top_p: Option<f32>) -> Vec<(Token, f64)> {
        if toks.is_empty() {
            // Llama.cpp doesn't like an empty advance.
            return self.first_candidates.clone();
        }
        let res = self.advance_and_predict(&toks, top_p);

        self.truncate_to_prompt();
        res
    }
}

#[test]
fn llm_test() {
    // TODO: gotta find a small model with more predictable behavior...
}
