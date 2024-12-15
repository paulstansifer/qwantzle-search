#![allow(dead_code)]
use lazy_static::lazy_static;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::{params::LlamaModelParams, LlamaModel};
use llama_cpp_2::token::LlamaToken;

lazy_static! {
    pub static ref BACKEND: LlamaBackend = LlamaBackend::init().unwrap();
}

pub fn model_from_gguf(path: impl AsRef<std::path::Path>, on_gpu: bool) -> LlamaModel {
    let _stderr_gag = gag::Gag::stderr().unwrap();
    let params: LlamaModelParams =
        LlamaModelParams::default().with_n_gpu_layers(if on_gpu { 1000000 } else { 0 });
    // TODO: wish we could set the defrag threshold. Manually defrag?

    LlamaModel::load_from_file(&BACKEND, path, &params).expect("loading model")
}

pub fn tok_to_str(t: LlamaToken, model: &LlamaModel) -> String {
    model
        .token_to_str(t, llama_cpp_2::model::Special::Plaintext)
        .unwrap_or("<???>".to_string())
}

pub fn toks_to_str(t: &[LlamaToken], model: &LlamaModel) -> String {
    model
        .tokens_to_str(t, llama_cpp_2::model::Special::Tokenize)
        .expect("stringifying tokens")
}

pub fn str_to_tokens(s: &str, model: &LlamaModel) -> Vec<LlamaToken> {
    model
        .str_to_token(s, llama_cpp_2::model::AddBos::Never)
        .unwrap()
}

pub fn str_to_tokens_maybe_with_prefix_space(
    s: &str,
    model: &LlamaModel,
) -> (Vec<LlamaToken>, bool) {
    let toks_without_space = str_to_tokens(s, model);

    let toks_with_space = str_to_tokens(&format!(" {s}"), model);

    if toks_without_space.len() < toks_with_space.len() {
        (toks_without_space, false)
    } else {
        (toks_with_space, true)
    }
}

pub struct Session<'a> {
    ctx: LlamaContext<'a>,
    // batch: LlamaBatch,
    prompt_toks: Option<usize>,
    empty: bool,
    pub advance_time: u128,
    pub predict_time: u128,
    pub score_time: u128,
    pub truncate_time: u128,
    toks: usize,
}

static PROMPT_SEQ_ID: i32 = 0;
static OTHER_SEQ_ID: i32 = 1;

impl<'a> Session<'a> {
    pub fn new(model: &'a LlamaModel, toks: u32) -> Session<'a> {
        let params: LlamaContextParams =
            LlamaContextParams::default().with_n_ctx(std::num::NonZero::new(toks));

        let sess_ctx = {
            let _stderr_gag = gag::Gag::stderr().unwrap();
            model
                .new_context(&BACKEND, params)
                .expect("Unable to build context")
        };
        Session {
            ctx: sess_ctx,
            // batch: LlamaBatch::new(toks as usize, 2),
            prompt_toks: None,
            empty: true,
            advance_time: 0,
            predict_time: 0,
            score_time: 0,
            truncate_time: 0,
            toks: 0,
        }
    }

    // Only mutable to update the time
    fn candidates(&mut self, top_p: Option<f32>) -> Vec<(LlamaToken, f32)> {
        let before_score = std::time::Instant::now();

        let mut candidates: Vec<(LlamaToken, f32)> =
            self.ctx.candidates().map(|c| (c.id(), c.logit())).collect();

        // Reversed; we want largest-first:
        candidates.sort_by(|l, r| r.1.partial_cmp(&l.1).unwrap());

        let max_logit = candidates.first().unwrap().1;

        let mut total_weight = 0.0;
        for (_, ref mut logit) in &mut candidates {
            *logit = f32::exp(*logit - max_logit);
            total_weight += *logit;
        }

        // Scale so it sums to zero:
        for (_, ref mut weight) in &mut candidates {
            *weight /= total_weight;
        }

        if let Some(top_p) = top_p {
            let mut total_p = 0.0;
            let mut elts_needed = 0;
            for (i, (_, p)) in candidates.iter().enumerate() {
                total_p += p;
                if total_p > top_p {
                    elts_needed = i + 1;
                    break;
                }
            }
            if elts_needed == 0 {
                println!("Warning: Unable to find {top_p} total probability.");
            } else {
                candidates.truncate(elts_needed);
            }
        }
        self.score_time += before_score.elapsed().as_micros();

        candidates
    }

    pub fn advance_and_predict_str(
        &mut self,
        text: &str,
        top_p: Option<f32>,
    ) -> Vec<(LlamaToken, f32)> {
        let toks = self
            .ctx
            .model
            .str_to_token(
                text,
                if self.empty {
                    AddBos::Always
                } else {
                    AddBos::Never
                },
            )
            .unwrap();
        self.advance_and_predict(&toks, top_p)
    }

    pub fn advance_and_predict(
        &mut self,
        toks: &[LlamaToken],
        top_p: Option<f32>,
    ) -> Vec<(LlamaToken, f32)> {
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
                .add(*t, self.toks as i32, &seq_ids, i + 1 == toks.len())
                .unwrap();
            self.toks += 1;
        }

        // self.batch
        //     .add_sequence(toks, seq_id, /*logits_all*/ false)
        //     .expect("advancing context");
        self.empty = false;

        self.advance_time += before_advance.elapsed().as_micros();
        {
            let _stderr_gag = gag::Gag::stderr().unwrap();
            let before_predict = std::time::Instant::now();
            self.ctx.decode(&mut batch).expect("predicting");
            self.predict_time += before_predict.elapsed().as_micros();
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
        self.truncate_time += before_truncate.elapsed().as_micros();

        self.toks = self.prompt_toks.expect("prompt saved");
    }

    pub fn model(&self) -> &LlamaModel {
        self.ctx.model
    }
}

#[test]
fn llm_test() {
    let m = model_from_gguf("maykeye-tl.gguf", true);

    let mut sess = Session::new(&m, 100);

    let candidates_4 = sess.advance_and_predict_str(" 1 2 3 ", Some(0.7));
    println!(
        "### {} {}",
        tok_to_str(candidates_4[0].0, &m),
        candidates_4[0].1
    );
    let candidates_5 = sess.advance_and_predict_str(" 4 ", Some(0.7));
    println!(
        "### {} {}",
        tok_to_str(candidates_5[0].0, &m),
        candidates_5[0].1
    );
    let candidates_6 = sess.advance_and_predict_str(" 5 ", Some(0.7));
    println!(
        "### {} {}",
        tok_to_str(candidates_6[0].0, &m),
        candidates_6[0].1
    );

    let candidates_0 = sess.advance_and_predict_str("Mary had", Some(0.9));
    assert_eq!(tok_to_str(candidates_0[0].0, &m), " a");
    sess.save_prompt();
    // let _ = sess.advance_and_predict_str(" a", Some(0.9));
    // let _ = sess.advance_and_predict_str(" a", Some(0.9));
    // let _ = sess.advance_and_predict_str(" a", Some(0.9));
    let candidates_1 = sess.advance_and_predict_str(" a", Some(0.9));
    assert_eq!(tok_to_str(candidates_1[0].0, &m), " little");
    sess.truncate_to_prompt();
    let candidates_2 = sess.advance_and_predict_str(" some", Some(0.9));
    assert_eq!(tok_to_str(candidates_2[0].0, &m), " little");
    sess.truncate_to_prompt();

    let candidates_1_again = sess.advance_and_predict_str(" a", Some(0.9));
    assert_eq!(candidates_1, candidates_1_again);
}
