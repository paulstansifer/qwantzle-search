use std::{
    fmt::Write,
    io::Write as _,
    sync::{Arc, Mutex},
};

use llama_cpp::{standard_sampler::StandardSampler, LlamaModel, LlamaParams, SessionParams, Token};
use llama_cpp_sys::llama_token_data;

pub struct PeekSampler {
    eos: Token,
    candidates: Arc<Mutex<Vec<llama_token_data>>>,
}

impl llama_cpp::Sampler for PeekSampler {
    fn sample(
        &mut self,
        context: *mut llama_cpp_sys::llama_context,
        _tokens: &[Token],
        mut candidates_p: llama_cpp_sys::llama_token_data_array,
    ) -> Token {
        unsafe {
            llama_cpp_sys::llama_sample_top_p(
                context,
                std::ptr::addr_of_mut!(candidates_p),
                0.9,
                5,
            );
        }

        for i in 0..candidates_p.size {
            self.candidates
                .lock()
                .unwrap()
                .push(unsafe { *candidates_p.data.wrapping_add(i) });
        }
        return self.eos; // Don't produce any tokens! (This sampler doesn't sample.)
    }
}

fn megabytes(bytes: usize) -> usize {
    return bytes >> 20;
}

fn main() {
    // Create a model from anything that implements `AsRef<Path>`:

    let model = LlamaModel::load_from_file("/workspace/micro_llama.gguf", LlamaParams::default())
        .expect("Could not load model");

    println!(
        "Estimated session size: {} / {}",
        megabytes(
            model
                .estimate_session_size(&SessionParams::default())
                .host_memory
        ),
        megabytes(
            model
                .estimate_session_size(&SessionParams::default())
                .device_memory
        )
    );

    // A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
    // several gigabytes large, a session is typically a few dozen to a hundred megabytes!
    let mut ctx = model
        .create_session(SessionParams::default())
        .expect("Failed to create session");

    println!("Initial context size: {}", megabytes(ctx.memory_size()));

    let toks = model
        .tokenize_bytes(
            "Once upon a time there was a DINOSAUR who had some THOUGHTS about SCIENCE.",
            false,
            false,
        )
        .unwrap();

    ctx.advance_context_with_tokens(&[model.bos()])
        .expect("Context BOS");

    let mut punchline: String = String::new();
    for tok in toks {
        let candidates = Arc::new(Mutex::new(vec![]));
        let peek_sampler = PeekSampler {
            eos: model.eos(),
            candidates: candidates.clone(),
        };

        let mut completion_res = ctx
            .start_completing_with(peek_sampler, 1)
            .expect("Completion error!");
        let _ = completion_res.next_token();

        let candidates: &Vec<llama_token_data> = &candidates.lock().unwrap();

        println!("{} ...: ({} candidates)", punchline, candidates.len());

        let mut tok_s = "   ".to_string();
        let mut prb_s = "   ".to_string();
        let mut lgt_s = "   ".to_string();

        for candidate in candidates.iter().take(10) {
            let one_tok = vec![Token(candidate.id)];
            write!(tok_s, "{:>10}", model.decode_tokens(one_tok.iter())).unwrap();
            if candidate.p < 0.01 {
                write!(prb_s, "{:>9.2}%", candidate.p * 100.0).unwrap();
            } else if candidate.p < 0.1 {
                write!(prb_s, "{:>9.1}%", candidate.p * 100.0).unwrap();
            } else {
                write!(prb_s, "{:>9.0}%", candidate.p * 100.0).unwrap();
            }
            write!(lgt_s, "{:>10.1}", candidate.logit).unwrap();
        }
        print!("{}\n{}\n{}\n", tok_s, prb_s, lgt_s);

        ctx.advance_context_with_tokens(&[tok])
            .expect("Advancement error!");
        punchline.push_str(&model.decode_tokens([tok].iter()));
    }

    // ctx.advance_context("").unwrap();

    // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
    let max_tokens = 1024;
    let mut decoded_tokens = 0;

    // `ctx.start_completing_with` creates a worker thread that generates tokens. When the completion
    // handle is dropped, tokens stop generating!
    let mut completions = ctx
        .start_completing_with(StandardSampler::default(), 1024)
        .unwrap()
        .into_strings();

    for completion in completions {
        print!("{completion}");
        let _ = std::io::stdout().flush();

        decoded_tokens += 1;

        if decoded_tokens > max_tokens {
            break;
        }
    }
}
