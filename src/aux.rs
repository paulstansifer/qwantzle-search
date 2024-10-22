// TODO: need to do some more refactoring to make this actually work in its own file.

use std::sync::{Arc, Mutex};

use llama_cpp::{LlamaModel, LlamaParams, LlamaSession, SessionParams, Token};
use llama_cpp_sys::llama_token_data;

mod llm;

use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the .gguf model file
    #[arg(short, long)]
    model: String,
}

fn display_top_toks(ctx: &mut LlamaSession) {
    let candidates = Arc::new(Mutex::new(vec![]));
    let peek_sampler = llm::PeekSampler {
        eos: ctx.model().eos(),
        candidates: candidates.clone(),
    };

    let mut completion_res = ctx
        .start_completing_with(peek_sampler, 1)
        .expect("Completion error!");
    let _ = completion_res.next_token();

    let candidates: &Vec<llama_token_data> = &candidates.lock().unwrap();

    for c in candidates.iter().take(7) {
        print!(
            " {} => {:.2}% ",
            ctx.model().decode_tokens([Token(c.id)]),
            c.p * 100.0
        );
    }
    println!();
}
fn check_truncation(model: &LlamaModel, toks: usize) {
    let params = SessionParams::default();

    let mut ctx = model
        .create_session(params)
        .expect("Failed to create session");

    println!("===truncation by {}===", toks);

    ctx.set_context("Hi -- wait, no, a volcano is about to")
        .unwrap();

    display_top_toks(&mut ctx);
    for _ in 0..toks {
        ctx.advance_context_with_tokens(&[Token(9999)]).unwrap();
    }
    ctx.truncate_context(ctx.context_size() - toks).unwrap();
    display_top_toks(&mut ctx);
    for _ in 0..toks {
        ctx.advance_context_with_tokens(&[Token(1001)]).unwrap();
    }
    ctx.truncate_context(ctx.context_size() - toks).unwrap();
    display_top_toks(&mut ctx);
    for _ in 0..toks {
        ctx.advance_context_with_tokens(&[Token(1001)]).unwrap();
    }
    ctx.truncate_context(ctx.context_size() - toks).unwrap();
    display_top_toks(&mut ctx);
}

fn main() {
    let args = Args::parse();
    let mut model_params = LlamaParams::default();
    model_params.n_gpu_layers = 10000;

    let model = LlamaModel::load_from_file(args.model, model_params).expect("Could not load model");

    for word in vec!["the fish.", "fish", " fish", "fish "] {
        let toks = model.tokenize_bytes(word, false, false).unwrap();

        for tok in toks {
            print!("'{}' => {} |", model.decode_tokens(&[tok]), tok.0)
        }
        println!();
    }

    // check_truncation(&model, 0);

    // check_truncation(&model, 1);

    // check_truncation(&model, 10);
}
