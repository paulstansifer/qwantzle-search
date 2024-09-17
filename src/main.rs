use std::io::Write;

use llama_cpp::{standard_sampler::StandardSampler, LlamaModel, LlamaParams, SessionParams};

fn main() {
    // Create a model from anything that implements `AsRef<Path>`:

    let model = LlamaModel::load_from_file("/workspace/tiny_llama.gguf", LlamaParams::default())
        .expect("Could not load model");

    // A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
    // several gigabytes large, a session is typically a few dozen to a hundred megabytes!
    let mut ctx = model
        .create_session(SessionParams::default())
        .expect("Failed to create session");

    // You can feed anything that implements `AsRef<[u8]>` into the model's context.
    ctx.advance_context("Once upon").unwrap();

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
