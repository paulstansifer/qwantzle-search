// TODO: need to do some more refactoring to make this actually work in its own file.

mod llm;

use clap::Parser;
use llama_cpp_2::model::LlamaModel;
use llm::Session;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the .gguf model file
    #[arg(short, long)]
    model: String,
}

fn display_top_toks(model: &LlamaModel, s: &str) {
    let mut sess = Session::new(model, 1000);

    let candidates = sess.advance_and_predict_str(s, Some(0.995));

    for (t, p) in candidates.iter().take(15) {
        let t_string = llm::tok_to_str(*t, model)
            .replace('\n', "\\n")
            .replace('\r', "\\r");
        print!("{:.2}%: {}  ", p * 100.0, t_string)
    }
    println!();
}

fn truncate_and_try(model: &LlamaModel, prompt: &str, suffixes: &[&str]) {
    let mut sess = Session::new(model, 1000);

    let candidates = sess.advance_and_predict_str(prompt, Some(0.995));
    sess.save_prompt();

    print!("prompt: '{prompt}'   ---   ");
    for (t, p) in candidates.iter().take(15) {
        print!("{:.2}%: {}  ", p * 100.0, llm::tok_to_str(*t, model))
    }
    println!();

    for suffix in suffixes {
        print!("suffix: '{suffix}'   ---   ");
        let candidates = sess.advance_and_predict_str(suffix, Some(0.995));
        for (t, p) in candidates.iter().take(15) {
            print!("{:.2}%: {}  ", p * 100.0, llm::tok_to_str(*t, model))
        }
        println!();
        sess.truncate_to_prompt();
    }
}

// fn check_truncation(model: &LlamaModel, toks: usize) {
//     let params = SessionParams::default();

//     let mut ctx = model
//         .create_session(params)
//         .expect("Failed to create session");

//     println!("===truncation by {}===", toks);

//     ctx.set_context("Hi -- wait, no, a volcano is about to")
//         .unwrap();

//     display_top_toks(&mut ctx);
//     for _ in 0..toks {
//         ctx.advance_context_with_tokens(&[Token(9999)]).unwrap();
//     }
//     ctx.truncate_context(ctx.context_size() - toks).unwrap();
//     display_top_toks(&mut ctx);
//     for _ in 0..toks {
//         ctx.advance_context_with_tokens(&[Token(1001)]).unwrap();
//     }
//     ctx.truncate_context(ctx.context_size() - toks).unwrap();
//     display_top_toks(&mut ctx);
//     for _ in 0..toks {
//         ctx.advance_context_with_tokens(&[Token(1001)]).unwrap();
//     }
//     ctx.truncate_context(ctx.context_size() - toks).unwrap();
//     display_top_toks(&mut ctx);
// }

fn main() {
    let args = Args::parse();

    let model = llm::model_from_gguf(args.model, true);

    truncate_and_try(
        &model,
        "T-Rex: I'm going to count! ",
        &["1 2 3 ", "1 ", "3 2 ", "2 "],
    );

    // display_top_toks(&model, "T-Rex: I'm going to count! 1 2 3 ");
    // display_top_toks(&model, "T-Rex: I'm going to count! 1 2 3 4 ");
    // display_top_toks(&model, "T-Rex: I'm going to count! 1 2 3 4 5 ");

    // for word in vec!["fundamental", " fundamental"] {
    //     let toks = model.tokenize_bytes(word, false, false).unwrap();

    //     for tok in toks {
    //         print!("'{}' => {} |", model.decode_tokens(&[tok]), tok.0)
    //     }
    //     println!();
    // }

    // check_truncation(&model, 0);

    // check_truncation(&model, 1);

    // check_truncation(&model, 10);
}
