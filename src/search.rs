use std::{
    cell::Cell,
    cmp::Ordering,
    fmt::Write,
    sync::{Arc, Mutex},
};

use clap::Parser;
use indicatif::ProgressBar;
use llama_cpp::{LlamaModel, SessionParams, Token};
use llama_cpp_sys::llama_token_data;
use priority_queue::PriorityQueue;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use crate::{llm, pool::LetterPool, strip::Strip};

#[derive(Parser, Debug)]
struct Args {
    /// Path to the .gguf model file
    #[arg(short, long)]
    model: String,

    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    /// Don't store the model on the GPU.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    no_gpu: bool,

    /// String to prepend to every strip. Use "l" to add a message about what the longest word in
    /// the punchline is.
    #[arg(short, long, default_value(""))]
    prompt_prefix: String,
}

// Reversed ordering for the priority queue, and pretending to really be `Cmp`.
#[derive(PartialOrd, PartialEq)]
struct Score(f64);

impl Eq for Score {}
impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap().reverse()
    }
}

struct Node {
    text: Vec<Token>,
    remaining: LetterPool,
    probability: f64, // f32 is not quite precise enough!
}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for t in &self.text {
            state.write_i32(t.0);
        }
        state.finish();
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        return self.text == other.text;
    }
}
impl Eq for Node {}

type Q = PriorityQueue<Node, Score>;

// 0.999 ^ 20 is around  0.98, so there's a 2% chance this loses a critical token.
const MIN_TOK_P: f32 = 0.001;

// Average at the end tends to be 5%, but it bounces around some, especially at the beginning.
// Perhaps we want some sort of grace period instead of setting this so low?
const MIN_AVG_P: f64 = 0.01;

// Note that `ctx.model()` clones the whole model! It's just an `Arc` plus a bunch of `usize`s, so it's not too bad, but it's an optimization opportunity.

thread_local! {
    pub static COPY_TIME : Cell<u128> = Cell::<u128>::default();
    pub static ADVANCE_TIME : Cell<u128> = Cell::<u128>::default();
    pub static PREDICT_TIME : Cell<u128> = Cell::<u128>::default();
}

impl Node {
    fn new(text: &str) -> Node {
        Node {
            text: vec![],
            remaining: LetterPool::from_text(text),
            probability: 1.0,
        }
    }

    fn push_token(&self, t: Token, prob: f32, model: &llama_cpp::LlamaModel) -> Option<Node> {
        let new_remaining = self.remaining.try_remove(t, model)?;
        let done = new_remaining.size() == 0;

        let mut res = Node {
            remaining: new_remaining,
            text: self.text.clone(),
            probability: self.probability * prob as f64,
        };
        res.text.push(t);

        // if done {
        //     println!(
        //         "=== === === {:.2}%: {}",
        //         f64::powf(res.probability as f64, 1.0 / (res.text.len() as f64)) * 100.0,
        //         model.decode_tokens(&res.text)
        //     );
        // }

        Some(res)
    }

    fn advance(&self, root_ctx: &llama_cpp::LlamaSession, q: &mut Q) {
        let mut my_ctx;
        {
            let before_copy = std::time::Instant::now();
            my_ctx = root_ctx.deep_copy().expect("Failed to copy context");
            COPY_TIME.replace(COPY_TIME.get() + before_copy.elapsed().as_micros());
        }
        {
            let before_advance = std::time::Instant::now();
            my_ctx
                .advance_context_with_tokens(&self.text)
                .expect("Failed to advance context");
            ADVANCE_TIME.replace(ADVANCE_TIME.get() + before_advance.elapsed().as_micros());
        }

        let candidates = Arc::new(Mutex::new(vec![]));
        let peek_sampler = llm::PeekSampler {
            eos: my_ctx.model().eos(),
            candidates: candidates.clone(),
        };

        {
            let before_predict = std::time::Instant::now();

            let mut completion_res = my_ctx
                .start_completing_with(peek_sampler, 1)
                .expect("Completion error!");
            let _ = completion_res.next_token();
            PREDICT_TIME.replace(PREDICT_TIME.get() + before_predict.elapsed().as_micros());
        }

        let candidates: &Vec<llama_token_data> = &candidates.lock().unwrap();

        for cand in candidates {
            if cand.p < MIN_TOK_P {
                break;
            }

            let avg_prob = f64::powf(
                cand.p as f64 * self.probability,
                1.0 / (self.text.len() as f64 + 1.0),
            );

            if avg_prob < MIN_AVG_P && self.text.len() >= 2 {
                break;
            }

            // if self.text.is_empty() {
            //     print!(
            //         "{:.1}% {}  ",
            //         cand.p * 100.0,
            //         root_ctx.model().decode_tokens(&[Token(cand.id)])
            //     )
            // }

            let score = f64::powf(
                cand.p as f64 * self.probability * 0.3 * 0.3,
                1.0 / (self.text.len() as f64 + 3.0),
            );

            if let Some(node) = self.push_token(Token(cand.id), cand.p, &root_ctx.model()) {
                q.push(node, Score(score));
            }
        }

        // if self.text.is_empty() {
        //     println!();
        // }
    }
}

pub fn practice_search(strip: &Strip, model: &LlamaModel, steps_limit: Option<usize>) {
    let mut q = Q::new();
    q.push(Node::new(&strip.punchline), Score(1.0));

    let leadup_toks = model.tokenize_bytes(&strip.leadup, true, false).unwrap();

    let mut params = SessionParams::default();
    params.n_ctx = leadup_toks.len() as u32 + 25;

    println!(
        "{} >>{}<<",
        strip
            .leadup
            .chars()
            .skip(strip.leadup.len() - 50)
            .collect::<String>()
            .replace("\n", "  "),
        strip.punchline
    );

    let progress = ProgressBar::new_spinner();

    progress.set_message(format!("Loading context"));

    let mut root_ctx = model.create_session(params).unwrap();
    root_ctx.set_context_to_tokens(&leadup_toks).unwrap();

    let mut step = 0;
    let mut log = String::new();
    loop {
        progress.tick();
        if let Some(lim) = steps_limit {
            if step > lim {
                progress.abandon_with_message(format!("Hit step limit: {}", step));
                break;
            }
        }
        if let Some((node, p)) = q.pop() {
            if node.remaining.size() == 0
                && model.decode_tokens(&node.text).trim() == strip.punchline.trim()
            {
                progress.abandon_with_message(format!(
                    "Search successful after {} steps. Prob: {:.2}%",
                    step,
                    p.0 * 100.0
                ));
                break;
            } else if node.remaining.size() == 0 {
                progress.println(format!(
                    "Non-answer: {} != {}",
                    model.decode_tokens(&node.text).trim(),
                    strip.punchline.trim()
                ));
            }

            let cur_str = format!(
                "{:7} {:.2}%: {}",
                step,
                p.0 * 100.0,
                model.decode_tokens(&node.text)
            );
            log.write_str(&format!("{}\n", cur_str)).unwrap();

            progress.set_message(cur_str);

            node.advance(&root_ctx, &mut q);
            step += 1;
        } else {
            progress
                .abandon_with_message(format!("Exhausted all reasonable nodes at {} steps", step));
            break;
        }
    }
    let per_step = |n: u128| (n as f32 / step as f32) / 1000.0;

    println!(
        "Per-step times: Copy: {:.0}ms  Advance: {:.0}ms  Predict: {:.0}ms",
        per_step(COPY_TIME.get()),
        per_step(ADVANCE_TIME.get()),
        per_step(PREDICT_TIME.get())
    );

    std::fs::write(
        format!("reports/search-{}.txt", strip.id),
        format!("{}\n{}", strip.punchline, log),
    )
    .unwrap();
}

/*
fn () {
    let args = Args::parse();
    init_tracing(&args);

    let mut model_params = LlamaParams::default();
    model_params.n_gpu_layers = if args.no_gpu { 0 } else { 1000000 };
    let model =
        LlamaModel::load_from_file(&args.model, model_params).expect("Could not load model");

    let strips = get_strips("corpus/strips.csv", &args.prompt_prefix);

    let mut strip_with_right_size: Option<Strip> = None;
    for strip in &strips {
        if !strip.punchline.contains("\n")
            && !strip.punchline.contains(":")
            && !strip.punchline.contains("tricky")
            // && !strip.punchline.contains("flossing")
            // && !strip.punchline.contains("TOES THAT HURT")
            // && !strip.punchline.contains("get that ALL")
            // && !strip.punchline.contains("just a little closer")
            // && !strip.punchline.contains("them my story, Utah")
            // && !strip.punchline.contains("you should quit")
            && strip.punchline.len() > 20
            && strip.punchline.len() <= 25
        {
            strip_with_right_size = Some(strip.clone());
            break;
        }
    }
    let strip = strip_with_right_size.expect("Unable to find a strip with the right size.");

    search(&strip, &model, Some(1e8 as usize));

    // let mut q = Q::new();
    // println!("{}", strip.leadup);

    // println!(">> {} <<", strip.punchline);
    // q.push(Node::new(&strip.punchline), Score(1.0));

    // let leadup_toks = model.tokenize_bytes(&strip.leadup, true, false).unwrap();

    // let mut params = SessionParams::default();
    // params.n_ctx = leadup_toks.len() as u32 + 25;

    // let mut root_ctx = model.create_session(params).unwrap();
    // root_ctx.set_context_to_tokens(&leadup_toks).unwrap();

    // loop {
    //     match q.pop() {
    //         Some((node, p)) => {
    //             println!("{:.2}%: {}", p.0 * 100.0, model.decode_tokens(&node.text));
    //             node.advance(&root_ctx, &mut q);
    //         }
    //         None => {
    //             break;
    //         }
    //     }
    // }
    // println!("Search complete.");
}
*/
