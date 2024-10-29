#![allow(dead_code)]

use std::{
    cell::Cell,
    cmp::Ordering,
    fmt::Write,
    sync::{Arc, Mutex},
};

use indicatif::ProgressBar;
use llama_cpp::{LlamaModel, SessionParams, Token};
use llama_cpp_sys::llama_token_data;
use priority_queue::PriorityQueue;

use crate::{
    corpus::Strip,
    llm,
    pool::{LetterPool, Vocab, VocabBuilder, WordState},
};

// Reversed ordering for the priority queue, and pretending to really be `Cmp`.
#[derive(PartialOrd, PartialEq, Clone, Copy)]
struct Score(f64);

impl Eq for Score {}
impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap().reverse()
    }
}

struct Node {
    text: Vec<Token>,
    word_state: WordState,
    remaining: LetterPool,
    probability: f64, // f32 is not quite precise enough!
    tok_probs: Vec<f32>,
    chars_so_far: u8,
}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for t in &self.text {
            state.write_i32(t.0);
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        return self.text == other.text;
    }
}
impl Eq for Node {}

type Q = PriorityQueue<Node, Score>;

const TOK_BUFFER: u32 = 25;

// How good does the best next token need to be to fast-forward it?
// Performance seems sensitive to this; maybe we need a better criterion?
const FF_MIN_P: f32 = 0.25;

// 0.999 ^ 20 is around  0.98, so there's a 2% chance this loses a critical token.
const TOK_TOP_P: f32 = 0.999;

// Token 0: (50%) 1.82%  (25%) 0.45%  (10%) 0.24%  (5%) 0.24% (1%) 0.10%
// Token 1: (50%) 5.31%  (25%) 1.58%  (10%) 1.07%  (5%) 1.07% (1%) 0.35%
// Token 2: (50%) 8.92%  (25%) 4.23%  (10%) 2.50%  (5%) 2.50% (1%) 2.21%
// Token 3: (50%) 9.00%  (25%) 4.54%  (10%) 1.48%  (5%) 1.48% (1%) 1.45%
// Token 4: (50%) 9.24%  (25%) 4.92%  (10%) 3.37%  (5%) 3.37% (1%) 3.33%
// Token 5: (50%) 6.77%  (25%) 4.83%  (10%) 4.13%  (5%) 4.13% (1%) 3.93%
// Token 6: (50%) 6.98%  (25%) 6.43%  (10%) 5.49%  (5%) 5.49% (1%) 4.14%
// Token 7: (50%) 8.93%  (25%) 7.39%  (10%) 6.16%  (5%) 6.16% (1%) 6.14%
// Token 8: (50%) 9.37%  (25%) 7.82%  (10%) 5.96%  (5%) 5.96% (1%) 5.59%
// Token 9: (50%) 11.72%  (25%) 6.33%  (10%) 6.28%  (5%) 6.28% (1%) 6.27%

// Average at the end tends to be 5%, but it bounces around some, especially at the beginning.
// Perhaps we want some sort of grace period instead of setting this so low?
const MIN_AVG_P: f64 = 0.01;

// Note that `ctx.model()` clones the whole model! It's just an `Arc` plus a bunch of `usize`s, so it's not too bad, but it's an optimization opportunity.

thread_local! {
    pub static COPY_TIME : Cell<u128> = Cell::<u128>::default();
    pub static ADVANCE_TIME : Cell<u128> = Cell::<u128>::default();
    pub static FF_ADVANCE_TIME : Cell<u128> = Cell::<u128>::default();
    pub static PREDICT_TIME : Cell<u128> = Cell::<u128>::default();
    pub static MISC_TIME : Cell<u128> = Cell::<u128>::default();
}

fn generous_score(probs: &Vec<f32>, chars_so_far: u8) -> Score {
    let mut probs: Vec<f64> = probs.iter().map(|p| *p as f64).collect();
    probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if probs.len() >= 1 {
        probs[0] += 0.07;
        if probs.len() >= 2 {
            probs[1] += 0.05;
        }
    }
    let prod: f64 = probs.iter().product();

    Score(f64::powf(
        prod * 0.3 * 0.3,
        1.0 / ((chars_so_far as f64 / 5.0) + 2.0),
    ))
}

fn compromise_score(probs: &Vec<f32>) -> Score {
    let mut probs: Vec<f64> = probs.iter().map(|p| *p as f64).collect();
    probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if probs.len() >= 1 {
        probs[0] += 0.07;
        if probs.len() >= 2 {
            probs[1] += 0.05;
        }
    }
    let prod: f64 = probs.iter().product();

    Score(f64::powf(prod, 1.0 / (f64::powf(probs.len() as f64, 0.5))))
}

fn prob_score(probs: &Vec<f32>, chars_so_far: u8) -> Score {
    let mut probs: Vec<f64> = probs.iter().map(|p| *p as f64).collect();
    probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Try this, but probably way the heck weaker:

    // if probs.len() >= 1 {
    //     probs[0] += 0.07;
    //     if probs.len() >= 2 {
    //         probs[1] += 0.05;
    //     }
    // }

    let mut prod: f64 = probs.iter().product();

    // Credit for distance elapsed is measured in characters, not in tokens. This hurts things with
    // digits in them a lot, but the Qwantzle has no digits!
    let mut chars_i = 0.0;
    while chars_i < chars_so_far as f32 {
        // The linear approximation for how much the anagram helps things is rough, but seems about accurate in practice.
        let filter_ratio: f32 = 0.05 + 0.55 * ((80.0 - chars_i) / 80.0);
        // 6.0 is just made-up, though.
        prod = f32::powf(6.0 / filter_ratio, 0.5) as f64 * prod;

        chars_i += 2.0; // TODO: try stepping down to 0.25 / += 1.0
    }

    Score(prod)
}

impl Node {
    fn new(text: &str) -> Node {
        Node {
            text: vec![],
            word_state: WordState::new_empty(),
            remaining: LetterPool::from_text(text),
            probability: 1.0,
            tok_probs: vec![],
            chars_so_far: 0,
        }
    }

    fn push_token(
        &self,
        t: Token,
        prob: f32,
        model: &llama_cpp::LlamaModel,
        vocab: &Vocab,
    ) -> Option<(Node, Score)> {
        let new_remaining = self.remaining.try_remove(t, model)?;
        // let done = new_remaining.size() == 0;
        let new_word_state = self.word_state.add_tok(t, vocab)?;

        let mut res = Node {
            remaining: new_remaining,
            word_state: new_word_state,
            text: self.text.clone(),
            probability: self.probability * prob as f64,
            tok_probs: self.tok_probs.clone(),
            chars_so_far: self.chars_so_far + model.decode_tokens(&[t]).trim().len() as u8,
        };
        res.text.push(t);
        res.tok_probs.push(prob);

        let score = prob_score(&res.tok_probs, res.chars_so_far);

        Some((res, score))
    }

    fn advance(&self, root_ctx: &mut llama_cpp::LlamaSession, q: &mut Q, vocab: &Vocab) {
        let orig_toks = root_ctx.context_size();
        {
            let before_advance = std::time::Instant::now();
            root_ctx
                .advance_context_with_tokens(&self.text)
                .expect("Failed to advance context");
            ADVANCE_TIME.replace(ADVANCE_TIME.get() + before_advance.elapsed().as_micros());
        }

        self.predict_with_ctx(root_ctx, q, vocab);

        {
            let before_truncate = std::time::Instant::now();
            root_ctx.truncate_context(orig_toks).unwrap();
            MISC_TIME.replace(MISC_TIME.get() + before_truncate.elapsed().as_micros());
        }
    }

    fn predict_with_ctx(&self, my_ctx: &mut llama_cpp::LlamaSession, q: &mut Q, vocab: &Vocab) {
        if self.text.len() >= (TOK_BUFFER - 1) as usize {
            return;
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

        let mut fast_forwarded = false;

        let mut total_p = 0.0;
        for cand in candidates {
            if total_p >= TOK_TOP_P {
                break;
            }
            total_p += cand.p;

            let avg_prob = f64::powf(
                cand.p as f64 * self.probability,
                1.0 / (self.text.len() as f64 + 1.0),
            );

            if avg_prob < MIN_AVG_P && self.text.len() >= 2 {
                break;
            }

            if let Some((next_node, score)) =
                self.push_token(Token(cand.id), cand.p, &my_ctx.model(), vocab)
            {
                // Re-use the context right away, if the best candidate is good enough.
                if cand.p > FF_MIN_P && !fast_forwarded && next_node.remaining.size() > 0 {
                    let before_advance = std::time::Instant::now();
                    my_ctx
                        .advance_context_with_tokens(&[Token(cand.id)])
                        .unwrap();
                    FF_ADVANCE_TIME
                        .replace(FF_ADVANCE_TIME.get() + before_advance.elapsed().as_micros());

                    next_node.predict_with_ctx(my_ctx, q, vocab);
                    fast_forwarded = true;
                } else {
                    q.push(next_node, score);
                }
            }
        }
    }
}

pub struct SearchResult {
    pub found: bool,
    pub steps: usize,
    pub seconds: f32,
}

pub fn practice_search(
    strip: &Strip,
    model: &LlamaModel,
    words: &Vec<String>,
    steps_limit: Option<usize>,
    report: &mut String,
) -> SearchResult {
    let mut q = Q::new();
    q.push(Node::new(&strip.punchline), Score(1.0));

    let leadup_toks = model.tokenize_bytes(&strip.leadup, true, false).unwrap();

    let mut params = SessionParams::default();
    params.defrag_threshold = 0.5;
    params.n_ctx = leadup_toks.len() as u32 + TOK_BUFFER;

    let strip_summary = format!(
        "...{} >>{}<<\n",
        strip
            .leadup
            .chars()
            .skip(strip.leadup.len() - 50)
            .collect::<String>()
            .replace("\n", "  "),
        strip.punchline
    );

    print!("{}", strip_summary);
    *report += &strip_summary;

    let progress = ProgressBar::new_spinner();

    progress.set_message(format!("Loading context"));

    let mut root_ctx = model.create_session(params).unwrap();
    root_ctx.set_context_to_tokens(&leadup_toks).unwrap();

    // Set up vocabulary restrictions
    let mut v_builder = VocabBuilder::default();

    let mut longest_word_len = 0;
    // If there are non-dictionary words, sneak 'em in:
    for word in regex::Regex::new(r"\b").unwrap().split(&strip.punchline) {
        v_builder.add_word(word, model, /*vary_case=*/ false);
        longest_word_len = std::cmp::max(longest_word_len, word.len());
    }

    for word in words {
        // Simulate knowing that all words are shorter than the longest in the vocabulary.
        if word.len() >= longest_word_len {
            continue;
        }
        v_builder.add_word(word, model, /*vary_case=*/ true);
    }

    let vocab = v_builder.build(/*disabled=*/ false);

    let search_start_time = std::time::Instant::now();
    COPY_TIME.replace(0);
    ADVANCE_TIME.replace(0);
    FF_ADVANCE_TIME.replace(0);
    PREDICT_TIME.replace(0);
    MISC_TIME.replace(0);
    let mut success = false;
    let mut step = 0;
    let mut score_progress_info = "Score progression: ".to_string();
    let mut log = String::new();
    loop {
        progress.tick();
        if let Some(lim) = steps_limit {
            if step > lim {
                let msg = format!("Hit step limit: {}", step);
                *report += &msg;
                progress.abandon_with_message(msg);
                break;
            }
        }
        if let Some((node, p)) = q.pop() {
            let cur_text = model.decode_tokens(&node.text);

            if node.remaining.size() == 0 && cur_text.trim() == strip.punchline.trim() {
                let msg = format!(
                    "Search successful after {} steps. Score: {:.2}%",
                    step,
                    p.0 * 100.0
                );
                success = true;
                *report += &msg;
                progress.abandon_with_message(msg);
                break;
            } else if node.remaining.size() == 0 {
                let msg = format!(
                    "Non-answer: {} != {}",
                    model.decode_tokens(&node.text).trim(),
                    strip.punchline.trim()
                );
                *report += &msg;
                *report += "\n";
                progress.println(msg);
            } else if strip.punchline.trim().starts_with(cur_text.trim()) && !node.text.is_empty() {
                score_progress_info += &format!(
                    "'{}' {:.2}%  ",
                    model.token_to_piece(*node.text.last().unwrap()),
                    p.0 * 100.0
                );
            }

            let cur_str = format!("{:.2}%: {}", p.0 * 100.0, cur_text);
            log.write_str(&format!("{}\n", cur_str)).unwrap();

            progress.set_message(format!("{:7} {}", step, cur_str));

            node.advance(&mut root_ctx, &mut q, &vocab);
            step += 1;
        } else {
            let msg = format!("Exhausted all reasonable nodes at {} steps", step);
            *report += &msg;
            progress.abandon_with_message(msg);
            break;
        }
    }
    *report += "\n";
    let per_step = |n: u128| (n as f32 / step as f32) / 1000.0;

    let time_info = format!(
        "Search time: {:.0}s.  (Non-ML: {:.0}s)  Per-step times: Copy: {:.0}ms  Advance: {:.0}ms  FF advance: {:.0}ms  Predict: {:.0}ms  Truncate: {:.0}.",
        search_start_time.elapsed().as_secs_f32(),
        (search_start_time.elapsed().as_micros() - (COPY_TIME.get() + ADVANCE_TIME.get() + FF_ADVANCE_TIME.get() + PREDICT_TIME.get() + MISC_TIME.get())) as f32 / 1000000.0,
        per_step(COPY_TIME.get()),
        per_step(ADVANCE_TIME.get()),
        per_step(FF_ADVANCE_TIME.get()),
        per_step(PREDICT_TIME.get()),
        per_step(MISC_TIME.get()),
    );

    *report += &time_info;
    *report += "\n";
    println!("{}", time_info);

    *report += &score_progress_info;
    *report += "\n";
    println!("{}", score_progress_info);

    std::fs::write(
        format!("reports/search-{}.txt", strip.id),
        format!("{}\n{}", strip.punchline, log),
    )
    .unwrap();

    return SearchResult {
        found: success,
        steps: step,
        seconds: search_start_time.elapsed().as_secs_f32(),
    };
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
