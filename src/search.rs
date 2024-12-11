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
use serde::{Deserialize, Serialize};

use crate::{
    corpus::Strip,
    llm,
    pool::{LetterPool, Vocab, VocabBuilder, WordState},
    remaining_letters_neural_net::LetterNet,
};

// Reversed ordering for the priority queue, and pretending to really be `Cmp`.
#[derive(PartialOrd, PartialEq, Clone, Copy, Serialize, Deserialize)]
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
    depth_at_pruning: u32,
}

#[derive(Serialize, Deserialize)]
struct SerNode {
    text: Vec<i32>,
    word_state: WordState,
    remaining: LetterPool,
    probability: f64, // f32 is not quite precise enough!
    tok_probs: Vec<f32>,
    chars_so_far: u8,
    depth_at_pruning: u32,
}

impl SerNode {
    fn from(n: &Node) -> SerNode {
        SerNode {
            text: n.text.iter().map(|t| t.0).collect(),
            word_state: n.word_state.clone(),
            remaining: n.remaining.clone(),
            probability: n.probability,
            tok_probs: n.tok_probs.clone(),
            chars_so_far: n.chars_so_far,
            depth_at_pruning: n.depth_at_pruning,
        }
    }
    fn to(&self) -> Node {
        Node {
            text: self.text.iter().map(|i| Token(*i)).collect(),
            word_state: self.word_state.clone(),
            remaining: self.remaining.clone(),
            probability: self.probability,
            tok_probs: self.tok_probs.clone(),
            chars_so_far: self.chars_so_far,
            depth_at_pruning: self.depth_at_pruning,
        }
    }
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

struct Q {
    ties_respecting: PriorityQueue<Node, Score>,
    non_ties_respecting: PriorityQueue<Node, Score>,
}

impl Q {
    fn new() -> Q {
        Q {
            ties_respecting: PriorityQueue::<Node, Score>::new(),
            non_ties_respecting: PriorityQueue::<Node, Score>::new(),
        }
    }
    fn trim(self, elts: usize) -> Q {
        Q {
            ties_respecting: PriorityQueue::<Node, Score>::from_iter(
                self.ties_respecting
                    .into_sorted_iter()
                    .enumerate()
                    .map(|(i, (mut node, score))| {
                        node.depth_at_pruning = std::cmp::max(node.depth_at_pruning, i as u32);
                        (node, score)
                    })
                    .take(elts),
            ),
            non_ties_respecting: PriorityQueue::<Node, Score>::from_iter(
                self.non_ties_respecting
                    .into_sorted_iter()
                    .enumerate()
                    .map(|(i, (mut node, score))| {
                        node.depth_at_pruning = std::cmp::max(node.depth_at_pruning, i as u32);
                        (node, score)
                    })
                    .take(elts),
            ),
        }
    }

    fn len(&self) -> usize {
        self.ties_respecting.len() + self.non_ties_respecting.len()
    }

    fn save(&self) {
        let mut bytes = 0;
        let mut elts = 0;
        for priority_queue in [&self.ties_respecting, &self.non_ties_respecting] {
            let srs: Vec<SerNode> = priority_queue.iter().map(|n| SerNode::from(&n.0)).collect();
            let serialized: Vec<u8> = bincode::serialize(&srs).unwrap();
            elts += priority_queue.len();
            bytes += serialized.len();
        }

        println!(
            "Serializing {} elements would use {} bytes ({} MB)",
            elts,
            bytes,
            bytes / (1024 * 1024)
        );
    }

    fn push(&mut self, n: Node, s: Score) {
        if n.remaining.respects_ties() {
            self.ties_respecting.push(n, s);
        } else {
            self.non_ties_respecting.push(n, s);
        }
    }

    fn pop(&mut self, require_tie_respecting: bool) -> Option<(Node, Score)> {
        if require_tie_respecting {
            return self
                .ties_respecting
                .pop()
                .or_else(|| self.non_ties_respecting.pop());
        }

        if let (Some((_, t_score)), Some((_, n_score))) =
            (self.ties_respecting.peek(), self.non_ties_respecting.peek())
        {
            if t_score < n_score {
                // `Scores` compare lower-is-better.
                self.ties_respecting.pop()
            } else {
                self.non_ties_respecting.pop()
            }
        } else {
            self.ties_respecting
                .pop()
                .or_else(|| self.non_ties_respecting.pop())
        }
    }
}

const TOK_BUFFER: u32 = 25;

// How good does the best next token need to be to fast-forward it?
// Performance seems sensitive to this; maybe we need a better criterion?
const FF_MIN_P: f32 = 0.25;

// 0.999 ^ 20 is around  0.98, so there's a 2% chance this loses a critical token.
// In practice, it seems like we need to be more careful than that.
const TOK_TOP_P: f32 = 0.9995;

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

fn prob_score(probs: &Vec<f32>, chars_so_far: u8, rlnn_mult: f32) -> Score {
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

        // This is unmotivated; purely empirical. Starts at 6.0, goes towards 4.0:
        let len_bonus = f32::max(0.0, ((100.0 - chars_i) / 100.0) * 2.0) + 3.0;

        prod = f32::powf((1.0 + len_bonus) / filter_ratio, 0.25) as f64 * prod;

        chars_i += 1.0;
    }

    Score(prod * rlnn_mult as f64)
}

impl Node {
    fn new(text: &str) -> Node {
        Node {
            text: vec![],
            word_state: WordState::new_empty(),
            remaining: LetterPool::from_text(text, /*look_at_ties=*/ false),
            probability: 1.0,
            tok_probs: vec![],
            chars_so_far: 0,
            depth_at_pruning: 0,
        }
    }
    fn new_with_longest_tok(text: &str, model: &LlamaModel) -> Node {
        let mut res = Node::new(text);
        res.remaining.set_longest_tok_from(text, model);
        res
    }

    fn push_token(
        &self,
        t: Token,
        prob: f32,
        model: &llama_cpp::LlamaModel,
        vocab: &Vocab,
        rlnn: &LetterNet,
    ) -> Option<(Node, Score)> {
        let new_remaining = self.remaining.try_remove(t, model)?;
        // let done = new_remaining.size() == 0;
        let new_word_state = self.word_state.add_tok(t, vocab)?;

        let rlnn_prob = rlnn.evaluate(&new_remaining);

        let mut res = Node {
            remaining: new_remaining,
            word_state: new_word_state,
            text: self.text.clone(),
            probability: self.probability * prob as f64,
            tok_probs: self.tok_probs.clone(),
            chars_so_far: self.chars_so_far + model.decode_tokens(&[t]).trim().len() as u8,
            depth_at_pruning: self.depth_at_pruning,
        };
        res.text.push(t);
        res.tok_probs.push(prob);

        let score = prob_score(&res.tok_probs, res.chars_so_far, rlnn_prob);

        Some((res, score))
    }

    fn advance(
        &self,
        root_ctx: &mut llama_cpp::LlamaSession,
        q: &mut Q,
        vocab: &Vocab,
        rlnn: &LetterNet,
    ) {
        let orig_toks = root_ctx.context_size();
        {
            let before_advance = std::time::Instant::now();
            root_ctx
                .advance_context_with_tokens(&self.text)
                .expect("Failed to advance context");
            ADVANCE_TIME.replace(ADVANCE_TIME.get() + before_advance.elapsed().as_micros());
        }

        self.predict_with_ctx(root_ctx, q, vocab, rlnn);

        {
            let before_truncate = std::time::Instant::now();
            root_ctx.truncate_context(orig_toks).unwrap();
            MISC_TIME.replace(MISC_TIME.get() + before_truncate.elapsed().as_micros());
        }
    }

    fn predict_with_ctx(
        &self,
        my_ctx: &mut llama_cpp::LlamaSession,
        q: &mut Q,
        vocab: &Vocab,
        rlnn: &LetterNet,
    ) {
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
                self.push_token(Token(cand.id), cand.p, &my_ctx.model(), vocab, rlnn)
            {
                // Re-use the context right away, if the best candidate is good enough.
                if cand.p > FF_MIN_P && !fast_forwarded && next_node.remaining.size() > 0 {
                    let before_advance = std::time::Instant::now();
                    my_ctx
                        .advance_context_with_tokens(&[Token(cand.id)])
                        .unwrap();
                    FF_ADVANCE_TIME
                        .replace(FF_ADVANCE_TIME.get() + before_advance.elapsed().as_micros());

                    next_node.predict_with_ctx(my_ctx, q, vocab, rlnn);
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
    q.push(
        Node::new_with_longest_tok(&strip.punchline, model),
        // Node::new(&strip.punchline),
        Score(1.0),
    );

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

    let mut longest_word_len = 0;
    // Set up vocabulary restrictions
    let vocab = {
        let mut v_builder = VocabBuilder::default();

        // If there are non-dictionary words, sneak 'em in:
        for word in regex::Regex::new(r"\b").unwrap().split(&strip.punchline) {
            v_builder.add_word(word, model, /*vary_case=*/ false);
            if word.len() > longest_word_len {
                longest_word_len = word.len();
            }
        }

        for word in words {
            // Simulate knowing that all words are shorter than the longest in the vocabulary.
            if word.len() >= longest_word_len {
                continue;
            }
            v_builder.add_word(word, model, /*vary_case=*/ true);
        }

        v_builder.build(/*disabled=*/ false)
    };

    // Set up the letter pool
    let rlnn = LetterNet::new_from_file("corpus/letter_pool.safetensors").unwrap();

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
    let mut deepest_node_accessed = 0;
    loop {
        progress.tick();
        if let Some(lim) = steps_limit {
            if step >= lim {
                let msg = format!("Hit step limit: {}", step);
                *report += &msg;
                progress.abandon_with_message(msg);
                break;
            }
        }

        if q.len() >= 3_000_000 {
            progress.println(format!(
                "Queue length is {}; trimming to 500k each.",
                q.len()
            ));
            q = q.trim(500_000);
            // save_queue(&q);
        }
        if let Some((node, p)) = q.pop(/*require_tie_respecting=*/ step % 4 == 0) {
            deepest_node_accessed = std::cmp::max(deepest_node_accessed, node.depth_at_pruning);
            let cur_text = model.decode_tokens(&node.text);

            if node.remaining.empty_of_letters() && cur_text.trim() == strip.punchline.trim() {
                let mut cur_text_alpha = cur_text.clone();
                let mut punchline_alpha = strip.punchline.clone();
                cur_text_alpha.retain(|c| c.is_alphabetic());
                punchline_alpha.retain(|c| c.is_alphabetic());

                if cur_text_alpha == punchline_alpha {
                    let msg = format!(
                        "Search successful after {} steps. Score: {:.2}%",
                        step,
                        p.0 * 100.0
                    );
                    success = true;
                    *report += &msg;
                    progress.abandon_with_message(msg);
                    break;
                } else {
                    let msg = format!(
                        "Non-answer: {} != {}",
                        model.decode_tokens(&node.text).trim(),
                        strip.punchline.trim()
                    );
                    *report += &msg;
                    *report += "\n";
                    progress.println(msg);
                }
            } else if strip.punchline.trim().starts_with(cur_text.trim()) && !node.text.is_empty() {
                score_progress_info += &format!(
                    "'{}' {:.3}%  ",
                    model.token_to_piece(*node.text.last().unwrap()),
                    p.0 * 100.0
                );
            }

            let cur_str = format!("{:.2}%: {}", p.0 * 100.0, cur_text);
            log.write_str(&format!("{}\n", cur_str)).unwrap();

            progress.set_message(format!("{:7} {}", step, cur_str));

            node.advance(&mut root_ctx, &mut q, &vocab, &rlnn);
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

    *report += &format!("Deepest node accessed: {deepest_node_accessed}\n");
    println!("Deepest node accessed: {deepest_node_accessed}");

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
