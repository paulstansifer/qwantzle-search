#![allow(dead_code)]

use std::{cell::Cell, cmp::Ordering, fs, io::Write as _, time::Duration};

use indicatif::ProgressBar;
use llama_cpp_2::{model::LlamaModel, token::LlamaToken};
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use thousands::Separable;

use crate::{
    corpus::Strip,
    llm::{self, str_to_tokens, tok_to_str, toks_to_str, Session, SessionTimers},
    pool::{LetterPool, Vocab, VocabBuilder, WordState},
    remaining_letters_neural_net::{self, LetterNet},
    TIME_TO_QUIT,
};

// Pretent to really be `Cmp`.
#[derive(PartialOrd, PartialEq, Clone, Copy, Serialize, Deserialize, Debug)]
struct Score(f64);

impl Eq for Score {}
impl Ord for Score {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Serialize, Deserialize, Clone)]
struct Node {
    text: Vec<i32>,
    word_state: WordState,
    remaining: LetterPool,
    probability: f64, // f32 is not quite precise enough!
    tok_probs: Vec<f32>,
    chars_so_far: u8,
    depth_at_pruning: u32,
}

impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for t in &self.text {
            state.write_i32(*t);
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Node) -> bool {
        return self.text == other.text;
    }
}
impl Eq for Node {}

#[derive(Serialize, Deserialize, Clone, Default)]
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
            if t_score > n_score {
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

#[derive(Serialize, Deserialize, Clone)]
pub struct Hints {
    pub context: String,
    pub goal: Option<String>,
    pub id: usize,
    pub letter_pool: LetterPool,
    pub vocab: Vocab,
    pub longest_word_len: Option<u8>,
    pub longest_token: Option<i32>, // instead of LLamaToken, for serializability
    pub token_size: usize,
}

impl Hints {
    pub fn from_strip(
        strip: &Strip,
        words: &Vec<String>,
        look_at_ties: bool,
        llm: &LlamaModel,
    ) -> Hints {
        let mut longest_word_len = 0;
        let vocab = {
            let mut v_builder = VocabBuilder::default();

            // If there are non-dictionary words, sneak 'em in:
            for word in regex::Regex::new(r"\b").unwrap().split(&strip.punchline) {
                v_builder.add_word(word, llm, /*vary_case=*/ false);
                if word.len() > longest_word_len {
                    longest_word_len = word.len();
                }
            }

            for word in words {
                // Simulate knowing that all words are shorter than the longest in the vocabulary.
                if word.len() >= longest_word_len {
                    continue;
                }
                v_builder.add_word(word, llm, /*vary_case=*/ true);
            }

            v_builder.build(/*disabled=*/ false, /*enforce_8_11=*/ false)
        };

        let toks = llm::str_to_tokens(&strip.punchline, llm);
        let mut longest_tok_len = 0;
        let mut longest_tok = toks[0];
        for t in toks {
            let t_len = llm::tok_to_str(t, llm).len();
            if t_len > longest_tok_len {
                longest_tok_len = t_len;
                longest_tok = t;
            }
        }

        let mut letter_pool = LetterPool::from_text(&strip.punchline, look_at_ties);
        letter_pool.set_longest_tok_from(&strip.punchline, llm);

        Hints {
            context: strip.context(),
            id: strip.id,
            goal: Some(strip.punchline.clone()),
            letter_pool: letter_pool,
            vocab,
            longest_word_len: Some(longest_word_len as u8),
            longest_token: Some(longest_tok.0),
            token_size: str_to_tokens(&strip.context(), llm).len() + TOK_BUFFER as usize,
        }
    }

    pub fn for_1663(words: &Vec<String>, look_at_ties: bool, llm: &LlamaModel) -> Hints {
        let vocab = {
            let mut v_builder = VocabBuilder::default();
            for word in words {
                if word.len() <= 8 {
                    v_builder.add_word(word, llm, /*vary_case=*/ true);
                }
            }
            v_builder.add_word(":", llm, false);
            v_builder.add_word(",", llm, false);
            v_builder.add_word("!!", llm, false);
            v_builder.add_word("fundamental", llm, false);
            v_builder.build(/*disabled=*/ false, /*enforce_8_11=*/ true)
        };

        let long_word_toks = llm::str_to_tokens(" fundamental", llm);
        if long_word_toks.len() != 1 {
            panic!("' fundamental' isn't one token");
        }

        let context = String::from_utf8(std::fs::read("corpus/1663-prefix.txt").unwrap()).unwrap();
        let token_size = str_to_tokens(&context, llm).len() + TOK_BUFFER as usize;
        let letters = "ttttttttttttooooooooooeeeeeeeeaaaaaaallllllnnnnnnuuuuuuiiiiisssssdddddhhhhhyyyyyIIrrrfffbbwwkcmvg:,!!";
        let mut letter_pool = LetterPool::just_letters_from_text(letters);
        letter_pool.set_longest_tok(*long_word_toks.first().unwrap(), llm);
        letter_pool.set_last_letter(b'w');
        if look_at_ties {
            letter_pool.set_ties(letters);
        }

        Hints {
            context: context,
            id: 1663,
            goal: None, //  Find this!!
            letter_pool: letter_pool,
            vocab,
            longest_word_len: Some(8),
            longest_token: Some(long_word_toks.first().unwrap().0),
            token_size: token_size,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]

struct HallOfFame {
    initial_nodes: Vec<String>,
    modulo_nodes: Vec<String>,
    high_score_nodes: PriorityQueue<String, Score>,
    high_score_tie_nodes: PriorityQueue<String, Score>,
    long_nodes: PriorityQueue<String, usize>,
    possible_completions: Vec<String>,
}

pub struct SearchState<'a> {
    q: Q,

    llm: &'a LlamaModel,
    rlnn: LetterNet,
    sess: llm::Session<'a>,

    hints: Hints,

    search_time: Duration,
    // Total probability mass that doesn't fit the formal restrictions:
    discard_prob_letters: f64,
    // Total probability mass that was too unlikely and got dropped:
    discard_prob_dregs: f64,
    step: usize,
    max_search: Option<usize>,
    deepest_node_accessed: u32,
    report: String,
    hall_of_fame: HallOfFame,
    progress: ProgressBar,
}

#[derive(Serialize, Deserialize)]
struct SearchStateSerializable {
    q: Q,
    sess_timers: SessionTimers,
    hints: Hints,
    search_time: Duration,
    #[serde(default)]
    discard_prob_letters: f64,
    #[serde(default)]
    discard_prob_dregs: f64,
    step: usize,
    deepest_node_accessed: u32,
    report: String,
    max_search: Option<usize>,
    hall_of_fame: HallOfFame,
}

impl SearchState<'_> {
    pub fn new(
        llm: &LlamaModel,
        hints: Hints,
        max_search: Option<usize>,
        start_prefixes: Vec<String>,
    ) -> SearchState {
        let token_size = hints.token_size as u32;

        let q = Q::new();
        let rlnn = remaining_letters_neural_net::LetterNet::new_from_file(
            "corpus/letter_pool.safetensors",
        )
        .unwrap();

        let root_node = Node::root_from_hints(&hints);
        let mut sess = Session::new(llm, token_size);

        // TODO: this is ad-hoc; we should at least make this configurable (and not have to be
        // synced between this and `load`)
        let colon_tok = *llm::str_to_tokens("totally:", llm).last().unwrap();
        sess.boost(colon_tok, 35.0);

        println!("Colon is boosted!");

        let first_candidates = sess.advance_and_predict_str(&hints.context, Some(TOK_TOP_P));
        sess.save_prompt();

        let mut search_state = SearchState {
            q,
            llm,
            rlnn,

            hints,

            sess: sess,
            search_time: Duration::from_nanos(0),
            discard_prob_letters: 0.0,
            discard_prob_dregs: 0.0,
            step: 0,
            deepest_node_accessed: 0,
            report: String::new(),
            max_search: max_search,
            hall_of_fame: HallOfFame::default(),

            progress: ProgressBar::new_spinner(),
        };

        if start_prefixes.is_empty() {
            root_node.consider_candidates(first_candidates, &mut search_state);
        } else {
            for prefix in start_prefixes {
                if prefix.trim().is_empty() || prefix.contains("#") {
                    continue;
                }
                let mut pfx_node = root_node.clone();
                let mut score = Score(1.0);
                for tok in llm::str_to_tokens(&prefix.trim_end(), llm) {
                    if let Some((next_node, next_score)) = pfx_node.push_token(
                        tok,
                        0.12,
                        llm,
                        &search_state.hints.vocab,
                        &search_state.rlnn,
                    ) {
                        pfx_node = next_node;
                        score = next_score;
                    } else {
                        panic!("Invalid prefix: {}", prefix);
                    }
                }
                println!("(Score {:.5}%) --> {:?} ", score.0 * 100.0, pfx_node.text);
                search_state.q.push(pfx_node, score);
            }
            println!("{}", search_state.q.len());
        }

        search_state
    }

    pub fn save(&self, filename: &str) {
        let sss = SearchStateSerializable {
            q: self.q.clone(),
            sess_timers: self.sess.timers,
            hints: self.hints.clone(),
            search_time: self.search_time,
            discard_prob_letters: self.discard_prob_letters,
            discard_prob_dregs: self.discard_prob_dregs,
            step: self.step,
            deepest_node_accessed: self.deepest_node_accessed,
            report: self.report.clone(),
            max_search: self.max_search,
            hall_of_fame: self.hall_of_fame.clone(),
        };

        let bytes = bincode::serialize(&sss).unwrap();
        fs::write(filename, &bytes).unwrap();
    }

    pub fn load<'a>(filename: &str, llm: &'a LlamaModel) -> SearchState<'a> {
        let bytes = fs::read(filename).unwrap();
        let sss: SearchStateSerializable = bincode::deserialize(&bytes).unwrap();
        let token_size = sss.hints.token_size as u32;

        println!("Loaded search queue with {} elements.", sss.q.len());

        let mut sess = Session::new(llm, token_size);

        let colon_tok = *llm::str_to_tokens("totally:", llm).last().unwrap();
        sess.boost(colon_tok, 35.0);

        // TODO: should probably have `advance` without `predict`
        let _ = sess.advance_and_predict_str(&sss.hints.context, Some(0.0));
        sess.save_prompt();
        sess.timers = sss.sess_timers;

        SearchState {
            q: sss.q,
            llm: llm,
            rlnn: LetterNet::new_from_file("corpus/letter_pool.safetensors").unwrap(),

            hints: sss.hints,

            sess: sess,
            search_time: sss.search_time,
            discard_prob_letters: sss.discard_prob_letters,
            discard_prob_dregs: sss.discard_prob_dregs,
            step: sss.step,
            report: sss.report,
            deepest_node_accessed: sss.deepest_node_accessed,
            max_search: sss.max_search,
            hall_of_fame: sss.hall_of_fame,

            progress: ProgressBar::new_spinner(),
        }
    }

    fn time_report(&mut self) {
        let per_step = |n: u128| (n as f32 / self.step as f32) / 1000.0;
        let report = format!("Search time: {:.0}s.  (Non-ML: {:.0}s)  Per-step times: Score: {:.0}ms  Advance: {:.0}ms  Predict: {:.0}ms  Truncate: {:.0}ms.",
            self.search_time.as_secs(),
            (self.search_time.as_micros() - (self.sess.timers.score_time + self.sess.timers.advance_time + self.sess.timers.predict_time + self.sess.timers.truncate_time)) as f32 / 1_000_000.0,
            per_step(self.sess.timers.score_time),
            per_step(self.sess.timers.advance_time),
            per_step(self.sess.timers.predict_time),
            per_step(self.sess.timers.truncate_time),
        );
        self.log_ln(&report);
    }

    fn log_ln(&mut self, s: &str) {
        self.report += s;
        self.report += "\n";
        self.progress.println(s);
    }

    fn full_string_complete(&mut self, text: &str, score: Score) -> bool {
        match &self.hints.goal {
            None => {
                let desc = format!("==> '{}!!' ({:.2}%)", text, score.0 * 100.0);
                self.log_ln("!!!!!!!!!!!!!!!!!!!!!!!!!!!");
                self.log_ln(&desc);
                self.hall_of_fame.possible_completions.push(desc);
                let mut file = fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/home/paul/qwantzle_completions.txt")
                    .unwrap();
                let _ = writeln!(&mut file, "{}!! ({:.2}%)", text, score.0 * 100.0);

                false // keep going!
            }
            Some(goal_text) => {
                let mut text_alpha = text.to_owned();
                let mut goal_alpha = goal_text.clone();
                text_alpha.retain(|c| c.is_alphabetic());
                goal_alpha.retain(|c| c.is_alphabetic());

                if text_alpha == goal_alpha {
                    self.log_ln(&format!("'{}' ({:.2}%)", text, score.0 * 100.0));
                    self.log_ln(&format!(
                        "Search successful after {} steps. Deepest node accessed: {}",
                        self.step.separate_with_commas(),
                        self.deepest_node_accessed.separate_with_commas()
                    ));

                    true
                } else {
                    self.log_ln(&format!(
                        "Non-answer: {} ({:.2}%) != {}",
                        text,
                        score.0 * 100.0,
                        goal_text
                    ));
                    false
                }
            }
        }
    }

    fn hof_update(&mut self, desc: String, len: usize, p: Score, respects_ties: bool) {
        let initial_node_limit = if self.hints.goal.is_none() {
            2_000
        } else {
            500_000
        };

        if self.hall_of_fame.initial_nodes.len() < initial_node_limit {
            self.hall_of_fame.initial_nodes.push(desc.clone());

            if self.step == 500
                || self.step == 1_000
                || self.step == 2_000
                || self.step % 100_000 == 0
            {
                std::fs::write("/tmp/hof-init", self.hall_of_fame.initial_nodes.join("\n"))
                    .unwrap();
            }
        }

        if self.step % 500 == 0 {
            self.hall_of_fame.modulo_nodes.push(format!(
                "{}   - {}",
                desc.clone(),
                self.deepest_node_accessed
            ));
        }

        self.hall_of_fame.high_score_nodes.push(desc.clone(), p);
        if respects_ties {
            self.hall_of_fame.high_score_tie_nodes.push(desc.clone(), p);
        }
        self.hall_of_fame.long_nodes.push(desc, len);

        if self.hall_of_fame.high_score_nodes.len() > 20_000 || self.step % 10_000 == 0 {
            self.hall_of_fame.high_score_nodes = PriorityQueue::<String, Score>::from_iter(
                self.hall_of_fame
                    .high_score_nodes
                    .clone()
                    .into_sorted_iter()
                    .take(5_000),
            );
            self.hall_of_fame.high_score_tie_nodes = PriorityQueue::<String, Score>::from_iter(
                self.hall_of_fame
                    .high_score_tie_nodes
                    .clone()
                    .into_sorted_iter()
                    .take(5_000),
            );
            self.hall_of_fame.long_nodes = PriorityQueue::<String, usize>::from_iter(
                self.hall_of_fame
                    .long_nodes
                    .clone()
                    .into_sorted_iter()
                    .take(5_000),
            );
        }

        if self.step % 10_000 == 0 {
            std::fs::write("/tmp/hof-mod", self.hall_of_fame.modulo_nodes.join("\n")).unwrap();

            std::fs::write(
                "/tmp/hof-best",
                format!(
                    "{}\n====\n{}",
                    self.hall_of_fame.possible_completions.join("\n---\n"),
                    self.hall_of_fame
                        .high_score_nodes
                        .clone()
                        .into_sorted_vec()
                        .join("\n")
                ),
            )
            .unwrap();
            std::fs::write(
                "/tmp/hof-best-ties",
                self.hall_of_fame
                    .high_score_tie_nodes
                    .clone()
                    .into_sorted_vec()
                    .join("\n"),
            )
            .unwrap();
            std::fs::write(
                "/tmp/hof-long",
                self.hall_of_fame
                    .long_nodes
                    .clone()
                    .into_sorted_vec()
                    .join("\n"),
            )
            .unwrap();
        }
    }

    fn search_step(&mut self) -> bool {
        self.progress.tick();
        let step_start = std::time::Instant::now();

        if let Some((node, p)) = self
            .q
            .pop(/*require_tie_respecting=*/ (self.step / 100) % 8 == 0)
        {
            self.deepest_node_accessed =
                std::cmp::max(self.deepest_node_accessed, node.depth_at_pruning);
            let cur_text = toks_to_str(&node.tokens(), &self.llm);

            if node.remaining.empty_of_letters() {
                if self.full_string_complete(&cur_text, p) {
                    return true;
                }
            }

            let desc = format!(
                "{:>10} {}{} {:.5}% ({:.4}%/{:.4}%)  {:<125} {}",
                self.step.separate_with_commas(),
                self.hall_of_fame.possible_completions.len(),
                if node.remaining.respects_ties() {
                    "T"
                } else {
                    "*"
                },
                p.0 * 100.0,
                self.discard_prob_letters * 100.0,
                self.discard_prob_dregs * 100.0,
                cur_text,
                node.remaining.print()
            );

            self.progress.set_message(desc.clone());

            self.hof_update(
                desc,
                101 - node.remaining.size(),
                p,
                node.remaining.respects_ties(),
            );

            node.advance(self);
            self.step += 1;
        } else {
            let msg = format!(
                "Exhausted all reasonable nodes at {} steps",
                self.step.separate_with_commas()
            );
            self.log_ln(&msg);

            self.progress.abandon_with_message(msg);
            return true;
        }

        let quit_now = TIME_TO_QUIT.load(std::sync::atomic::Ordering::SeqCst);

        if self.q.len() > 5_000_000 || quit_now {
            self.q = std::mem::take(&mut self.q).trim(2_000_000);
        }

        self.search_time += step_start.elapsed();

        // Only save for 1663.
        if self.hints.goal.is_none() {
            if quit_now || self.step % 1_000_000 == 0 {
                self.time_report();
                self.progress.set_message("Saving...");
                let filename = format!(
                    "/home/paul/.qwantzle/{}-{}.search",
                    self.hints.id,
                    if quit_now {
                        "in_progress"
                    } else {
                        "checkpoint"
                    }
                );
                self.save(&filename);
                self.progress.set_message(format!("Saved to {}", filename));
            }
        }

        if quit_now {
            self.progress.abandon();
            return true;
        }

        // Have we reached the end?
        if let Some(limit) = self.max_search {
            self.step >= limit
        } else {
            false
        }
    }

    pub fn search(&mut self) {
        while !self.search_step() {}
        self.time_report();
    }
}

const TOK_BUFFER: u32 = 35;

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

// TODO: remove most of these
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

#[derive(clap::Parser, Debug)]
pub struct ScoreArgs {
    /// Exponent to apply to the probability from the RLNN
    #[arg(long, default_value("1.0"))]
    pub rlnn_strength: f32,

    /// By default, we apply an approximate adjustment that gives credit to longer strings
    /// for the fact that they've survived this far without being invalidated
    /// by the anagram constraint; this turns that off.
    ///
    /// There's presumably no need to adjust the strength of this, because it
    /// follows the same curve as the bonus_exponent.
    #[arg(long, action = clap::ArgAction::SetTrue)]
    pub no_filter_strenth: bool,

    /// An exponential bonus for length (value at the beginning of the sequence)
    /// (For backwards-compatibility, this is measured per 4 characters, even though )
    #[arg(long, default_value("4.5"))]
    pub bonus_exponent_initial: f32,

    /// An exponential bonus for length (value at character 100 (the "end"))
    #[arg(long, default_value("4.5"))]
    pub bonus_exponent_final: f32,
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
        let filter_ratio: f32 = 0.05 + 0.55 * f32::max((80.0 - chars_i) / 80.0, 0.75);

        // This is unmotivated; purely empirical. Starts at 5.0, goes towards 4.0:
        //let len_bonus = f32::max(0.0, ((100.0 - chars_i) / 100.0) * 1.0) + 3.0;
        let len_bonus = 4.5;

        prod = f32::powf((1.0 + len_bonus) / filter_ratio, 0.25) as f64 * prod;

        chars_i += 1.0;
    }

    Score(prod * rlnn_mult as f64)
}

impl Node {
    fn root_from_hints(hints: &Hints) -> Node {
        Node {
            text: vec![],
            word_state: WordState::new_empty(),
            remaining: hints.letter_pool.clone(),
            probability: 1.0,
            tok_probs: vec![],
            chars_so_far: 0,
            depth_at_pruning: 0,
        }
    }

    // TODO: get rid of this one
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

    fn tokens(&self) -> Vec<LlamaToken> {
        self.text.iter().map(|t| LlamaToken(*t)).collect()
    }

    fn new_with_longest_tok(text: &str, model: &LlamaModel) -> Node {
        let mut res = Node::new(text);
        res.remaining.set_longest_tok_from(text, model);
        res
    }

    fn push_token(
        &self,
        t: LlamaToken,
        prob: f32,
        model: &LlamaModel,
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
            chars_so_far: self.chars_so_far + tok_to_str(t, model).trim().len() as u8,
            depth_at_pruning: self.depth_at_pruning,
        };
        res.text.push(t.0);
        res.tok_probs.push(prob);

        let score = prob_score(&res.tok_probs, res.chars_so_far, rlnn_prob);

        Some((res, score))
    }

    fn advance(&self, search_state: &mut SearchState) {
        if self.text.len() >= (TOK_BUFFER - 1) as usize {
            return;
        }

        let candidates = search_state
            .sess
            .advance_and_predict(&self.tokens(), Some(TOK_TOP_P));

        self.consider_candidates(candidates, search_state);

        search_state.sess.truncate_to_prompt();
    }

    fn consider_candidates(
        &self,
        candidates: Vec<(LlamaToken, f64)>,
        search_state: &mut SearchState,
    ) {
        let mut fast_forwarded = true; // TODO: why does fast-forwarding cause NoKvCacheSlot?

        let mut remaining_prob: f64 = 1.0;

        for (tok, p) in candidates {
            let avg_prob = f64::powf(
                p as f64 * self.probability,
                1.0 / (self.text.len() as f64 + 1.0),
            );

            if avg_prob < MIN_AVG_P && self.text.len() >= 2 {
                search_state.discard_prob_dregs += self.probability * remaining_prob;
                break;
            }

            remaining_prob -= p;

            if let Some((next_node, score)) = self.push_token(
                tok,
                p as f32,
                &search_state.sess.model(),
                &search_state.hints.vocab,
                &search_state.rlnn,
            ) {
                // Re-use the context right away, if the best candidate is good enough.
                if p as f32 > FF_MIN_P && !fast_forwarded && next_node.remaining.size() > 0 {
                    let before_advance = std::time::Instant::now();
                    let ff_candidates = search_state
                        .sess
                        .advance_and_predict(&[tok], Some(TOK_TOP_P));
                    FF_ADVANCE_TIME
                        .replace(FF_ADVANCE_TIME.get() + before_advance.elapsed().as_micros());
                    next_node.consider_candidates(ff_candidates, search_state);
                    fast_forwarded = true;
                } else {
                    search_state.q.push(next_node, score);
                }
            } else {
                search_state.discard_prob_letters += self.probability * p;
            }
        }
    }
}

pub struct SearchResult {
    pub found: bool,
    pub steps: usize,
    pub seconds: f32,
    pub prob_letters: f64,
    pub prob_dregs: f64,
}

pub fn practice_search(
    strip: &Strip,
    model: &LlamaModel,
    words: &Vec<String>,
    steps_limit: Option<usize>,
    report: &mut String,
) -> SearchResult {
    let strip_summary = format!(
        "...{} >>{}<<\n",
        strip
            .context()
            .chars()
            .skip(strip.context().len() - 50)
            .collect::<String>()
            .replace("\n", "  "),
        strip.punchline
    );

    print!("{}", strip_summary);
    *report += &strip_summary;

    let hints = if strip.id == 1663 {
        Hints::for_1663(&words, /*look_at_ties=*/ false, &model)
    } else {
        Hints::from_strip(strip, &words, /*look_at_ties=*/ false, &model)
    };

    let mut search = SearchState::new(&model, hints, steps_limit, vec![]);
    search.search();

    *report += &format!("Deepest node accessed: {}\n", search.deepest_node_accessed);
    println!("Deepest node accessed: {}", search.deepest_node_accessed);

    std::fs::write(
        format!("reports/search-logs/search-{}.txt", strip.id),
        format!(
            "{}\n{}",
            strip.punchline,
            search.hall_of_fame.initial_nodes.join("\n")
        ),
    )
    .unwrap();

    return SearchResult {
        found: search.report.contains("Search successful after"), // hack, but it works!
        steps: search.step,
        seconds: search.search_time.as_secs_f32(),
        prob_letters: search.discard_prob_letters,
        prob_dregs: search.discard_prob_dregs,
    };
}

pub fn manual_search(llm: &LlamaModel, hints: Hints, prefix: &str) {
    let rlnn =
        remaining_letters_neural_net::LetterNet::new_from_file("corpus/letter_pool.safetensors")
            .unwrap();

    let mut sess = Session::new(llm, hints.token_size as u32);

    let mut node = Node::root_from_hints(&hints);

    let mut cands = sess.advance_and_predict_str(&hints.context, None);

    let spaced_prefix = format!(" {}", prefix.trim());
    for tok in str_to_tokens(&spaced_prefix, llm) {
        print!("'{}' ", tok_to_str(tok, llm));

        let mut p = 0.0;
        for (possible_tok, possible_p) in &cands {
            if *possible_tok == tok {
                p = *possible_p;
                break;
            }
        }

        if let Some((next_node, score)) = node.push_token(tok, p as f32, llm, &hints.vocab, &rlnn) {
            print!(
                "({}{:.3}%) ",
                if next_node.remaining.respects_ties() {
                    "T"
                } else {
                    ""
                },
                score.0 * 100.0
            );
            node = next_node;
        } else if p > 0.0 {
            if node.remaining.try_remove(tok, llm).is_none() {
                print!("(pool-invalid '{}'", node.remaining.print());
            } else if node.word_state.add_tok(tok, &hints.vocab).is_none() {
                print!("(word-invalid ");
            } else {
                print!("(can't happen!? ");
            }

            print!(" {:.2}%)", p * 100.0);
        } else {
            print!("(unavailable token??)");
        }
        cands = sess.advance_and_predict(&[tok], None);
    }

    println!();

    let mut ok_s = String::new();
    let mut not_ok_s = String::new();

    for (tok, p) in cands.iter().take(60) {
        let node_or = node.push_token(*tok, *p as f32, llm, &hints.vocab, &rlnn);
        if let Some((possible_node, score)) = node_or {
            ok_s += &format!(
                "'{}' {:.2}%->{}{:.3}%  ",
                tok_to_str(*tok, llm),
                p * 100.0,
                if possible_node.remaining.respects_ties() {
                    "T"
                } else {
                    ""
                },
                score.0 * 100.0
            );
        } else {
            not_ok_s += &format!("'{}' {:.2}%  ", tok_to_str(*tok, llm), p * 100.0);
        }
    }

    println!("{}\n{}", ok_s, not_ok_s);
}
