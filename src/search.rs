#![allow(dead_code)]

use std::{
    cell::Cell,
    cmp::Ordering,
    fmt::Debug,
    fs,
    io::Write as _,
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};

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
    pub leadup: String,
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
            leadup: strip.leadup.clone(),
            id: strip.id,
            goal: Some(strip.punchline.clone()),
            letter_pool: letter_pool,
            vocab,
            longest_word_len: Some(longest_word_len as u8),
            longest_token: Some(longest_tok.0),
            token_size: str_to_tokens(&strip.leadup, llm).len() + TOK_BUFFER as usize,
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

        let leadup = String::from_utf8(std::fs::read("corpus/1663-prefix.txt").unwrap()).unwrap();
        let token_size = str_to_tokens(&leadup, llm).len() + TOK_BUFFER as usize;
        let letters = "ttttttttttttooooooooooeeeeeeeeaaaaaaallllllnnnnnnuuuuuuiiiiisssssdddddhhhhhyyyyyIIrrrfffbbwwkcmvg:,!!";
        let mut letter_pool = LetterPool::just_letters_from_text(letters);
        letter_pool.set_longest_tok(*long_word_toks.first().unwrap(), llm);
        letter_pool.set_last_letter(b'w');
        if look_at_ties {
            letter_pool.set_ties(letters);
        }

        Hints {
            leadup: leadup,
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

    hints: Hints,

    search_time: Duration,
    step: usize,
    max_search: Option<usize>,
    deepest_node_accessed: u32,
    report: String,
    hall_of_fame: HallOfFame,
    progress: ProgressBar,
}

impl<'a> Debug for SearchState<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchState")
            .field("q [size]", &self.q.len())
            .field("llm", &self.llm)
            .field("search_time", &self.search_time)
            .field("step", &self.step)
            .field("max_search", &self.max_search)
            .field("deepest_node_accessed", &self.deepest_node_accessed)
            .field("report", &self.report)
            .field("progress", &self.progress)
            .finish()
    }
}

#[derive(Serialize, Deserialize)]
struct SearchStateSerializable {
    q: Q,
    sess_timers: SessionTimers,
    hints: Hints,
    search_time: Duration,
    step: usize,
    deepest_node_accessed: u32,
    report: String,
    max_search: Option<usize>,
    hall_of_fame: HallOfFame,
}

struct SearchThreadLocal<'a> {
    sess: llm::Session<'a>,
    hints: Hints, // Immutable, so let's only copy it at the start
    rlnn: LetterNet,
}

impl SearchState<'_> {
    pub fn new(
        llm: &LlamaModel,
        hints: Hints,
        max_search: Option<usize>,
        start_prefixes: Vec<String>,
    ) -> SearchState {
        let token_size = hints.token_size as u32;

        let mut q = Q::new();
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

        let first_candidates = sess.advance_and_predict_str(&hints.leadup, Some(TOK_TOP_P));
        sess.save_prompt();

        if start_prefixes.is_empty() {
            for (node, score) in
                root_node.consider_candidates(first_candidates, &mut sess, &hints.vocab, &rlnn)
            {
                q.push(node, score);
            }
        } else {
            for prefix in start_prefixes {
                if prefix.trim().is_empty() || prefix.contains("#") {
                    continue;
                }
                let mut pfx_node = root_node.clone();
                let mut score = Score(1.0);
                for tok in llm::str_to_tokens(&prefix.trim_end(), llm) {
                    if let Some((next_node, next_score)) =
                        pfx_node.push_token(tok, 0.12, llm, &hints.vocab, &rlnn)
                    {
                        pfx_node = next_node;
                        score = next_score;
                    } else {
                        panic!("Invalid prefix: {}", prefix);
                    }
                }
                println!("(Score {:.5}%) --> {:?} ", score.0 * 100.0, pfx_node.text);
                q.push(pfx_node, score);
            }
            println!("{}", q.len());
        }

        SearchState {
            q,
            llm,
            rlnn,

            hints,

            search_time: Duration::from_nanos(0),
            step: 0,
            deepest_node_accessed: 0,
            report: String::new(),
            max_search: max_search,
            hall_of_fame: HallOfFame::default(),

            progress: ProgressBar::new_spinner(),
        }
    }

    pub fn save(&self, quit_now: bool) {
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

        let sss = SearchStateSerializable {
            q: self.q.clone(),
            sess_timers: SessionTimers::default(), // TODO; don't save these at all!
            hints: self.hints.clone(),
            search_time: self.search_time,
            step: self.step,
            deepest_node_accessed: self.deepest_node_accessed,
            report: self.report.clone(),
            max_search: self.max_search,
            hall_of_fame: self.hall_of_fame.clone(),
        };

        let bytes = bincode::serialize(&sss).unwrap();
        fs::write(&filename, &bytes).unwrap();
        self.progress.set_message(format!("Saved to {}", filename));
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
        let _ = sess.advance_and_predict_str(&sss.hints.leadup, Some(0.0));
        sess.save_prompt();
        sess.timers = sss.sess_timers;

        SearchState {
            q: sss.q,
            llm: llm,
            rlnn: LetterNet::new_from_file("corpus/letter_pool.safetensors").unwrap(),

            hints: sss.hints,

            search_time: sss.search_time,
            step: sss.step,
            report: sss.report,
            deepest_node_accessed: sss.deepest_node_accessed,
            max_search: sss.max_search,
            hall_of_fame: sss.hall_of_fame,

            progress: ProgressBar::new_spinner(),
        }
    }

    fn time_report(&mut self, sess_timers: &SessionTimers) {
        let per_step = |n: u128| (n as f32 / self.step as f32) / 1000.0;
        let report = format!("Search time: {:.0}s.  (Non-ML: {:.0}s)  Per-step times: Score: {:.0}ms  Advance: {:.0}ms  Predict: {:.0}ms  Truncate: {:.0}ms.",
            self.search_time.as_secs(),
            (self.search_time.as_micros() - (sess_timers.score_time + sess_timers.advance_time + sess_timers.predict_time + sess_timers.truncate_time)) as f32 / 1_000_000.0,
            per_step(sess_timers.score_time),
            per_step(sess_timers.advance_time),
            per_step(sess_timers.predict_time),
            per_step(sess_timers.truncate_time),
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

    fn process_node(state: Arc<RwLock<SearchState>>, local: &mut SearchThreadLocal) -> bool {
        let (node, p, step) = {
            let mut write_state = state.write().unwrap();
            write_state.step += 1;

            let require_tie_respecting = (write_state.step / 100) % 8 == 0;

            let (node, p) = match write_state.q.pop(require_tie_respecting) {
                Some(x) => x,
                None => {
                    let msg = format!(
                        "Exhausted all reasonable nodes at {} steps",
                        write_state.step.separate_with_commas()
                    );
                    write_state.log_ln(&msg);

                    write_state.progress.abandon_with_message(msg);
                    return true;
                }
            };

            let cur_text = toks_to_str(&node.tokens(), &write_state.llm);

            let desc = format!(
                "{:>10} {}{} {:.5}% {:<125} {}",
                write_state.step.separate_with_commas(),
                write_state.hall_of_fame.possible_completions.len(),
                if node.remaining.respects_ties() {
                    "T"
                } else {
                    "*"
                },
                p.0 * 100.0,
                cur_text,
                node.remaining.print()
            );

            write_state.progress.set_message(desc.clone());

            write_state.hof_update(
                desc,
                101 - node.remaining.size(),
                p,
                node.remaining.respects_ties(),
            );
            write_state.deepest_node_accessed =
                std::cmp::max(write_state.deepest_node_accessed, node.depth_at_pruning);

            (node, p, write_state.step)
        };
        // We have to use `step` after this point, because `write_state.step` may have changed.

        if node.remaining.empty_of_letters() {
            let mut write_state = state.write().unwrap();
            let full_string = &toks_to_str(&node.tokens(), &write_state.llm);
            if write_state.full_string_complete(full_string, p) {
                // HACK to make the other threads quit:
                TIME_TO_QUIT.store(true, std::sync::atomic::Ordering::SeqCst);
                return true;
            }
        }

        let new_nodes = node.advance(&mut local.sess, &local.hints.vocab, &local.rlnn);

        {
            let mut write_state = state.write().unwrap();

            for (new_node, score) in new_nodes {
                write_state.q.push(new_node, score);
            }

            let quit_now = TIME_TO_QUIT.load(std::sync::atomic::Ordering::SeqCst)
                || write_state.max_search.map(|l| step >= l).unwrap_or(false);

            if write_state.q.len() > Q_MAX_LEN || quit_now {
                write_state.q = std::mem::take(&mut write_state.q).trim(Q_MIN_LEN);
            }

            if step % 1_000_000 == 0 {
                // This will lose nodes in flight in other threads, which is not ideal. Ctrl-C is
                // safe, though.
                write_state.save(/*quit_now=*/ false);
            }

            return quit_now;
        }
    }

    pub fn search(self, num_threads: usize) -> Self {
        self.progress.set_message("Beginning search...");
        let state = Arc::new(RwLock::new(self));

        let search_time_start = std::time::Instant::now();
        let total_timer = Arc::new(Mutex::new(SessionTimers::default()));

        let state_copy = state.clone();

        std::thread::scope(move |scope| {
            let mut threads = vec![];

            for _ in 0..num_threads {
                let state = state.clone();
                let total_timer = total_timer.clone();
                threads.push(scope.spawn(move || {
                    let mut thread_local = {
                        let read_state = state.read().unwrap();
                        SearchThreadLocal {
                            sess: Session::new_with_prompt(
                                read_state.llm,
                                read_state.hints.token_size as u32,
                                &read_state.hints.leadup,
                            ),
                            hints: read_state.hints.clone(),
                            rlnn: read_state.rlnn.clone(),
                        }
                    };

                    while !SearchState::process_node(state.clone(), &mut thread_local) {} // Search!

                    total_timer.lock().unwrap().add(&thread_local.sess.timers);
                }));
            }

            for t in threads {
                t.join().unwrap();
            }

            let mut state_write = state.write().unwrap();
            state_write.search_time += search_time_start.elapsed();
            state_write.time_report(&total_timer.lock().unwrap());

            state_write.save(/*quit_now=*/ true);
        });

        Arc::try_unwrap(state_copy).unwrap().into_inner().unwrap()
    }
}

const TOK_BUFFER: u32 = 35;

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

const Q_MAX_LEN: usize = 5_000_000;
const Q_MIN_LEN: usize = 2_000_000;

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
        let len_bonus = f32::max(0.0, ((100.0 - chars_i) / 100.0) * 1.0) + 3.0;

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

    fn advance(
        &self,
        root_sess: &mut Session,
        vocab: &Vocab,
        rlnn: &LetterNet,
    ) -> Vec<(Node, Score)> {
        if self.text.len() >= (TOK_BUFFER - 1) as usize {
            return vec![];
        }

        let candidates = root_sess.advance_and_predict(&self.tokens(), Some(TOK_TOP_P));

        let res = self.consider_candidates(candidates, root_sess, vocab, rlnn);

        root_sess.truncate_to_prompt();

        res
    }

    fn consider_candidates(
        &self,
        candidates: Vec<(LlamaToken, f64)>,
        sess: &Session,
        vocab: &Vocab,
        rlnn: &LetterNet,
    ) -> Vec<(Node, Score)> {
        let mut res = vec![];
        for (tok, p) in candidates {
            let avg_prob = f64::powf(
                p as f64 * self.probability,
                1.0 / (self.text.len() as f64 + 1.0),
            );

            if avg_prob < MIN_AVG_P && self.text.len() >= 2 {
                break;
            }

            if let Some(node_and_score) = self.push_token(tok, p as f32, &sess.model(), vocab, rlnn)
            {
                res.push(node_and_score);
            }
        }
        res
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

    let hints = if strip.id == 1663 {
        Hints::for_1663(&words, /*look_at_ties=*/ false, &model)
    } else {
        Hints::from_strip(strip, &words, /*look_at_ties=*/ false, &model)
    };

    let search = SearchState::new(&model, hints, steps_limit, vec![]);
    let search = search.search(/*threads=*/ 3);

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
    };
}

pub fn manual_search(llm: &LlamaModel, hints: Hints, prefix: &str) {
    let rlnn =
        remaining_letters_neural_net::LetterNet::new_from_file("corpus/letter_pool.safetensors")
            .unwrap();

    let mut sess = Session::new(llm, hints.token_size as u32);

    let mut node = Node::root_from_hints(&hints);

    let mut cands = sess.advance_and_predict_str(&hints.leadup, None);

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
