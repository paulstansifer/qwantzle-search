#![allow(dead_code)]

use assoc::AssocExt;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

use crate::llm;

#[derive(PartialEq, Eq, Hash, Clone, Copy, Deserialize, Serialize, Debug)]
struct Char(u8);

impl Char {
    fn is_lc_letter(self) -> bool {
        self.0 >= b'a' && self.0 <= b'z'
    }
    fn is_uc_letter(self) -> bool {
        self.0 >= b'A' && self.0 <= b'Z'
    }
    fn is_letter(self) -> bool {
        self.is_lc_letter() || self.is_uc_letter()
    }
}

// TODO: no longer true; use it!
// Token doesn't implement Ord or Hash, so we use the underlying i32.
#[derive(Default)]
pub struct VocabBuilder {
    word_lengths: HashMap<Vec<i32>, u8>,
    allowed_tokens: HashSet<i32>,
    word_starters: HashSet<i32>,
    word_enders: HashSet<i32>,
    nonletters: HashSet<i32>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
enum TokenRole {
    StartsWord,
    EndsWord,
    Continues,
    NonLetter,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Vocab {
    valid_word_lengths: HashMap<Vec<i32>, u8>,
    valid_prefixes: HashSet<Vec<i32>>,
    token_roles: HashMap<i32, TokenRole>,
    enforce_8_11: bool,
    disabled: bool,
}

impl VocabBuilder {
    fn eval_token(&mut self, tok: i32, model: &LlamaModel) {
        if self.allowed_tokens.contains(&tok) {
            return; // already processed
        }
        self.allowed_tokens.insert(tok);

        let solo_token = llm::tok_to_str(LlamaToken(tok), model);

        if !regex::Regex::new(r"[a-zA-Z]")
            .unwrap()
            .is_match(&solo_token)
        {
            self.nonletters.insert(tok);
            return;
        }

        if solo_token.contains(" ") {
            if solo_token.starts_with(" ") {
                self.word_starters.insert(tok);
            } else if solo_token.ends_with(" ") {
                self.word_enders.insert(tok);
            } else {
                panic!("Space in the middle of a token?!?!?");
            }
        } else {
            let xes = llm::str_to_tokens("xxxxxxxxxxxxxxx", model);
            let mid_token = xes[xes.len() / 2];

            // Does the token add a space when attached to something?
            if llm::toks_to_str(&[mid_token, LlamaToken(tok)], model).contains(" ") {
                self.word_starters.insert(tok);
            }
            if llm::toks_to_str(&[LlamaToken(tok), mid_token], model).contains(" ") {
                self.word_enders.insert(tok);
            }
        }
    }

    pub fn add_word(&mut self, word: &str, model: &LlamaModel, vary_case: bool) {
        if word.is_empty() {
            return; // Titlecasing would get unhappy.
        }

        for word in [word.to_string(), format!(" {}", word), format!("@{}", word)] {
            let pop_front_tok = word.starts_with("@");
            self.add_literal_word(&word, model, pop_front_tok);

            if vary_case {
                self.add_literal_word(&word.to_ascii_lowercase(), model, pop_front_tok);
                self.add_literal_word(&word.to_ascii_uppercase(), model, pop_front_tok);

                let mut title_case = word.to_owned();
                unsafe {
                    title_case.as_bytes_mut()[0].make_ascii_uppercase();
                }
                self.add_literal_word(&title_case, model, pop_front_tok);
            }
        }
    }
    fn add_literal_word(&mut self, word: &str, model: &LlamaModel, pop_front_tok: bool) {
        let mut toks: Vec<i32> = llm::str_to_tokens(word, model)
            .into_iter()
            .map(|t| t.0)
            .collect();

        if pop_front_tok {
            toks.remove(0);
        }

        self.word_lengths
            .insert(toks.clone(), word.trim().len() as u8);

        for tok in toks {
            self.eval_token(tok, model);
        }
    }

    pub fn build(self, disabled: bool, enforce_8_11: bool) -> Vocab {
        let mut valid_prefixes = HashSet::new();
        for word in self.word_lengths.keys() {
            for i in 1..word.len() + 1 {
                valid_prefixes.insert(word[0..i].to_vec());
            }
        }
        Vocab {
            valid_word_lengths: self.word_lengths.clone(),
            valid_prefixes: valid_prefixes,
            token_roles: self
                .allowed_tokens
                .iter()
                .map(|t| {
                    if self.nonletters.contains(t) {
                        (*t, TokenRole::NonLetter)
                    } else if self.word_starters.contains(t) {
                        (*t, TokenRole::StartsWord)
                    } else if self.word_enders.contains(t) {
                        (*t, TokenRole::EndsWord)
                    } else {
                        (*t, TokenRole::Continues)
                    }
                })
                .collect(),
            disabled: disabled,
            enforce_8_11: enforce_8_11,
        }
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct WordState {
    cur_word: Vec<i32>,
    seen_8: bool,
    seen_11: bool,
}

impl WordState {
    pub fn new_empty() -> WordState {
        WordState {
            cur_word: vec![],
            seen_8: false,
            seen_11: false,
        }
    }

    fn finish_with_length(
        ws: &WordState,
        toks: Vec<i32>,
        len: u8,
        vocab: &Vocab,
    ) -> Option<WordState> {
        let mut res = WordState {
            cur_word: toks,
            seen_8: ws.seen_8,
            seen_11: ws.seen_11,
        };
        if !vocab.enforce_8_11 {
            return Some(res);
        }
        if len == 11 && ws.seen_11 || len == 8 && ws.seen_8 {
            return None; // Only one of each
        }
        if ws.seen_11 && ws.seen_8 {
            return Some(res); // Already done
        }
        if (ws.seen_11 || ws.seen_8) && len != 11 && len != 8 {
            return None; // Must be consecutive
        }

        res.seen_11 |= len == 11;
        res.seen_8 |= len == 8;
        Some(res)
    }

    pub fn add_tok(&self, tok: LlamaToken, voc: &Vocab) -> Option<WordState> {
        if voc.disabled {
            return Some(self.clone());
        }
        match voc.token_roles.get(&tok.0) {
            None => None,
            Some(TokenRole::StartsWord) => {
                let prev_word_len = if self.cur_word.is_empty() {
                    Some(0)
                } else {
                    voc.valid_word_lengths.get(&self.cur_word).cloned()
                };
                if let Some(len) = prev_word_len {
                    WordState::finish_with_length(self, vec![tok.0], len, voc)
                } else {
                    None
                }
            }
            Some(TokenRole::NonLetter) => {
                let prev_word_len = if self.cur_word.is_empty() {
                    Some(0)
                } else {
                    voc.valid_word_lengths.get(&self.cur_word).cloned()
                };
                if let Some(len) = prev_word_len {
                    WordState::finish_with_length(self, vec![], len, voc)
                } else {
                    None
                }
            }
            Some(TokenRole::EndsWord) => {
                let mut full_word = self.cur_word.clone();
                full_word.push(tok.0);

                if let Some(word_len) = voc.valid_word_lengths.get(&full_word) {
                    WordState::finish_with_length(self, full_word, *word_len, voc)
                } else {
                    None
                }
            }
            Some(TokenRole::Continues) => {
                let mut lengthened_word = self.cur_word.clone();
                lengthened_word.push(tok.0);
                if voc.valid_prefixes.contains(&lengthened_word) {
                    Some(WordState {
                        cur_word: lengthened_word,
                        seen_8: self.seen_8,
                        seen_11: self.seen_11,
                    })
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Deserialize, Serialize)]
pub struct LetterPool {
    lowercase: [u8; 26],
    other_chars: Vec<(Char, u8)>,
    long_tok: Option<i32>,
    last_letter: Option<Char>,
    tie_sequences: Option<Vec<Vec<Char>>>, // if we need to save memory, do Option<[u8; 5]>
}

#[derive(Default)]
struct PoolTok {
    /// Stored in order of first occurrence
    chars: Vec<(Char, u8)>,
}

thread_local! {
    pub static POOL_TOK_CACHE : RefCell<TokCache> = RefCell::<TokCache>::default();
}

static LOWERCASE: std::ops::Range<u8> = b'a'..b'z';

impl PoolTok {
    fn from_str(s: &str) -> PoolTok {
        let mut res = PoolTok::default();
        for b in s.as_bytes() {
            let c = Char(*b);

            if *b == b' ' {
                continue;
            }
            if res.chars.get(&c).is_none() {
                res.chars.push((c, 1));
            } else {
                *res.chars.get_mut(&c).unwrap() += 1;
            }
        }
        return res;
    }

    fn size(&self) -> u8 {
        let mut res = 0;
        for (_, count) in &self.chars {
            res += count;
        }
        return res;
    }
}

impl LetterPool {
    pub fn print(&self) -> String {
        let mut res = String::new();
        for i in 0..26 {
            if self.lowercase[i] > 0 {
                for _ in 0..self.lowercase[i] {
                    res.push((b'a' + i as u8) as char);
                }
            }
        }
        for (ch, count) in &self.other_chars {
            for _ in 0..*count {
                res.push(ch.0 as char);
            }
        }

        res
    }

    pub fn print_ties_info(&self) -> String {
        if let Some(ties) = &self.tie_sequences {
            let mut res = String::new();
            for tie in ties {
                for ch in tie {
                    res.push(ch.0 as char);
                }
                res.push_str(" ");
            }
            return res;
        } else {
            return "No ties".to_string();
        }
    }

    // TODO: use this, rather than `size`, to terminate search.
    pub fn empty_of_letters(&self) -> bool {
        return self.long_tok == None && self.letter_size() == 0;
    }

    pub fn respects_ties(&self) -> bool {
        self.tie_sequences.is_some()
    }

    pub fn size(&self) -> usize {
        let mut res: usize = 0;
        for count in self.lowercase {
            res += count as usize;
        }
        for (_, count) in &self.other_chars {
            res += *count as usize;
        }
        return res;
    }

    // Number of letters remaining
    pub fn letter_size(&self) -> usize {
        let mut res: usize = 0;
        for count in self.lowercase {
            res += count as usize;
        }
        for (ch, count) in &self.other_chars {
            if ch.is_uc_letter() {
                res += *count as usize;
            }
        }
        return res;
    }

    fn index_of(c: Char) -> Option<usize> {
        if LOWERCASE.contains(&c.0) {
            Some((c.0 - b'a') as usize)
        } else {
            None
        }
    }

    fn entry(&mut self, c: Char) -> &mut u8 {
        match Self::index_of(c) {
            Some(i) => {
                return &mut self.lowercase[i];
            }
            None => {
                if self.other_chars.get(&c).is_none() {
                    self.other_chars.push((c, 0));
                }
                return self.other_chars.get_mut(&c).unwrap();
            }
        }
    }

    fn lookup(&self, c: Char) -> u8 {
        match Self::index_of(c) {
            Some(i) => {
                return self.lowercase[i];
            }
            None => {
                return *self.other_chars.get(&c).unwrap_or(&0);
            }
        }
    }

    pub fn char_count(&self, c: u8) -> u8 {
        self.lookup(Char(c))
    }

    /// This must be called after `set_longest_tok` (TODO: fix or enforce!)
    pub fn set_ties(&mut self, text: &str) {
        let mut tie_seqs = vec![vec![]; 7];

        let mut text_occurences = [0; 26];
        for byte in text.as_bytes() {
            if *byte == b' ' || !Char(*byte).is_lc_letter() {
                continue;
            }
            text_occurences[(*byte - b'a') as usize] += 1;
        }

        for occurences in 1..7 {
            for byte in text.as_bytes() {
                if *byte == b' '
                    || !Char(*byte).is_lc_letter()
                    || text_occurences[(byte - b'a') as usize] != occurences
                {
                    continue;
                }
                let ch = Char(*byte);
                if tie_seqs[occurences as usize].contains(&ch) {
                    continue;
                }
                tie_seqs[occurences as usize].push(ch);
            }
        }
        tie_seqs.retain(|v| !v.is_empty());

        self.tie_sequences = Some(tie_seqs);
    }

    pub fn just_letters_from_text(text: &str) -> LetterPool {
        let mut res = LetterPool {
            lowercase: [0; 26],
            other_chars: vec![],
            long_tok: None,
            last_letter: None,
            tie_sequences: None,
        };

        for byte in text.as_bytes() {
            let c = Char(*byte);
            // Spaces were not included in the original anagram!
            if *byte == b' ' {
                continue;
            }

            *res.entry(c) += 1;
        }
        res
    }

    // TODO TODO: Don't use this
    // Add the model as an optional for finding the longest token.
    pub fn from_text(text: &str, look_at_ties: bool) -> LetterPool {
        let mut res = Self::just_letters_from_text(text);

        for byte in text.as_bytes() {
            let c = Char(*byte);
            if c.is_letter() {
                res.last_letter = Some(c);
            }
        }

        if look_at_ties {
            res.set_ties(text);
        }

        return res;
    }

    // HUGE HACK; should pull this concept out of `from_text`
    pub fn set_last_letter(&mut self, b: u8) {
        self.last_letter = Some(Char(b))
    }

    pub fn set_longest_tok(&mut self, longest_tok: LlamaToken, model: &LlamaModel) {
        self.remove(longest_tok, model);
        self.long_tok = Some(longest_tok.0);
    }

    pub fn set_longest_tok_from(&mut self, text: &str, model: &LlamaModel) {
        let toks = llm::str_to_tokens(text, model);
        let mut longest_tok_len = 0;
        let mut longest_tok = toks[0];
        for t in toks {
            let t_len = llm::tok_to_str(t, model).len();
            if t_len > longest_tok_len {
                longest_tok_len = t_len;
                longest_tok = t;
            }
        }

        self.remove(longest_tok, model); // need to do this before the next line!
        self.long_tok = Some(longest_tok.0);
    }

    /// Does not respect `.long_tok`!
    fn has_pt(&self, tok: &PoolTok) -> bool {
        for (c, count) in &tok.chars {
            let available = self.lookup(*c);
            if available < *count {
                return false;
            } else if Some(*c) == self.last_letter
                && available == *count
                && self.letter_size() > tok.size().into()
            {
                return false; // Used the last letter, but we're not the last word.
            }
        }
        return true;
    }

    pub fn has(&self, tok: LlamaToken, model: &LlamaModel) -> bool {
        if Some(tok.0) == self.long_tok {
            // the theory of ties doesn't affect `has`!
            return true;
        }

        POOL_TOK_CACHE.with_borrow_mut(|ptc| {
            let pt = ptc.get_tok(tok, model);
            self.has_pt(pt)
        })
    }

    pub fn has_str(&self, _s: &str, _model: &LlamaModel) -> bool {
        todo!()
    }

    /// Panics if the letters aren't available.
    fn remove_pt(&mut self, tok: &PoolTok, is_the_long_tok: bool) {
        if is_the_long_tok {
            self.long_tok = None;
        } else {
            for (c, count) in &tok.chars {
                let entry = self.entry(*c);
                *entry -= *count; // panics if this was invalid
            }
        }

        // Do we still respect tied-letter order?  (It matters even if this is the long token.)
        let mut ties_invalidated = false;
        if let Some(ties) = self.tie_sequences.as_mut() {
            'char_in_tok: for c in tok.chars.keys() {
                for occ_list in &mut *ties {
                    if occ_list.first() == Some(c) {
                        occ_list.remove(0); // Great, the letter we were hoping for.
                        break; // Only one occ_list could be affected.
                    } else if occ_list.contains(c) {
                        ties_invalidated = true;
                        break 'char_in_tok;
                    }
                }
            }
        }

        if ties_invalidated {
            self.tie_sequences = None;
        }
    }

    /// Panics if the letters aren't available.
    pub fn remove(&mut self, tok: LlamaToken, model: &LlamaModel) {
        POOL_TOK_CACHE.with_borrow_mut(|ptc| {
            let pt = ptc.get_tok(tok, model);
            self.remove_pt(pt, Some(tok.0) == self.long_tok)
        })
    }

    /// Intended for testing.
    /// Panics if the letters aren't available.
    pub fn remove_str(&mut self, s: &str) {
        assert!(self.long_tok == None); // Can't respect this, if present!
        self.remove_pt(&PoolTok::from_str(s), false);
    }

    /// Creates a copy, unless it's not possible
    fn try_remove_pt(&self, tok: &PoolTok, is_the_long_tok: bool) -> Option<Self> {
        if is_the_long_tok || self.has_pt(tok) {
            let mut res = (*self).clone();
            res.remove_pt(tok, is_the_long_tok);
            return Some(res);
        }
        return None;
    }

    pub fn try_remove(&self, tok: LlamaToken, model: &LlamaModel) -> Option<Self> {
        POOL_TOK_CACHE.with_borrow_mut(|ptc| {
            let pt = ptc.get_tok(tok, model);
            self.try_remove_pt(pt, Some(tok.0) == self.long_tok)
        })
    }
}

#[derive(Default)]
struct TokCache {
    toks: HashMap<i32, PoolTok>,
}

impl TokCache {
    fn get_tok(&mut self, tok: LlamaToken, model: &LlamaModel) -> &PoolTok {
        self.toks
            .entry(tok.0)
            .or_insert(PoolTok::from_str(&llm::tok_to_str(tok, model)))
    }
}

#[test]
fn pool_test() {
    let mut pool = LetterPool::from_text("an okay story, but as a Terminator sequel, it profoundly disappoints on every conceivable level!", true);

    assert!(pool.has_pt(&PoolTok::from_str("okay")));
    assert!(pool.has_pt(&PoolTok::from_str(" okay")));
    assert!(pool.has_pt(&PoolTok::from_str("disappoints")));
    assert!(pool.has_pt(&PoolTok::from_str("level")));
    assert!(!pool.has_pt(&PoolTok::from_str("xerox")));
    for w in "an okay story, but as a Terminator sequel, it profoundly disappoints on every conceivable level !".split(" ") {
        if w == "!" {
            assert!(pool.empty_of_letters());
        }

        assert!(pool.has_pt(&PoolTok::from_str(w)));
        assert!(pool.respects_ties());
        pool.remove_pt(&PoolTok::from_str(w), /*is_the_long_tok*/ false);
    }

    assert!(!pool.has_pt(&PoolTok::from_str("o")));
    assert!(!pool.has_pt(&PoolTok::from_str("e")));
    assert!(pool.empty_of_letters());
    assert_eq!(pool.size(), 0);

    let mut pool = LetterPool::from_text("haha wow!", false);
    assert!(!pool.has_pt(&PoolTok::from_str("wow"))); // 'w' has to be the last letter.
    assert!(pool.has_pt(&PoolTok::from_str("haha")));
    pool.remove_pt(&PoolTok::from_str("haha"), false);
    assert!(pool.has_pt(&PoolTok::from_str("wow")));
    pool.remove_pt(&PoolTok::from_str("wow"), false);

    let mut pool_1663 = LetterPool::from_text("ttttttttttttooooooooooeeeeeeeeaaaaaaallllllnnnnnnuuuuuuiiiiisssssdddddhhhhhyyyyyIIIrrrfffbbwwkcmvg:,!!", false);

    for w in "I have the fundamental theories that you study final toenail: I doubt null, I hold you frog ditty tons stoic break yes wow!!".split(" ") {
        pool_1663.remove_str(w);
    }
    assert!(pool_1663.empty_of_letters());
    assert!(!pool_1663.respects_ties());
}

#[test]
fn ties_test() {
    use crate::llm::{self, str_to_tokens};

    let abcdef_pool = LetterPool::from_text("aaa bbb ccc dd ee ff x", /*look_at_ties=*/ true);
    assert!(abcdef_pool.respects_ties());

    {
        let mut pool = abcdef_pool.clone();
        pool.remove_str("abc");
        assert!(pool.respects_ties());
        pool.remove_str("def");
        assert!(pool.respects_ties());
        pool.remove_str("x");
        assert!(pool.respects_ties());
    }

    {
        // Letters that occur different numbers of times can happen in any order:
        let mut pool = abcdef_pool.clone();
        pool.remove_str("x");
        assert!(pool.respects_ties());
        pool.remove_str("def");
        assert!(pool.respects_ties());
        pool.remove_str("abc");
        assert!(pool.respects_ties());
    }

    {
        let mut pool = abcdef_pool.clone();
        pool.remove_str("cba");
        assert!(!pool.respects_ties());
    }

    {
        let mut pool = abcdef_pool.clone();
        pool.remove_str("b");
        assert!(!pool.respects_ties());
    }

    {
        let mut pool = abcdef_pool.clone();
        pool.remove_str("abc");
        assert!(pool.respects_ties());
        pool.remove_str("cba");
        assert!(pool.respects_ties()); // Only the first letters matter!
        pool.remove_str("f");
        assert!(!pool.respects_ties()); // But we still know d->e->f!
    }

    let model = llm::model_from_gguf("maykeye-tl.gguf", false);

    let letters_for_1663 = "ttttttttttttooooooooooeeeeeeeeaaaaaaallllllnnnnnnuuuuuuiiiiisssssdddddhhhhhyyyyyIIrrrfffbbwwkcmvg:,!!";
    let mut pool_for_1663 = LetterPool::just_letters_from_text(&letters_for_1663);
    pool_for_1663.set_longest_tok(
        *str_to_tokens(" fundamental", &model).last().unwrap(),
        &model,
    );
    pool_for_1663.set_ties(&letters_for_1663);
    pool_for_1663.set_last_letter(b'w');

    assert_eq!(pool_for_1663.print_ties_info(), "kcmvg bw rf isdhy lnu ");

    {
        let mut pool = pool_for_1663.clone();
        assert!(pool.respects_ties());
        pool.remove(*str_to_tokens(" bet", &model).last().unwrap(), &model);
        assert!(pool.respects_ties());
    }
    {
        let mut pool = pool_for_1663.clone();
        assert!(pool.respects_ties());
        pool.remove(*str_to_tokens(" can", &model).last().unwrap(), &model);
        assert!(!pool.respects_ties());
    }

    // "fundamental" needs to come after some things to work
    {
        let mut pool = pool_for_1663.clone();
        assert!(pool.respects_ties());
        pool.remove(
            *str_to_tokens(" fundamental", &model).last().unwrap(),
            &model,
        );
        assert!(!pool.respects_ties());
    }

    {
        let mut pool = pool_for_1663.clone();
        assert!(pool.respects_ties());
        pool.remove(*str_to_tokens(" r", &model).last().unwrap(), &model);
        assert!(pool.respects_ties());
        for tok in str_to_tokens(" ln", &model) {
            pool.remove(tok, &model);
            assert!(pool.respects_ties());
        }
        for tok in str_to_tokens(" is", &model) {
            pool.remove(tok, &model);
            assert!(pool.respects_ties());
        }
        for tok in str_to_tokens(" kc", &model) {
            pool.remove(tok, &model);
            assert!(pool.respects_ties());
        }
        assert!(pool.respects_ties());
        pool.remove(
            *str_to_tokens(" fundamental", &model).last().unwrap(),
            &model,
        );
        assert!(pool.respects_ties());
    }
}

#[test]
fn word_state_test() {
    let mut vb = VocabBuilder::default();
    let model = llm::model_from_gguf("maykeye-tl.gguf", false);
    vb.add_word("!", &model, false);
    vb.add_word("the", &model, false);
    vb.add_word("fundamental", &model, false);
    vb.add_word("theories", &model, false);
    vb.add_word("frog", &model, false); // 2 tokens
    vb.add_word("loripels", &model, false); // 8 letters, 2 tokens
    let vocab = vb.build(/*disabled=*/ false, /*enforce_8_11=*/ true);

    let seq_is_ok = |seq: &str| {
        let mut ws_or = Some(WordState::new_empty());
        for tok in llm::str_to_tokens(&format!("{}!", seq), &model) {
            if let Some(ws) = ws_or {
                ws_or = ws.add_tok(LlamaToken(tok.0), &vocab);
            } else {
                return false;
            }
        }
        ws_or.is_some()
    };

    assert!(seq_is_ok("the"));
    assert!(seq_is_ok("frog"));
    assert!(!seq_is_ok("forp")); // non-word
    assert!(!seq_is_ok("theories theories")); // two 8-letter words
    assert!(seq_is_ok("fundamental theories"));
    assert!(seq_is_ok("the fundamental theories the"));
    assert!(seq_is_ok("the fundamental loripels the"));
    assert!(!seq_is_ok("fundamental the theories"));
    assert!(seq_is_ok("the theories fundamental"));
    assert!(seq_is_ok("the theories fundamental frog"));
    assert!(seq_is_ok("frog loripels fundamental frog"));
    assert!(!seq_is_ok("frog loripels the fundamental frog"));
    assert!(!seq_is_ok("frog loripels frog fundamental frog"));
}
