use small_map::SmallMap;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use trie_rs::{Trie, TrieBuilder};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct Char(u8);

// Token doesn't implement Ord or Hash, so we use the underlying i32.

#[derive(Default)]
pub struct VocabBuilder {
    tb: TrieBuilder<i32>,
    allowed_tokens: HashSet<i32>,
    word_starters: HashSet<i32>,
    word_enders: HashSet<i32>,
    nonletters: HashSet<i32>,
}

enum TokenRole {
    StartsWord,
    EndsWord,
    Continues,
    NonLetter,
}

pub struct Vocab {
    valid_words: Trie<i32>,
    token_roles: HashMap<i32, TokenRole>,
    disabled: bool,
}

impl VocabBuilder {
    fn eval_token(&mut self, tok: i32, model: &llama_cpp::LlamaModel) {
        if self.allowed_tokens.contains(&tok) {
            return; // already processed
        }
        self.allowed_tokens.insert(tok);

        let solo_token = model.token_to_piece(llama_cpp::Token(tok));

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
            let xes = model
                .tokenize_bytes("xxxxxxxxxxxxxxx", false, false)
                .unwrap();
            let mid_token = xes[xes.len() / 2];

            // Does the token add a space when attached to something?
            if model
                .decode_tokens(&[mid_token, llama_cpp::Token(tok)])
                .contains(" ")
            {
                self.word_starters.insert(tok);
            }
            if model
                .decode_tokens(&[llama_cpp::Token(tok), mid_token])
                .contains(" ")
            {
                self.word_enders.insert(tok);
            }
        }
    }

    pub fn add_word(&mut self, word: &str, model: &llama_cpp::LlamaModel, vary_case: bool) {
        if word.is_empty() {
            return; // Titlecasing would get unhappy.
        }

        // // If it starts with a letter, add a space.
        // let word = if regex::Regex::new(r"^[a-zA-Z]").unwrap().is_match(word) {
        //     format!(" {}", word)
        // } else {
        //     word.to_string()
        // };

        for word in [word.to_string(), format!(" {}", word)] {
            self.add_literal_word(&word, model);

            if vary_case {
                self.add_literal_word(&word.to_ascii_lowercase(), model);
                self.add_literal_word(&word.to_ascii_uppercase(), model);

                let mut title_case = word.to_owned();
                unsafe {
                    title_case.as_bytes_mut()[0].make_ascii_uppercase();
                }
                self.add_literal_word(&title_case, model);
            }
        }
    }
    fn add_literal_word(&mut self, word: &str, model: &llama_cpp::LlamaModel) {
        let toks: Vec<i32> = model
            .tokenize_bytes(word, false, false)
            .unwrap()
            .into_iter()
            .map(|t| t.0)
            .collect();

        self.tb.push(toks.clone());

        for tok in toks {
            self.eval_token(tok, model);
        }
    }

    pub fn build(self, disabled: bool) -> Vocab {
        Vocab {
            valid_words: self.tb.build(),
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
        }
    }
}

pub struct WordState {
    cur_word: Vec<i32>,
}

impl WordState {
    pub fn new_empty() -> WordState {
        WordState { cur_word: vec![] }
    }

    pub fn add_tok(&self, tok: llama_cpp::Token, voc: &Vocab) -> Option<WordState> {
        if voc.disabled {
            return Some(WordState { cur_word: vec![] });
        }
        match voc.token_roles.get(&tok.0) {
            None => None,
            Some(TokenRole::StartsWord) => {
                if self.cur_word.is_empty() || voc.valid_words.exact_match(&self.cur_word) {
                    Some(WordState {
                        cur_word: vec![tok.0],
                    })
                } else {
                    None
                }
            }
            Some(TokenRole::NonLetter) => {
                if self.cur_word.is_empty() || voc.valid_words.exact_match(&self.cur_word) {
                    Some(WordState::new_empty())
                } else {
                    None
                }
            }
            Some(TokenRole::EndsWord) => {
                let mut full_word = self.cur_word.clone();
                full_word.push(tok.0);
                if voc.valid_words.exact_match(&full_word) {
                    println!("Ending: {:?}", full_word);
                    Some(WordState::new_empty())
                } else {
                    None
                }
            }
            Some(TokenRole::Continues) => {
                let mut lengthened_word = self.cur_word.clone();
                lengthened_word.push(tok.0);
                if voc.valid_words.exact_match(&lengthened_word)
                    || voc.valid_words.is_prefix(&lengthened_word)
                {
                    Some(WordState {
                        cur_word: lengthened_word,
                    })
                } else {
                    None
                }
            }
        }
    }
}

impl Vocab {
    pub fn summary(&self, model: &llama_cpp::LlamaModel) -> String {
        let mut s = String::new();
        let words: Vec<Vec<i32>> = self.valid_words.iter().collect();
        for w in words.iter().take(15) {
            s += &format!(
                "({:?}) '{}'  ",
                w,
                model.decode_tokens(w.iter().map(|t| llama_cpp::Token(*t)).collect::<Vec<_>>())
            );
        }

        s += &format!(" ({})\n", words.len());

        for t in self.token_roles.iter().take(25) {
            s += &format!(
                "({}) '{}'  ",
                t.0,
                model.token_to_piece(llama_cpp::Token(*t.0))
            );
        }

        s += &format!("{} tokens", self.token_roles.len());
        return s;
    }

    pub fn check_expressibility(&self, text: &str, model: &llama_cpp::LlamaModel) {
        let toks = model.tokenize_bytes(text, false, false).unwrap();
        for tok in toks {
            if !self.token_roles.contains_key(&tok.0) {
                println!("Missing token: ({}) {}", tok.0, model.token_to_piece(tok));
            }
        }

        for word in regex::Regex::new(r"\b").unwrap().split(text) {
            let toks: Vec<i32> = model
                .tokenize_bytes(&format!(" {}", word), false, false)
                .unwrap()
                .into_iter()
                .map(|t| t.0)
                .collect();

            if !self.valid_words.exact_match(&toks) {
                println!("Cannot find ({:?}) '{}'", toks, word);
            }
        }
    }
}

#[derive(Clone)]
pub struct LetterPool {
    lowercase: [u8; 26],
    other_chars: SmallMap<8, Char, u8>,
}

#[derive(Default)]
struct PoolTok {
    chars: SmallMap<8, Char, u8>,
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
                res.chars.insert(c, 1);
            } else {
                *res.chars.get_mut(&c).unwrap() += 1;
            }
        }
        return res;
    }
}

impl LetterPool {
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
                    self.other_chars.insert(c, 0);
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

    pub fn from_text(text: &str) -> LetterPool {
        let mut res = LetterPool {
            lowercase: [0; 26],
            other_chars: SmallMap::new(),
        };

        for byte in text.as_bytes() {
            let c = Char(*byte);
            // Spaces were not included in the original anagram!
            if *byte == b' ' {
                continue;
            }

            *res.entry(c) += 1;
        }

        return res;
    }

    fn has_pt(&self, tok: &PoolTok) -> bool {
        for (c, count) in &tok.chars {
            if self.lookup(*c) < *count {
                return false;
            }
        }
        return true;
    }

    pub fn has(&self, tok: llama_cpp::Token, model: &llama_cpp::LlamaModel) -> bool {
        POOL_TOK_CACHE.with_borrow_mut(|ptc| {
            let pt = ptc.get_tok(tok, model);
            self.has_pt(pt)
        })
    }

    /// Panics if the letters aren't available.
    fn remove_pt(&mut self, tok: &PoolTok) {
        for (c, count) in &tok.chars {
            let entry = self.entry(*c);
            *entry -= *count; // panics if this was invalid
        }
    }

    /// Panics if the letters aren't available.
    pub fn remove(&mut self, tok: llama_cpp::Token, model: &llama_cpp::LlamaModel) {
        POOL_TOK_CACHE.with_borrow_mut(|ptc| {
            let pt = ptc.get_tok(tok, model);
            self.remove_pt(pt)
        })
    }

    /// Creates a copy, unless it's not possible
    fn try_remove_pt(&self, tok: &PoolTok) -> Option<Self> {
        if self.has_pt(tok) {
            let mut res = (*self).clone();
            res.remove_pt(tok);
            return Some(res);
        }
        return None;
    }

    pub fn try_remove(&self, tok: llama_cpp::Token, model: &llama_cpp::LlamaModel) -> Option<Self> {
        POOL_TOK_CACHE.with_borrow_mut(|ptc| {
            let pt = ptc.get_tok(tok, model);
            self.try_remove_pt(pt)
        })
    }
}

#[derive(Default)]
struct TokCache {
    toks: HashMap<i32, PoolTok>,
}

impl TokCache {
    fn get_tok(&mut self, tok: llama_cpp::Token, model: &llama_cpp::LlamaModel) -> &PoolTok {
        self.toks
            .entry(tok.0)
            .or_insert(PoolTok::from_str(&model.decode_tokens([tok])))
    }
}

#[test]
fn pool_test() {
    let mut pool = LetterPool::from_text("an okay story, but as a Terminator sequel, it profoundly disappoints on every conceivable level!");

    assert!(pool.has_pt(&PoolTok::from_str("okay")));
    assert!(pool.has_pt(&PoolTok::from_str(" okay")));
    assert!(pool.has_pt(&PoolTok::from_str("disappoints")));
    assert!(!pool.has_pt(&PoolTok::from_str("xerox")));
    for w in "an okay story, but as a Terminator sequel, it profoundly disappoints on every conceivable level!".split(" ") {
        assert!(pool.has_pt(&PoolTok::from_str(w)));
        pool.remove_pt(&PoolTok::from_str(w));
    }

    assert!(!pool.has_pt(&PoolTok::from_str("o")));
    assert!(!pool.has_pt(&PoolTok::from_str("e")));
    assert_eq!(pool.size(), 0)
}
