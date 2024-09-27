use small_map::SmallMap;
use std::cell::RefCell;
use std::hash::Hash;
use std::{collections::HashMap, default};

#[derive(PartialEq, Eq, Hash, Clone, Copy)]
struct Char(u8);

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
    assert!(pool.has_pt(&PoolTok::from_str("disappoints")));
    assert!(!pool.has_pt(&PoolTok::from_str("xerox")));
    for w in "an okay story, but as a Terminator sequel, it profoundly disappoints on every conceivable level!".split(" ") {
        assert!(pool.has_pt(&PoolTok::from_str(w)));
        pool.remove_pt(&PoolTok::from_str(w));
    }

    assert!(!pool.has_pt(&PoolTok::from_str("o")));
    assert!(!pool.has_pt(&PoolTok::from_str("e")));
}
