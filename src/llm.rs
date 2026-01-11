use serde::{Deserialize, Serialize};

/// Common token type across all backends
pub type Token = i32;

#[derive(Serialize, Deserialize, Default, Clone, Copy)]
pub struct SessionTimers {
    pub advance_time: u128,
    pub predict_time: u128,
    pub score_time: u128,
    pub truncate_time: u128,
}

pub fn str_to_tokens_maybe_with_prefix_space(s: &str, model: &dyn Model) -> (Vec<Token>, bool) {
    let toks_without_space = model.str_to_tokens(s);

    let toks_with_space = model.str_to_tokens(&format!(" {s}"));

    if toks_without_space.len() < toks_with_space.len() {
        (toks_without_space, false)
    } else {
        (toks_with_space, true)
    }
}

/// Common interface for model backends
#[allow(dead_code)]
pub trait Model {
    /// Convert a string to tokens
    fn str_to_tokens(&self, s: &str) -> Vec<Token> {
        self.str_to_tokens_bos(s, false)
    }

    fn str_to_tokens_bos(&self, s: &str, bos: bool) -> Vec<Token>;

    /// Convert tokens back to a string
    fn toks_to_str(&self, toks: &[Token]) -> String;

    /// Convert a single token to a string
    fn tok_to_str(&self, tok: Token) -> String;

    /// Create a new session with the given prompt
    fn new_session<'b>(&'b self, prompt: &str, extra_toks: usize) -> Box<dyn Session<'b> + 'b>;
}

/// Common interface for inference sessions
#[allow(dead_code)]
pub trait Session<'a> {
    /// Add a boost multiplier for a specific token
    fn boost(&mut self, tok: Token, boost: f64);

    fn timers(&self) -> &SessionTimers;
    fn timers_mut(&mut self) -> &mut SessionTimers;

    /// Predict next tokens from a string prompt
    fn predict_str(&mut self, text: &str, top_p: Option<f32>) -> Vec<(Token, f64)>;

    /// Predict next tokens from token IDs
    fn predict(&mut self, toks: &[Token], top_p: Option<f32>) -> Vec<(Token, f64)>;
}
