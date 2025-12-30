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

/// Common interface for model backends
#[allow(dead_code)]
pub trait Model {
    /// Convert a string to tokens
    fn str_to_tokens(&mut self, s: &str) -> Vec<Token>;

    /// Convert tokens back to a string
    fn toks_to_str(&mut self, toks: &[Token]) -> String;

    /// Convert a single token to a string
    fn tok_to_str(&mut self, tok: Token) -> String;

    /// Create a new session with the given prompt
    fn new_session<'b>(&'b mut self, prompt: &str) -> Box<dyn Session<'b> + 'b>;
}

/// Common interface for inference sessions
#[allow(dead_code)]
pub trait Session<'a> {
    /// Add a boost multiplier for a specific token
    fn boost(&mut self, tok: Token, boost: f64);

    /// Get timing information from the session
    fn timers(&self) -> &SessionTimers;

    /// Predict next tokens from a string prompt
    fn predict_str(&mut self, text: &str, top_p: Option<f32>) -> Vec<(Token, f64)>;

    /// Predict next tokens from token IDs
    fn predict(&mut self, toks: &[Token], top_p: Option<f32>) -> Vec<(Token, f64)>;
}
