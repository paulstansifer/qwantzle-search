use llama_cpp::Token;
use llama_cpp_sys::llama_token_data;
use std::sync::{Arc, Mutex};

pub struct PeekSampler {
    pub eos: Token,
    pub candidates: Arc<Mutex<Vec<llama_token_data>>>,
}

impl llama_cpp::Sampler for PeekSampler {
    fn sample(
        &mut self,
        context: *mut llama_cpp_sys::llama_context,
        _tokens: &[Token],
        mut candidates_p: llama_cpp_sys::llama_token_data_array,
    ) -> Token {
        unsafe {
            // Doing this forces Llama.cpp to sort the candidates.
            llama_cpp_sys::llama_sample_top_p(
                context,
                std::ptr::addr_of_mut!(candidates_p),
                0.9995,
                5,
            );
        }

        for i in 0..candidates_p.size {
            self.candidates
                .lock()
                .unwrap()
                .push(unsafe { *candidates_p.data.wrapping_add(i) });
        }
        return self.eos; // Don't produce any tokens! (This sampler doesn't sample.)
    }
}
