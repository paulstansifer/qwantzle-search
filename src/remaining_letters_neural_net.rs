// Adapted and debugged from something that Claude generated:
// https://claude.site/artifacts/6a6d91f5-bc73-4e1c-8f2d-da3dfa67e860

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{Module, VarBuilder};
use safetensors::SafeTensors;
use std::fs;

use crate::pool::LetterPool;

// Define a flexible neural network structure
pub struct LetterNet {
    layers: Vec<candle_nn::Linear>,
}

static CHAR_FEATURE_ORDER: &[u8] = b"!',-.:?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

impl LetterNet {
    pub fn new_from_file(model_path: &str) -> Result<Self> {
        // Load the safetensors file
        // let model_path = "/home/paul/src/qwantzle-search/corpus/letter_pool.safetensors";
        let model_bytes = fs::read(model_path)
            .with_context(|| format!("Failed to read model file: {}", model_path))?;
        let safetensors = SafeTensors::deserialize(&model_bytes)?;

        // Create a variable builder from the loaded weights
        let device = Device::Cpu;
        let vb =
            VarBuilder::from_slice_safetensors(&model_bytes, candle_core::DType::F32, &device)?;

        LetterNet::new_internal(vb, &safetensors)
    }

    pub fn evaluate(&self, remaining: &LetterPool) -> f32 {
        let mut features = vec![0.0; CHAR_FEATURE_ORDER.len() + 3];

        let mut vowels = 0.0;
        let mut consonants = 0.0;
        for (i, ch) in CHAR_FEATURE_ORDER.iter().enumerate() {
            let ch = *ch;
            let ch_count = remaining.char_count(ch) as f32;
            if ch == b'a' || ch == b'e' || ch == b'i' || ch == b'o' || ch == b'u' {
                vowels += ch_count;
            } else
            /*if ch >= b'a' && ch <= b'z'*/  // TODO: fix this in sync with the data
            {
                consonants += ch_count;
            }
            features[i] = ch_count as f32;
        }
        // Matches the order in
        // https://colab.research.google.com/drive/1rGLSxC_esmQAL2OKruSIET5gMy4iaZHL#scrollTo=yW38Co-4Ll6f&line=28&uniqifier=1
        features[CHAR_FEATURE_ORDER.len()] = remaining.size() as f32;
        features[CHAR_FEATURE_ORDER.len() + 1] = vowels;
        features[CHAR_FEATURE_ORDER.len() + 2] = consonants;

        // println!("FEATURES: {:?}", features);

        let device = Device::cuda_if_available(0).unwrap();
        let input = Tensor::from_slice(&features, (1, features.len()), &device).unwrap();

        // Run inference
        let output = self.forward(&input).unwrap().to_vec2::<f32>().unwrap()[0][0];

        // Apply sigmoid, which isn't visible in the tensors file!
        return 1.0 / (1.0 + (-output).exp());
    }

    fn new_internal(vb: VarBuilder, tensors: &SafeTensors) -> Result<Self> {
        // Find all weight matrices to determine network structure
        let mut weight_names: Vec<String> = tensors
            .names()
            .into_iter()
            .filter(|name| name.ends_with(".weight"))
            .cloned()
            .collect();
        weight_names.sort(); // Ensure consistent layer ordering

        let mut layers = Vec::new();

        for weight_name in weight_names {
            let layer_prefix = weight_name.strip_suffix(".weight").unwrap();
            let weight_tensor = tensors.tensor(&weight_name)?;
            let weight_shape = weight_tensor.shape();

            if weight_shape.len() != 2 {
                anyhow::bail!(
                    "Expected weight matrix to be 2D, got shape: {:?}",
                    weight_shape
                );
            }

            println!(
                "Found layer '{}' with shape: {:?}",
                layer_prefix, weight_shape
            );

            // Create linear layer with dimensions from the weight matrix
            let layer = candle_nn::linear(
                weight_shape[1], // input features
                weight_shape[0], // output features
                vb.pp(layer_prefix),
            )?;

            layers.push(layer);
        }

        if layers.is_empty() {
            anyhow::bail!("No valid weight matrices found in safetensors file");
        }

        Ok(Self { layers })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut current = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current)?;

            // Apply ReLU to all but the last layer
            if i < self.layers.len() - 1 {
                current = current.relu()?;
            }
        }

        Ok(current)
    }
}

#[test]
fn model_distrusts_toenails() {
    let rlnn = LetterNet::new_from_file("corpus/letter_pool.safetensors").unwrap();

    let mut pool_1663 = LetterPool::from_text("ttttttttttttooooooooooeeeeeeeeaaaaaaallllllnnnnnnuuuuuuiiiiisssssdddddhhhhhyyyyyIIrrrfffbbwwkcmvg:,!!");

    // Ignore starting score; for all we know, the string does contain "toenails" once!
    pool_1663.remove_str("toenails");
    let score_1 = rlnn.evaluate(&pool_1663);
    pool_1663.remove_str("toenails");
    let score_2 = rlnn.evaluate(&pool_1663);
    pool_1663.remove_str("toenails");
    let score_3 = rlnn.evaluate(&pool_1663);
    pool_1663.remove_str("toenails");
    let score_4 = rlnn.evaluate(&pool_1663);
    pool_1663.remove_str("toenails");
    let score_5 = rlnn.evaluate(&pool_1663);

    assert!(score_2 < score_1);
    assert!(score_3 < score_2);
    assert!(score_4 < score_3);
    assert!(score_5 < score_4);

    let no_vowels = LetterPool::from_text(
        "ttttttttttttllllllnnnnnnuuuuuusssssdddddhhhhhyyyyyrrrfffbbwwkcmvg:,!!",
    );
    let no_vowels_short = LetterPool::from_text("ttttt!!");

    assert!(rlnn.evaluate(&no_vowels) < 0.01);
    assert!(rlnn.evaluate(&no_vowels_short) < 0.01);

    // Empty string *ought* to be alright, but it's an edge case...
    assert!(rlnn.evaluate(&LetterPool::from_text("")) > 0.25);

    let sentence = [
        "I", "am", "saying", "a", "normal", "thing", "for", "a", "T-Rex", "to", "say!!",
    ];

    for i in 0..sentence.len() {
        assert!(
            rlnn.evaluate(&LetterPool::from_text(
                &sentence[i..sentence.len()].join("")
            )) > 0.65 // In practice, I saw this bottom out at 7.8.
        );
    }

    assert!(rlnn.evaluate(&LetterPool::from_text("Sphinx of qwartz, judge my vow!!")) < 0.2);
}
