#!/bin/bash

cd /app
git clone https://github.com/paulstansifer/qwantzle-search.git
cd qwantzle-search
cargo build --release --features "cuda"
cd /app

# We need this in order to get the conversion script
git clone https://github.com/ggerganov/llama.cpp


huggingface-cli download "$HF_MODEL_PATH" --local-dir "${HF_MODEL_PATH#*/}"
python3 llama.cpp/convert_hf_to_gguf.py "${HF_MODEL_PATH#*/}" --outfile model.gguf --outtype "$TYPE"

qwantzle-search/target/release/qwantzle-search



local spec="${HF_MODEL_PATH#*/}-$TYPE"

huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

huggingface-cli download paulstansifer/estimation-results --repo-type dataset
echo "foo" > estimation-results/foo.txt
huggingface-cli upload paulstansifer/estimation-results estimation-results/foo.txt foo.txt --repo-type=dataset

echo "Process completed successfully!"