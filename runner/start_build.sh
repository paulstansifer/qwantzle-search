#!/bin/bash

cd /app/qwantzle-search
git pull
cargo build --release --features "cuda"
cd /app


# huggingface-cli download "$HF_MODEL_PATH" --local-dir "${HF_MODEL_PATH#*/}"
# python3 llama.cpp/convert_hf_to_gguf.py "${HF_MODEL_PATH#*/}" --outfile model.gguf --outtype "$TYPE"
