#! /bin/bash
cd /app/models/

huggingface-cli download TheBloke/Mixtral-8x7B-v0.1-GGUF  mixtral-8x7b-v0.1.Q3_K_M.gguf  --local-dir .
huggingface-cli download TheBloke/Yi-34B-GGUF  yi-34b.Q4_0.gguf  --local-dir .
# huggingface-cli download mradermacher/gemma-2-27b-GGUF  gemma-2-27b.Q4_K_S.gguf  --local-dir .
huggingface-cli download mradermacher/Meta-Llama-3.1-70B-i1-GGUF Meta-Llama-3.1-70B.i1-IQ2_XS.gguf  --local-dir .
huggingface-cli download mradermacher/Qwen2.5-32B-GGUF  Qwen2.5-32B.Q4_K_S.gguf  --local-dir .
