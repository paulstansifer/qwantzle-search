#! /bin/bash
huggingface-cli download QuantFactory/TinyLlama_v1.1-GGUF  TinyLlama_v1.1.Q8_0.gguf  --local-dir .
huggingface-cli download paul-stansifer/qw-us-mistral-7b-gguf  unsloth.Q4_K_M.gguf  --local-dir .
mv unsloth.Q4_K_M.gguf qw-mistral.Q4_K_M.gguf
huggingface-cli download paul-stansifer/qw-us-tinyllama-1b-gguf   unsloth.F16.gguf  --local-dir .
mv unsloth.F16.gguf qw-tinyllama.Q4_K_M.gguf
huggingface-cli download paul-stansifer/qw-us-tinyllama-1b-gguf   unsloth.Q8_0.gguf  --local-dir .
mv unsloth.Q8_0.gguf qw-tinyllama.Q8_0.gguf
huggingface-cli download TheBloke/Mistral-7B-v0.1-GGUF  mistral-7b-v0.1.Q8_0.gguf  --local-dir .
huggingface-cli download TheBloke/Mistral-7B-v0.1-GGUF  mistral-7b-v0.1.Q4_0.gguf  --local-dir .
huggingface-cli download QuantFactory/Mistral-7B-v0.3-GGUF  Mistral-7B-v0.3.Q4_K_S.gguf  --local-dir .
huggingface-cli download TheBloke/MythoMax-L2-13B-GGUF  mythomax-l2-13b.Q3_K_M.gguf --local-dir .
huggingface-cli download TheBloke/Llama-2-13B-GGUF  llama-2-13b.Q4_0.gguf --local-dir .
huggingface-cli download mradermacher/pythia-1.4b-deduped-GGUF  pythia-1.4b-deduped.f16.gguf  --local-dir .
huggingface-cli download mradermacher/pythia-2.8b-deduped-GGUF  pythia-2.8b-deduped.f16.gguf  --local-dir .
huggingface-cli download mradermacher/pythia-6.9b-deduped-GGUF  pythia-6.9b-deduped.Q8_0.gguf  --local-dir .
huggingface-cli download TheBloke/Nous-Capybara-7B-v1.9-GGUF   nous-capybara-7b-v1.9.Q4_K_S.gguf --local-dir .
huggingface-cli download TheBloke/phi-2-GGUF   phi-2.Q8_0.gguf --local-dir .
huggingface-cli download Aloshe/Sheared-LLaMA-1.3B-Q8_0-GGUF  sheared-llama-1.3b-q8_0.gguf  --local-dir .
huggingface-cli download DevQuasar/Llama-3.2-3B-GGUF  Llama-3.2-3B.Q4_K_M.gguf --local-dir .
huggingface-cli download QuantFactory/gemma-2-9b-GGUF   gemma-2-9b.Q4_K_S.gguf --local-dir .
huggingface-cli download Aryanne/Sheared-LLaMA-2.7B-gguf  q4_0-sheared-llama-2.7b.gguf   --local-dir .
 huggingface-cli download TheBloke/Mistral-7B-OpenOrca-GGUF  mistral-7b-openorca.Q4_0.gguf --local-dir .
