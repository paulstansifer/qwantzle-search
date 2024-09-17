

Pre-quantized and gguf-ed versions of MicroLlama (though you can also have the Python script quantize while it converts to GGUF)
* https://huggingface.co/Felladrin/gguf-MicroLlama/tree/main

```
cd /workspace/
git clone https://huggingface.co/Maykeye/TinyLLama-v0
python /workspace/llama.cpp/convert_hf_to_gguf.py /workspace/TinyLLama-v0 --outfile /workspace/tiny_llama.gguf
```