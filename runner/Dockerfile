# # based on https://github.com/utilityai/llama-cpp-rs/blob/main/test-build.Dockerfile
# ARG CUDA_VERSION=12.3.1
# ARG UBUNTU_VERSION=22.04
# FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base-cuda

# # Install requirements for rustup install + bindgen: https://rust-lang.github.io/rust-bindgen/requirements.html
# RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y curl llvm-dev libclang-dev clang pkg-config libssl-dev
# RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
# ENV PATH=/root/.cargo/bin:$PATH

# COPY .. .
# RUN cargo build --bin qwantzle-search --features cuda

# FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} as base-cuda-runtime

# COPY --from=base-cuda /target/debug/simple /usr/local/bin/simple

# ENTRYPOINT ["/usr/local/bin/simple"]



ARG CUDA_VERSION=12.3.1
ARG UBUNTU_VERSION=22.04
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} as base-cuda

RUN DEBIAN_FRONTEND=noninteractive apt update -y && apt install -y curl llvm-dev libclang-dev clang pkg-config libssl-dev
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH=/root/.cargo/bin:$PATH

# Possibly redundant
RUN apt-get update && apt-get install -y python3 python3-pip
RUN apt-get install -y git zile

# Install requirements for the llama.cpp converter (possibly redundant)
RUN pip3 install "numpy~=1.26.4"
RUN pip3 install "sentencepiece~=0.2.0"
RUN pip3 install "transformers>=4.40.1,<5.0.0"
RUN pip3 install "gguf>=0.1.0"
RUN pip3 install "protobuf>=4.21.0,<5.0.0"
RUN pip3 install --extra-index-url https://download.pytorch.org/whl/cpu "torch~=2.2.1"

RUN pip3 install -U "huggingface_hub[cli]"

# Set the working directory in the container
WORKDIR /app

RUN git clone https://github.com/paulstansifer/qwantzle-search.git
WORKDIR /app/qwantzle-search
RUN cargo build --release

WORKDIR /app
RUN git clone https://github.com/ggerganov/llama.cpp


# Copy the shell script into the container
COPY estimation.sh .

# Make the shell script executable
RUN chmod +x estimation.sh

ENV TYPE=f16
# shell variables:
# * $HF_MODEL_PATH (should be of the form 'username/modelname')
# * $TYPE (one of "f32", "f16", "bf16", "q8_0", "tq1_0", "tq2_0", "auto")
#     (used as an argument in `convert_hf_to_gguf.py`)
# * $HF_TOKEN

# Set the entrypoint to the shell script
ENTRYPOINT ["./estimation.sh"]

