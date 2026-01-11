fn main() {
    println!("cargo:rustc-env=PYTHON_SYS_EXECUTABLE=venv/bin/python3.12");
    println!("cargo:rustc-link-arg=-Wl,-rpath,/home/paul/src/qwantzle-search/.venv/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/");
}
