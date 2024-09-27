#! /bin/bash

for m in /app/models/*.gguf; do
  echo "=== $m ==="
  echo "=== $m ===" >> /tmp/report
  /app/qwantzle-search/target/release/main --model $m | tee -a /tmp/report
done