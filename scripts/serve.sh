#!/usr/bin/env bash
# Production profile measured on one NVIDIA DGX Spark (GB10 / SM121):
# Qwen3.8-27B NVFP4 + in-checkpoint MTP k=3 + Cortex-X5 CPU pin + prefix cache.
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:v0.27.1}"
# Recorded digest for this campaign:
# vllm/vllm-openai@sha256:2c211a1273b48e8929f893b267aeb1509e6b84654cdbde1bad56d79e3964224d
NAME="${NAME:-vllm-qwen38-27b}"
PORT="${PORT:-8000}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the NVFP4 snapshot (needs model.safetensors and model_mtp.safetensors)}"

test -f "$MODEL_DIR/model.safetensors"
test -f "$MODEL_DIR/model_mtp.safetensors"
test -f "$MODEL_DIR/config.json"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null
fi

docker run -d \
  --name "$NAME" \
  --gpus all \
  --ipc=host \
  --cpuset-cpus 5-9,15-19 \
  -p "$PORT:8000" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v "$MODEL_DIR":/models/Qwen3.8-27B-NVFP4:ro \
  "$IMAGE" \
  /models/Qwen3.8-27B-NVFP4 \
  --served-model-name Qwen3.8-27B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype auto \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --enable-flashinfer-autotune \
  --gpu-memory-utilization 0.88 \
  --max-model-len 65536 \
  --max-num-seqs 24 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'

echo "Launched $NAME on http://127.0.0.1:$PORT (model Qwen3.8-27B, MTP k=3)"
