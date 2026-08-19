# Qwen3.8-27B NVFP4 + MTP on one DGX Spark (vLLM)

Measured serving recipe for **Qwen3.8-27B NVFP4** on one NVIDIA DGX Spark (GB10 / SM121) using **vLLM 0.27.1**.

This repository pins the container image, model files, launch flags, CPU pin, and the throughput numbers from a single-box A/B on 2026-08-19. It is a serving qualification, not a training or fine-tune tree.

**Winner:** in-checkpoint **MTP k=3**, prefix cache on, container pinned to the ten Cortex-X5 cores (`5-9,15-19`).

## Pinned stack

| Piece | Value |
|---|---|
| Target | Local NVFP4 snapshot `Qwen3.8-27B-NVFP4` served as `Qwen3.8-27B` |
| Target files | `model.safetensors` + in-checkpoint `model_mtp.safetensors` + `config.json` |
| Architecture | `Qwen3_5ForConditionalGeneration` (mixed NVFP4 MLP, FP8 attention) |
| Image | `vllm/vllm-openai:v0.27.1` |
| Image digest | `sha256:2c211a1273b48e8929f893b267aeb1509e6b84654cdbde1bad56d79e3964224d` |
| Engine commit | `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Topology | 1x GB10, TP=1, `--max-model-len 65536` (native card window is 262144) |
| Speculation (winner) | MTP, 3 draft tokens |
| CPU pin (winner) | Docker `--cpuset-cpus 5-9,15-19` (3.9 GHz X5; leave 0-4 and 10-14 for the 2.8 GHz A725 cores) |

DSpark draft weights (`Qwen3.8-27B-DSpark-vLLM`, vLLM architecture `Qwen3DSparkModel`, block 7) were used only in configs A and B. The winner does **not** load a draft model.

## Serve (winner)

Place the NVFP4 snapshot on disk so it contains `model.safetensors` and `model_mtp.safetensors`, then:

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 bash scripts/serve.sh
```

Equivalent flags:

```bash
docker run -d --name vllm-qwen38-27b \
  --gpus all --ipc=host --cpuset-cpus 5-9,15-19 \
  -p 8000:8000 \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -v /path/to/Qwen3.8-27B-NVFP4:/models/Qwen3.8-27B-NVFP4:ro \
  vllm/vllm-openai:v0.27.1 \
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
```

Clients call `http://127.0.0.1:8000/v1` with `"model": "Qwen3.8-27B"`.

Sampling used by the checkpoint `generation_config.json` is `temperature=1.0`, `top_p=0.95`, `top_k=20`. The timed claim rows below force `temperature=0`. For instruct/non-thinking requests send `chat_template_kwargs.enable_thinking=false`.

`VLLM_MARLIN_USE_ATOMIC_ADD=1` is required on SM121 for the Marlin/NVFP4 path.

## Measured results

One GB10. Thinking off. Same harness for every row (`evidence/bench-*.json`). Throughput is `usage.completion_tokens / wall_seconds` on non-streamed `chat/completions`. These are synthetic fixed-length serving tests, not model-quality scores.

| Config | Spec | Extra | C1 median tok/s | C8 aggregate tok/s | Last mean accept | Last draft accept |
|---|---|---|---:|---:|---:|---:|
| A | DSpark k=7, `draft_sample_method=probabilistic` | prefix on, no CPU pin | 24.68 | 96.12 | 2.39 | 19.8% |
| B | DSpark k=7, default draft sampling | prefix on, no CPU pin | 23.17 | 91.91 | 3.23 | 31.8% |
| C | MTP k=3 | prefix on, no CPU pin | 25.51 | 116.67 | 3.21 | 73.6% |
| **D (winner)** | **MTP k=3** | **prefix on, X5 pin** | **25.63** | **122.18** | **3.39** | **79.8%** |
| E | MTP k=3 | prefix **off**, X5 pin | 24.29 | 112.36 | 2.57 | 52.3% |

C1 is the median of three 512-in / 256-out greedy runs. C8 is the median of two 8-way waves (1024-in / 256-out); aggregate is total completion tokens divided by the wall time of the slowest request in that wave.

Versus A, D is about **+4%** single-stream and **+27%** eight-way aggregate. Switching DSpark k=7 to MTP k=3 is the large move. The X5 pin adds a smaller C8 gain on top of MTP. Turning prefix cache off (E) lost both speed and accept rate.

A sixth knob (`--max-num-batched-tokens 16384`) was skipped: the winner boot did not emit the `max_num_scheduled_tokens is set to 8048` warning that appeared on the DSpark boots.

### Sampling row (not the claim row)

One c1-shaped request at the checkpoint sampling defaults (`temperature=1.0`, `top_p=0.95`, `top_k=20`):

| Config | tok/s |
|---|---:|
| A | 19.08 |
| B | 23.06 |
| C | 22.12 |
| D | 22.19 |
| E | 20.14 |

### Tool-call smoke (winner)

One non-stream request with a single `list_dir` tool, thinking off: HTTP 200, `finish_reason=tool_calls`, structured `tool_calls[0].function.name=list_dir` with `{"path": "."}`. Parser: `qwen3_coder`.

## Shared launch contract (every config)

Unchanged across A-E:

- `--gpu-memory-utilization 0.88`
- `--max-model-len 65536`
- `--max-num-seqs 24` (queue depth; the eight-way test is the measured concurrency)
- `--max-num-batched-tokens 8192`
- `--kv-cache-dtype fp8`
- `--attention-backend flashinfer --enable-flashinfer-autotune`
- `--enable-chunked-prefill`
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
- `--tensor-parallel-size 1 --trust-remote-code --dtype auto`
- Docker `--gpus all --ipc=host`
- Env `VLLM_MARLIN_USE_ATOMIC_ADD=1`

Only speculation method, prefix-cache on/off, and the X5 cpuset changed.

## How the harness works

Prompt sentence: `Explain hash maps, concurrency, and lock-free queues. ` (13 tokens). Repeated 39 times for c1 (~512 user tokens) and 79 times for c8 (~1024). After `/v1/models` is 200, one 16-token warmup is discarded.

Every request: `stream=false`, `seed=0`, `chat_template_kwargs.enable_thinking=false`. Count server `usage.completion_tokens`, not SSE events.

Raw per-config JSON: `evidence/bench-*.json`. Numeric rollup: `evidence/summary.json`.

## What this is not

- Not a quality leaderboard. No GSM8K / HumanEval / SWE numbers here.
- Not SGLang, not DFlash2, not an engine swap.
- Not a claim that MTP output matches speculation-off byte for byte.
- Not a dual-instance recipe. One 27B NVFP4+spec process fills this box; do not expect a second full copy alongside it.

## License

MIT for the scripts and notes in this tree. Model weights follow their own cards (Qwen / NVFP4 checkpoint licenses) and are not stored here.
