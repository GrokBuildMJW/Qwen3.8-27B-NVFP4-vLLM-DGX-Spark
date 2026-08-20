# Qwen3.8-27B NVFP4 + MTP on one DGX Spark (vLLM)

Measured serving recipe for **Qwen3.8-27B NVFP4** on one NVIDIA DGX Spark (GB10 / SM121) using **vLLM 0.27.1**.

This repository pins the container image, model files, launch flags, CPU pin, and the throughput numbers from a single-box A/B on 2026-08-19. It is a serving qualification, not a training or fine-tune tree.

**Winner:** in-checkpoint **MTP k=3**, prefix cache on, container pinned to the ten Cortex-X5 cores (`5-9,15-19`).

## Pinned stack

| Piece | Value |
|---|---|
| Target | Local NVFP4 snapshot `Qwen3.8-27B-NVFP4` served as `Qwen3.8-27B` |
| Snapshot | `unsloth/Qwen3.8-27B-NVFP4` (Unsloth Dynamic mixed NVFP4 MLP + FP8 attention) |
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

## Follow-up (2026-08-20)

Same image digest, same Unsloth snapshot, same harness. **D stays the winner.** The rows below are negative results and one same-day D re-measure. Do not treat an 11-hour-old process as the k=3 baseline: after ~11 h uptime, live D had dropped to mean accept 2.92 / 64% draft and C8 112.6. A fresh k=3 boot the same day reproduced C1 at **25.62** (published D was 25.63). Fresh C8 was 116.0 vs 122.2 on 08-19; C1 is the stable claim, C8 moves a few percent day to day.

`tokenizer.json` on this Unsloth snapshot has `"truncation": null`. The community 2048 silent-cut bug does not apply here.

| Config | Spec | Extra | C1 median tok/s | C8 aggregate tok/s | Last mean accept | Last draft accept |
|---|---|---|---:|---:|---:|---:|
| D_fresh (same-day D) | MTP k=3 | prefix on, X5 pin | **25.62** | 116.0 | — | — |
| F | MTP k=4 | prefix on, X5 pin | 24.08 | 100.8 | 3.50 | 62.6% |
| F | MTP k=5 | prefix on, X5 pin | 24.31 | 114.5 | 3.80 | 55.9% |
| I | MTP k=5 | prefix on, X5 pin, `--async-scheduling` | 24.34 | 121.3 | 3.83 | 56.7% |
| G | MTP k=3 | **Inferact** NVFP4 snapshot, X5 pin | 17.37 | 99.2 | 3.16 | 71.9% |
| G | MTP k=5 | **Inferact** NVFP4 snapshot, X5 pin | 16.29 | 80.3 | 3.32 | 46.4% |
| G | MTP k=3 | **RadixArk** ModelOpt NVFP4, X5 pin | — | — | — | — |

RadixArk never became ready: `/v1/models` stayed non-200 for 15 minutes on `v0.27.1`, then the boot was skipped.

What this says:

- **Do not swap Unsloth mixed NVFP4+FP8 for Inferact W4A4 on this box expecting a speedup.** Inferact is the snapshot some vLLM recipes name for Qwen3.8-27B NVFP4. On one GB10 it lost about **32%** C1 (25.62 → 17.37) and about **14%** C8. Higher draft-accept % on Inferact k=3 (71.9%) did not translate into tokens per second.
- **Raising MTP k above 3 does not raise single-stream decode.** k=4 and k=5 lifted mean accept length (3.5–3.8 vs D's 3.39 on 08-19) and cut draft-accept %. Extra verify work ate the gain. C1 stayed ~24.1–24.3 vs 25.6.
- **`--async-scheduling` is a C8-only knob.** On k=5 it moved C8 114.5 → 121.3 (~+6%) and left C1 at 24.3. Versus same-day fresh k=3, C8 is +4.6% (inside the 5% window this tree uses) while C1 is −5%. Kept off the winner.
- Weight swaps and k-sweeps are exhausted for this vLLM pin. A jump toward ~34 tok/s is an engine change (not measured here), not another NVFP4 scale.

Raw JSON: `evidence/bench-D_fresh_k3.json`, `evidence/bench-F_mtp4.json`, `evidence/bench-F_mtp5.json`, `evidence/bench-I_async.json`, `evidence/bench-G_inferact_mtp*.json`. Live-process row (not a claim): `evidence/bench-D_live.json`.

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
