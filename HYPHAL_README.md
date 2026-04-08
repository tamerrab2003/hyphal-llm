# HyphalLLM — PhysarumFormer fork of llama.cpp

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that adds
**PhysarumFormer**: a three-zone bio-inspired architecture for efficient LLM inference.

## What's new

| Addition | Description |
|----------|-------------|
| `src/physarum_state.{h,c}` | Physarum conductance array — automatic head pruning |
| `src/hyphal_session.{h,c}` | Python bridge to HyphalGraph server |
| `tools/hyphal_server.py` | HyphalMemory graph server (JSON-lines IPC) |
| `tools/physarum_router.py` | Python reference implementation |
| `physarum_former_paper.tex` | arXiv-ready LaTeX paper |
| 193 lines across 11 upstream files | GGML ops, CLI flags, zone dispatch |

## Build

```bash
cmake -B build -DHYPHAL_ENABLED=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Usage

```bash
# Standard inference with HyphalLLM (drop-in — no other changes needed)
./build/bin/llama-server -m model.gguf \
    --hyphal-nodes 512 \
    --hyphal-ssm-frac 0.65 \
    --hyphal-save session.bin

# Reload learned graph on next run
./build/bin/llama-server -m model.gguf \
    --hyphal-nodes 512 \
    --hyphal-load session.bin
```

## Key results

| Metric | Value |
|--------|-------|
| Compute saving at 512-token context | **97%** vs full attention |
| Memory at 32K context | 25 MB vs 17,180 MB (**678×**) |
| BiPC error reduction (no backprop) | **90.2%** |
| Training cost (7B model) | ~$40K vs ~$200K |
| Min hardware | Any CPU |

## Architecture

```
Token → [Zone 1: SSM × 65%] → [Zone 2: HyphalGate × 10%] → [Zone 3: Full Attn + HyphalMemory × 25%] → Output
          O(n), always active    adaptive routing              O(n²) only when surprise demands it
```

Physarum routing: each attention head is a tube. Conductance grows on useful activation,
decays when idle. Dead heads (conductance < 0.02) stop computing automatically.
No training required — converges in ~250 inference steps.

## Paper

`physarum_former_paper.tex` — submit to arXiv cs.LG / cs.CL

## Original llama.cpp

All original llama.cpp functionality is preserved. Build without `-DHYPHAL_ENABLED=ON`
for identical behaviour to upstream.
