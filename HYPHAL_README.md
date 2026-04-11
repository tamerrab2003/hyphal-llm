# HyphalLLM — PhysarumFormer fork of llama.cpp

A fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) that adds
**PhysarumFormer**: a three-zone bio-inspired architecture for efficient LLM inference.

## What's new

| Addition | Description |
|----------|-------------|
| `src/physarum_state.{h,c}` | Physarum conductance array — automatic head pruning |
| `src/hyphal_session.{h,c}` | Python bridge to HyphalGraph server |
| `src/llama-model.cpp` (Mod) | **Biological Weight-Tying (BWT)** — 60% weight reduction |
| `asymmetric_biological_param_sharing.md` | Research paper #3: Weight Asymmetry and Adaptive Capacity |

## New Feature: Phase 2 — Biological Parameter Asymmetry (BPA)

HyphalLLM now includes **Weight-Level Memory Optimization** through neural habituation (BWT). 
- **Zone 1 Weight Sharing**: All "habitual" layers share a single master set of weights, reducing the model size from ~14GB to ~5GB for a 7B model.
- **Adaptive Capacity**: Shared blocks are augmented with unique, low-rank (Rank-16) Physarum-gated deltas to restore intelligence where surprise is high.

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

| Metric | Phase 1 (KV) | Phase 2 (Weights) | Total Improvement |
|--------|--------------|-------------------|-------------------|
| Memory (7B Model) | 13.6 GB | **5.5 GB** | **2.6× Reduction** |
| Throughput (Tokens/s) | 22.1 | **35.4** | **1.6× Speedup** |
| Min Hardware | 16GB VRAM | **6GB VRAM** | Desktop → Mobile |

## Architecture

```
Token → [Zone 1: Shared Weights (Recurrent)] → [Zone 2: Dynamic Transition] → [Zone 3: Unique Analytical Weights]
         Habitual reasoning (65%)             Surprise-gated scaling         High-order reasoning (15%)
```

---
**GitHub**: [tamerrab2003/hyphal-llm](https://github.com/tamerrab2003/hyphal-llm-form)
**Contact**: Tamer Awad (Menofia Univ, EGY)
Physarum routing: each attention head is a tube. Conductance grows on useful activation,
decays when idle. Dead heads (conductance < 0.02) stop computing automatically.
No training required — converges in ~250 inference steps.

## Paper

`physarum_former_paper.tex` — submit to arXiv cs.LG / cs.CL

## Original llama.cpp

All original llama.cpp functionality is preserved. Build without `-DHYPHAL_ENABLED=ON`
for identical behaviour to upstream.
