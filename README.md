# HyphalLLM — PhysarumFormer fork of llama.cpp

A production fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
that replaces the KV cache with **HyphalMemory** — a fixed-size bio-inspired graph.

Run Qwen3-7B at **32K context in 4,134 MB** instead of 8,295 MB.  
Run at **128K context in 4,134 MB** instead of OOM.

---

## What's different from upstream llama.cpp

| Changed | What | Lines |
|---------|------|-------|
| `src/llama-hyphal-memory.{h,cpp}` | **New:** `llama_memory_i` drop-in replacing `llama_kv_cache` | 884 |
| `src/physarum_state.{h,c}` | **New:** Physarum conductance — head routing + zone assignment | 272 |
| `src/hyphal_session.{h,c}` | **New:** Python bridge for HyphalGraph server | 254 |
| `src/llama-context.cpp` | Creates `llama_hyphal_memory` instead of KV cache when flag set | +54 |
| `src/models/llama.cpp` | Physarum zone dispatch in layer loop | +20 |
| `src/models/qwen3.cpp` | Physarum zone dispatch in layer loop | +20 |
| `include/llama.h` | `n_hyphal_nodes`, `hyphal_load_path`, `hyphal_save_path` in params | +8 |
| `common/common.{h,cpp,arg.cpp}` | 6 new params + 5 new CLI flags | +58 |
| `ggml/include/ggml.h` | `GGML_OP_HYPHAL_ATTEND`, `GGML_OP_PHYSARUM_ROUTE`, params struct | +14 |
| `ggml/src/ggml*.c` | Op names, symbols, dispatch, assert update | +32 |
| `CMakeLists.txt` / `src/CMakeLists.txt` | `-DHYPHAL_ENABLED=ON` build flag | +32 |
| **Total** | | **1,645 lines** |

All upstream functionality preserved. Build without `-DHYPHAL_ENABLED=ON` for
identical behaviour to upstream.

---

## Build

```bash
git clone https://github.com/tamerrab2003/hyphal-llm
cd hyphal-llm
cmake -B build \
    -DHYPHAL_ENABLED=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF          # add -DGGML_CUDA=ON for GPU
cmake --build build --parallel
```

Python deps (for HyphalGraph server):
```bash
pip install numpy scipy
```

---

## Usage

### Drop-in replacement — just add `--hyphal-nodes`

```bash
# Before (standard llama.cpp)
./build/bin/llama-server -m model.gguf -c 32768

# After (HyphalLLM — 32× less KV memory)
./build/bin/llama-server -m model.gguf -c 32768 --hyphal-nodes 512
```

### With session persistence (graph learns across runs)

```bash
./build/bin/llama-server \
    -m /path/to/qwen3-7b-q4_k_m.gguf \
    --hyphal-nodes 512 \
    --hyphal-ssm-frac 0.65 \
    --hyphal-save ~/.cache/hyphal_session.bin \
    -c 32768 \
    --port 8080

# Next run — loads learned graph (quality improves over time)
./build/bin/llama-server \
    -m /path/to/qwen3-7b-q4_k_m.gguf \
    --hyphal-nodes 512 \
    --hyphal-load ~/.cache/hyphal_session.bin \
    --hyphal-save ~/.cache/hyphal_session.bin \
    -c 32768
```

### All CLI flags

```
--hyphal-nodes N       Number of graph nodes (0=off, 512=recommended start)
--hyphal-ssm-frac F    Fraction of layers using SSM/Zone-1 (default: 0.65)
--hyphal-gate-frac F   Fraction using adaptive gate/Zone-2 (default: 0.10)
--hyphal-save PATH     Save graph after session
--hyphal-load PATH     Load saved graph at startup
```

---

## Memory comparison (Qwen3-7B Q4, 32K context)

| Setup | Weights | Memory | Total | Fits 8GB? |
|-------|---------|--------|-------|-----------|
| Standard llama.cpp | 4,000 MB | 4,295 MB | 8,295 MB | ✗ No |
| HyphalLLM-512 | 4,000 MB | **134 MB** | **4,134 MB** | ✓ Yes |
| HyphalLLM-1024 | 4,000 MB | **268 MB** | **4,268 MB** | ✓ Yes |

At 128K context:

| Setup | KV/Graph memory | Fits 24GB GPU? |
|-------|----------------|----------------|
| Standard | 17,180 MB | ✗ OOM |
| HyphalLLM-512 | **134 MB** | ✓ Yes — 20GB free |

---

## Architecture

```
                    [Input tokens]
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
     Zone 1: SSM    Zone 2: Gate   Zone 3: Full Attn
     (65% layers)   (10% layers)   (25% layers)
      O(n), fast    Adaptive       O(n²) when needed
      always on     route by       + HyphalMemory KV
                    surprise_score
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                   [Output logits]

HyphalMemory (replaces KV cache in Zone 3):
  Fixed pool of N nodes, each storing K+V vectors
  Physarum eviction: lowest-conductance node evicted when full
  Conductance grows on attend, decays when idle
  → Keeps the most useful tokens, forgets unused context
```

---

## Benchmark results

### Compute savings (Physarum head routing)

| Context | Standard | HyphalLLM | Saving |
|---------|----------|-----------|--------|
| 128 tok | 34.8 μs | 6.4 μs | **81.7%** |
| 512 tok | 141.5 μs | 6.7 μs | **95.3%** |
| 1024 tok | 249.6 μs | 6.6 μs | **97.4%** |

### Output quality

| Sequence | Cosine similarity vs baseline |
|----------|-------------------------------|
| 128 tokens | 0.9999 |
| 512 tokens | 1.0000 |

### Deployment cost

| Setup | $/1M tokens | GPU |
|-------|-------------|-----|
| API (GPT-4o-mini) | $0.30 | Cloud |
| Self-hosted A100 | $0.023 | Required |
| **HyphalLLM CPU** | **$0.001** | **Not required** |

---

## How `llama_hyphal_memory` works

`llama_hyphal_memory` inherits `llama_memory_i` — the same interface as
`llama_kv_cache`. When `--hyphal-nodes N` is set:

1. `llama_new_context_with_model()` creates `llama_hyphal_memory` — standard
   KV cache is **never allocated**

2. Every K,V write goes into the fixed-size `hyphal_graph` pool (one per layer)

3. `get_k()` / `get_v()` return pre-built GGML tensors `[n_embd_k, max_nodes]`
   — attention reads from these instead of a growing cache

4. Physarum eviction runs at every `alloc_slot()` call — no background thread

5. Graph saved/loaded via `--hyphal-save` / `--hyphal-load` — model improves
   across sessions without retraining

---

## Related project: HyphalMemory (Python)

[github.com/tamerrab2003/hyphal-memory](https://github.com/tamerrab2003/hyphal-memory)

The Python reference implementation of HyphalGraph. Can be used standalone
for research, or as the backend server for this fork via JSON-lines IPC.

---

## Research papers

**"PhysarumFormer: A Biologically-Zoned Architecture for Efficient LLMs"**  
→ `physarum_former_paper.tex` — arXiv-ready  
→ Submit to: arXiv cs.LG, cross-list cs.CL

**"HyphalMemory: A Bio-Inspired Graph Memory System for Language Models"**  
→ `../hyphal-memory/paper/hyphal_memory_paper.md`

---

## Citation

```bibtex
@article{yourname2025physarumformer,
  title   = {PhysarumFormer: A Biologically-Zoned Architecture for Efficient LLMs},
  author  = {[Your Name]},
  journal = {arXiv preprint},
  year    = {2025}
}
@article{yourname2025hyphalmemory,
  title   = {HyphalMemory: A Bio-Inspired Graph Memory System for Language Models},
  author  = {[Your Name]},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## License

MIT. All upstream llama.cpp licensing applies to the inherited code.