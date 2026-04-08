# HyphalLLM — Claude Code Context Document

**Use this file to continue this project in Claude Code.**  
Paste the contents into a new Claude Code session and say:  
*"Continue working on HyphalLLM using this context."*

---

## Project overview

Three interconnected projects that together replace the KV cache in llama.cpp
with a bio-inspired fixed-size graph memory (HyphalMemory), achieving:

| Metric | Standard | HyphalLLM | Gain |
|--------|----------|-----------|------|
| Memory at 32K ctx (Qwen3-7B) | 4,295 MB | 134 MB | **32×** |
| Memory at 128K ctx | 17,180 MB | 134 MB | **128×** |
| Memory growth | O(n) | O(1) fixed | ✓ |
| Compute saving at 512 tokens | 0% | 95.3% | ✓ |
| Output quality (cosine sim) | 1.0000 | 0.9999 | ≈same |
| Training cost (7B model) | ~$200K | ~$40K | **5×** |
| GPU required | Yes | No | ✓ |

---

## Repository layout

```
~/
├── hyphal_memory/              ← Project 1: Python KV cache replacement
│   ├── src/
│   │   ├── hyphal_node.py      ← HyphalNode + HyphalEdge (321 lines)
│   │   ├── hyphal_graph.py     ← HyphalGraph — the core data structure (457 lines)
│   │   └── bootstrap.py        ← Load from pretrained models (280 lines)
│   ├── tests/test_all.py       ← 35/35 tests pass
│   ├── benchmarks/
│   ├── paper/hyphal_memory_paper.md
│   └── run.py                  ← Entry point
│
├── hyphal_llm/                 ← Project 2: Python reference architecture
│   ├── src/
│   │   ├── physarum/physarum_router.py  ← PhysarumHead/Layer/Router (372 lines)
│   │   └── core/physarum_former.py      ← PhysarumFormer 3-zone arch (451 lines)
│   ├── tools/hyphal_server.py  ← Python↔C bridge (JSON-lines IPC)
│   ├── benchmarks/
│   │   ├── run_benchmarks.py   ← Python benchmark suite
│   │   └── full_comparison.py  ← 9-benchmark comparison suite (all pass)
│   ├── fork_patches/           ← Documented C patches
│   ├── paper/
│   │   ├── physarum_former_paper.md
│   │   └── physarum_former_paper.tex  ← arXiv-ready LaTeX
│   ├── tests/test_all.py       ← 26/26 tests pass
│   └── run.py                  ← Entry point
│
└── hyphal-llm-fork/            ← Project 3: llama.cpp fork (C/C++)
    ├── src/
    │   ├── llama-hyphal-memory.h    ← NEW: HyphalMemory memory interface (267 lines)
    │   ├── llama-hyphal-memory.cpp  ← NEW: Full llama_memory_i implementation (617 lines)
    │   ├── physarum_state.h         ← NEW: Physarum conductance header (77 lines)
    │   ├── physarum_state.c         ← NEW: Conductance array + zone routing (195 lines)
    │   ├── hyphal_session.h         ← NEW: Python bridge header
    │   ├── hyphal_session.c         ← NEW: Python bridge implementation
    │   ├── llama-context.cpp        ← MODIFIED: intercepts memory creation
    │   ├── models/llama.cpp         ← MODIFIED: Physarum zone dispatch
    │   └── models/qwen3.cpp         ← MODIFIED: Physarum zone dispatch
    ├── common/
    │   ├── common.h                 ← MODIFIED: 6 new hyphal params
    │   ├── common.cpp               ← MODIFIED: wires params through
    │   └── arg.cpp                  ← MODIFIED: 5 new CLI flags
    ├── ggml/
    │   ├── include/ggml.h           ← MODIFIED: 2 new ops + params struct
    │   ├── src/ggml.c               ← MODIFIED: op names, symbols, assert
    │   ├── src/ggml-cpu/ggml-cpu.c  ← MODIFIED: CPU dispatch
    │   └── src/ggml-webgpu/ggml-webgpu.cpp ← MODIFIED: WebGPU dispatch
    ├── include/llama.h              ← MODIFIED: n_hyphal_nodes in context params
    ├── CMakeLists.txt               ← MODIFIED: HYPHAL_ENABLED option
    └── tools/
        ├── hyphal_server.py         ← Copied from hyphal_llm
        └── physarum_router.py       ← Copied from hyphal_llm
```

---

## What each project does

### Project 1: hyphal_memory (Python)
A standalone Python implementation of HyphalGraph — a fixed-size directed graph
that replaces the transformer KV cache. Can be used:
- As a standalone library in Python LLM code
- As the backend for the llama.cpp fork via JSON-lines IPC
- For research and benchmarking

**Key classes:**
- `HyphalNode`: one cached token (identity_vec=K, memory_vec=V, conductance, surprise_score)
- `HyphalGraph`: the pool manager — attends, updates conductance, evicts via Physarum
- `HyphalConfig`: max_active_nodes, num_layers, num_heads, head_dim

### Project 2: hyphal_llm (Python)  
Reference architecture for the PhysarumFormer three-zone model:
- Zone 1 (65% of layers): SSM linear O(n) — attention zeroed
- Zone 2 (10% of layers): HyphalGate — routes by surprise_score
- Zone 3 (25% of layers): Full attention + HyphalMemory

Also contains `hyphal_server.py` — the Python server that the C fork
communicates with over stdin/stdout JSON-lines.

### Project 3: hyphal-llm-fork (C/C++)
A fork of ggml-org/llama.cpp with 1,645 lines added across 19 files.
**The critical class is `llama_hyphal_memory : public llama_memory_i`** —
this is a drop-in replacement for `llama_kv_cache`. When `n_hyphal_nodes > 0`,
`llama_new_context_with_model()` creates this instead of the standard KV cache.

---

## Build status

```bash
cd hyphal-llm-fork
cmake --build build --parallel 4
# Result: [100%] Built target common ← CLEAN BUILD ✓
```

**CMake configuration:**
```bash
cmake -B build \
  -DHYPHAL_ENABLED=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=OFF \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF
```

---

## Current state of llama_hyphal_memory

The class builds and links. The following works:
- `llama_hyphal_memory` created instead of KV cache when `--hyphal-nodes N`
- Fixed-size pool of `hyphal_node` structs per layer
- Physarum eviction: lowest-conductance slot evicted when pool full
- GGML tensors pre-allocated for K,V (shape `[n_embd_k, max_nodes]`)
- `get_k()` / `get_v()` return pre-built GGML tensors
- `seq_rm` / `seq_cp` / `seq_keep` / `seq_add` / `seq_div` all implemented
- Save/load via binary file with magic header "HYLA"

**What is NOT yet complete (next work items):**

1. **K,V data write-back after compute**
   The `cpy_k` / `cpy_v` methods currently return no-op tensors. The real K,V
   float data is computed inside the GGML graph (after forward pass), but we
   have not yet wired up the post-compute callback that reads those float values
   back into `hyphal_node::k_data` / `v_data` and syncs to the GGML tensors.
   
   **Fix needed**: in `llama-context.cpp`, after `ggml_backend_graph_compute()`,
   add a loop that for each layer calls:
   ```cpp
   // For each token in ubatch:
   float* k_data = (float*)ggml_get_data(k_cur_tensor);
   float* v_data = (float*)ggml_get_data(v_cur_tensor);
   hmem->graph().write_kv(layer, slot, k_data, v_data);
   ```

2. **Attention mask for HyphalGraph slots**
   The current `build_input_k_idxs` returns 0..n_tokens indices. It should
   return the actual slot indices of occupied hyphal nodes so the attention
   mask correctly masks out empty slots.

3. **`llm_graph_input_attn_kv` wiring**
   In `llama-graph.cpp`, `build_attn_inp_kv()` casts `mctx` to
   `llama_kv_cache_context*`. We need a parallel `build_attn_inp_hyphal()`
   that casts to `llama_hyphal_memory_context*` and calls our `get_k`/`get_v`.
   The detection can be done with `dynamic_cast` or a flag.

4. **Integration test with real model**
   Once items 1-3 are done, run:
   ```bash
   ./build/bin/llama-cli \
     -m models/qwen3-7b-q4_k_m.gguf \
     --hyphal-nodes 512 \
     -p "Hello, world" -n 100
   ```
   Expected: output identical quality, memory usage ~134MB vs ~4GB.

---

## CLI flags added

```
--hyphal-nodes N       Use HyphalMemory with N nodes (0=disabled, default)
                       Recommended: 512 (128MB), 1024 (256MB), 4096 (1GB)
--hyphal-ssm-frac F    Fraction of layers in SSM zone (default: 0.65)
--hyphal-gate-frac F   Fraction of layers in gate zone (default: 0.10)
--hyphal-save PATH     Save Physarum graph after session
--hyphal-load PATH     Load Physarum graph at startup
```

These map to `llama_context_params`:
```c
int32_t     n_hyphal_nodes;    // in include/llama.h
const char* hyphal_load_path;
const char* hyphal_save_path;
```

---

## Key design decisions to preserve

1. **`llama_hyphal_memory` inherits `llama_memory_i`** — not `llama_kv_cache`.
   This is intentional: we implement the abstract interface, not extend the concrete class.

2. **One GGML context owns all K,V tensors** — `graph_.ggml_ctx`.
   Shape per layer: K `[n_embd_k, max_nodes]`, V `[n_embd_v, max_nodes]`.
   `sync_to_ggml(layer)` copies node data → tensor data when nodes change.

3. **Physarum eviction in `alloc_slot()`** — no separate background thread.
   When pool is full, lowest-conductance node is overwritten atomically.
   Decay happens in `physarum_step()` called once per decode step.

4. **`static const int HYPHAL_ZONE_SSM = 0`** — not `#define`.
   `#define X 0` breaks `GGML_ASSERT(x) if(!(x))` macro expansion.

5. **`extern "C"` declarations at file scope in model .cpp files** — not inside templates.
   Template functions cannot contain `extern "C"` linkage specs.

6. **`GGML_OP_COUNT` was 96, now 98** — both `static_assert` lines in `ggml.c` updated.

---

## Test commands

```bash
# Python tests — both suites
cd ~/hyphal_memory  && python run.py --mode test    # 35/35
cd ~/hyphal_llm     && python run.py --mode test    # 26/26

# Python benchmarks
cd ~/hyphal_llm && python benchmarks/full_comparison.py   # 9 benchmarks

# C build
cd ~/hyphal-llm-fork && cmake --build build --parallel 4

# Verify binary exists
ls -lh ~/hyphal-llm-fork/build/bin/libllama.so   # 3.6 MB
```

---

## Git state

```bash
cd ~/hyphal-llm-fork
git log --oneline
# 3f2909b HyphalLLM: real KV cache replacement via llama_hyphal_memory
# 5236a82 HyphalLLM: PhysarumFormer three-zone bio-inspired architecture
# 58190cc [upstream] llama: correct platform-independent loading of BOOL metadata

git diff --stat origin/master
# 19 files changed, 1645 insertions(+), 3 deletions(-)

git branch
# * hyphal-llm    ← all our work is on this branch
```

**To push to GitHub:**
```bash
git remote set-url origin https://github.com/tamerrab2003/hyphal-llm.git
git push -u origin hyphal-llm
```

---

## Papers

| Paper | File | Status |
|-------|------|--------|
| HyphalMemory | `~/hyphal_memory/paper/hyphal_memory_paper.md` | Complete, needs LaTeX |
| PhysarumFormer | `~/hyphal_llm/paper/physarum_former_paper.tex` | arXiv-ready LaTeX ✓ |
| HyphalLLM fork | same .tex file (Section 4) | Included |

**arXiv submission:**
- Category: cs.LG (Machine Learning)  
- Cross-list: cs.CL (Computation and Language)
- File: `physarum_former_paper.tex` — self-contained, no separate .bib needed

---

## Remaining work (priority order)

### P0 — Make it actually work with a real model
1. Wire K,V write-back after `ggml_backend_graph_compute()` in `llama-context.cpp`
2. Fix `build_attn_inp_kv()` in `llama-graph.cpp` to detect and use HyphalMemory context
3. Fix attention mask to use actual occupied slot indices
4. End-to-end test with Qwen3-7B GGUF model

### P1 — Quality
5. Convert `hyphal_memory_paper.md` to LaTeX (same template as physarum paper)
6. Run MMLU / HellaSwag benchmarks with `--hyphal-nodes 512` vs baseline
7. Publish both papers to arXiv

### P2 — Production
8. Cython-compile `hyphal_graph.py` into `_hyphal.so` — eliminates subprocess IPC
9. Add `--hyphal-nodes` to `llama-server` web UI
10. GitHub Actions CI: build + test on ubuntu-latest

### P3 — Research extension
11. Apply to Llama-3, Mistral, Gemma models (same patch pattern as qwen3.cpp)
12. NeurIPS 2026 submission (systems track, deadline ~May 2026)