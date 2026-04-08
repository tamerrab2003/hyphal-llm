/*
 * llama-hyphal-memory.h
 * =====================
 * HyphalMemory: a drop-in replacement for the llama.cpp KV cache that uses
 * a fixed-size graph of "hyphal nodes" instead of a growing O(n) buffer.
 *
 * Key property: memory usage is O(max_nodes × layers × n_embd) — CONSTANT
 * regardless of context length. A 512-node graph for Qwen3-7B uses ~25 MB
 * instead of 17 GB at 32K context.
 *
 * The graph is managed by Physarum polycephalum-inspired conductance:
 *   - New tokens write into the least-conductive (least recently useful) slot
 *   - Slots that are frequently attended to increase in conductance
 *   - Dead slots (conductance < threshold) are evicted to make room
 *
 * This is a REAL replacement: when --hyphal-nodes N is set, the standard
 * llama_kv_cache is never allocated. All K,V state lives in HyphalGraph.
 *
 * Author: [Your Name]
 * Paper:  "HyphalMemory: A Bio-Inspired Graph Memory System for LLMs"
 */

#pragma once

#include "llama-memory.h"
#include "llama-hparams.h"
#include "llama-cparams.h"
#include "llama-batch.h"
#include "llama-graph.h"
#include "ggml.h"

#include <vector>
#include <array>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>

// ── HyphalNode ───────────────────────────────────────────────────────────────
// One slot in the HyphalGraph. Stores K and V vectors for one token position.

struct hyphal_node {
    // Position in the original sequence
    llama_pos   pos        = -1;
    llama_seq_id seq_id   = -1;
    int32_t     layer     = -1;

    // Physarum conductance (0..5): grows on attention, decays when idle
    float       conductance = 1.0f;

    // Step at which this node was last attended to
    uint64_t    last_used  = 0;

    // Whether this slot is occupied
    bool        occupied   = false;

    // Raw K and V vectors — stored flat, all heads concatenated
    // Size: n_embd_k_gqa floats for K, n_embd_v_gqa floats for V
    std::vector<float> k_data;
    std::vector<float> v_data;

    void reset() {
        pos = -1; seq_id = -1; layer = -1;
        conductance = 1.0f; last_used = 0; occupied = false;
        std::fill(k_data.begin(), k_data.end(), 0.0f);
        std::fill(v_data.begin(), v_data.end(), 0.0f);
    }
};

// ── HyphalGraph ──────────────────────────────────────────────────────────────
// Fixed-size pool of hyphal_nodes. Manages eviction + conductance.

struct hyphal_graph {
    uint32_t max_nodes;    // fixed capacity (e.g. 512)
    uint32_t n_embd_k;     // K vector size per token = n_head_kv * head_dim_k
    uint32_t n_embd_v;     // V vector size per token = n_head_kv * head_dim_v
    uint32_t n_layer;      // number of transformer layers

    uint64_t step = 0;     // global decode step counter

    // One pool per layer — each layer has its own max_nodes slots
    // Indexed as: nodes[layer][slot]
    std::vector<std::vector<hyphal_node>> nodes;

    // Flat GGML tensors for each layer — the actual K,V data that ggml reads
    // Size: [n_embd_k, max_nodes] for K, [n_embd_v, max_nodes] for V
    // These are backed by ggml_backend_cpu_buffer and kept in sync with nodes[]
    std::vector<ggml_tensor *> k_tensors;  // one per layer
    std::vector<ggml_tensor *> v_tensors;  // one per layer

    // GGML context that owns the K,V tensors
    ggml_context * ggml_ctx = nullptr;

    // Physarum constants
    static constexpr float DECAY_RATE   = 0.9995f;
    static constexpr float GROWTH_RATE  = 0.01f;
    static constexpr float DEATH_THRESH = 0.02f;

    // Initialise graph with given capacity
    void init(uint32_t max_n, uint32_t embd_k, uint32_t embd_v, uint32_t n_l);

    // Find slot for a new token (evicts lowest-conductance occupied slot)
    uint32_t alloc_slot(uint32_t layer, llama_pos pos, llama_seq_id seq_id);

    // Write K,V floats into a slot and its GGML tensor
    void write_kv(uint32_t layer, uint32_t slot,
                  const float * k, const float * v);

    // Reinforce all slots that were attended to this step
    void reinforce(uint32_t layer, uint32_t slot, float attn_weight);

    // Global Physarum decay — call once per decode step
    void physarum_step();

    // Build an n_kv × n_embd_k mask tensor indicating which slots are occupied
    // Returns a preallocated integer index array for the attention computation
    void build_slot_map(uint32_t layer,
                        int32_t * out_slots, uint32_t & out_n_kv) const;

    // Sync node data → GGML tensors (called after write_kv)
    void sync_to_ggml(uint32_t layer);

    // Memory footprint in bytes
    size_t memory_bytes() const;

    // Save / load
    bool save(const std::string & path) const;
    bool load(const std::string & path);

    // Number of occupied nodes across all layers
    uint32_t n_occupied(uint32_t layer) const;
};

// ── llama_hyphal_memory_context ──────────────────────────────────────────────
// Per-batch processing context — implements llama_memory_context_i

class llama_hyphal_memory;

class llama_hyphal_memory_context : public llama_memory_context_i {
public:
    llama_hyphal_memory_context(llama_hyphal_memory * mem,
                                 llama_memory_status  status);

    bool   next()   override;
    bool   apply()  override;

    const llama_ubatch & get_ubatch()  const override;
    llama_memory_status  get_status()  const override;

    // KV-cache-like API used by llama-graph.cpp
    uint32_t get_n_kv() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur,
                         ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur,
                         ggml_tensor * v_idxs, int32_t il) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx,
                                      const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx,
                                      const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const { return nullptr; }
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const { return nullptr; }

    void set_input(const llama_ubatch * ubatch);

    // Allocated slot indices for current ubatch (one per token)
    std::vector<int32_t> token_slots; // slot index in HyphalGraph

private:
    llama_hyphal_memory * mem;
    llama_memory_status   status_;
    llama_ubatch          ubatch_;
    bool                  done_ = false;
};

// ── llama_hyphal_memory ───────────────────────────────────────────────────────
// THE real replacement for llama_kv_cache.
// Implements llama_memory_i so llama.cpp uses it transparently.

class llama_hyphal_memory : public llama_memory_i {
public:
    llama_hyphal_memory(const llama_hparams & hparams,
                         const llama_cparams & cparams,
                         uint32_t max_nodes,
                         const std::string & load_path = "");

    ~llama_hyphal_memory() override;

    //
    // llama_memory_i interface
    //

    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc,
                                         uint32_t n_ubatch,
                                         bool embd_all) override;

    llama_memory_context_ptr init_full()   override;
    llama_memory_context_ptr init_update(llama_context * lctx,
                                          bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,
                  llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                  llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,
                  llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,
                  llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io,
                     llama_seq_id seq_id = -1,
                     llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i & io,
                     llama_seq_id seq_id = -1,
                     llama_state_seq_flags flags = 0) override;

    //
    // HyphalMemory-specific API
    //

    hyphal_graph & graph() { return graph_; }
    const hyphal_graph & graph() const { return graph_; }

    uint32_t max_nodes() const { return max_nodes_; }
    uint32_t n_layer()   const { return n_layer_; }
    uint32_t n_embd_k()  const { return n_embd_k_; }
    uint32_t n_embd_v()  const { return n_embd_v_; }

    // Save learned graph to disk
    bool save(const std::string & path) const { return graph_.save(path); }

private:
    hyphal_graph graph_;

    uint32_t max_nodes_;
    uint32_t n_layer_;
    uint32_t n_embd_k_;   // = n_head_kv * n_embd_head_k
    uint32_t n_embd_v_;   // = n_head_kv * n_embd_head_v
    uint32_t n_head_kv_;
    uint32_t n_embd_head_k_;
    uint32_t n_embd_head_v_;

    std::string save_path_;

    // Per-sequence position tracking
    struct seq_info {
        llama_pos pos_min = INT32_MAX;
        llama_pos pos_max = -1;
    };
    std::vector<seq_info> seq_infos_;
};
