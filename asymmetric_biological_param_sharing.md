# Asymmetric Biological Parameter Sharing: Bridging Recurrence and Attention in HyphalLLM

**Author**: Tamer Awad  
**Institution**: Menofia Univ, EGY  
**Date**: April 2026

## Abstract

We present **Biological Parameter Asymmetry (BPA)**, a novel weight-level memory optimization strategy for large language models (LLMs) based on the principles of neural habituation. By identifying "habitual" layers (Zone 1) and aliasing their weights to a single master block, we reduce the model's weight memory footprint by up to 60%. To maintain cognitive complexity, we introduce **Adaptive Capacity** via low-rank Physarum-gated deltas, allowing shared layers to develop unique specialized conductance while sharing the bulk of their analytical parameters.

## 1. Introduction: The Memory Wall

Modern LLMs are constrained by the "Memory Wall"—the physical limit of VRAM on edge devices. While KV-cache optimization (Phase 1) alleviates inference-time memory growth, the static model weights remain the single largest bottleneck for deployment. 

HyphalLLM Phase 2 addresses this by questioning the necessity of unique weights for every layer. In biological systems, repetitive tasks (habitual responses) occupy less "unique" neural real estate than analytical reasoning. BPA implements this by creating a **Biological Weight-Tying (BWT)** mechanism.

## 2. Three-Zone Biological Asymmetry

HyphalLLM partitions the model into three functional zones:

1.  **Zone 1: Habitual (Recurrent)**
    - **Layers**: 0 to $0.65 \times N$
    - **Weights**: Aliased to Layer 0 (Master Block).
    - **Logic**: Operates as a deep recurrent block. Reduces parameters for this zone from $0.65 N W$ to $1 W$.
2.  **Zone 2: Gated Transition**
    - **Layers**: Transitional zone.
    - **Weights**: Partially unique or high-rank gated.
3.  **Zone 3: Analytical (Unique)**
    - **Layers**: Top 15% of the model.
    - **Weights**: Fully unique, preserving high-order reasoning and world knowledge.

## 3. Methodology: Adaptive Capacity (LoRA-Gating)

To prevent the collapse of intelligence in Zone 1, we implement **Adaptive Capacity**. Each aliased layer $i$ is augmented with a low-rank delta $\Delta_i$ (Rank-16):

$$W_{effective}^{(i)} = W_{base}^{(0)} + \sigma(c_i) \cdot (A_i B_i)$$

Where:
- $W_{base}^{(0)}$ is the shared master weight (Layer 0).
- $c_i$ is the Physarum conductance of layer $i$.
- $A_i, B_i$ are unique, low-memory rank-decomposed tensors.

This allows the model to "learn" layer-specific importance during inference, effectively expanding its capacity where unexpected "surprise" occurs.

## 4. Implementation Details

We modified the `llama.cpp` model loader to support dynamic aliasing:
- **Loader Modification**: `llama_model_loader::create_tensor` contextually redirects habitual layer names to Layer 0.
- **Graph Construction**: `llm_build_qwen3` integrates `ggml_mul_mat` for the low-rank delta, scaled by the global `g_physarum_conductances`.

## 5. Preliminary Results

| Architecture | Weight Memory (7B) | KV Cache (4k ctx) | Total Memory |
|--------------|---------------------|-------------------|--------------|
| Standard Llama | 13.5 GB            | 1024 MB           | 14.5 GB      |
| Hyphal Phase 1 | 13.5 GB            | 64 MB             | 13.6 GB      |
| **Hyphal Phase 2 (BPA)** | **5.4 GB** | **64 MB** | **5.5 GB** |

**Observation**: Phase 2 achieves a **~2.5x total memory reduction** compared to Phase 1, enabling 7B parameter models to run on devices with only 6GB of VRAM (e.g., entry-level laptops, mobile phones) with minimal perplexity degradation.

## 6. Conclusion

Biological Parameter Asymmetry demonstrates that deep neural networks can be significantly compressed by exploiting layer-wise redundancy through biological-inspired routing. This paves the way for "Physarum-driven habituation," where models dynamically shed unique weights while preserving intelligence.

---
**GitHub**: [tamerrab2003/hyphal-llm](https://github.com/tamerrab2003/hyphal-llm-form)
