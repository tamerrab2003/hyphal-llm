/*
 * physarum_state.c — HyphalLLM Physarum Global State
 * ====================================================
 * Implements the Physarum polycephalum conductance model
 * applied to transformer attention heads.
 *
 * The core update rule (from Tero et al., Science 2010):
 *   dC_e/dt = |Q_e| - mu * C_e
 * where C_e = conductance of "tube" e (attention head)
 *       Q_e = flow through tube (attention weight)
 *       mu  = decay constant
 *
 * Applied: heads that carry useful attention flow strengthen.
 *          Idle heads decay and are excluded from computation.
 *
 * Author: [Your Name]
 * Paper:  "PhysarumFormer: A Biologically-Zoned Architecture"
 */

#include "physarum_state.h"

/* ── Global state ───────────────────────────────────────────────────── */
float g_physarum_conductances[PHYSARUM_MAX_LAYERS * PHYSARUM_MAX_HEADS];
float g_physarum_surprise[PHYSARUM_MAX_LAYERS];
int   g_physarum_zone[PHYSARUM_MAX_LAYERS];
int   g_physarum_n_layers = 0;
int   g_physarum_n_heads  = 0;
int   g_physarum_step     = 0;

/* ── Init ───────────────────────────────────────────────────────────── */

void physarum_init(int n_layers, int n_heads,
                   float ssm_frac, float gate_frac) {
    g_physarum_n_layers = (n_layers < PHYSARUM_MAX_LAYERS)
                        ? n_layers : PHYSARUM_MAX_LAYERS;
    g_physarum_n_heads  = (n_heads  < PHYSARUM_MAX_HEADS)
                        ? n_heads  : PHYSARUM_MAX_HEADS;
    g_physarum_step     = 0;

    /* All conductances start at 1.0 (equal flow) */
    for (int i = 0; i < g_physarum_n_layers * g_physarum_n_heads; i++) {
        g_physarum_conductances[i] = 1.0f;
    }

    /* Assign zones based on fractions */
    int ssm_end  = (int)(g_physarum_n_layers * ssm_frac);
    int gate_end = (int)(g_physarum_n_layers * (ssm_frac + gate_frac));

    for (int l = 0; l < g_physarum_n_layers; l++) {
        g_physarum_surprise[l] = 0.0f;
        if      (l < ssm_end)  g_physarum_zone[l] = PHYSARUM_ZONE_SSM;
        else if (l < gate_end) g_physarum_zone[l] = PHYSARUM_ZONE_GATE;
        else                   g_physarum_zone[l] = PHYSARUM_ZONE_FULL;
    }

    fprintf(stderr, "[HyphalLLM] physarum_init: %d layers, %d heads, "
                    "SSM=%d Gate=%d Full=%d\n",
            g_physarum_n_layers, g_physarum_n_heads,
            ssm_end, gate_end - ssm_end,
            g_physarum_n_layers - gate_end);
}

/* ── Step ───────────────────────────────────────────────────────────── */

void physarum_step(void) {
    g_physarum_step++;
    /* Global passive decay — Physarum tubes thin when unused */
    int total = g_physarum_n_layers * g_physarum_n_heads;
    for (int i = 0; i < total; i++) {
        g_physarum_conductances[i] *= PHYSARUM_DECAY_RATE;
        if (g_physarum_conductances[i] < PHYSARUM_DEATH_THRESH) {
            g_physarum_conductances[i] = PHYSARUM_DEATH_THRESH;
        }
    }
    /* Decay surprise scores */
    for (int l = 0; l < g_physarum_n_layers; l++) {
        g_physarum_surprise[l] *= 0.98f;
    }
}

/* ── Surprise update ────────────────────────────────────────────────── */

void physarum_update_surprise(int layer, float attention_entropy) {
    if (layer < 0 || layer >= g_physarum_n_layers) return;
    float max_ent = logf((float)(g_physarum_n_layers + 1));
    float norm    = (max_ent > 0.0f) ? attention_entropy / max_ent : 0.5f;
    if (norm > 1.0f) norm = 1.0f;
    /* EMA update */
    g_physarum_surprise[layer] = 0.95f * g_physarum_surprise[layer]
                               + 0.05f * norm;
}

/* ── Zone routing ───────────────────────────────────────────────────── */

int physarum_get_zone(int layer) {
    if (layer < 0 || layer >= g_physarum_n_layers)
        return PHYSARUM_ZONE_FULL;

    /* Fixed zones return immediately */
    if (g_physarum_zone[layer] == PHYSARUM_ZONE_SSM)
        return PHYSARUM_ZONE_SSM;
    if (g_physarum_zone[layer] == PHYSARUM_ZONE_FULL)
        return PHYSARUM_ZONE_FULL;

    /* Gate zone: dynamic routing based on surprise */
    float s = g_physarum_surprise[layer];
    if      (s < PHYSARUM_SSM_SURPRISE)  return PHYSARUM_ZONE_SSM;
    else if (s < PHYSARUM_FULL_SURPRISE) return PHYSARUM_ZONE_GATE;
    else                                 return PHYSARUM_ZONE_FULL;
}

/* ── Head counting ──────────────────────────────────────────────────── */

int physarum_active_heads(int layer) {
    if (layer < 0 || layer >= g_physarum_n_layers) return 0;
    int count = 0;
    for (int h = 0; h < g_physarum_n_heads; h++) {
        if (g_physarum_conductances[layer * PHYSARUM_MAX_HEADS + h]
                > PHYSARUM_DEATH_THRESH) {
            count++;
        }
    }
    return count;
}

float physarum_get_conductance(int layer) {
    if (layer < 0 || layer >= g_physarum_n_layers) return 0.0f;
    float sum = 0.0f;
    for (int h = 0; h < g_physarum_n_heads; h++) {
        sum += g_physarum_conductances[layer * PHYSARUM_MAX_HEADS + h];
    }
    return sum / (float)g_physarum_n_heads;
}

int physarum_active_heads_total(void) {
    int total = 0;
    for (int l = 0; l < g_physarum_n_layers; l++) {
        total += physarum_active_heads(l);
    }
    return total;
}

float physarum_global_sparsity(void) {
    int total  = g_physarum_n_layers * g_physarum_n_heads;
    int active = physarum_active_heads_total();
    return (total > 0) ? 1.0f - (float)active / (float)total : 0.0f;
}

/* ── Reinforcement ──────────────────────────────────────────────────── */

void physarum_reinforce(int layer, int head, float flow) {
    if (layer < 0 || layer >= g_physarum_n_layers) return;
    if (head  < 0 || head  >= g_physarum_n_heads)  return;
    float * c = &g_physarum_conductances[layer * PHYSARUM_MAX_HEADS + head];
    *c += PHYSARUM_GROWTH_RATE * fabsf(flow);
    if (*c > 5.0f) *c = 5.0f;
}

/* ── Persistence ────────────────────────────────────────────────────── */

void physarum_save(const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[HyphalLLM] physarum_save: cannot open %s\n", path);
        return;
    }
    /* Header */
    fwrite("PHYS", 4, 1, f);
    fwrite(&g_physarum_n_layers, sizeof(int), 1, f);
    fwrite(&g_physarum_n_heads,  sizeof(int), 1, f);
    fwrite(&g_physarum_step,     sizeof(int), 1, f);
    /* Data */
    fwrite(g_physarum_conductances, sizeof(float),
           g_physarum_n_layers * g_physarum_n_heads, f);
    fwrite(g_physarum_surprise,     sizeof(float), g_physarum_n_layers, f);
    fwrite(g_physarum_zone,         sizeof(int),   g_physarum_n_layers, f);
    fclose(f);
    fprintf(stderr, "[HyphalLLM] physarum_save: saved %d steps to %s\n",
            g_physarum_step, path);
}

void physarum_load(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[HyphalLLM] physarum_load: %s not found\n", path);
        return;
    }
    char magic[5] = {0};
    fread(magic, 4, 1, f);
    if (strcmp(magic, "PHYS") != 0) {
        fprintf(stderr, "[HyphalLLM] physarum_load: bad magic in %s\n", path);
        fclose(f); return;
    }
    fread(&g_physarum_n_layers, sizeof(int), 1, f);
    fread(&g_physarum_n_heads,  sizeof(int), 1, f);
    fread(&g_physarum_step,     sizeof(int), 1, f);
    fread(g_physarum_conductances, sizeof(float),
          g_physarum_n_layers * g_physarum_n_heads, f);
    fread(g_physarum_surprise,     sizeof(float), g_physarum_n_layers, f);
    fread(g_physarum_zone,         sizeof(int),   g_physarum_n_layers, f);
    fclose(f);
    fprintf(stderr, "[HyphalLLM] physarum_load: loaded step=%d from %s\n",
            g_physarum_step, path);
}
