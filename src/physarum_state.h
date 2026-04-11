/*
 * physarum_state.h — HyphalLLM Physarum Global State
 * ====================================================
 * Global conductance array and zone management for the
 * PhysarumFormer three-zone architecture.
 *
 * Include guard: PHYSARUM_STATE_H
 * Compile with: -DHYPHAL_ENABLED (via CMake -DHYPHAL_ENABLED=ON)
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* ── Constants ──────────────────────────────────────────────────────── */
#define PHYSARUM_MAX_LAYERS  128
#define PHYSARUM_MAX_HEADS   128
#define PHYSARUM_DECAY_RATE  0.9998f
#define PHYSARUM_GROWTH_RATE 0.005f
#define PHYSARUM_DEATH_THRESH 0.02f
#define PHYSARUM_SSM_SURPRISE  0.15f
#define PHYSARUM_FULL_SURPRISE 0.40f

/* Zones */
#define PHYSARUM_ZONE_SSM  0
#define PHYSARUM_ZONE_GATE 1
#define PHYSARUM_ZONE_FULL 2

/* ── Global state (thread-safe read, write from decode thread only) ── */
extern float g_physarum_conductances[PHYSARUM_MAX_LAYERS * PHYSARUM_MAX_HEADS];
extern float g_physarum_surprise[PHYSARUM_MAX_LAYERS];
extern int   g_physarum_zone[PHYSARUM_MAX_LAYERS];
extern int   g_physarum_n_layers;
extern int   g_physarum_n_heads;
extern int   g_physarum_step;

/* ── API ────────────────────────────────────────────────────────────── */

/* Initialise with zone fractions (call once after model load) */
void physarum_init(int n_layers, int n_heads,
                   float ssm_frac, float gate_frac);

/* Call after each token decode step — applies global passive decay */
void physarum_step(void);

/* Update surprise score for a layer (called after attention) */
void physarum_update_surprise(int layer, float attention_entropy);

/* Get effective zone for a layer (respects gate dynamics) */
int physarum_get_zone(int layer);

/* Count active heads for a layer */
int physarum_active_heads(int layer);
float physarum_get_conductance(int layer);

/* Count total active heads across all layers */
int physarum_active_heads_total(void);

/* Compute global sparsity (0.0 = no pruning, 1.0 = all dead) */
float physarum_global_sparsity(void);

/* Reinforce a head that produced useful output */
void physarum_reinforce(int layer, int head, float flow);

/* Save/load conductance state for session persistence */
void physarum_save(const char * path);
void physarum_load(const char * path);

#ifdef __cplusplus
}
#endif
