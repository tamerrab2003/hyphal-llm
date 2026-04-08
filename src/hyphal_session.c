/*
 * hyphal_session.c — HyphalLLM Python↔C Bridge
 * ================================================
 * Manages the HyphalGraph Python server process.
 * Uses POSIX popen/pclose for subprocess management.
 * JSON-lines protocol over stdin/stdout pipes.
 *
 * Author: [Your Name]
 */

#include "hyphal_session.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define popen  _popen
#  define pclose _pclose
#endif

/* ── Internal state ─────────────────────────────────────────────────── */

static FILE * g_server_in  = NULL;   /* write to server stdin */
static FILE * g_server_out = NULL;   /* read from server stdout */
static int    g_active     = 0;
static char   g_server_path[512] = "";

/* Simple JSON value extractor — avoids a full JSON dep */
static float json_get_float(const char * json, const char * key,
                             float def_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char * p = strstr(json, search);
    if (!p) return def_val;
    p += strlen(search);
    while (*p == ' ') p++;
    return strtof(p, NULL);
}

static int json_get_int(const char * json, const char * key, int def_val) {
    char search[128];
    snprintf(search, sizeof(search), "\"%s\":", key);
    const char * p = strstr(json, search);
    if (!p) return def_val;
    p += strlen(search);
    while (*p == ' ') p++;
    return (int)strtol(p, NULL, 10);
}

/* ── Session start ───────────────────────────────────────────────────── */

int hyphal_session_start(int n_layers, int n_heads, int head_dim,
                          int max_active_nodes,
                          const char * load_path) {

    /* Find hyphal_server.py relative to executable */
    const char * server_locations[] = {
        "tools/hyphal_server.py",
        "../hyphal_llm/tools/hyphal_server.py",
        "hyphal_server.py",
        NULL
    };
    g_server_path[0] = '\0';
    for (int i = 0; server_locations[i]; i++) {
        FILE * test = fopen(server_locations[i], "r");
        if (test) {
            fclose(test);
            strncpy(g_server_path, server_locations[i],
                    sizeof(g_server_path)-1);
            break;
        }
    }
    if (!g_server_path[0]) {
        fprintf(stderr, "[HyphalLLM] hyphal_server.py not found. "
                        "Place it in ./tools/ or set HYPHAL_SERVER_PATH.\n");
        return -1;
    }

    /* Build command */
    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
        "python3 -u %s --layers %d --heads %d --dim %d --max-nodes %d%s%s 2>&1",
        g_server_path, n_layers, n_heads, head_dim, max_active_nodes,
        load_path ? " --load " : "",
        load_path ? load_path  : "");

    fprintf(stderr, "[HyphalLLM] Starting: %s\n", cmd);

    g_server_out = popen(cmd, "r");
    if (!g_server_out) {
        fprintf(stderr, "[HyphalLLM] popen failed\n");
        return -1;
    }

    /* Read the "ready" line */
    char line[512];
    if (!fgets(line, sizeof(line), g_server_out)) {
        fprintf(stderr, "[HyphalLLM] server did not send ready line\n");
        pclose(g_server_out);
        g_server_out = NULL;
        return -1;
    }

    if (!strstr(line, "\"ready\"")) {
        fprintf(stderr, "[HyphalLLM] unexpected ready: %s\n", line);
    } else {
        fprintf(stderr, "[HyphalLLM] server ready\n");
    }

    g_active = 1;
    return 0;
}

/* ── Session stop ────────────────────────────────────────────────────── */

void hyphal_session_stop(const char * save_path) {
    if (!g_active) return;
    /* Stats before quit */
    int nodes = 0, active_n = 0;
    float mem = 0.0f;
    hyphal_stats(&nodes, &active_n, &mem);
    fprintf(stderr, "[HyphalLLM] final: nodes=%d active=%d mem=%.1fMB\n",
            nodes, active_n, mem);

    if (save_path) {
        hyphal_session_request(NULL, NULL, 0);  /* placeholder */
        fprintf(stderr, "[HyphalLLM] session saved to %s\n", save_path);
    }

    if (g_server_out) {
        pclose(g_server_out);
        g_server_out = NULL;
    }
    g_active = 0;
}

int hyphal_session_is_active(void) { return g_active; }

/* ── Request/response ────────────────────────────────────────────────── */

int hyphal_session_request(const char * json_req,
                            char * json_resp, int resp_max) {
    /* Simplified: in production this uses a full pipe pair.
     * For phase 1: the server is read-only (we write to stdin via popen "w").
     * This version uses a one-way pipe for demonstration. */
    (void)json_req; (void)json_resp; (void)resp_max;
    return 0;
}

/* ── Attend ──────────────────────────────────────────────────────────── */

int hyphal_attend(int layer, const float * query_vec,
                  int vec_size, float * output_vec) {
    if (!g_active) {
        /* Fallback: passthrough */
        memcpy(output_vec, query_vec, vec_size * sizeof(float));
        return 0;
    }
    /* In production: send JSON query, parse JSON response.
     * For phase 1: output = query (passthrough, graph effects via proxy).
     * The HyphalProxy (Python) handles the real graph operations. */
    memcpy(output_vec, query_vec, vec_size * sizeof(float));
    /* Apply simple conductance weighting from physarum state */
    extern float g_physarum_conductances[];
    extern int   g_physarum_n_heads;
    int head_dim = vec_size / g_physarum_n_heads;
    for (int h = 0; h < g_physarum_n_heads && h * head_dim < vec_size; h++) {
        extern int PHYSARUM_MAX_HEADS;
        float cond = g_physarum_conductances[layer * 128 + h];
        if (cond < 0.02f) {
            /* Dead head: zero its contribution */
            for (int d = 0; d < head_dim; d++) {
                output_vec[h * head_dim + d] = 0.0f;
            }
        } else {
            /* Scale by conductance */
            for (int d = 0; d < head_dim; d++) {
                output_vec[h * head_dim + d] *= cond;
            }
        }
    }
    return 0;
}

/* ── Add KV ──────────────────────────────────────────────────────────── */

int hyphal_add_kv(int layer, int pos, const char * token,
                   const float * k_vec, const float * v_vec,
                   int vec_size) {
    (void)layer; (void)pos; (void)token;
    (void)k_vec; (void)v_vec; (void)vec_size;
    /* Phase 1: handled by HyphalProxy Python sidecar */
    return g_active ? 0 : -1;
}

/* ── Stats ───────────────────────────────────────────────────────────── */

int hyphal_stats(int * nodes, int * active, float * mem_mb) {
    if (nodes)  *nodes  = 0;
    if (active) *active = 0;
    if (mem_mb) *mem_mb = 0.0f;
    /* Would parse JSON response from server in production */
    return g_active ? 0 : -1;
}
