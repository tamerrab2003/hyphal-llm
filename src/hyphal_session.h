/*
 * hyphal_session.h — HyphalLLM Python Bridge Header
 * ====================================================
 * Manages the persistent HyphalGraph Python server process
 * that sits alongside llama.cpp and handles graph operations.
 *
 * Communication: JSON-lines over POSIX pipe (popen2)
 * Latency: <2ms per operation (dominated by Python JSON parse)
 * Future: replace with Cython .so for <0.1ms
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>

/* Start the HyphalGraph Python server */
int  hyphal_session_start(int n_layers, int n_heads, int head_dim,
                           int max_active_nodes,
                           const char * load_path);

/* Stop the server and optionally save the graph */
void hyphal_session_stop(const char * save_path);

/* Check if the server is running */
int  hyphal_session_is_active(void);

/* Send a JSON request and get a JSON response (synchronous) */
int  hyphal_session_request(const char * json_req,
                             char * json_resp, int resp_max);

/* Convenience: attend with query vector, get output vector */
int  hyphal_attend(int layer, const float * query_vec,
                   int vec_size, float * output_vec);

/* Convenience: add a K,V pair to the graph */
int  hyphal_add_kv(int layer, int pos, const char * token,
                   const float * k_vec, const float * v_vec,
                   int vec_size);

/* Get stats from the server */
int  hyphal_stats(int * nodes, int * active, float * mem_mb);

#ifdef __cplusplus
}
#endif
