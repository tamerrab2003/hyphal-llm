"""
HyphalLLM Server — Python↔C Bridge
=====================================
Runs as a subprocess alongside llama.cpp.
Receives KV vectors via stdin, maintains HyphalGraph,
returns attention outputs.

Protocol (JSON-lines over stdin/stdout):
  → {"op": "attend", "layer": 0, "query": [...], "step": 42}
  ← {"output": [...], "active_nodes": 12, "mem_mb": 1.2}

  → {"op": "add_kv", "layer": 0, "pos": 5, "token": "hello", "k": [...], "v": [...]}
  ← {"status": "ok"}

  → {"op": "save", "path": "/tmp/session.pkl"}
  ← {"status": "saved"}

  → {"op": "load", "path": "/tmp/session.pkl"}
  ← {"status": "loaded", "nodes": 512}

  → {"op": "stats"}
  ← {"nodes": 512, "edges": 4096, "mem_mb": 25.4, "step": 1000}

  → {"op": "quit"}
  ← {"status": "bye"}

Usage (called by llama.cpp via hyphal_session_start()):
    python3 tools/hyphal_server.py --layers 32 --heads 32 --dim 128

Author: [Your Name]
"""

import sys
import os
import json
import argparse
import numpy as np

# Add hyphal_memory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from hyphal_memory.src.hyphal_graph import HyphalGraph, HyphalConfig


def serve(config: HyphalConfig, load_path: str = None):
    """Main server loop — reads JSON-lines from stdin, writes to stdout."""
    graph = None

    # Load existing session if specified
    if load_path and os.path.exists(load_path):
        try:
            graph = HyphalGraph.load(load_path)
            sys.stderr.write(f"[hyphal_server] Loaded graph: {graph}\n")
        except Exception as e:
            sys.stderr.write(f"[hyphal_server] Load failed: {e}\n")

    if graph is None:
        graph = HyphalGraph(config)
        sys.stderr.write(f"[hyphal_server] New graph: {graph}\n")

    # Signal ready
    print(json.dumps({"status": "ready", "nodes": len(graph.nodes)}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"error": str(e)}), flush=True)
            continue

        op = req.get("op", "")

        if op == "attend":
            layer  = req["layer"]
            q_flat = np.array(req["query"], dtype=np.float32)
            step   = req.get("step", graph.step)

            # Reshape query to [num_heads, head_dim]
            q = q_flat.reshape(config.num_heads, config.head_dim)

            # Attend
            output, info = graph.attend(q, layer=layer % max(config.num_layers, 1))
            print(json.dumps({
                "output":       output.flatten().tolist(),
                "active_nodes": info.get("active_nodes", 0),
                "total_nodes":  info.get("total_nodes", 0),
                "mem_mb":       round(graph.memory_mb()["total_mb"], 3),
                "step":         graph.step,
            }), flush=True)

        elif op == "add_kv":
            layer   = req["layer"]
            pos     = req["pos"]
            token   = req.get("token", f"pos_{pos}")
            tok_id  = req.get("token_id", hash(token) % 32000)
            k_flat  = np.array(req["k"], dtype=np.float32)
            v_flat  = np.array(req["v"], dtype=np.float32)

            k = k_flat.reshape(config.num_heads, config.head_dim)
            v = v_flat.reshape(config.num_heads, config.head_dim)

            graph.add_node_from_kv(
                token_id=tok_id,
                token_text=token,
                position=pos,
                layer=layer % max(config.num_layers, 1),
                k_vec=k,
                v_vec=v,
            )
            print(json.dumps({"status": "ok", "nodes": len(graph.nodes)}), flush=True)

        elif op == "save":
            path = req.get("path", "/tmp/hyphal_session.pkl")
            graph.save(path)
            print(json.dumps({"status": "saved", "path": path, "nodes": len(graph.nodes)}), flush=True)

        elif op == "load":
            path = req.get("path", "/tmp/hyphal_session.pkl")
            if os.path.exists(path):
                graph = HyphalGraph.load(path)
                print(json.dumps({"status": "loaded", "nodes": len(graph.nodes), "step": graph.step}), flush=True)
            else:
                print(json.dumps({"status": "error", "msg": f"file not found: {path}"}), flush=True)

        elif op == "stats":
            m = graph.memory_mb()
            s = graph.stats()
            print(json.dumps({
                "nodes":         m["num_nodes"],
                "active_set":    m["active_set_size"],
                "edges":         m["num_edges"],
                "mem_mb":        m["total_mb"],
                "step":          graph.step,
                "total_created": s.get("total_nodes_created", 0),
                "total_pruned":  s.get("total_nodes_pruned", 0),
                "total_clones":  s.get("total_clones", 0),
            }), flush=True)

        elif op == "reset":
            graph = HyphalGraph(config)
            print(json.dumps({"status": "reset"}), flush=True)

        elif op == "quit":
            print(json.dumps({"status": "bye"}), flush=True)
            break

        else:
            print(json.dumps({"error": f"unknown op: {op}"}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="HyphalLLM Server")
    parser.add_argument("--layers",    type=int, default=32)
    parser.add_argument("--heads",     type=int, default=32)
    parser.add_argument("--dim",       type=int, default=128)
    parser.add_argument("--max-nodes", type=int, default=512)
    parser.add_argument("--load",      type=str, default=None)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    config = HyphalConfig(
        num_layers=args.layers,
        num_heads=args.heads,
        head_dim=args.dim,
        max_active_nodes=args.max_nodes,
        seed=args.seed,
    )

    serve(config, load_path=args.load)


if __name__ == "__main__":
    main()
