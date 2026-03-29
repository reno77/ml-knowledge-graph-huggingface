#!/usr/bin/env python3
"""Generate static/graph.json from raw_models.json."""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PIPELINE_COLORS, ORG_COLOR, DEFAULT_PIPELINE_COLOR


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def log_scale(value: float, base: float = 10.0) -> float:
    """Log-scale a value, returning 0 for non-positive inputs."""
    if value <= 0:
        return 0.0
    return math.log(value + 1) / math.log(base)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def build_graph(models: list[dict]) -> dict:
    nodes = []
    edges = []
    node_ids = set()

    # ---- Aggregate org-level stats ----
    org_downloads: dict[str, int] = {}
    org_models_map: dict[str, list[dict]] = {}
    for m in models:
        org = m["author"]
        org_downloads[org] = org_downloads.get(org, 0) + m["downloads"]
        org_models_map.setdefault(org, []).append(m)

    # ---- Create ORG nodes ----
    org_likes: dict[str, int] = {}
    for m in models:
        org = m["author"]
        org_likes[org] = org_likes.get(org, 0) + m["likes"]

    for org, total_dl in org_downloads.items():
        nodes.append({
            "id": org,
            "label": org,
            "type": "org",
            "category": "organization",
            "val": round(clamp(log_scale(total_dl), 4, 12), 2),
            "color": ORG_COLOR,
            "downloads": total_dl,
            "likes": org_likes.get(org, 0),
            "modelCount": len(org_models_map.get(org, [])),
        })
        node_ids.add(org)

    # ---- Create MODEL nodes ----
    for m in models:
        model_id = m["modelId"]
        if not model_id:
            continue

        pipeline = m.get("pipeline_tag") or "unknown"
        short_label = model_id.split("/", 1)[1] if "/" in model_id else model_id
        raw_val = log_scale(m["downloads"])
        val = round(clamp(raw_val, 2, 8), 2)

        nodes.append({
            "id": model_id,
            "label": short_label,
            "type": "model",
            "category": pipeline,
            "color": PIPELINE_COLORS.get(pipeline, DEFAULT_PIPELINE_COLOR),
            "val": val,
            "downloads": m["downloads"],
            "likes": m["likes"],
            "lastModified": m["lastModified"],
            "author": m["author"],
            "pipeline_tag": pipeline,
        })
        node_ids.add(model_id)

    # ---- Create EDGES ----
    for m in models:
        model_id = m["modelId"]
        if not model_id:
            continue
        org = m["author"]

        # org -> model (owns)
        if org in node_ids:
            edges.append({
                "source": org,
                "target": model_id,
                "type": "owns",
            })

        # base_model -> model (derived)
        for base in m.get("base_models", []):
            if base in node_ids and base != model_id:
                edges.append({
                    "source": base,
                    "target": model_id,
                    "type": "derived",
                })

    return {"nodes": nodes, "edges": edges}


def main():
    raw_path = Path(__file__).resolve().parent / "raw_models.json"
    if not raw_path.exists():
        print(f"ERROR: {raw_path} not found. Run fetch_models.py first.", file=sys.stderr)
        sys.exit(1)

    with open(raw_path) as f:
        models = json.load(f)

    print(f"Loaded {len(models)} models from raw_models.json")

    graph = build_graph(models)

    out_dir = PROJECT_ROOT / "static"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "graph.json"

    with open(out_path, "w") as f:
        json.dump(graph, f, indent=2)

    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    n_orgs = sum(1 for n in graph["nodes"] if n["type"] == "org")
    n_models = sum(1 for n in graph["nodes"] if n["type"] == "model")
    n_derived = sum(1 for e in graph["edges"] if e["type"] == "derived")

    print(f"Graph: {n_nodes} nodes ({n_orgs} orgs, {n_models} models), {n_edges} edges ({n_derived} derived)")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
