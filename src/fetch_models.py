#!/usr/bin/env python3
"""Fetch top models from HuggingFace API for each target organization."""

import json
import time
import sys
from pathlib import Path

import requests

# Allow running as `python3 src/fetch_models.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import HF_API_BASE, TARGET_ORGS, MAX_MODELS_PER_ORG


def fetch_org_models(org: str) -> list[dict]:
    """Fetch top models for a single org, sorted by downloads."""
    url = HF_API_BASE
    params = {
        "author": org,
        "sort": "downloads",
        "limit": MAX_MODELS_PER_ORG,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    models = []
    for m in raw:
        tags = m.get("tags", [])
        # Extract base_model references from tags (format: "base_model:org/model")
        base_models = []
        for tag in tags:
            if tag.startswith("base_model:"):
                base_models.append(tag.split(":", 1)[1])

        models.append({
            "modelId": m.get("modelId", ""),
            "pipeline_tag": m.get("pipeline_tag", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": tags,
            "base_models": base_models,
            "author": m.get("author", org),
            "lastModified": m.get("lastModified", ""),
        })
    return models


def main():
    all_models = []
    total = len(TARGET_ORGS)

    for i, org in enumerate(TARGET_ORGS, 1):
        print(f"[{i}/{total}] Fetching models for {org}...")
        try:
            models = fetch_org_models(org)
            print(f"  -> {len(models)} models fetched")
            all_models.extend(models)
        except requests.RequestException as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)

        if i < total:
            time.sleep(0.5)

    out_path = Path(__file__).resolve().parent / "raw_models.json"
    with open(out_path, "w") as f:
        json.dump(all_models, f, indent=2)

    print(f"\nTotal models fetched: {len(all_models)}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
