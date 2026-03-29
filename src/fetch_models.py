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

# Known pipeline tags that can appear in the tags array
KNOWN_PIPELINES = {
    "text-generation", "text2text-generation", "text-classification",
    "token-classification", "question-answering", "fill-mask",
    "summarization", "translation", "conversational", "feature-extraction",
    "image-classification", "object-detection", "image-segmentation",
    "image-to-text", "text-to-image", "text-to-video", "text-to-audio",
    "text-to-speech", "automatic-speech-recognition", "audio-classification",
    "image-to-image", "depth-estimation", "zero-shot-classification",
    "zero-shot-image-classification", "sentence-similarity",
    "reinforcement-learning", "video-classification",
    "visual-question-answering", "image-text-to-text", "any-to-any",
    "image-feature-extraction", "time-series-forecasting",
}


def infer_pipeline_from_tags(tags: list[str]) -> str:
    """Infer pipeline_tag from the tags array if not explicitly set."""
    for tag in tags:
        if tag in KNOWN_PIPELINES:
            return tag
    # Heuristic fallbacks based on common tag patterns
    tag_set = set(tags)
    if "diffusers" in tag_set or "stable-diffusion" in tag_set:
        return "text-to-image"
    if "text-generation-inference" in tag_set or "vllm" in tag_set:
        return "text-generation"
    # Model architecture heuristics
    llm_archs = {"llama", "mistral", "qwen2", "gpt2", "gpt_neox", "falcon", "bloom",
                 "gemma", "gemma2", "phi", "phi3", "starcoder2", "deepseek", "internlm2",
                 "baichuan", "chatglm", "yi", "mpt", "rwkv", "codegen", "codellama"}
    if tag_set & llm_archs:
        return "text-generation"
    embed_archs = {"bert", "roberta", "xlm-roberta", "e5", "bge", "nomic-bert"}
    if tag_set & embed_archs:
        return "feature-extraction"
    vision_tags = {"vision", "clip", "siglip", "vit", "image-classification"}
    if tag_set & vision_tags:
        return "image-classification"
    return ""


def fetch_model_detail(model_id: str, session: requests.Session) -> dict:
    """Fetch individual model detail to get pipeline_tag and description."""
    try:
        resp = session.get(f"{HF_API_BASE}/{model_id}", timeout=15)
        if resp.ok:
            data = resp.json()
            # Extract description from cardData or model card
            desc = ""
            card_data = data.get("cardData", {})
            if isinstance(card_data, dict):
                desc = card_data.get("description", "") or card_data.get("summary", "")
            return {
                "pipeline_tag": data.get("pipeline_tag", ""),
                "tags": data.get("tags", []),
                "description": desc,
            }
    except Exception:
        pass
    return {}


def fetch_org_models(org: str, session: requests.Session) -> list[dict]:
    """Fetch top models for a single org, sorted by downloads."""
    params = {
        "author": org,
        "sort": "downloads",
        "limit": MAX_MODELS_PER_ORG,
    }
    resp = session.get(HF_API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    raw = resp.json()

    models = []
    for m in raw:
        tags = m.get("tags", [])
        base_models = [tag.split(":", 1)[1] for tag in tags if tag.startswith("base_model:")]

        pipeline = m.get("pipeline_tag", "")
        if not pipeline:
            pipeline = infer_pipeline_from_tags(tags)

        models.append({
            "modelId": m.get("modelId", ""),
            "pipeline_tag": pipeline,
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "tags": tags,
            "base_models": base_models,
            "author": m.get("author", org),
            "createdAt": m.get("createdAt", ""),
        })
    return models


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "ml-knowledge-graph/1.0"})

    all_models = []
    total = len(TARGET_ORGS)

    for i, org in enumerate(TARGET_ORGS, 1):
        print(f"[{i}/{total}] Fetching models for {org}...")
        try:
            models = fetch_org_models(org, session)
            print(f"  -> {len(models)} models fetched")
            all_models.extend(models)
        except requests.RequestException as e:
            print(f"  -> ERROR: {e}", file=sys.stderr)
        if i < total:
            time.sleep(0.5)

    # Second pass: fetch detail for all models (descriptions + missing pipeline_tag)
    print(f"\nFetching model details for descriptions...")
    for j, m in enumerate(all_models, 1):
        detail = fetch_model_detail(m["modelId"], session)
        if detail.get("description"):
            m["description"] = detail["description"]
        if not m["pipeline_tag"]:
            if detail.get("pipeline_tag"):
                m["pipeline_tag"] = detail["pipeline_tag"]
            elif detail.get("tags"):
                inferred = infer_pipeline_from_tags(detail["tags"])
                if inferred:
                    m["pipeline_tag"] = inferred
        if j % 50 == 0:
            print(f"  {j}/{len(all_models)} fetched...", flush=True)
        time.sleep(0.15)

    still_unknown = sum(1 for m in all_models if not m["pipeline_tag"])
    has_desc = sum(1 for m in all_models if m.get("description"))
    print(f"\nTotal models: {len(all_models)}, still unknown: {still_unknown}, with description: {has_desc}")

    out_path = Path(__file__).resolve().parent / "raw_models.json"
    with open(out_path, "w") as f:
        json.dump(all_models, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
