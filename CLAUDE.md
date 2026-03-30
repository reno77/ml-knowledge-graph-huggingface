# HuggingFace Model Knowledge Graph

Interactive 3D knowledge graph of **967 ML models** from **36 AI organizations** (US, Europe, China), visualised with 3D-force-graph + Three.js.

## Architecture

- **Frontend**: Single-file vanilla JS app in `static/index.html` (no build step, CDN deps)
- **Backend**: FastAPI serving `static/` as a static site (`server.py`)
- **Data**: Pre-generated `static/graph.json` — 967 nodes (36 orgs + 931 models), 1041 edges
- **Deployment**: Vercel static site (`vercel.json`)

### Data pipeline (offline)

```
src/config.py          (org list, pipeline colors, constants)
       ↓
src/fetch_models.py    (HuggingFace API → src/raw_models.json)
       ↓
src/generate_graph.py  (raw_models.json → static/graph.json)
```

**Important**: The HuggingFace API returns `createdAt`, NOT `lastModified`. The timeline view depends on this field.

## Running

```bash
pip install fastapi uvicorn
python3 server.py
# or
python3 -m http.server 8001 --directory static
```

## Regenerating graph data

```bash
pip install requests

# Fetch models from HuggingFace API (~36 orgs × 30 models each)
python3 src/fetch_models.py

# Build graph.json from raw data
python3 src/generate_graph.py
```

To add new organizations, edit `TARGET_ORGS` in `src/config.py` and re-run both scripts.

## Deploying

```bash
npx vercel --prod --yes
```

## Key files

| File | Purpose |
|------|---------|
| `src/config.py` | Target orgs, pipeline colors, constants |
| `src/fetch_models.py` | Fetches top models per org from HuggingFace API |
| `src/generate_graph.py` | Builds graph.json with org/model nodes and edges |
| `static/index.html` | Entire frontend application |
| `static/graph.json` | Pre-built graph (served directly to browser) |
| `server.py` | FastAPI static file server |
| `vercel.json` | Vercel deployment config |

## Graph schema

- **Org nodes**: id, label, type="org", category="organization", color, val, downloads, likes, modelCount
- **Model nodes**: id, label, type="model", category (pipeline_tag), color, val, downloads, likes, createdAt, author, pipeline_tag
- **Edges**: source, target, type ("owns" = org→model, "derived" = base_model→fine-tune)

## Layout modes

- **Force**: Standard 3D force-directed layout (default)
- **Timeline**: X-axis = createdAt date (month granularity), Y/Z = org/hash jitter. Node dragging disabled.

## Organizations (36)

**US**: meta-llama, google, openai, microsoft, nvidia, apple, amazon, EleutherAI, allenai, NousResearch, teknium, databricks, Salesforce, xai-org, togethercomputer, lmsys

**Europe**: mistralai, stabilityai, black-forest-labs, bigscience, HuggingFaceTB, HuggingFaceM4, CohereForAI, tiiuae, DeepFloyd

**China**: Qwen, deepseek-ai, THUDM, baichuan-inc, 01-ai, internlm, openbmb, BAAI, zhipuai, WizardLMTeam, FlagAlpha, Skywork, modelscope, PaddlePaddle, Tongyi-MAI

## UI patterns

- Dark theme (#0d1117 bg, #FFD21E accent)
- Sidebar click = isolate (show only that org/pipeline), click again = restore all
- All/None buttons for manual multi-select
- Dimmed opacity = 0.15 (must be higher than light-theme projects due to dark bg)
- Rotation gizmo (X/Y/Z axis snap)

## No test suite, CI, or requirements.txt lock

Dependencies are listed above per task. `src/raw_models.json` is gitignored.

## HF Token

Set `HF_TOKEN` environment variable when running fetch_models.py to avoid rate limits and access gated models:

```bash
HF_TOKEN=hf_xxx python3 src/fetch_models.py
```
