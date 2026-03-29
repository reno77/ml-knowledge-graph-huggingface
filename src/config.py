"""Configuration constants for the HuggingFace Model Knowledge Graph."""

HF_API_BASE = "https://huggingface.co/api/models"

TARGET_ORGS = [
    # ── US Companies ─────────────────────────────────────────────────
    "meta-llama",           # Meta / Facebook — Llama family
    "google",               # Google — Gemma, T5, BERT, PaLM
    "openai",               # OpenAI — GPT, CLIP, Whisper
    "microsoft",            # Microsoft — Phi, Florence, Orca
    "nvidia",               # NVIDIA — Megatron, NeMo, Nemotron
    "apple",                # Apple — OpenELM, AXLearn
    "amazon",               # Amazon — Titan
    "EleutherAI",           # EleutherAI — GPT-NeoX, Pythia
    "allenai",              # Allen AI — OLMo, Longformer
    "NousResearch",         # Nous Research — Hermes, Capybara
    "teknium",              # Teknium — OpenHermes
    "databricks",           # Databricks — DBRX, Dolly
    "Salesforce",           # Salesforce — CodeGen, BLIP
    "anthropic",            # Anthropic — Claude (limited on HF)
    "xai-org",              # xAI — Grok
    "togethercomputer",     # Together AI — RedPajama
    "lmsys",                # LMSYS — Vicuna, FastChat

    # ── European Companies ───────────────────────────────────────────
    "mistralai",            # Mistral AI (France) — Mistral, Mixtral
    "stabilityai",          # Stability AI (UK) — Stable Diffusion
    "black-forest-labs",    # Black Forest Labs (Germany) — FLUX
    "bigscience",           # BigScience (INRIA, France) — BLOOM
    "HuggingFaceTB",        # HuggingFace (France) — SmolLM, FineWeb
    "HuggingFaceM4",        # HuggingFace Multimodal — Idefics
    "CohereForAI",          # Cohere (Canada) — Aya, Command-R
    "tiiuae",               # TII (UAE) — Falcon
    "DeepFloyd",            # DeepFloyd (Stability, Germany) — IF

    # ── Chinese Companies ────────────────────────────────────────────
    "Qwen",                 # Alibaba — Qwen, Qwen2
    "deepseek-ai",          # DeepSeek — DeepSeek-V3, R1, Coder
    "THUDM",                # Tsinghua — ChatGLM, GLM-4
    "baichuan-inc",         # Baichuan — Baichuan-2
    "01-ai",                # Yi (01.AI, Kai-Fu Lee) — Yi-34B
    "internlm",             # InternLM (Shanghai AI Lab)
    "openbmb",              # OpenBMB (Tsinghua) — MiniCPM
    "BAAI",                 # Beijing Academy of AI — bge embeddings
    "zhipuai",              # Zhipu AI — GLM, CogVideo
    "WizardLMTeam",         # WizardLM
    "FlagAlpha",            # FlagAlpha — Llama2-Chinese
    "Skywork",              # Skywork (Kunlun Tech)
    "modelscope",           # ModelScope (Alibaba DAMO)
]

MAX_MODELS_PER_ORG = 30

ORG_COLOR = "#1a6985"

PIPELINE_COLORS = {
    "text-generation": "#e6194b",
    "text2text-generation": "#f58231",
    "text-classification": "#ffe119",
    "token-classification": "#bfef45",
    "question-answering": "#3cb44b",
    "fill-mask": "#42d4f4",
    "summarization": "#4363d8",
    "translation": "#911eb4",
    "conversational": "#f032e6",
    "feature-extraction": "#a9a9a9",
    "image-classification": "#9a6324",
    "object-detection": "#800000",
    "image-segmentation": "#469990",
    "image-to-text": "#dcbeff",
    "text-to-image": "#fabed4",
    "text-to-video": "#aaffc3",
    "text-to-audio": "#ffd8b1",
    "automatic-speech-recognition": "#fffac8",
    "audio-classification": "#808000",
    "image-to-image": "#e6beff",
    "depth-estimation": "#aa6e28",
    "zero-shot-classification": "#00bcd4",
    "sentence-similarity": "#ff6f61",
    "reinforcement-learning": "#c7e59a",
    "video-classification": "#6b8e23",
    "visual-question-answering": "#cd5c5c",
    "image-text-to-text": "#ff8c00",
    "any-to-any": "#7b68ee",
    "text-to-speech": "#ff69b4",
    "image-feature-extraction": "#daa520",
}

# Fallback color for unknown pipeline tags
DEFAULT_PIPELINE_COLOR = "#cccccc"
