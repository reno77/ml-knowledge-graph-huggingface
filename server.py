"""FastAPI server for the HuggingFace Model Knowledge Graph."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="HuggingFace Model Knowledge Graph")

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
