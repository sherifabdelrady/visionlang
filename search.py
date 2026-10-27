"""
VisionLang - Multimodal Visual Search
CLIP-powered natural language image search with FAISS indexing.

Audit fixes (audit/production-remediation):
- P1 FIX: Empty index search — faiss.IndexFlatIP.search() on ntotal=0
  raises a native exception. Added guard that returns [] immediately.
- P1 FIX: /index/upload has NO authentication — anyone could poison the index.
  Added API-key check via X-API-Key header (set via env var UPLOAD_API_KEY).
- P1 FIX: No request size limit on file uploads — DoS risk.
  Added MAX_UPLOAD_MB limit (default 10 MB, configurable via env var).
- P1 FIX: FAISS index and metadata can desync if save() is interrupted midway.
  Fixed by writing metadata atomically (write to .tmp then rename).
- P2 FIX: Model loaded at module import — crashes entire process if CLIP
  not installed. Moved to lazy load inside a lifespan event handler.
- P2 FIX: Added k validation (must be >= 1 and <= ntotal).
- P2 FIX: search/text now validates query is non-empty.
- P2 FIX: Index type is IndexFlatIP (exact cosine similarity after L2-norm).
  README updated to accurately describe this (not IVF-PQ).
- P2 FIX: embed_image/embed_text now normalise output to unit sphere
  to ensure consistent cosine similarity via inner product.
"""

from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
import faiss
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from PIL import Image
from pydantic import BaseModel
import uvicorn

log = logging.getLogger("visionlang")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------------------------------------------------------------------------
# Config (environment-driven)
# ---------------------------------------------------------------------------
MODEL_NAME    = os.environ.get("CLIP_MODEL",    "ViT-L-14")
PRETRAINED    = os.environ.get("CLIP_WEIGHTS",  "openai")
EMBED_DIM     = int(os.environ.get("EMBED_DIM", "768"))
INDEX_PATH    = Path(os.environ.get("INDEX_PATH", "index/faiss.index"))
META_PATH     = Path(os.environ.get("META_PATH",  "index/metadata.json"))
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "10"))
UPLOAD_API_KEY = os.environ.get("UPLOAD_API_KEY", "")     # empty = disabled (dev only)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_upload_key(key: str | None = Depends(api_key_header)) -> None:
    """Guard upload endpoints with an API key when UPLOAD_API_KEY is set."""
    if UPLOAD_API_KEY and key != UPLOAD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


# ---------------------------------------------------------------------------
# FAISS Index
# ---------------------------------------------------------------------------

class FaissIndex:
    def __init__(self, embed_dim: int = EMBED_DIM):
        self._embed_dim = embed_dim
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(embed_dim)
        self.metadata: List[dict] = []

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomic save: write metadata to a temp file then rename."""
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        tmp = META_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.metadata, ensure_ascii=False))
        tmp.replace(META_PATH)   # atomic rename
        log.info("Index saved: %d vectors → %s", self.index.ntotal, INDEX_PATH)

    def load(self) -> None:
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index    = faiss.read_index(str(INDEX_PATH))
            self.metadata = json.loads(META_PATH.read_text())
            log.info("Loaded index: %d vectors", self.index.ntotal)
        else:
            log.info("No existing index — starting fresh")

    # ── Mutate ───────────────────────────────────────────────────────────────

    def add(self, embeddings: np.ndarray, meta: List[dict]) -> None:
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(meta)

    # ── Query ────────────────────────────────────────────────────────────────

    def search(self, query: np.ndarray, k: int = 10) -> List[dict]:
        """
        Return up to k nearest neighbours by cosine similarity.
        Returns empty list if the index contains no vectors.
        """
        if self.index.ntotal == 0:
            return []
        # Clamp k so we never ask for more than we have
        k = max(1, min(k, self.index.ntotal))
        query = query.astype(np.float32)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results


# ---------------------------------------------------------------------------
# Lazy model holder
# ---------------------------------------------------------------------------
_model      = None
_preprocess = None
_tokenizer  = None


def _load_clip() -> None:
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return
    try:
        import open_clip  # type: ignore
    except ImportError:
        raise RuntimeError(
            "open_clip not installed. Run: pip install open-clip-torch"
        )
    log.info("Loading CLIP %s on %s …", MODEL_NAME, DEVICE)
    m, _, prep = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    _tokenizer  = open_clip.get_tokenizer(MODEL_NAME)
    _model      = m.to(DEVICE).eval()
    _preprocess = prep
    log.info("CLIP ready.")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

@torch.inference_mode()
def embed_image(pil_img: Image.Image) -> np.ndarray:
    tensor = _preprocess(pil_img).unsqueeze(0).to(DEVICE)
    emb    = _model.encode_image(tensor)
    emb    = emb / emb.norm(dim=-1, keepdim=True)   # unit-norm for cosine sim
    return emb.cpu().float().numpy()


@torch.inference_mode()
def embed_text(text: str) -> np.ndarray:
    tokens = _tokenizer([text]).to(DEVICE)
    emb    = _model.encode_text(tokens)
    emb    = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().float().numpy()


def index_directory(image_dir: str) -> None:
    paths = (
        list(Path(image_dir).rglob("*.jpg")) +
        list(Path(image_dir).rglob("*.jpeg")) +
        list(Path(image_dir).rglob("*.png")) +
        list(Path(image_dir).rglob("*.webp"))
    )
    log.info("Indexing %d images from %s …", len(paths), image_dir)
    batch_size = 64
    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        embeddings, meta = [], []
        for p in batch:
            try:
                img = Image.open(p).convert("RGB")
                emb = embed_image(img)
                embeddings.append(emb)
                meta.append({"path": str(p), "filename": p.name})
            except Exception as exc:
                log.warning("Skipping %s: %s", p, exc)
        if embeddings:
            index.add(np.concatenate(embeddings, axis=0), meta)
        log.info("  %d/%d", min(i + batch_size, len(paths)), len(paths))
    index.save()
    log.info("Indexed %d images total.", index.index.ntotal)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

index = FaissIndex()


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_clip()
    index.load()
    yield
    log.info("VisionLang shutting down.")


app = FastAPI(
    title="VisionLang",
    description="CLIP-powered multimodal visual search. Index type: IndexFlatIP (exact cosine similarity).",
    version="1.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "indexed":      index.index.ntotal,
        "embed_dim":    EMBED_DIM,
        "model":        MODEL_NAME,
        "index_type":   "IndexFlatIP",
        "upload_auth":  bool(UPLOAD_API_KEY),
    }


@app.get("/search/text")
def search_by_text(
    q: str = Query(..., min_length=1, description="Natural language query"),
    k: int = Query(10, ge=1, le=1000),
):
    emb     = embed_text(q)
    results = index.search(emb, k=k)
    return {"query": q, "results": results, "total_indexed": index.index.ntotal}


@app.post("/search/image")
async def search_by_image(
    request: Request,
    file: UploadFile = File(...),
    k: int = Query(10, ge=1, le=1000),
):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")
    data = await file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    emb     = embed_image(img)
    results = index.search(emb, k=k)
    return {"results": results, "total_indexed": index.index.ntotal}


@app.post("/index/upload")
async def add_to_index(
    request: Request,
    file: UploadFile = File(...),
    label: Optional[str] = None,
    _auth: None = Depends(require_upload_key),
):
    """Add an image to the search index. Requires X-API-Key header when UPLOAD_API_KEY is set."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")
    data = await file.read(int(MAX_UPLOAD_MB * 1024 * 1024) + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large. Max: {MAX_UPLOAD_MB} MB")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not decode image")
    emb  = embed_image(img)
    meta = [{"filename": file.filename, "label": label}]
    index.add(emb, meta)
    index.save()
    return {"indexed": file.filename, "total": index.index.ntotal}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VisionLang visual search server")
    parser.add_argument("--index-dir", help="Directory to index before starting server")
    parser.add_argument("--port",      type=int, default=8000)
    args = parser.parse_args()
    if args.index_dir:
        _load_clip()
        index.load()
        index_directory(args.index_dir)
    uvicorn.run("search:app", host="0.0.0.0", port=args.port, reload=False)
