"""
VisionLang — Multimodal Visual Search
CLIP-powered natural language image search with FAISS indexing.
Supports image-to-image and text-to-image retrieval.
"""

import numpy as np
import faiss
import torch
import open_clip
from pathlib import Path
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json
import io

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME   = "ViT-L-14"
PRETRAINED   = "openai"
EMBED_DIM    = 768
INDEX_PATH   = Path("index/faiss.index")
META_PATH    = Path("index/metadata.json")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ─────────────────────────────────────────────────────────────────────
print(f"Loading CLIP {MODEL_NAME} on {DEVICE}...")
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(DEVICE).eval()
print("CLIP ready.")


# ── Index ─────────────────────────────────────────────────────────────────────
class FaissIndex:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBED_DIM)  # Inner product for cosine sim
        self.metadata: List[dict] = []

    def add(self, embeddings: np.ndarray, meta: List[dict]):
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend(meta)

    def search(self, query: np.ndarray, k: int = 10):
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results

    def save(self):
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_PATH))
        META_PATH.write_text(json.dumps(self.metadata))

    def load(self):
        if INDEX_PATH.exists() and META_PATH.exists():
            self.index = faiss.read_index(str(INDEX_PATH))
            self.metadata = json.loads(META_PATH.read_text())
            print(f"Loaded index: {self.index.ntotal} vectors")
        else:
            print("No existing index — starting fresh")


index = FaissIndex()
index.load()


# ── Embeddings ────────────────────────────────────────────────────────────────
@torch.inference_mode()
def embed_image(pil_img: Image.Image) -> np.ndarray:
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    emb = model.encode_image(tensor)
    return emb.cpu().float().numpy()


@torch.inference_mode()
def embed_text(text: str) -> np.ndarray:
    tokens = tokenizer([text]).to(DEVICE)
    emb = model.encode_text(tokens)
    return emb.cpu().float().numpy()


def index_directory(image_dir: str):
    """Index all images in a directory."""
    paths = list(Path(image_dir).rglob("*.jpg")) + \
            list(Path(image_dir).rglob("*.png")) + \
            list(Path(image_dir).rglob("*.jpeg"))
    print(f"Indexing {len(paths)} images...")
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
            except Exception as e:
                print(f"Skip {p}: {e}")
        if embeddings:
            index.add(np.concatenate(embeddings, axis=0), meta)
        print(f"  {min(i + batch_size, len(paths))}/{len(paths)}")
    index.save()
    print(f"Indexed {index.index.ntotal} images.")


# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="VisionLang", description="CLIP-powered visual search", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "indexed": index.index.ntotal}


@app.get("/search/text")
def search_by_text(q: str = Query(..., description="Natural language query"), k: int = 10):
    emb = embed_text(q)
    results = index.search(emb, k=k)
    return {"query": q, "results": results}


@app.post("/search/image")
async def search_by_image(file: UploadFile = File(...), k: int = 10):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = embed_image(img)
    results = index.search(emb, k=k)
    return {"results": results}


@app.post("/index/upload")
async def add_to_index(file: UploadFile = File(...), label: Optional[str] = None):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = embed_image(img)
    meta = [{"filename": file.filename, "label": label}]
    index.add(emb, meta)
    index.save()
    return {"indexed": file.filename, "total": index.index.ntotal}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", help="Directory to index before starting server")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.index_dir:
        index_directory(args.index_dir)
    uvicorn.run("search:app", host="0.0.0.0", port=args.port, reload=False)
