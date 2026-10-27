"""
tests/test_search.py — VisionLang search unit tests
Runs without CLIP/GPU (mocks embedding model).
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFaissIndex:
    def test_empty_index_returns_empty(self):
        """REGRESSION: faiss.search() on ntotal=0 raised native exception."""
        from search import FaissIndex
        idx = FaissIndex(embed_dim=16)
        query = np.random.randn(1, 16).astype(np.float32)
        results = idx.search(query, k=5)
        assert results == []

    def test_add_and_search(self):
        from search import FaissIndex
        import faiss
        idx = FaissIndex(embed_dim=16)
        emb = np.random.randn(5, 16).astype(np.float32)
        meta = [{"filename": f"img{i}.jpg"} for i in range(5)]
        idx.add(emb, meta)
        assert idx.index.ntotal == 5
        query = np.random.randn(1, 16).astype(np.float32)
        results = idx.search(query, k=3)
        assert len(results) == 3

    def test_k_clamped_to_ntotal(self):
        """k > ntotal should not crash — clamp to ntotal."""
        from search import FaissIndex
        idx = FaissIndex(embed_dim=16)
        emb = np.random.randn(3, 16).astype(np.float32)
        idx.add(emb, [{"filename": f"i{j}"} for j in range(3)])
        results = idx.search(np.random.randn(1, 16).astype(np.float32), k=100)
        assert len(results) <= 3

    def test_embeddings_normalized(self):
        """After add(), all stored vectors should have L2-norm ≈ 1."""
        from search import FaissIndex
        import faiss
        idx = FaissIndex(embed_dim=4)
        emb = np.array([[3.0, 0, 0, 0]], dtype=np.float32)
        idx.add(emb, [{"filename": "x.jpg"}])
        # Retrieve the stored vector
        reconstructed = np.zeros((1, 4), dtype=np.float32)
        idx.index.reconstruct(0, reconstructed[0])
        norm = np.linalg.norm(reconstructed)
        assert norm == pytest.approx(1.0, abs=1e-5), "Stored embedding not unit-normalised"

    def test_save_load_roundtrip(self, tmp_path, monkeypatch):
        """save() then load() should restore identical index and metadata."""
        import search as s
        monkeypatch.setattr(s, "INDEX_PATH", tmp_path / "faiss.index")
        monkeypatch.setattr(s, "META_PATH",  tmp_path / "metadata.json")
        from search import FaissIndex
        idx = FaissIndex(embed_dim=8)
        emb = np.random.randn(4, 8).astype(np.float32)
        idx.add(emb, [{"filename": f"img{i}.png"} for i in range(4)])
        idx.save()
        idx2 = FaissIndex(embed_dim=8)
        idx2.load()
        assert idx2.index.ntotal == 4
        assert len(idx2.metadata) == 4

    def test_atomic_save_metadata(self, tmp_path, monkeypatch):
        """Metadata should be written via .tmp then renamed (no partial write)."""
        import search as s
        meta_path  = tmp_path / "metadata.json"
        index_path = tmp_path / "faiss.index"
        monkeypatch.setattr(s, "INDEX_PATH", index_path)
        monkeypatch.setattr(s, "META_PATH",  meta_path)
        from search import FaissIndex
        idx = FaissIndex(embed_dim=8)
        emb = np.random.randn(2, 8).astype(np.float32)
        idx.add(emb, [{"filename": "a.jpg"}, {"filename": "b.jpg"}])
        # tmp file should NOT exist after successful save
        tmp = meta_path.with_suffix(".json.tmp")
        idx.save()
        assert not tmp.exists(), ".json.tmp should be renamed after save()"
        assert meta_path.exists()
