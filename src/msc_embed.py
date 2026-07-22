"""Shared sentence-embedding helper with an on-disk cache.

All pipeline stages embed short utterances with the same model
(``all-MiniLM-L6-v2``, 384-d). Embeddings are cached to disk keyed by the exact
text so that re-runs -- and different scripts (features, labeling, the demo) --
never recompute the same vector.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = Path(__file__).resolve().parent.parent / "data" / "processed" / "emb_cache.pkl"

_model = None
_cache: dict[str, np.ndarray] | None = None


def _device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        print(f"Loading embedding model {MODEL_NAME} on {_device()} ...")
        _model = SentenceTransformer(MODEL_NAME, device=_device())
    return _model


def _load_cache() -> dict[str, np.ndarray]:
    global _cache
    if _cache is None:
        if CACHE.exists():
            with open(CACHE, "rb") as fh:
                _cache = pickle.load(fh)
        else:
            _cache = {}
    return _cache


def _save_cache() -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as fh:
        pickle.dump(_cache, fh)


def embed(texts: list[str], batch_size: int = 256) -> np.ndarray:
    """Return raw (un-normalised) embeddings for ``texts`` in input order."""
    cache = _load_cache()
    missing = [t for t in set(texts) if t not in cache]
    if missing:
        model = get_model()
        try:
            vecs = model.encode(
                missing, batch_size=batch_size, show_progress_bar=len(missing) > 2000,
                convert_to_numpy=True, normalize_embeddings=False,
            )
        except Exception as exc:  # noqa: BLE001 - MPS op gaps -> retry on CPU
            print(f"Embedding on accelerator failed ({exc}); retrying on CPU.")
            model = model.to("cpu")
            vecs = model.encode(
                missing, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=False,
            )
        for t, v in zip(missing, vecs):
            cache[t] = v.astype(np.float32)
        _save_cache()
    return np.vstack([cache[t] for t in texts])


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity for equal-length stacks (or broadcast a 1xd)."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=-1)
