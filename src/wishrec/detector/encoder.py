"""Frozen Qwen3-Embedding-0.6B encoder, CPU-only and never trained.

Qwen3-Embedding-0.6B is a decoder-style 1024-dimensional encoder.  We use the pooling and
normalization shipped in its sentence-transformers configuration and encode the raw L3
wish; no retrieval instruction is added, because the task is symmetric classification, not
query-document retrieval.  Embeddings are cached by model revision and exact text list so
repeated detector fits do not re-encode.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import numpy as np

from ..config import RESULTS_DIR
from .constants import (
    ENCODER_BATCH_SIZE,
    ENCODER_DIM,
    ENCODER_NAME,
    ENCODER_REVISION,
)

_LOCK = threading.Lock()
_MODEL = None
EMBEDDING_CACHE_DIR = RESULTS_DIR / "cache" / "detector" / "embeddings"


def _load_model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # The checkpoint is stored in bfloat16, but CPU bfloat16 inference is
            # emulated and much slower on this hardware.  float32 is the right CPU
            # choice and does not change the frozen encoder beyond numeric precision.
            _MODEL = SentenceTransformer(
                ENCODER_NAME,
                device="cpu",
                model_kwargs={"dtype": torch.float32},
            )
            _MODEL.max_seq_length = 512
            _MODEL.eval()
        return _MODEL


def embedding_dimension() -> int:
    model = _load_model()
    getter = (
        getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    )
    dimension = int(getter())
    if dimension != ENCODER_DIM:
        raise RuntimeError(
            f"{ENCODER_NAME} returned {dimension} dimensions, expected {ENCODER_DIM}"
        )
    return dimension


def encode_texts(texts: list[str], *, batch_size: int = ENCODER_BATCH_SIZE) -> np.ndarray:
    if not texts:
        return np.zeros((0, ENCODER_DIM), dtype=np.float32)
    model = _load_model()
    with _LOCK:
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
            device="cpu",
        )
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != ENCODER_DIM:
        raise RuntimeError(f"unexpected embedding shape {matrix.shape}")
    return matrix


def _cache_key(texts: list[str]) -> str:
    payload = json.dumps(
        {
            "model": ENCODER_NAME,
            "revision": ENCODER_REVISION,
            "texts": list(texts),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def embed_cached(texts: list[str], *, cache_dir: Path | None = None) -> np.ndarray:
    directory = Path(cache_dir) if cache_dir else EMBEDDING_CACHE_DIR
    key = _cache_key(texts)
    path = directory / f"{key}.npy"
    if path.is_file():
        matrix = np.load(path)
        if matrix.shape == (len(texts), ENCODER_DIM):
            return matrix.astype(np.float32)
    matrix = encode_texts(texts)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, matrix)
    temporary.replace(path)
    return matrix


def encoder_info() -> dict:
    return {
        "model": ENCODER_NAME,
        "revision": ENCODER_REVISION,
        "dimension": ENCODER_DIM,
        "frozen": True,
        "device": "cpu",
        "normalize_embeddings": True,
        "pooling": "model default (last token)",
        "precision": "float32 on CPU",
        "max_seq_length": 512,
        "prompt": "raw L3 wish; no retrieval instruction",
        "batch_size": ENCODER_BATCH_SIZE,
    }


__all__ = [
    "EMBEDDING_CACHE_DIR",
    "embed_cached",
    "embedding_dimension",
    "encode_texts",
    "encoder_info",
]
