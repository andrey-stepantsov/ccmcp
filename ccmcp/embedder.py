from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.sparse.sparse_embedding_base import SparseEmbedding

from ccmcp.metrics import EMBED_BATCH_SIZE, EMBED_SECONDS


class Embedder:
    def __init__(self, dense_model: str, sparse_model: str, rotation_matrix_path: str):
        self._dense_model_name = dense_model
        self._sparse_model_name = sparse_model
        self._dense_embedder: TextEmbedding | None = None
        self._sparse_embedder: SparseTextEmbedding | None = None
        self._rotation_path = str(Path(rotation_matrix_path).expanduser())
        self._R: np.ndarray | None = None
        if Path(self._rotation_path).exists():
            self._R = np.load(self._rotation_path)

    def _load(self):
        if self._dense_embedder is None:
            self._dense_embedder = TextEmbedding(self._dense_model_name)
        if self._sparse_embedder is None:
            self._sparse_embedder = SparseTextEmbedding(self._sparse_model_name)

    @property
    def dim(self) -> int:
        return 384

    def setup(self):
        """Generate and save rotation matrix. Raises FileExistsError if it already exists."""
        if Path(self._rotation_path).exists():
            raise FileExistsError(
                f"{self._rotation_path} already exists. "
                "Regenerating invalidates all indexed vectors — run 'ccmcp reset' first."
            )
        R, _ = np.linalg.qr(np.random.randn(self.dim, self.dim))
        Path(self._rotation_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(self._rotation_path, R)
        self._R = R

    def embed(self, texts: list[str]) -> tuple[np.ndarray, list[SparseEmbedding]]:
        self._load()
        EMBED_BATCH_SIZE.observe(len(texts))
        t0 = time.perf_counter()
        dense = np.array(list(self._dense_embedder.embed(texts)), dtype=np.float32)
        if self._R is not None:
            dense = dense @ self._R
        sparse = list(self._sparse_embedder.embed(texts))
        EMBED_SECONDS.observe(time.perf_counter() - t0)
        return dense, sparse
