"""Text embedding pipeline using fastembed ONNX models."""

import numpy as np
from fastembed import SparseTextEmbedding, TextEmbedding

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL = "Qdrant/bm25"


def load_models(dense_model=DENSE_MODEL, sparse_model=SPARSE_MODEL):
    """Load dense and sparse embedding models from local ONNX cache."""
    dense = TextEmbedding(dense_model)
    sparse = SparseTextEmbedding(sparse_model)
    return dense, sparse


def embed_texts(texts, dense_model, sparse_model, rotation_matrix=None):
    """Embed a list of texts into dense and sparse vectors.

    Applies an optional orthogonal rotation to dense vectors to improve
    INT8 quantisation effectiveness across all dimensions.
    """
    dense_vecs = np.array(list(dense_model.embed(texts)), dtype=np.float32)
    if rotation_matrix is not None:
        dense_vecs = dense_vecs @ rotation_matrix
    sparse_vecs = list(sparse_model.embed(texts))
    return dense_vecs, sparse_vecs


def generate_rotation_matrix(dim: int, path: str):
    """Generate and save a random orthogonal rotation matrix.

    The matrix is generated via QR decomposition of a random Gaussian matrix.
    Once generated it must never be regenerated: all stored vectors become
    invalid if the rotation matrix changes.
    """
    R, _ = np.linalg.qr(np.random.randn(dim, dim))
    np.save(path, R)
    return R
