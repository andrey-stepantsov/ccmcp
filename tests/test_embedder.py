import numpy as np
import pytest

from ccmcp.embedder import Embedder


@pytest.mark.slow
def test_embed_returns_correct_shape(tmp_path):
    e = Embedder(
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm25",
        rotation_matrix_path=str(tmp_path / "rotation.npy"),
    )
    e.setup()
    dense, sparse = e.embed(["hello world", "foo bar"])
    assert dense.shape == (2, 384)
    assert len(sparse) == 2


@pytest.mark.slow
def test_rotation_applied(tmp_path):
    e = Embedder(
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm25",
        rotation_matrix_path=str(tmp_path / "rotation.npy"),
    )
    e.setup()
    d_rotated, _ = e.embed(["test"])

    # Reload without rotation to compare
    e2 = Embedder(
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm25",
        rotation_matrix_path=str(tmp_path / "nonexistent.npy"),
    )
    d_plain, _ = e2.embed(["test"])

    # Rotation should change the vector
    assert not np.allclose(d_rotated, d_plain)
    # But norm should be preserved (orthogonal rotation)
    assert abs(np.linalg.norm(d_rotated[0]) - np.linalg.norm(d_plain[0])) < 1e-4


@pytest.mark.slow
def test_sparse_has_nonzero_values(tmp_path):
    e = Embedder(
        dense_model="BAAI/bge-small-en-v1.5",
        sparse_model="Qdrant/bm25",
        rotation_matrix_path=str(tmp_path / "nonexistent.npy"),
    )
    _, sparse = e.embed(["Python programming language"])
    assert len(sparse[0].indices) > 0
    assert len(sparse[0].values) > 0


def test_setup_refuses_to_overwrite(tmp_path):
    rotation_path = str(tmp_path / "rotation.npy")
    # Create a dummy file
    np.save(rotation_path, np.eye(384))
    e = Embedder("BAAI/bge-small-en-v1.5", "Qdrant/bm25", rotation_path)
    with pytest.raises(FileExistsError):
        e.setup()
