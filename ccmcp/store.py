from __future__ import annotations

import hashlib
import uuid

import numpy as np
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from qdrant_client import QdrantClient, models

from ccmcp.chunker import Chunk
from ccmcp.state import _now

_ARTIFACT_COLLECTION = "ccmcp-artifacts"


def _sparse_vec(s: SparseEmbedding) -> models.SparseVector:
    return models.SparseVector(
        indices=s.indices.tolist(),
        values=s.values.tolist(),
    )


class VectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str = "",
        artifact_collection: str = _ARTIFACT_COLLECTION,
    ):
        self._client = QdrantClient(url=url, api_key=api_key or None)
        self._collection = collection
        self._artifact_collection = artifact_collection

    def setup(self, dense_dim: int = 384):
        existing = {c.name for c in self._client.get_collections().collections}

        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim,
                        distance=models.Distance.COSINE,
                        quantization_config=models.ScalarQuantizationConfig(
                            scalar=models.ScalarQuantization(
                                type=models.ScalarType.INT8,
                                quantile=0.99,
                                always_ram=True,
                            )
                        ),
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

        if self._artifact_collection not in existing:
            self._client.create_collection(
                collection_name=self._artifact_collection,
                vectors_config={
                    "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )

    def drop_collections(self):
        for name in (self._collection, self._artifact_collection):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass

    def doc_id(self, source_uri: str) -> str:
        return hashlib.sha256(source_uri.encode()).hexdigest()

    def upsert(
        self,
        chunks: list[Chunk],
        dense: np.ndarray,
        sparse: list[SparseEmbedding],
        version: int,
        source_type: str,
    ) -> int:
        if not chunks:
            return 0
        now = _now()
        points = []
        for chunk, d, s in zip(chunks, dense, sparse):
            point_id = str(uuid.uuid5(
                uuid.NAMESPACE_URL,
                chunk.source_uri + str(chunk.chunk_index),
            ))
            doc_id = self.doc_id(chunk.source_uri)
            content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()
            points.append(models.PointStruct(
                id=point_id,
                vector={"dense": d.tolist(), "sparse": _sparse_vec(s)},
                payload={
                    "text": chunk.text,
                    "source_uri": chunk.source_uri,
                    "source_type": source_type,
                    "doc_id": doc_id,
                    "chunk_index": chunk.chunk_index,
                    "section": chunk.section,
                    "content_hash": f"sha256:{content_hash}",
                    "version": version,
                    "ingested_at": now,
                },
            ))
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def _doc_filter(self, source_uri: str) -> models.Filter:
        return models.Filter(must=[
            models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=self.doc_id(source_uri))
            ),
        ])

    def delete_old_version(self, source_uri: str, old_version: int):
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=self.doc_id(source_uri))
                ),
                models.FieldCondition(
                    key="version", match=models.MatchValue(value=old_version)
                ),
            ])),
        )

    def delete_doc(self, source_uri: str):
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=self._doc_filter(source_uri)),
        )

    def search(
        self,
        dense: np.ndarray,
        sparse: SparseEmbedding,
        limit: int = 10,
    ) -> list[dict]:
        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(query=dense.tolist(), using="dense", limit=20),
                models.Prefetch(query=_sparse_vec(sparse), using="sparse", limit=20),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [p.payload for p in results.points if p.payload]

    def store_artifact(
        self,
        text: str,
        dense: np.ndarray,
        sparse: SparseEmbedding,
        session_id: str,
        metadata: dict,
    ) -> str:
        point_id = str(uuid.uuid4())
        self._client.upsert(
            collection_name=self._artifact_collection,
            points=[models.PointStruct(
                id=point_id,
                vector={"dense": dense.tolist(), "sparse": _sparse_vec(sparse)},
                payload={
                    **metadata,           # caller-supplied fields first
                    "text": text,         # system fields always win
                    "source_type": "artifact",
                    "session_id": session_id,
                    "ingested_at": _now(),
                },
            )],
        )
        return point_id

    def cleanup_artifacts(self, cutoff_iso: str):
        self._client.delete(
            collection_name=self._artifact_collection,
            points_selector=models.FilterSelector(filter=models.Filter(must=[
                models.FieldCondition(
                    key="ingested_at",
                    range=models.DatetimeRange(lt=cutoff_iso),
                )
            ])),
        )

    def collection_info(self) -> dict:
        info = self._client.get_collection(self._collection)
        return {
            "points_count": info.points_count,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": str(info.status),
        }
