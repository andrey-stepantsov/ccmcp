# Hybrid Dense and Sparse Search

Hybrid search combines two complementary retrieval methods: dense vector search
using semantic embeddings, and sparse BM25-based keyword search.

## Dense Retrieval

Dense retrieval encodes queries and documents into fixed-size vectors using a
neural language model. Semantically similar content clusters together in the
vector space, enabling retrieval of relevant passages even when exact keywords
differ from the query.

## Sparse Retrieval (BM25)

BM25 term frequency scoring captures exact vocabulary matches. Technical terms
like hostnames, CLI flags, model names, and error codes are reliably found by
sparse retrieval when they would be missed by semantic embeddings.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges ranked results from both retrieval methods.
Each result receives a score of 1/(k + rank), where k dampens the influence of
high-ranked outliers. The combined list consistently outperforms either method
alone for technical documentation retrieval.
