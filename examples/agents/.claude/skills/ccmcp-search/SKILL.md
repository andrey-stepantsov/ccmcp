---
name: ccmcp-search
description: >-
  Search the local ccmcp knowledge base (hybrid vector search over indexed
  repos/docs) instead of grepping. Use when a question's answer likely lives in
  another repo, internal docs, or a large tree not open in the workspace.
---

# Searching with ccmcp

1. **Discover scope first.** Call `qdrant_list_scopes()` to list indexed projects
   (names, tags). Don't search blindly across the whole corpus.

2. **Search.** `qdrant_find(query, scope=...)`:
   - Omit `scope` to auto-scope to the current workspace.
   - Pass `scope=["proj", "tag"]` to target specific projects (e.g. a shared lib).
   - Use `scope=["*"]` only when you truly need the whole corpus — it is the
     noisiest option.

3. **Craft the query.** ccmcp is hybrid (dense semantic + sparse BM25). Include
   exact symbols, filenames, and flags VERBATIM (BM25 matches them precisely)
   AND describe the concept in prose (dense finds semantically related content).
   A single query covers both channels.

4. **Read the citation.** Each result includes a `source_uri`. Open it for full
   context instead of relying only on the snippet.

5. **Don't over-search.** If the answer is in a file already open in the
   workspace, just read it. ccmcp is for things you can't already see.

6. **Persist context.** Use `qdrant_store(text, title, session_id)` to save a
   decision or summary back into the index. Entries auto-expire after the
   configured TTL (default 30 days) and live in a separate artifacts collection.
