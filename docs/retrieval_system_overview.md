# Retrieval System Overview

The handoff package contains two related retrieval paths:

- `src/chroma_retriever.py` is the primary natural-language retrieval path. It encodes manifest rows into structured text tokens, parses user queries into constraints, filters feasible candidate rows, ranks with Chroma, and applies an accept/fallback decision layer.
- `src/VectorDB.py` is the original numeric masked nearest-neighbor retriever and Gradio dashboard. It searches only dimensions explicitly present in a structured query.

The Chroma path is the recommended review target because it contains the current query parsing, constraint support checks, candidate filtering, ranking, and confidence decision behavior.

Generated Chroma database files are excluded. Run `python src/chroma_retriever.py build --force` or any demo/evaluation command to rebuild `data/chroma/` from `data/manifest.csv`.
