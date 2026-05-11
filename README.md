# SoilX GPR Retrieval System

## Overview

This repository contains a cleaned handoff of the SoilX Ground Penetrating Radar retrieval system. The code searches over a synthetic GPR soil simulation manifest and returns candidate simulation rows that best match a user query.

Each manifest row represents a three-layer soil profile with per-layer properties such as thickness, sand/silt/clay percentages, volumetric water content, density, and organic fraction.

## Repository Structure

- `src/chroma_retriever.py` - Chroma/vector retrieval path, query parsing, constraint extraction, candidate filtering, and decision logic.
- `src/VectorDB.py` - original masked numeric retriever and Gradio dashboard.
- `src/eval_query_sets.py` - built-in 24-query held-out set and CSV query-set loader.
- `src/evaluate_chroma_ranking.py` - ranking metrics evaluation.
- `src/evaluate_decision_layer.py` - accept/fallback decision-layer evaluation.
- `src/evaluate_retriever.py` - lightweight evaluation for the original numeric retriever.
- `src/demo_retrieval.py` - one-command Chroma decision demo.
- `demo/` - original demo entry points.
- `data/manifest.csv` - 10,000-row synthetic simulation manifest.
- `data/eval/query_set_expanded_150_v1.csv` - broader 150-query benchmark set.
- `results/` - selected summary/comparison files only.
- `docs/` - concise handoff documentation.

## What the Retrieval System Does

The retrieval system parses user text into layer/property constraints such as `layer 2 silt 30 percent`. Chroma/vector retrieval ranks candidate manifest rows using structured row documents generated from the manifest. Constraint-aware filtering narrows candidates when supported by the manifest ranges.

The decision layer combines top-result vector similarity with support alignment. It accepts a retrieval when confidence passes the threshold and falls back when constraints are unsupported or confidence is too low.

The 24-query set is the initial held-out check. The included 150-query set is a broader benchmark for supported, partially supported, and unsupported queries.

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run a Demo Query

```bash
python src/demo_retrieval.py
```

Equivalent direct CLI:

```bash
python src/chroma_retriever.py decide --text "layer 2 silt 30 percent" --top-k 3
```

The first run creates `data/chroma/` from `data/manifest.csv`. The generated Chroma database is intentionally ignored by Git.

## How to Run Evaluation

Built-in 24-query decision evaluation:

```bash
python src/evaluate_decision_layer.py
```

Built-in 24-query ranking evaluation:

```bash
python src/evaluate_chroma_ranking.py
```

Broader 150-query decision evaluation:

```bash
python src/evaluate_decision_layer.py --query-set-csv data/eval/query_set_expanded_150_v1.csv --out-csv results/decision_eval_150_v1.csv --out-summary results/decision_eval_150_v1_summary.md
```

Broader 150-query ranking evaluation:

```bash
python src/evaluate_chroma_ranking.py --query-set-csv data/eval/query_set_expanded_150_v1.csv --out-prefix results/chroma_ranking_eval_150_v1
```

## Data Notes

`data/manifest.csv` is included because it is small enough for review and needed to rebuild the vector index. Persisted Chroma files under `data/chroma/` are not included because they are generated artifacts and can be rebuilt from the manifest.

## Limitations

The system retrieves over synthetic manifest metadata, not raw GPR signal files. Query parsing is rule-based and supports the properties encoded in the manifest. Evaluation labels are useful for benchmark inspection but should not be treated as field validation.

## Contact / Author

Repository prepared for Vijay review by `vdhuang`.
