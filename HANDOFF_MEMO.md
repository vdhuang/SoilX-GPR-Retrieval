# Handoff Memo

## Included

This clean repository includes the primary Chroma retrieval system, the original numeric retriever/dashboard, evaluation scripts, a 10,000-row manifest, one expanded 150-query benchmark CSV, selected result summaries, and concise documentation.

## Excluded

Generated Chroma database binaries, caches, review scratch files, duplicate result CSVs, report-writing artifacts, and archived intermediate memos were excluded. The original `MQP Database` folder was not modified destructively.

## How to Run

Install dependencies with `pip install -r requirements.txt`.

Run a demo:

```bash
python src/demo_retrieval.py
```

Run the 24-query evaluations:

```bash
python src/evaluate_decision_layer.py
python src/evaluate_chroma_ranking.py
```

Run the broader 150-query evaluations using `data/eval/query_set_expanded_150_v1.csv`.

## Verification Status

See `VERIFICATION.md` for exact commands and outcomes.

## GitHub URL

https://github.com/vdhuang/SoilX-GPR-Retrieval

## Caveats

The generated Chroma index is not committed. It is rebuilt locally from `data/manifest.csv` on first demo/evaluation run. The retrieval system searches manifest metadata rather than raw signal binaries.
