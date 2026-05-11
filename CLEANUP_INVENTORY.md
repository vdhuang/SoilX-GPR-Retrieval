# Cleanup Inventory

## Important Retrieval Files Kept

- `src/chroma_retriever.py`
- `src/VectorDB.py`
- `src/demo_retrieval.py`
- `demo/run_chroma_demo.py`
- `demo/run_demo.py`

## Evaluation Files Kept

- `src/eval_query_sets.py`
- `src/evaluate_chroma_ranking.py`
- `src/evaluate_decision_layer.py`
- `src/evaluate_retriever.py`
- `data/eval/query_set_expanded_150_v1.csv`

## Data Files Kept

- `data/manifest.csv`

## Result Summaries Kept

- `results/chroma_ranking_eval_24_comparison.md`
- `results/decision_eval_24_summary.md`
- `results/chroma_ranking_eval_150_v1_comparison.md`
- `results/decision_eval_150_v1_summary.md`

## Files Excluded

- `data/chroma/` persisted Chroma database files
- `src/__pycache__/` and Python bytecode
- review workflow scripts such as `review_triage.py`, `prepare_blind_review.py`, session queue/status scripts, and label comparison utilities
- `results/review/` review work queues and triage outputs
- per-query evaluation CSV outputs
- duplicate/repro copies of expanded query sets and result outputs
- archive result charts and old architecture/report memos

## Uncertain Files

- `src/VectorDB.py` is retained because it is the original dashboard/numeric retriever, even though the Chroma path is the primary handoff target.
- `src/evaluate_retriever.py` is retained because it evaluates the original numeric retriever and is lightweight.
