# Evaluation Summary

Included evaluation entry points:

- `python src/evaluate_decision_layer.py` runs the built-in 24-query decision-layer check.
- `python src/evaluate_chroma_ranking.py` runs the built-in 24-query Chroma ranking check.
- `python src/evaluate_decision_layer.py --query-set-csv data/eval/query_set_expanded_150_v1.csv --out-csv results/decision_eval_150_v1.csv --out-summary results/decision_eval_150_v1_summary.md` runs the broader 150-query decision benchmark.
- `python src/evaluate_chroma_ranking.py --query-set-csv data/eval/query_set_expanded_150_v1.csv --out-prefix results/chroma_ranking_eval_150_v1` runs the broader 150-query ranking benchmark.

Selected historical summaries are kept in `results/`. Per-query CSV outputs and review scratch files are excluded from the handoff package to keep the repository focused.
