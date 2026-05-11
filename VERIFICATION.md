# Verification

Commands run from `C:\Users\discc\Downloads\Productive\School\SoilX-GPR-Retrieval-clean`.

## Compile Check

```bash
python -m compileall src
```

Status: pass. All files in `src/` compiled successfully.

## Dependency Import Check

```bash
python -c "import chromadb, pandas, numpy, sklearn; print('deps ok')"
```

Status: pass.

## Demo Query

```bash
python src\demo_retrieval.py
```

Status: pass.

Observed output summary:

- Built Chroma collection with 10,000 rows.
- Query: `layer 2 silt 30 percent`
- Decision: `accept`
- Branch: `threshold_pass`
- Confidence: `0.882`
- Top sample indices: `1526`, `9084`, `6499`

## Lightweight Evaluation

```bash
python src\evaluate_decision_layer.py --out-csv results\verification_decision_eval_24.csv --out-summary results\verification_decision_eval_24_summary.md
```

Status: pass.

Observed output summary:

- Query set: `QUERY_SET_24`
- Queries: `24`
- Branch counts: `threshold_pass=21`, `unsupported=3`
- Sanity checks: `false_reject_good=0`, `false_accept_bad=0`

The temporary verification CSV/summary files were removed after the run to keep the repository clean.

## What Vijay Should Run

```bash
pip install -r requirements.txt
python src\demo_retrieval.py
python src\evaluate_decision_layer.py
python src\evaluate_chroma_ranking.py
```

The Chroma index under `data/chroma/` is generated locally from `data/manifest.csv` and is not committed.
