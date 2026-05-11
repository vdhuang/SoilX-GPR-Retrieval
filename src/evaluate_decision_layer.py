from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from chroma_retriever import ChromaManifestRetriever
from eval_query_sets import QUERY_SET_24, EvalQuery, load_query_set


def evaluate_decision_layer(threshold: float, query_set: List[EvalQuery]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    retriever = ChromaManifestRetriever()

    rows: List[Dict[str, Any]] = []
    for item in query_set:
        decision = retriever.decide(item.text, top_k=3, confidence_threshold=threshold)
        top = decision["top_results"][0] if decision["top_results"] else {}

        rows.append(
            {
                "query_id": item.query_id,
                "family": item.family,
                "query_label": item.label,
                "query": item.text,
                "expected_decision": item.expected_decision,
                "expected_branch": item.expected_branch,
                "label_confidence": item.label_confidence,
                "notes": item.notes,
                "decision": decision["decision"],
                "branch": decision["branch"],
                "confidence": float(decision["confidence"]),
                "confidence_threshold": float(decision["confidence_threshold"]),
                "feasibility_status": top.get("feasibility_status", ""),
                "top1_support_alignment": float(top.get("support_alignment", 0.0)),
                "top1_similarity_score": float(top.get("similarity_score", 0.0)),
                "top1_sample_index": str(top.get("sample_index", "")),
                "top1_filter_stage": str(top.get("filter_stage", "")),
            }
        )

    summary: Dict[str, Any] = {
        "n_queries": len(rows),
        "threshold": threshold,
        "branch_counts": {},
        "family_averages": {},
    }

    branch_counts: Dict[str, int] = {}
    for r in rows:
        b = str(r["branch"])
        branch_counts[b] = branch_counts.get(b, 0) + 1
    summary["branch_counts"] = branch_counts

    families = sorted({str(r["family"]) for r in rows})
    family_avgs: Dict[str, Dict[str, float]] = {}
    for fam in families:
        fam_rows = [r for r in rows if str(r["family"]) == fam]
        if not fam_rows:
            continue
        avg_conf = sum(float(r["confidence"]) for r in fam_rows) / len(fam_rows)
        avg_align = sum(float(r["top1_support_alignment"]) for r in fam_rows) / len(fam_rows)
        family_avgs[fam] = {
            "n": float(len(fam_rows)),
            "avg_confidence": avg_conf,
            "avg_top1_support_alignment": avg_align,
        }
    summary["family_averages"] = family_avgs

    false_reject_good = sum(1 for r in rows if r["query_label"] == "good" and r["decision"] != "accept")
    false_accept_bad = sum(1 for r in rows if r["query_label"] == "bad" and r["decision"] == "accept")
    summary["false_reject_good"] = false_reject_good
    summary["false_accept_bad"] = false_accept_bad

    expected_decision_rows = [r for r in rows if str(r["expected_decision"]).strip()]
    expected_branch_rows = [r for r in rows if str(r["expected_branch"]).strip()]
    summary["expected_decision_count"] = len(expected_decision_rows)
    summary["expected_branch_count"] = len(expected_branch_rows)
    summary["decision_mismatch_count"] = sum(
        1
        for r in expected_decision_rows
        if str(r["decision"]).strip() != str(r["expected_decision"]).strip()
    )
    summary["branch_mismatch_count"] = sum(
        1
        for r in expected_branch_rows
        if str(r["branch"]).strip() != str(r["expected_branch"]).strip()
    )

    return rows, summary


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_inspection_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    inspection_rows: List[Dict[str, Any]] = []

    for r in rows:
        expected_decision = str(r.get("expected_decision", "")).strip()
        expected_branch = str(r.get("expected_branch", "")).strip()
        observed_decision = str(r.get("decision", "")).strip()
        observed_branch = str(r.get("branch", "")).strip()

        decision_match = (not expected_decision) or (expected_decision == observed_decision)
        branch_match = (not expected_branch) or (expected_branch == observed_branch)

        inspection_rows.append(
            {
                "query_id": r.get("query_id", ""),
                "family": r.get("family", ""),
                "query": r.get("query", ""),
                "expected_decision": expected_decision,
                "observed_decision": observed_decision,
                "decision_match": decision_match,
                "expected_branch": expected_branch,
                "observed_branch": observed_branch,
                "branch_match": branch_match,
                "confidence": r.get("confidence", 0.0),
                "top1_support_alignment": r.get("top1_support_alignment", 0.0),
                "top1_sample_index": r.get("top1_sample_index", ""),
            }
        )

    if not inspection_rows:
        return

    fieldnames = list(inspection_rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inspection_rows)


def write_summary_md(summary: Dict[str, Any], out_md: Path, query_set_name: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("# Decision-Layer Evaluation")
    lines.append("")
    lines.append(f"- Query set: `{query_set_name}`")
    lines.append(f"- Queries: `{summary['n_queries']}`")
    lines.append(f"- Threshold: `{summary['threshold']:.2f}`")
    lines.append("")
    lines.append("## Branch Counts")
    for branch, count in sorted(summary["branch_counts"].items()):
        lines.append(f"- `{branch}`: `{count}`")
    lines.append("")
    lines.append("## Family Averages")
    for fam, vals in sorted(summary["family_averages"].items()):
        lines.append(
            f"- `{fam}`: n={int(vals['n'])}, avg_confidence={vals['avg_confidence']:.4f}, "
            f"avg_top1_support_alignment={vals['avg_top1_support_alignment']:.4f}"
        )
    lines.append("")
    lines.append("## Labeled Sanity Checks")
    lines.append(f"- `false_reject_good`: `{summary['false_reject_good']}`")
    lines.append(f"- `false_accept_bad`: `{summary['false_accept_bad']}`")
    lines.append("")
    lines.append("## Expected-vs-Observed Checks")
    lines.append(f"- `expected_decision_count`: `{summary['expected_decision_count']}`")
    lines.append(f"- `decision_mismatch_count`: `{summary['decision_mismatch_count']}`")
    lines.append(f"- `expected_branch_count`: `{summary['expected_branch_count']}`")
    lines.append(f"- `branch_mismatch_count`: `{summary['branch_mismatch_count']}`")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate decision-layer outcomes on a query set.")
    parser.add_argument("--threshold", type=float, default=ChromaManifestRetriever.DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--query-set-csv", type=Path, default=None, help="Optional CSV query set path.")
    parser.add_argument("--out-csv", type=Path, default=Path("results") / "decision_eval_24.csv")
    parser.add_argument("--out-summary", type=Path, default=Path("results") / "decision_eval_24_summary.md")
    parser.add_argument(
        "--out-inspection",
        type=Path,
        default=None,
        help="Per-query inspection CSV with expected-vs-observed checks.",
    )
    args = parser.parse_args()

    query_set = load_query_set(args.query_set_csv, default_query_set=QUERY_SET_24)
    query_set_name = str(args.query_set_csv) if args.query_set_csv else "QUERY_SET_24"
    out_inspection = args.out_inspection or Path(str(args.out_csv).replace(".csv", "_inspection.csv"))

    rows, summary = evaluate_decision_layer(threshold=args.threshold, query_set=query_set)
    write_csv(rows, args.out_csv)
    write_inspection_csv(rows, out_inspection)
    write_summary_md(summary, args.out_summary, query_set_name=query_set_name)

    print("Decision-Layer Evaluation")
    print(f"query_set={query_set_name} queries={summary['n_queries']} threshold={summary['threshold']:.2f}")
    print("Branch counts:")
    for branch, count in sorted(summary["branch_counts"].items()):
        print(f"  {branch}: {count}")
    print("Family averages:")
    for fam, vals in sorted(summary["family_averages"].items()):
        print(
            f"  {fam}: n={int(vals['n'])} "
            f"avg_confidence={vals['avg_confidence']:.4f} "
            f"avg_top1_support_alignment={vals['avg_top1_support_alignment']:.4f}"
        )
    print(
        f"Sanity: false_reject_good={summary['false_reject_good']} "
        f"false_accept_bad={summary['false_accept_bad']}"
    )
    print(
        f"Expected-vs-observed: decision_mismatch={summary['decision_mismatch_count']}/"
        f"{summary['expected_decision_count']} branch_mismatch={summary['branch_mismatch_count']}/"
        f"{summary['expected_branch_count']}"
    )
    print(f"Wrote: {args.out_csv}")
    print(f"Wrote: {args.out_summary}")
    print(f"Wrote: {out_inspection}")


if __name__ == "__main__":
    main()
