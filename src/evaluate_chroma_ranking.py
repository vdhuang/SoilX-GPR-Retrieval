from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from chroma_retriever import ChromaManifestRetriever
from eval_query_sets import QUERY_SET_24, load_query_set


def _dcg_at_k(relevances: List[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances[:k], start=1):
        dcg += (2.0**rel - 1.0) / math.log2(i + 1.0)
    return dcg


def _is_relevant_baseline(rel: float, relevance_threshold: float) -> bool:
    return rel >= relevance_threshold


def _is_relevant_strict(
    rel: float,
    satisfied_targets: int,
    query_target_count: int,
    relevance_threshold: float,
    strict_min_targets: int,
) -> bool:
    required_targets = min(strict_min_targets, max(1, query_target_count))
    return (rel >= relevance_threshold) and (satisfied_targets >= required_targets)


def _build_relevant_ids(
    protocol: str,
    support_rows: List[Dict[str, Any]],
    query_target_count: int,
    relevance_threshold: float,
    strict_min_targets: int,
    bounded_top_n: int,
) -> List[str]:
    if protocol == "baseline":
        return [
            r["sid"]
            for r in support_rows
            if _is_relevant_baseline(float(r["rel"]), relevance_threshold)
        ]

    if protocol == "strict":
        return [
            r["sid"]
            for r in support_rows
            if _is_relevant_strict(
                rel=float(r["rel"]),
                satisfied_targets=int(r["sat_targets"]),
                query_target_count=query_target_count,
                relevance_threshold=relevance_threshold,
                strict_min_targets=strict_min_targets,
            )
        ]

    if protocol == "bounded_recall":
        feasible = [r for r in support_rows if float(r["rel"]) > 0.0]
        feasible.sort(key=lambda x: (-float(x["rel"]), -int(x["sat_targets"]), str(x["sid"])))
        top_n = max(1, bounded_top_n)
        return [str(r["sid"]) for r in feasible[:top_n]]

    raise ValueError(f"Unknown protocol: {protocol}")


def _evaluate_query_ranking(
    retriever: ChromaManifestRetriever,
    query_text: str,
    protocol: str,
    relevance_threshold: float,
    strict_min_targets: int,
    bounded_top_n: int,
) -> Dict[str, Any]:
    query_document = retriever._query_to_document(query_text)
    constraints = retriever._parse_constraints_from_document(query_document)
    supported, unsupported, feasibility_status = retriever._assess_constraint_feasibility(constraints)
    query_target_count = len({c["key"] for c in supported})

    support_rows: List[Dict[str, Any]] = []
    relevance_by_id: Dict[str, float] = {}

    for sid in retriever._all_sample_ids:
        diag = retriever._evaluate_result_support(str(sid), supported)
        rel = float(diag["support_alignment"])
        sat_targets = len(diag.get("supported_satisfied", []))
        support_rows.append({"sid": str(sid), "rel": rel, "sat_targets": sat_targets})
        relevance_by_id[str(sid)] = rel

    relevant_ids = _build_relevant_ids(
        protocol=protocol,
        support_rows=support_rows,
        query_target_count=query_target_count,
        relevance_threshold=relevance_threshold,
        strict_min_targets=strict_min_targets,
        bounded_top_n=bounded_top_n,
    )

    top3 = retriever.retrieve(query_text, top_k=3)
    full_ranked = retriever.retrieve(query_text, top_k=len(retriever._all_sample_ids))

    top3_ids = [str(r["sample_index"]) for r in top3]
    top3_rels = [relevance_by_id.get(sid, 0.0) for sid in top3_ids]

    relevant_id_set = set(relevant_ids)
    relevant_in_top3 = sum(1 for sid in top3_ids if sid in relevant_id_set)
    precision_at_3 = relevant_in_top3 / 3.0
    recall_at_3 = (relevant_in_top3 / len(relevant_ids)) if relevant_ids else 0.0

    dcg = _dcg_at_k(top3_rels, 3)
    ideal_rels = sorted(relevance_by_id.values(), reverse=True)
    idcg = _dcg_at_k(ideal_rels, 3)
    ndcg_at_3 = (dcg / idcg) if idcg > 0.0 else 0.0

    rr = 0.0
    if relevant_ids:
        for rank, row in enumerate(full_ranked, start=1):
            sid = str(row["sample_index"])
            if sid in relevant_id_set:
                rr = 1.0 / rank
                break

    return {
        "protocol": protocol,
        "query": query_text,
        "feasibility_status": feasibility_status,
        "constraint_count": len(constraints),
        "supported_constraint_count": len(supported),
        "unsupported_constraint_count": len(unsupported),
        "query_target_count": query_target_count,
        "relevant_count": len(relevant_ids),
        "precision_at_3": precision_at_3,
        "recall_at_3": recall_at_3,
        "ndcg_at_3": ndcg_at_3,
        "mrr": rr,
        "top3_ids": " | ".join(top3_ids),
        "top3_relevance": " | ".join(f"{x:.3f}" for x in top3_rels),
    }


def evaluate_ranking_set(
    protocol: str,
    relevance_threshold: float,
    strict_min_targets: int,
    bounded_top_n: int,
    query_set,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    retriever = ChromaManifestRetriever()

    rows: List[Dict[str, Any]] = []
    for q in query_set:
        result = _evaluate_query_ranking(
            retriever,
            q.text,
            protocol=protocol,
            relevance_threshold=relevance_threshold,
            strict_min_targets=strict_min_targets,
            bounded_top_n=bounded_top_n,
        )
        result["query_id"] = q.query_id
        result["family"] = q.family
        result["query_label"] = q.label
        rows.append(result)

    summary: Dict[str, Any] = {
        "protocol": protocol,
        "n_queries": len(rows),
        "relevance_threshold": relevance_threshold,
        "strict_min_targets": strict_min_targets,
        "bounded_top_n": bounded_top_n,
        "overall": {
            "ndcg_at_3": mean(r["ndcg_at_3"] for r in rows),
            "precision_at_3": mean(r["precision_at_3"] for r in rows),
            "recall_at_3": mean(r["recall_at_3"] for r in rows),
            "mrr": mean(r["mrr"] for r in rows),
        },
        "by_family": {},
    }

    for family in sorted({r["family"] for r in rows}):
        fam_rows = [r for r in rows if r["family"] == family]
        summary["by_family"][family] = {
            "n": len(fam_rows),
            "ndcg_at_3": mean(r["ndcg_at_3"] for r in fam_rows),
            "precision_at_3": mean(r["precision_at_3"] for r in fam_rows),
            "recall_at_3": mean(r["recall_at_3"] for r in fam_rows),
            "mrr": mean(r["mrr"] for r in fam_rows),
        }

    return rows, summary


def write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_summary_md(summary: Dict[str, Any], out_md: Path, query_set_name: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Chroma Ranking Metrics Evaluation ({summary['protocol']})")
    lines.append("")
    lines.append(f"- Query set: `{query_set_name}`")
    lines.append(f"- Queries: `{summary['n_queries']}`")
    lines.append(f"- Relevance threshold: `{summary['relevance_threshold']:.2f}`")
    lines.append(f"- Strict min targets: `{summary['strict_min_targets']}`")
    lines.append(f"- Bounded top-N: `{summary['bounded_top_n']}`")
    lines.append("")
    lines.append("## Overall")
    overall = summary["overall"]
    lines.append(f"- NDCG@3: `{overall['ndcg_at_3']:.4f}`")
    lines.append(f"- Precision@3: `{overall['precision_at_3']:.4f}`")
    lines.append(f"- Recall@3: `{overall['recall_at_3']:.4f}`")
    lines.append(f"- MRR: `{overall['mrr']:.4f}`")
    lines.append("")
    lines.append("## By Family")
    for family, vals in summary["by_family"].items():
        lines.append(
            f"- `{family}` (n={vals['n']}): "
            f"NDCG@3={vals['ndcg_at_3']:.4f}, "
            f"P@3={vals['precision_at_3']:.4f}, "
            f"R@3={vals['recall_at_3']:.4f}, "
            f"MRR={vals['mrr']:.4f}"
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_comparison_md(summaries: Dict[str, Dict[str, Any]], out_md: Path, query_set_name: str) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    def _metric_line(metric: str) -> str:
        return (
            f"- {metric}: "
            f"baseline={summaries['baseline']['overall'][metric]:.4f}, "
            f"strict={summaries['strict']['overall'][metric]:.4f}, "
            f"bounded_recall={summaries['bounded_recall']['overall'][metric]:.4f}"
        )

    lines: List[str] = []
    lines.append("# Chroma Ranking Protocol Comparison")
    lines.append("")
    lines.append(f"- Query set: `{query_set_name}`")
    lines.append("")
    lines.append("## Protocols")
    lines.append("- baseline: relevant if `support_alignment >= relevance_threshold`.")
    lines.append(
        "- strict: relevant if `support_alignment >= relevance_threshold` and "
        "`satisfied_targets >= min(strict_min_targets, query_target_count)`."
    )
    lines.append(
        "- bounded_recall: relevant set is top-N feasible candidates by support score "
        "(`support_alignment`, tie-broken by satisfied targets)."
    )
    lines.append("")
    lines.append("## Overall Comparison")
    lines.append(_metric_line("ndcg_at_3"))
    lines.append(_metric_line("precision_at_3"))
    lines.append(_metric_line("recall_at_3"))
    lines.append(_metric_line("mrr"))
    lines.append("")
    lines.append("## By Family")
    for family in sorted(summaries["baseline"]["by_family"].keys()):
        b = summaries["baseline"]["by_family"][family]
        s = summaries["strict"]["by_family"][family]
        br = summaries["bounded_recall"]["by_family"][family]
        lines.append(
            f"- `{family}`: "
            f"NDCG@3 {b['ndcg_at_3']:.4f}/{s['ndcg_at_3']:.4f}/{br['ndcg_at_3']:.4f}, "
            f"P@3 {b['precision_at_3']:.4f}/{s['precision_at_3']:.4f}/{br['precision_at_3']:.4f}, "
            f"R@3 {b['recall_at_3']:.4f}/{s['recall_at_3']:.4f}/{br['recall_at_3']:.4f}, "
            f"MRR {b['mrr']:.4f}/{s['mrr']:.4f}/{br['mrr']:.4f}"
        )
    lines.append("")
    lines.append("Legend in By Family: baseline/strict/bounded_recall")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Chroma ranking metrics with baseline + strict + bounded-recall protocols."
    )
    parser.add_argument("--query-set-csv", type=Path, default=None, help="Optional CSV query set path.")
    parser.add_argument("--relevance-threshold", type=float, default=1.0)
    parser.add_argument("--strict-min-targets", type=int, default=2)
    parser.add_argument("--bounded-top-n", type=int, default=50)
    parser.add_argument("--out-prefix", type=Path, default=Path("results") / "chroma_ranking_eval_24")
    args = parser.parse_args()

    query_set = load_query_set(args.query_set_csv, default_query_set=QUERY_SET_24)
    query_set_name = str(args.query_set_csv) if args.query_set_csv else "QUERY_SET_24"

    protocols = ["baseline", "strict", "bounded_recall"]
    summaries: Dict[str, Dict[str, Any]] = {}

    for protocol in protocols:
        rows, summary = evaluate_ranking_set(
            protocol=protocol,
            relevance_threshold=args.relevance_threshold,
            strict_min_targets=args.strict_min_targets,
            bounded_top_n=args.bounded_top_n,
            query_set=query_set,
        )
        summaries[protocol] = summary
        out_csv = Path(str(args.out_prefix) + f"_{protocol}.csv")
        out_md = Path(str(args.out_prefix) + f"_{protocol}_summary.md")
        write_csv(rows, out_csv)
        write_summary_md(summary, out_md, query_set_name=query_set_name)

    cmp_md = Path(str(args.out_prefix) + "_comparison.md")
    write_comparison_md(summaries, cmp_md, query_set_name=query_set_name)

    print("Chroma Ranking Protocol Evaluation")
    print(
        f"query_set={query_set_name} queries={summaries['baseline']['n_queries']} "
        f"threshold={args.relevance_threshold:.2f} "
        f"strict_min_targets={args.strict_min_targets} bounded_top_n={args.bounded_top_n}"
    )
    for protocol in protocols:
        o = summaries[protocol]["overall"]
        print(
            f"{protocol:<14} ndcg@3={o['ndcg_at_3']:.4f} "
            f"p@3={o['precision_at_3']:.4f} r@3={o['recall_at_3']:.4f} mrr={o['mrr']:.4f}"
        )
    print(f"Wrote: {cmp_md}")


if __name__ == "__main__":
    main()
