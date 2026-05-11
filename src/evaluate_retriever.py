from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from VectorDB import GPR_Retriever


@dataclass
class EvalResult:
    mode: str
    n_queries: int
    top1_hit_rate: float
    top3_hit_rate: float
    mean_rank: float


def _clamp_physical(column: str, value: float) -> float:
    """Keep noisy values inside the same physical constraints used by the retriever."""
    if "thickness_m" in column:
        return max(0.0, value)
    if column.endswith("_pct"):
        return min(100.0, max(0.0, value))
    if "theta_v" in column:
        return min(1.0, max(0.0, value))
    if "bulk_density_gcm3" in column or "particle_density_gcm3" in column:
        return max(1e-6, value)
    return value


def _row_to_full_query(retriever: GPR_Retriever, row_idx: int) -> Dict[str, float]:
    query: Dict[str, float] = {}
    for col in retriever.search_columns:
        cidx = retriever.column_to_index[col]
        query[col] = float(retriever.raw_matrix[row_idx, cidx])
    return query


def _row_to_partial_query(retriever: GPR_Retriever, row_idx: int, rng: np.random.Generator) -> Dict[str, float]:
    columns = retriever.search_columns
    k = int(rng.integers(3, min(8, len(columns)) + 1))
    chosen = rng.choice(columns, size=k, replace=False).tolist()
    query: Dict[str, float] = {}
    for col in chosen:
        cidx = retriever.column_to_index[col]
        query[col] = float(retriever.raw_matrix[row_idx, cidx])
    return query


def _row_to_noisy_query(retriever: GPR_Retriever, row_idx: int, rng: np.random.Generator) -> Dict[str, float]:
    columns = retriever.search_columns
    k = int(rng.integers(3, min(8, len(columns)) + 1))
    chosen = rng.choice(columns, size=k, replace=False).tolist()
    query: Dict[str, float] = {}

    for col in chosen:
        cidx = retriever.column_to_index[col]
        base_val = float(retriever.raw_matrix[row_idx, cidx])
        col_std = float(retriever.scale_[cidx])
        noise = float(rng.normal(0.0, 0.05 * col_std))
        noisy_val = _clamp_physical(col, base_val + noise)
        query[col] = noisy_val
    return query


def _rank_of_target(retriever: GPR_Retriever, query: Dict[str, float], target_sample_index: str) -> int:
    matches = retriever.search(query, top_k=retriever.total_rows, min_score=0.0)
    for i, match in enumerate(matches, start=1):
        if str(match["sample_index"]) == str(target_sample_index):
            return i
    return retriever.total_rows + 1


def _evaluate_mode(
    retriever: GPR_Retriever,
    row_indices: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> EvalResult:
    ranks: List[int] = []

    for row_idx in row_indices:
        target_sample_index = str(retriever.sample_indices[int(row_idx)])
        if mode == "full":
            query = _row_to_full_query(retriever, int(row_idx))
        elif mode == "partial":
            query = _row_to_partial_query(retriever, int(row_idx), rng)
        elif mode == "noisy":
            query = _row_to_noisy_query(retriever, int(row_idx), rng)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        rank = _rank_of_target(retriever, query, target_sample_index)
        ranks.append(rank)

    n = len(ranks)
    top1 = sum(1 for r in ranks if r == 1) / n
    top3 = sum(1 for r in ranks if r <= 3) / n
    mean_rank = float(np.mean(ranks))

    return EvalResult(
        mode=mode,
        n_queries=n,
        top1_hit_rate=top1,
        top3_hit_rate=top3,
        mean_rank=mean_rank,
    )


def run_evaluation(sample_size: int, seed: int) -> List[EvalResult]:
    retriever = GPR_Retriever()
    rng = np.random.default_rng(seed)

    if sample_size < 1:
        raise ValueError("sample_size must be >= 1")
    sample_size = min(sample_size, retriever.total_rows)

    row_indices = rng.choice(retriever.total_rows, size=sample_size, replace=False)
    modes = ["full", "partial", "noisy"]
    return [_evaluate_mode(retriever, row_indices, mode, rng) for mode in modes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal retrieval evaluation harness for SoilX GPR retriever.")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of manifest rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    results = run_evaluation(sample_size=args.sample_size, seed=args.seed)

    print("SoilX Retrieval Evaluation")
    print(f"sample_size={args.sample_size} seed={args.seed}")
    print("-" * 72)
    print(f"{'Mode':<10} {'Queries':>8} {'Top1':>10} {'Top3':>10} {'MeanRank':>12}")
    print("-" * 72)
    for res in results:
        print(
            f"{res.mode:<10} {res.n_queries:>8d} "
            f"{res.top1_hit_rate * 100:>9.2f}% {res.top3_hit_rate * 100:>9.2f}% "
            f"{res.mean_rank:>12.2f}"
        )
    print("-" * 72)


if __name__ == "__main__":
    main()
