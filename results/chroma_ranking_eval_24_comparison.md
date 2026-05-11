# Chroma Ranking Protocol Comparison

## Protocols
- baseline: relevant if `support_alignment >= relevance_threshold`.
- strict: relevant if `support_alignment >= relevance_threshold` and `satisfied_targets >= min(strict_min_targets, query_target_count)`.
- bounded_recall: relevant set is top-N feasible candidates by support score (`support_alignment`, tie-broken by satisfied targets).

## Overall Comparison
- ndcg_at_3: baseline=0.8750, strict=0.8750, bounded_recall=0.8750
- precision_at_3: baseline=0.8750, strict=0.8750, bounded_recall=0.0139
- recall_at_3: baseline=0.0011, strict=0.0011, bounded_recall=0.0008
- mrr: baseline=0.6783, strict=0.6783, bounded_recall=0.0310

## By Family
- `edge`: NDCG@3 1.0000/1.0000/1.0000, P@3 1.0000/1.0000/0.0000, R@3 0.0010/0.0010/0.0000, MRR 0.8889/0.8889/0.0413
- `mixed`: NDCG@3 1.0000/1.0000/1.0000, P@3 1.0000/1.0000/0.0000, R@3 0.0018/0.0018/0.0000, MRR 0.5604/0.5604/0.0242
- `partial`: NDCG@3 0.5000/0.5000/0.5000, P@3 0.5000/0.5000/0.0556, R@3 0.0005/0.0005/0.0033, MRR 0.3750/0.3750/0.0199
- `supported`: NDCG@3 1.0000/1.0000/1.0000, P@3 1.0000/1.0000/0.0000, R@3 0.0011/0.0011/0.0000, MRR 0.8889/0.8889/0.0387

Legend in By Family: baseline/strict/bounded_recall
