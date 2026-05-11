# Chroma Ranking Protocol Comparison

- Query set: `data\eval\query_set_expanded_150_v1.csv`

## Protocols
- baseline: relevant if `support_alignment >= relevance_threshold`.
- strict: relevant if `support_alignment >= relevance_threshold` and `satisfied_targets >= min(strict_min_targets, query_target_count)`.
- bounded_recall: relevant set is top-N feasible candidates by support score (`support_alignment`, tie-broken by satisfied targets).

## Overall Comparison
- ndcg_at_3: baseline=0.6333, strict=0.6333, bounded_recall=0.6333
- precision_at_3: baseline=0.6333, strict=0.6333, bounded_recall=0.0644
- recall_at_3: baseline=0.0015, strict=0.0015, bounded_recall=0.0039
- mrr: baseline=0.4525, strict=0.4525, bounded_recall=0.0302

## By Family
- `partial_support`: NDCG@3 0.9718/0.9718/0.9718, P@3 0.9718/0.9718/0.0563, R@3 0.0020/0.0020/0.0034, MRR 0.6775/0.6775/0.0459
- `supported`: NDCG@3 1.0000/1.0000/1.0000, P@3 1.0000/1.0000/0.1250, R@3 0.0047/0.0047/0.0075, MRR 0.6109/0.6109/0.0486
- `unsupported`: NDCG@3 0.1587/0.1587/0.1587, P@3 0.1587/0.1587/0.0582, R@3 0.0001/0.0001/0.0035, MRR 0.1587/0.1587/0.0078

Legend in By Family: baseline/strict/bounded_recall
