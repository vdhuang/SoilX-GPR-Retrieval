# Decision-Layer Evaluation

- Query set: `data\eval\query_set_expanded_150_v1.csv`
- Queries: `150`
- Threshold: `0.75`

## Branch Counts
- `threshold_fail`: `2`
- `threshold_pass`: `95`
- `unsupported`: `53`

## Family Averages
- `partial_support`: n=71, avg_confidence=0.8757, avg_top1_support_alignment=0.9718
- `supported`: n=16, avg_confidence=0.8910, avg_top1_support_alignment=1.0000
- `unsupported`: n=63, avg_confidence=0.1356, avg_top1_support_alignment=0.1587

## Labeled Sanity Checks
- `false_reject_good`: `0`
- `false_accept_bad`: `0`

## Expected-vs-Observed Checks
- `expected_decision_count`: `150`
- `decision_mismatch_count`: `79`
- `expected_branch_count`: `150`
- `branch_mismatch_count`: `79`
