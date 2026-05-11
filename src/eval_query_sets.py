from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    family: str
    label: str
    text: str
    expected_decision: str = ""
    expected_branch: str = ""
    label_confidence: str = ""
    notes: str = ""


QUERY_SET_24: List[EvalQuery] = [
    EvalQuery("Q24_001", "supported", "good", "top layer 0.7m, middle 1.4m, bottom 2.6m"),
    EvalQuery("Q24_002", "supported", "good", "layer 2 silt 30 percent"),
    EvalQuery("Q24_003", "supported", "good", "layer 1 clay 12 percent and high sand"),
    EvalQuery("Q24_004", "supported", "good", "bulk density 1.6 in layer 2"),
    EvalQuery("Q24_005", "supported", "good", "particle density 2.58 in layer 1"),
    EvalQuery("Q24_006", "supported", "good", "moist top layer theta_v 0.14"),
    EvalQuery("Q24_007", "edge", "good", "layer 1 thickness 0.50m"),
    EvalQuery("Q24_008", "edge", "good", "layer 2 thickness 1.00m"),
    EvalQuery("Q24_009", "edge", "good", "layer 3 thickness 2.00m"),
    EvalQuery("Q24_010", "edge", "good", "layer 1 theta_v 0.15"),
    EvalQuery("Q24_011", "edge", "good", "layer 2 bulk density 1.67"),
    EvalQuery("Q24_012", "edge", "good", "layer 2 silt 40 percent"),
    EvalQuery("Q24_013", "partial", "mixed", "top 0.25m middle 0.65m bottom 1.2m"),
    EvalQuery("Q24_014", "partial", "bad", "layer 2 bulk density 1.75"),
    EvalQuery("Q24_015", "partial", "bad", "layer 1 theta_v 0.45 with dry lower layers"),
    EvalQuery("Q24_016", "partial", "mixed", "layer 1 sand 80 percent layer 1 clay 10 percent"),
    EvalQuery("Q24_017", "partial", "mixed", "very thin top layer over thick lower layers"),
    EvalQuery("Q24_018", "partial", "bad", "layer 3 particle density 2.8"),
    EvalQuery("Q24_019", "mixed", "good", "dry clay over wet sand"),
    EvalQuery("Q24_020", "mixed", "good", "wet top layer with theta_v 0.13 and dense middle layer"),
    EvalQuery("Q24_021", "mixed", "good", "silt-rich middle layer around 40 percent silt"),
    EvalQuery("Q24_022", "mixed", "good", "high sand top layer around 70 percent with low clay"),
    EvalQuery("Q24_023", "mixed", "good", "thin surface layer 0.05m over thicker middle and bottom layers"),
    EvalQuery("Q24_024", "mixed", "good", "all three layers around 0.5m thickness"),
]


def load_query_set_csv(path: Path) -> List[EvalQuery]:
    required_cols = {
        "query_id",
        "query_text",
        "bucket",
        "expected_decision",
        "expected_branch",
        "label_confidence",
        "notes",
    }

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_cols - fieldnames)
        if missing:
            raise ValueError(f"Query-set CSV missing required columns: {missing}")

        rows: List[EvalQuery] = []
        seen_ids: set[str] = set()
        for idx, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            query_text = (row.get("query_text") or "").strip()
            bucket = (row.get("bucket") or "").strip()
            expected_decision = (row.get("expected_decision") or "").strip()
            expected_branch = (row.get("expected_branch") or "").strip()
            label_confidence = (row.get("label_confidence") or "").strip()
            notes = (row.get("notes") or "").strip()

            if not query_id:
                raise ValueError(f"Row {idx}: query_id is empty.")
            if query_id in seen_ids:
                raise ValueError(f"Duplicate query_id found: {query_id}")
            if not query_text:
                raise ValueError(f"Row {idx}: query_text is empty.")
            if not bucket:
                raise ValueError(f"Row {idx}: bucket is empty.")

            seen_ids.add(query_id)
            rows.append(
                EvalQuery(
                    query_id=query_id,
                    family=bucket,
                    label="external",
                    text=query_text,
                    expected_decision=expected_decision,
                    expected_branch=expected_branch,
                    label_confidence=label_confidence,
                    notes=notes,
                )
            )
    return rows


def load_query_set(query_set_csv: Path | None, default_query_set: Sequence[EvalQuery] | None = None) -> List[EvalQuery]:
    if query_set_csv is None:
        return list(default_query_set if default_query_set is not None else QUERY_SET_24)
    return load_query_set_csv(query_set_csv)
