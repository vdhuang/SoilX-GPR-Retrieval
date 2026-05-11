from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Dict, List

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception as exc:  # pragma: no cover - import guard for environments missing dependency
    raise RuntimeError(
        "chromadb is required for vector retrieval. Install with: pip install chromadb"
    ) from exc


class ChromaManifestRetriever:
    """Parallel vector-retrieval path over manifest rows using Chroma embeddings."""

    MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.csv"
    PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
    COLLECTION_NAME = "soilx_manifest"

    LAYER_LABELS = {
        "top": [1],
        "upper": [1],
        "surface": [1],
        "layer 1": [1],
        "middle": [2],
        "mid": [2],
        "layer 2": [2],
        "bottom": [3],
        "lower": [3],
        "layer 3": [3],
        "lower layers": [2, 3],
        "all layers": [1, 2, 3],
        "all three": [1, 2, 3],
    }

    MATERIAL_HINTS = {"sand", "silt", "clay", "wet", "dry", "moisture", "dense"}

    NUMERIC_SUFFIXES = [
        "thickness_m",
        "sand_pct",
        "silt_pct",
        "clay_pct",
        "theta_v",
        "bulk_density_gcm3",
        "particle_density_gcm3",
        "organic_fraction",
    ]

    BUCKET_STEPS = {
        "thickness_m": 0.25,
        "sand_pct": 10.0,
        "silt_pct": 10.0,
        "clay_pct": 10.0,
        "theta_v": 0.05,
        "bulk_density_gcm3": 0.05,
        "particle_density_gcm3": 0.05,
        "organic_fraction": 0.02,
    }

    HARD_TOLERANCE = {
        "thickness_m": 0.20,
        "sand_pct": 8.0,
        "silt_pct": 8.0,
        "clay_pct": 8.0,
        "theta_v": 0.05,
        "bulk_density_gcm3": 0.08,
        "particle_density_gcm3": 0.08,
        "organic_fraction": 0.05,
    }

    SOFT_TOLERANCE = {
        "thickness_m": 0.45,
        "sand_pct": 15.0,
        "silt_pct": 15.0,
        "clay_pct": 15.0,
        "theta_v": 0.10,
        "bulk_density_gcm3": 0.15,
        "particle_density_gcm3": 0.15,
        "organic_fraction": 0.10,
    }

    PREFERRED_TOLERANCE = {
        "thickness_m": 0.70,
        "sand_pct": 20.0,
        "silt_pct": 20.0,
        "clay_pct": 20.0,
        "theta_v": 0.14,
        "bulk_density_gcm3": 0.20,
        "particle_density_gcm3": 0.20,
        "organic_fraction": 0.14,
    }

    PRIORITY_POOL_MIN = 400
    DECISION_ALIGNMENT_WEIGHT = 0.5
    DECISION_SIMILARITY_WEIGHT = 0.5
    DEFAULT_CONFIDENCE_THRESHOLD = 0.75

    def __init__(
        self,
        manifest_path: Path | None = None,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.manifest_path = manifest_path or self.MANIFEST_PATH
        self.persist_dir = persist_dir or self.PERSIST_DIR
        self.collection_name = collection_name or self.COLLECTION_NAME

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self._refresh_collection()
        self._load_manifest_numeric_index()

    def _refresh_collection(self) -> None:
        """Reacquire a live collection handle."""
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _safe_float(raw: str | float | int | None) -> float | None:
        if raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None

    @staticmethod
    def _quantize(value: float, step: float) -> float:
        return round(value / step) * step

    def _load_manifest_numeric_index(self) -> None:
        """Load manifest numeric values for constraint-aware candidate filtering."""
        self._row_numeric: Dict[str, Dict[str, float]] = {}
        self._all_sample_ids: List[str] = []
        self._field_minmax: Dict[str, tuple[float, float]] = {}
        self._field_band_values: Dict[str, set[str]] = {}

        with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = str(row["sample_index"])
                self._all_sample_ids.append(sid)

                numeric_map: Dict[str, float] = {}
                for layer in (1, 2, 3):
                    for suffix in self.NUMERIC_SUFFIXES:
                        key = f"layer_{layer}_{suffix}"
                        value = self._safe_float(row.get(key, ""))
                        if value is None:
                            continue

                        numeric_map[key] = value
                        if key not in self._field_minmax:
                            self._field_minmax[key] = (value, value)
                        else:
                            lo, hi = self._field_minmax[key]
                            self._field_minmax[key] = (min(lo, value), max(hi, value))

                        band = self._band_for_field(suffix, value)
                        if band is not None:
                            self._field_band_values.setdefault(key, set()).add(band)

                self._row_numeric[sid] = numeric_map

    @classmethod
    def _band_for_field(cls, suffix: str, value: float) -> str | None:
        if suffix == "thickness_m":
            if value < 0.3:
                return "very_thin"
            if value < 0.8:
                return "thin"
            if value < 1.6:
                return "medium"
            return "thick"
        if suffix == "theta_v":
            if value < 0.12:
                return "very_dry"
            if value < 0.18:
                return "dry"
            if value < 0.26:
                return "moist"
            return "wet"
        if suffix == "bulk_density_gcm3":
            if value < 1.55:
                return "low"
            if value < 1.63:
                return "mid"
            return "high"
        if suffix == "particle_density_gcm3":
            if value < 2.55:
                return "low"
            if value < 2.62:
                return "mid"
            return "high"
        if suffix in {"sand_pct", "silt_pct", "clay_pct"}:
            if value < 20:
                return "low"
            if value < 40:
                return "mid"
            return "high"
        return None

    @classmethod
    def _encode_numeric_field_tokens(cls, field_key: str, suffix: str, raw_value: str | float | int) -> List[str]:
        tokens = [f"{field_key}={raw_value}"]
        value = cls._safe_float(raw_value)
        if value is None:
            return tokens

        step = cls.BUCKET_STEPS.get(suffix)
        if step:
            q = cls._quantize(value, step)
            tokens.append(f"{field_key}_bucket={q:.4f}")

        band = cls._band_for_field(suffix, value)
        if band:
            tokens.append(f"{field_key}_band={band}")

        return tokens

    @classmethod
    def _append_unique(cls, target: List[str], new_tokens: List[str]) -> None:
        for token in new_tokens:
            if token not in target:
                target.append(token)

    @classmethod
    def _row_to_document(cls, row: Dict[str, str]) -> str:
        """Convert one manifest row into structured exact + bucketed field tokens."""
        tokens: List[str] = [f"sample_index={row.get('sample_index', '')}"]
        for layer in (1, 2, 3):
            for suffix in cls.NUMERIC_SUFFIXES:
                key = f"layer_{layer}_{suffix}"
                raw = row.get(key, "")
                cls._append_unique(tokens, cls._encode_numeric_field_tokens(key, suffix, raw))
        tokens = cls._resolve_band_conflicts(tokens)
        return " ".join(tokens)

    @staticmethod
    def _extract_first_number(text: str) -> float | None:
        m = re.search(r"(\d*\.?\d+)", text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    @classmethod
    def _detect_layers_in_segment(cls, segment: str, default_layers: List[int]) -> List[int]:
        seg = segment.lower()
        found: List[int] = []
        for label, layers in cls.LAYER_LABELS.items():
            if label in seg:
                for layer in layers:
                    if layer not in found:
                        found.append(layer)
        return found if found else list(default_layers)

    @classmethod
    def _add_query_value_tokens(cls, tokens: List[str], layer: int, suffix: str, value: float) -> None:
        key = f"layer_{layer}_{suffix}"
        cls._append_unique(tokens, cls._encode_numeric_field_tokens(key, suffix, value))

    @classmethod
    def _add_query_band_token(cls, tokens: List[str], layer: int, suffix: str, band: str) -> None:
        cls._append_unique(tokens, [f"layer_{layer}_{suffix}_band={band}"])

    @classmethod
    def _extract_pct_for_material(cls, segment: str, material: str) -> float | None:
        seg = segment.lower()
        patterns = [
            rf"\b{material}(?:\s+content)?\b(?:[^\d]{{0,30}})(\d*\.?\d+)",
            rf"\b(\d*\.?\d+)\s*(?:%|percent)\s*{material}\b",
            rf"\b(\d*\.?\d+)\s*{material}\b",
            rf"\b{material}\b\s*(?:around|near|about)?\s*(\d*\.?\d+)\b",
        ]
        for p in patterns:
            m = re.search(p, seg)
            if m:
                value = cls._safe_float(m.group(1))
                if value is not None and 0.0 <= value <= 100.0:
                    return value
        return None

    @classmethod
    def _extract_thickness_value(cls, segment: str) -> float | None:
        seg = segment.lower()
        patterns = [
            r"thickness\s*(?:around|near|about)?\s*(\d*\.?\d+)",
            r"(\d*\.?\d+)\s*(?:m|meter|meters)\b",
            r"(\d*\.?\d+)\s*thickness",
        ]
        for p in patterns:
            m = re.search(p, seg)
            if m:
                return cls._safe_float(m.group(1))
        return None

    @classmethod
    def _extract_theta_value(cls, segment: str) -> float | None:
        seg = segment.lower()
        patterns = [
            r"theta[_\s-]?v\s*(?:around|near|about)?\s*(\d*\.?\d+)",
            r"moisture\s*(?:around|near|about)?\s*(\d*\.?\d+)",
        ]
        for p in patterns:
            m = re.search(p, seg)
            if m:
                return cls._safe_float(m.group(1))
        return None

    @classmethod
    def _extract_density_value(cls, segment: str, particle: bool) -> float | None:
        seg = segment.lower()
        if particle:
            patterns = [r"particle\s*density\s*(?:around|near|about)?\s*(\d*\.?\d+)"]
        else:
            patterns = [r"bulk\s*density\s*(?:around|near|about)?\s*(\d*\.?\d+)"]
        for p in patterns:
            m = re.search(p, seg)
            if m:
                return cls._safe_float(m.group(1))
        return None

    @classmethod
    def _extract_segment_constraints(
        cls,
        segment: str,
        layers: List[int],
        tokens: List[str],
    ) -> None:
        seg = segment.lower()

        # Thickness numeric and comparative thickness descriptors.
        t_value = cls._extract_thickness_value(seg)
        if t_value is not None:
            for layer in layers:
                cls._add_query_value_tokens(tokens, layer, "thickness_m", t_value)
        if "thinner" in seg or "thin" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "thickness_m", "thin")
        if "thicker" in seg or re.search(r"\bthick\b", seg):
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "thickness_m", "thick")
        # Moisture numeric and qualitative moisture.
        # Conservative gated split: only split clauses when layered labels and
        # true wet-vs-dry conflict are both explicit.
        def _apply_moisture_clause(clause: str, clause_layers: List[int]) -> None:
            theta_v = cls._extract_theta_value(clause)
            if theta_v is not None:
                for layer in clause_layers:
                    cls._add_query_value_tokens(tokens, layer, "theta_v", theta_v)

            if "very dry" in clause:
                for layer in clause_layers:
                    cls._add_query_band_token(tokens, layer, "theta_v", "very_dry")
            elif "dry" in clause:
                for layer in clause_layers:
                    cls._add_query_band_token(tokens, layer, "theta_v", "dry")

            if "very wet" in clause:
                for layer in clause_layers:
                    cls._add_query_band_token(tokens, layer, "theta_v", "wet")
            elif "wet" in clause:
                for layer in clause_layers:
                    cls._add_query_band_token(tokens, layer, "theta_v", "wet")
            elif "moist" in clause:
                for layer in clause_layers:
                    cls._add_query_band_token(tokens, layer, "theta_v", "moist")

        moisture_terms = ("theta_v", "moisture", "wet", "dry", "moist")
        clauses = [c.strip() for c in re.split(r"\band\b", seg) if c.strip()]

        def _has_layer_label(clause: str) -> bool:
            for label in cls.LAYER_LABELS:
                if label in clause:
                    return True
            return False

        def _has_wet_term(clause: str) -> bool:
            return ("very wet" in clause) or ("wet" in clause) or ("moist" in clause)

        def _has_dry_term(clause: str) -> bool:
            return ("very dry" in clause) or ("dry" in clause)

        split_is_justified = False
        if len(clauses) > 1 and any(term in seg for term in moisture_terms):
            clause_has_label = [_has_layer_label(c) for c in clauses]
            clause_has_wet = [_has_wet_term(c) for c in clauses]
            clause_has_dry = [_has_dry_term(c) for c in clauses]
            split_is_justified = (
                sum(1 for v in clause_has_label if v) >= 2
                and any(clause_has_wet)
                and any(clause_has_dry)
            )

        if split_is_justified:
            for clause in clauses:
                clause_layers = cls._detect_layers_in_segment(clause, default_layers=layers)
                _apply_moisture_clause(clause, clause_layers)
        else:
            _apply_moisture_clause(seg, layers)

        # Density numeric and qualitative density.
        bulk_v = cls._extract_density_value(seg, particle=False)
        if bulk_v is not None:
            for layer in layers:
                cls._add_query_value_tokens(tokens, layer, "bulk_density_gcm3", bulk_v)
        elif "dense" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "bulk_density_gcm3", "high")

        part_v = cls._extract_density_value(seg, particle=True)
        if part_v is not None:
            for layer in layers:
                cls._add_query_value_tokens(tokens, layer, "particle_density_gcm3", part_v)

        # Composition numeric.
        for material, suffix in (("sand", "sand_pct"), ("silt", "silt_pct"), ("clay", "clay_pct")):
            v = cls._extract_pct_for_material(seg, material)
            if v is not None:
                for layer in layers:
                    cls._add_query_value_tokens(tokens, layer, suffix, v)

        # Composition qualitative descriptors.
        if "clay-heavy" in seg or "clayey" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "clay_pct", "high")
        if "sandy" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "sand_pct", "high")
        if "silty" in seg or "silt-rich" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "silt_pct", "high")
        if "low clay" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "clay_pct", "low")
        if "high clay" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "clay_pct", "high")
        if "low sand" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "sand_pct", "low")
        if "high sand" in seg:
            for layer in layers:
                cls._add_query_band_token(tokens, layer, "sand_pct", "high")


    @classmethod
    def _resolve_band_conflicts(cls, tokens: List[str]) -> List[str]:
        """Ensure one consistent band token per layer+field."""
        numeric_values: Dict[str, float] = {}
        band_occurrences: Dict[str, List[str]] = {}

        for token in tokens:
            if "=" not in token:
                continue
            key, raw = token.split("=", 1)
            if key.endswith("_bucket") or key.endswith("_band"):
                continue
            if not re.match(r"^layer_\d+_.+", key):
                continue
            value = cls._safe_float(raw)
            if value is not None:
                numeric_values[key] = value

        for token in tokens:
            m = re.match(r"^(layer_\d+_(.+))_band=([a-z_]+)$", token)
            if not m:
                continue
            base_key = m.group(1)
            band = m.group(3)
            band_occurrences.setdefault(base_key, []).append(band)

        chosen_band: Dict[str, str] = {}
        for base_key, bands in band_occurrences.items():
            suffix_m = re.match(r"^layer_\d+_(.+)$", base_key)
            suffix = suffix_m.group(1) if suffix_m else ""

            derived = None
            if base_key in numeric_values:
                derived = cls._band_for_field(suffix, numeric_values[base_key])

            if derived is not None:
                chosen_band[base_key] = derived
            else:
                chosen_band[base_key] = bands[-1]

        out: List[str] = []
        emitted: set[str] = set()
        for token in tokens:
            m = re.match(r"^(layer_\d+_.+)_band=([a-z_]+)$", token)
            if not m:
                out.append(token)
                continue

            base_key = m.group(1)
            if base_key in emitted:
                continue
            out.append(f"{base_key}_band={chosen_band.get(base_key, m.group(2))}")
            emitted.add(base_key)

        return out

    @classmethod
    def _query_to_document(cls, query_text: str) -> str:
        """Convert query text into structured exact + bucketed tokens compatible with stored rows."""
        q = query_text.strip().lower()
        tokens: List[str] = []

        over_segments = [seg.strip() for seg in re.split(r"\bover\b", q) if seg.strip()]
        if len(over_segments) >= 2:
            seg_layers_defaults: List[List[int]] = [[1]] + [[2, 3] for _ in over_segments[1:]]
            segments = over_segments
        else:
            segments = [seg.strip() for seg in re.split(r",|;", q) if seg.strip()]
            seg_layers_defaults = [[1] for _ in segments]

        if not segments:
            segments = [q]
            seg_layers_defaults = [[1]]

        for idx, seg in enumerate(segments):
            layers = cls._detect_layers_in_segment(seg, default_layers=seg_layers_defaults[idx])
            cls._extract_segment_constraints(seg, layers, tokens)

        # Sequential thickness assignment for compact "0.2m, 0.5m, 1.2m" style.
        meter_numbers = [cls._safe_float(n) for n in re.findall(r"(\d*\.?\d+)\s*(?:m|meter|meters)", q)]
        meter_numbers = [v for v in meter_numbers if v is not None]
        explicit_thickness_layers = {
            int(m.group(1))
            for token in tokens
            for m in [re.match(r"layer_(\d)_thickness_m=", token)]
            if m
        }
        if len(meter_numbers) >= 2 and len(explicit_thickness_layers) < 3:
            for layer, value in enumerate(meter_numbers[:3], start=1):
                if layer not in explicit_thickness_layers:
                    cls._add_query_value_tokens(tokens, layer, "thickness_m", value)

        if ("all three" in q or "all layers" in q) and "thickness" in q and meter_numbers:
            for layer in (1, 2, 3):
                cls._add_query_value_tokens(tokens, layer, "thickness_m", meter_numbers[0])

        # Fallback lexical hints when nothing extracted.
        if not tokens:
            hints = [word for word in re.findall(r"[a-z_]+", q) if word in cls.MATERIAL_HINTS]
            if not hints:
                hints = re.findall(r"[a-z_]+", q)[:8]
            cls._append_unique(tokens, [f"hint={h}" for h in hints])

        tokens = cls._resolve_band_conflicts(tokens)
        return " ".join(tokens)

    @staticmethod
    def _parse_constraints_from_document(query_document: str) -> List[Dict[str, Any]]:
        constraints_map: Dict[str, Dict[str, Any]] = {}
        for token in query_document.split():
            if "=" not in token:
                continue
            key, value_raw = token.split("=", 1)

            m_band = re.match(r"^layer_(\d)_(.+)_band$", key)
            if m_band:
                layer = int(m_band.group(1))
                suffix = m_band.group(2)
                if suffix in ChromaManifestRetriever.NUMERIC_SUFFIXES:
                    base_key = f"layer_{layer}_{suffix}"
                    constraints_map[key] = {
                        "type": "band",
                        "key": base_key,
                        "layer": layer,
                        "suffix": suffix,
                        "band": value_raw,
                    }
                continue

            if key.endswith("_bucket"):
                continue

            m_num = re.match(r"^layer_(\d)_(.+)$", key)
            if not m_num:
                continue
            layer = int(m_num.group(1))
            suffix = m_num.group(2)
            if suffix not in ChromaManifestRetriever.NUMERIC_SUFFIXES:
                continue
            try:
                value = float(value_raw)
            except ValueError:
                continue
            constraints_map[key] = {
                "type": "numeric",
                "key": key,
                "layer": layer,
                "suffix": suffix,
                "value": value,
            }

        return list(constraints_map.values())


    @classmethod
    def _constraints_to_document(cls, constraints: List[Dict[str, Any]]) -> str:
        tokens: List[str] = []
        for c in constraints:
            if c.get("type") == "numeric":
                cls._append_unique(tokens, cls._encode_numeric_field_tokens(c["key"], c["suffix"], c["value"]))
                continue
            if c.get("type") == "band":
                cls._append_unique(tokens, [f"{c['key']}_band={c['band']}"])
        tokens = cls._resolve_band_conflicts(tokens)
        return " ".join(tokens)


    def _project_numeric_to_feasible_band(self, numeric_c: Dict[str, Any]) -> Dict[str, Any] | None:
        # Bounded weak-case refinement: project only out-of-support thickness constraints.
        if numeric_c.get("suffix") != "thickness_m":
            return None

        key = numeric_c["key"]
        minmax = self._field_minmax.get(key)
        if minmax is None:
            return None

        lo, hi = minmax
        clamped = min(max(float(numeric_c["value"]), lo), hi)
        band = self._band_for_field(numeric_c["suffix"], clamped)
        if band is None:
            return None

        if band not in self._field_band_values.get(key, set()):
            return None

        return {
            "type": "band",
            "key": key,
            "layer": numeric_c["layer"],
            "suffix": numeric_c["suffix"],
            "band": band,
            "projected": True,
            "source_value": numeric_c["value"],
            "projected_value": clamped,
        }

    def _assess_constraint_feasibility(
        self, constraints: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], List[str], str]:
        if not constraints:
            return [], [], "no_constraints"

        supported: List[Dict[str, Any]] = []
        unsupported: List[str] = []
        eps = 2e-2

        by_key: Dict[str, Dict[str, Any]] = {}
        for c in constraints:
            key = c["key"]
            if key not in by_key:
                by_key[key] = {"numeric": None, "band": None}
            if c["type"] == "numeric":
                by_key[key]["numeric"] = c
            elif c["type"] == "band":
                by_key[key]["band"] = c

        for key, entry in by_key.items():
            numeric_c = entry.get("numeric")
            band_c = entry.get("band")
            minmax = self._field_minmax.get(key)

            if minmax is None:
                unsupported.append(f"{key}")
                continue

            if numeric_c is not None:
                lo, hi = minmax
                if numeric_c["value"] < (lo - eps) or numeric_c["value"] > (hi + eps):
                    unsupported.append(f"{key}={numeric_c['value']} outside[{lo:.4f},{hi:.4f}]")
                    projected = self._project_numeric_to_feasible_band(numeric_c)
                    if projected is not None:
                        supported.append(projected)
                        unsupported.append(
                            f"{key} projected_to_band={projected['band']}@{projected['projected_value']:.4f}"
                        )
                else:
                    # Numeric-primary support target for this key.
                    supported.append(numeric_c)
                continue

            if band_c is not None:
                if band_c["band"] in self._field_band_values.get(key, set()):
                    supported.append(band_c)
                else:
                    unsupported.append(f"{key}_band={band_c['band']} unsupported")

        if len(supported) == len(constraints):
            status = "fully_supported"
        elif len(supported) == 0:
            status = "unsupported"
        else:
            status = "partially_supported"

        return supported, unsupported, status

    def _apply_tolerance_filter(
        self,
        constraints: List[Dict[str, Any]],
        tolerance_map: Dict[str, float],
    ) -> List[str]:
        kept: List[str] = []
        for sid in self._all_sample_ids:
            row_map = self._row_numeric.get(sid, {})
            ok = True
            for c in constraints:
                key = c["key"]
                row_value = row_map.get(key)
                if row_value is None:
                    ok = False
                    break

                if c["type"] == "band":
                    row_band = self._band_for_field(c["suffix"], row_value)
                    if row_band != c["band"]:
                        ok = False
                        break
                    continue

                tol = tolerance_map.get(c["suffix"], 0.1)
                if abs(row_value - c["value"]) > tol:
                    ok = False
                    break
            if ok:
                kept.append(sid)
        return kept

    def _apply_numeric_tolerance_filter(
        self,
        numeric_constraints: List[Dict[str, Any]],
        tolerance_map: Dict[str, float],
    ) -> List[str]:
        if not numeric_constraints:
            return list(self._all_sample_ids)

        kept: List[str] = []
        for sid in self._all_sample_ids:
            row_map = self._row_numeric.get(sid, {})
            ok = True
            for c in numeric_constraints:
                key = c["key"]
                row_value = row_map.get(key)
                if row_value is None:
                    ok = False
                    break
                tol = tolerance_map.get(c["suffix"], 0.1)
                if abs(row_value - c["value"]) > tol:
                    ok = False
                    break
            if ok:
                kept.append(sid)
        return kept

    def _apply_band_match_filter(
        self,
        candidate_ids: List[str],
        band_constraints: List[Dict[str, Any]],
        min_match_count: int,
    ) -> List[str]:
        if not band_constraints:
            return list(candidate_ids)

        needed = max(1, min(min_match_count, len(band_constraints)))
        kept: List[str] = []
        for sid in candidate_ids:
            row_map = self._row_numeric.get(sid, {})
            match_count = 0
            for c in band_constraints:
                row_value = row_map.get(c["key"])
                if row_value is None:
                    continue
                row_band = self._band_for_field(c["suffix"], row_value)
                if row_band == c["band"]:
                    match_count += 1
            if match_count >= needed:
                kept.append(sid)
        return kept

    def _build_candidate_pool(
        self,
        constraints: List[Dict[str, Any]],
        top_k: int,
    ) -> tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        supported, unsupported, feasibility_status = self._assess_constraint_feasibility(constraints)

        info: Dict[str, Any] = {
            "filter_stage": "none",
            "feasibility_status": feasibility_status,
            "constraint_count": len(constraints),
            "supported_constraint_count": len(supported),
            "unsupported_constraint_count": len(unsupported),
            "unsupported_constraints": unsupported,
            "candidate_count": len(self._all_sample_ids),
        }

        if not constraints:
            return self._all_sample_ids, info, supported

        if not supported:
            info["filter_stage"] = "unsupported_only"
            return self._all_sample_ids, info, supported

        numeric_supported = [c for c in supported if c["type"] == "numeric"]
        numeric_keys = {c["key"] for c in numeric_supported}
        band_supported = [c for c in supported if c["type"] == "band" and c["key"] not in numeric_keys]

        hard_numeric_ids = self._apply_numeric_tolerance_filter(numeric_supported, self.HARD_TOLERANCE)
        if band_supported:
            hard_ids = self._apply_band_match_filter(
                hard_numeric_ids,
                band_supported,
                min_match_count=len(band_supported),
            )
        else:
            hard_ids = hard_numeric_ids
        if len(hard_ids) >= top_k:
            info["filter_stage"] = "hard"
            info["candidate_count"] = len(hard_ids)
            return hard_ids, info, supported

        soft_numeric_ids = self._apply_numeric_tolerance_filter(numeric_supported, self.SOFT_TOLERANCE)
        if band_supported:
            soft_band_needed = max(1, math.ceil(0.5 * len(band_supported)))
            soft_ids = self._apply_band_match_filter(
                soft_numeric_ids,
                band_supported,
                min_match_count=soft_band_needed,
            )
        else:
            soft_ids = soft_numeric_ids
        if soft_ids:
            info["filter_stage"] = "soft"
            info["candidate_count"] = len(soft_ids)
            return soft_ids, info, supported

        preferred_numeric_ids = self._apply_numeric_tolerance_filter(numeric_supported, self.PREFERRED_TOLERANCE)
        if band_supported:
            preferred_ids = self._apply_band_match_filter(
                preferred_numeric_ids,
                band_supported,
                min_match_count=1,
            )
        else:
            preferred_ids = preferred_numeric_ids
        if preferred_ids:
            info["filter_stage"] = "preferred"
            info["candidate_count"] = len(preferred_ids)
            return preferred_ids, info, supported

        info["filter_stage"] = "no_match_fallback_all"
        info["candidate_count"] = len(self._all_sample_ids)
        return self._all_sample_ids, info, supported


    def _constraint_priority_score(self, sid: str, constraints: List[Dict[str, Any]]) -> float:
        if not constraints:
            return 0.0

        row_map = self._row_numeric.get(sid, {})
        score = 0.0
        for c in constraints:
            row_value = row_map.get(c["key"])
            if row_value is None:
                continue

            if c["type"] == "band":
                row_band = self._band_for_field(c["suffix"], row_value)
                if row_band == c["band"]:
                    score += 1.0
                continue

            hard_tol = self.HARD_TOLERANCE.get(c["suffix"], 0.1)
            soft_tol = self.SOFT_TOLERANCE.get(c["suffix"], max(hard_tol, 0.1))
            diff = abs(row_value - c["value"])
            closeness = max(0.0, 1.0 - (diff / max(soft_tol * 1.5, 1e-9)))
            if diff <= hard_tol:
                closeness += 0.25
            score += min(closeness, 1.25)

        return score / float(len(constraints))

    def _prioritize_candidates(
        self,
        candidate_ids: List[str],
        supported_constraints: List[Dict[str, Any]],
        top_k: int,
    ) -> tuple[List[str], Dict[str, Any]]:
        info: Dict[str, Any] = {
            "priority_applied": False,
            "priority_pool_size": len(candidate_ids),
            "priority_best_score": 0.0,
            "priority_worst_score": 0.0,
        }

        if not supported_constraints or len(candidate_ids) <= top_k:
            return candidate_ids, info

        scored: List[tuple[float, str]] = []
        for sid in candidate_ids:
            s = self._constraint_priority_score(sid, supported_constraints)
            scored.append((s, sid))

        scored.sort(key=lambda x: (-x[0], x[1]))
        pool_size = min(len(scored), max(self.PRIORITY_POOL_MIN, top_k * 80))
        prioritized_ids = [sid for _, sid in scored[:pool_size]]

        info["priority_applied"] = True
        info["priority_pool_size"] = len(prioritized_ids)
        info["priority_best_score"] = scored[0][0] if scored else 0.0
        info["priority_worst_score"] = scored[pool_size - 1][0] if pool_size and scored else 0.0
        return prioritized_ids, info


    @staticmethod
    def _constraint_label(c: Dict[str, Any]) -> str:
        if c.get("type") == "numeric":
            return f"{c['key']}={c['value']:.4f}"
        return f"{c['key']}_band={c['band']}"

    def _evaluate_result_support(
        self,
        sid: str,
        supported_constraints: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not supported_constraints:
            return {
                "supported_satisfied": [],
                "supported_missed": [],
                "support_alignment": 0.0,
                "support_alignment_pct": 0.0,
            }

        # Metric cleanup: collapse constraints into one target per layer+field key.
        # Numeric + band for the same key count as one support target (numeric-primary),
        # preventing inflation from duplicate/redundant emissions.
        targets: Dict[str, Dict[str, Any]] = {}
        for c in supported_constraints:
            key = c["key"]
            if key not in targets:
                targets[key] = {"numeric": None, "band": None, "suffix": c["suffix"]}
            if c["type"] == "numeric":
                targets[key]["numeric"] = c
            elif c["type"] == "band":
                targets[key]["band"] = c

        row_map = self._row_numeric.get(str(sid), {})
        satisfied: List[str] = []
        missed: List[str] = []

        for key in sorted(targets.keys()):
            t = targets[key]
            row_value = row_map.get(key)
            if row_value is None:
                missed.append(f"{key}=missing")
                continue

            numeric_c = t.get("numeric")
            band_c = t.get("band")

            # Numeric-primary target evaluation for this field.
            if numeric_c is not None:
                hard_tol = self.HARD_TOLERANCE.get(numeric_c["suffix"], 0.1)
                is_ok = abs(row_value - numeric_c["value"]) <= hard_tol
                label = f"{key}={numeric_c['value']:.4f}"
                if band_c is not None:
                    label = f"{label} (band={band_c['band']})"
            elif band_c is not None:
                row_band = self._band_for_field(band_c["suffix"], row_value)
                is_ok = row_band == band_c["band"]
                label = f"{key}_band={band_c['band']}"
            else:
                # Defensive fallback: no evaluable constraint in target.
                is_ok = False
                label = f"{key}=unknown"

            if is_ok:
                satisfied.append(label)
            else:
                missed.append(label)

        total_targets = len(satisfied) + len(missed)
        alignment = (len(satisfied) / float(total_targets)) if total_targets else 0.0
        return {
            "supported_satisfied": satisfied,
            "supported_missed": missed,
            "support_alignment": alignment,
            "support_alignment_pct": alignment * 100.0,
        }

    def _safe_upsert(self, ids: List[str], docs: List[str], metas: List[Dict[str, Any]]) -> None:
        """Upsert with one refresh-and-retry if handle goes stale."""
        try:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        except Exception:
            self._refresh_collection()
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def build_collection(
        self,
        force_rebuild: bool = False,
        batch_size: int = 500,
        max_rows: int | None = None,
    ) -> int:
        """Build or rebuild the Chroma collection from manifest rows."""
        if force_rebuild:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

        self._refresh_collection()

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        total = 0
        seen = 0
        with self.manifest_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_rows is not None and seen >= max_rows:
                    break
                seen += 1

                sid = str(row["sample_index"])
                ids.append(sid)
                docs.append(self._row_to_document(row))
                metas.append(
                    {
                        "sample_index": sid,
                        "filename": row.get("filename", ""),
                        "filepath": row.get("filepath", ""),
                    }
                )

                if len(ids) >= batch_size:
                    self._safe_upsert(ids=ids, docs=docs, metas=metas)
                    total += len(ids)
                    ids, docs, metas = [], [], []

        if ids:
            self._safe_upsert(ids=ids, docs=docs, metas=metas)
            total += len(ids)

        return total

    def count(self) -> int:
        try:
            return self.collection.count()
        except Exception:
            self._refresh_collection()
            return self.collection.count()

    def retrieve(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Embed query text and return top-k nearest rows from Chroma with optional candidate filtering."""
        query_document = self._query_to_document(query_text)
        constraints = self._parse_constraints_from_document(query_document)
        candidate_ids, filter_info, supported_constraints = self._build_candidate_pool(constraints, top_k=top_k)

        retrieval_query_document = query_document
        if filter_info["feasibility_status"] in {"partially_supported", "unsupported"}:
            supported_query_document = self._constraints_to_document(supported_constraints)
            if supported_query_document:
                retrieval_query_document = supported_query_document
                filter_info["query_signal_mode"] = "supported_only"
            else:
                filter_info["query_signal_mode"] = "raw_no_supported"
        else:
            filter_info["query_signal_mode"] = "raw_all_supported"

        prioritized_ids, priority_info = self._prioritize_candidates(
            candidate_ids,
            supported_constraints,
            top_k=top_k,
        )
        filter_info.update(priority_info)

        n_results = max(1, min(top_k, len(prioritized_ids)))
        use_where = len(prioritized_ids) < len(self._all_sample_ids)

        query_kwargs: Dict[str, Any] = {
            "query_texts": [retrieval_query_document],
            "n_results": n_results,
            "include": ["metadatas", "distances", "documents"],
        }
        if use_where:
            query_kwargs["where"] = {"sample_index": {"$in": prioritized_ids}}

        try:
            result = self.collection.query(**query_kwargs)
        except Exception:
            self._refresh_collection()
            result = self.collection.query(
                query_texts=[retrieval_query_document],
                n_results=top_k,
                include=["metadatas", "distances", "documents"],
            )
            filter_info["filter_stage"] = f"{filter_info['filter_stage']}|where_fallback"

        ids = result.get("ids", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        out: List[Dict[str, Any]] = []
        for i, row_id in enumerate(ids):
            dist = float(dists[i]) if i < len(dists) else float("nan")
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            sid = str(meta.get("sample_index", row_id))
            support_diag = self._evaluate_result_support(sid, supported_constraints)
            out.append(
                {
                    "rank": i + 1,
                    "id": row_id,
                    "sample_index": meta.get("sample_index", row_id),
                    "filename": meta.get("filename", ""),
                    "filepath": meta.get("filepath", ""),
                    "distance": dist,
                    "similarity_score": 100.0 / (1.0 + max(0.0, dist)),
                    "filter_stage": filter_info["filter_stage"],
                    "feasibility_status": filter_info["feasibility_status"],
                    "query_signal_mode": filter_info.get("query_signal_mode", "raw_all_supported"),
                    "candidate_count": filter_info["candidate_count"],
                    "priority_applied": filter_info.get("priority_applied", False),
                    "priority_pool_size": filter_info.get("priority_pool_size", filter_info["candidate_count"]),
                    "priority_best_score": filter_info.get("priority_best_score", 0.0),
                    "priority_worst_score": filter_info.get("priority_worst_score", 0.0),
                    "constraint_count": filter_info["constraint_count"],
                    "supported_constraint_count": filter_info["supported_constraint_count"],
                    "unsupported_constraint_count": filter_info["unsupported_constraint_count"],
                    "unsupported_constraints": "; ".join(filter_info["unsupported_constraints"]),
                    "support_alignment": support_diag["support_alignment"],
                    "support_alignment_pct": support_diag["support_alignment_pct"],
                    "supported_constraints_satisfied": " | ".join(support_diag["supported_satisfied"]),
                    "supported_constraints_missed": " | ".join(support_diag["supported_missed"]),
                }
            )
        return out

    def decide(
        self,
        query_text: str,
        top_k: int = 3,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """Apply bounded decision-layer logic over retrieval outputs."""
        rows = self.retrieve(query_text, top_k=top_k)
        if not rows:
            return {
                "query": query_text,
                "decision": "fallback",
                "branch": "no_results",
                "reason": "No retrieval candidates returned.",
                "confidence": 0.0,
                "confidence_threshold": confidence_threshold,
                "top_results": [],
            }

        top = rows[0]
        feasibility = str(top.get("feasibility_status", ""))
        if feasibility == "unsupported":
            return {
                "query": query_text,
                "decision": "fallback",
                "branch": "unsupported",
                "reason": "Query constraints are unsupported by dataset ranges.",
                "confidence": 0.0,
                "confidence_threshold": confidence_threshold,
                "top_results": rows,
            }

        similarity_conf = float(top.get("similarity_score", 0.0)) / 100.0
        alignment_conf = float(top.get("support_alignment", 0.0))
        confidence = (
            self.DECISION_SIMILARITY_WEIGHT * similarity_conf
            + self.DECISION_ALIGNMENT_WEIGHT * alignment_conf
        )

        if confidence >= confidence_threshold:
            decision = "accept"
            branch = "threshold_pass"
            reason = "Top retrieval passed confidence threshold."
        else:
            decision = "fallback"
            branch = "threshold_fail"
            reason = "Top retrieval below confidence threshold."

        return {
            "query": query_text,
            "decision": decision,
            "branch": branch,
            "reason": reason,
            "confidence": confidence,
            "confidence_threshold": confidence_threshold,
            "top_results": rows,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build/query Chroma retrieval over manifest.csv")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build Chroma collection from manifest")
    p_build.add_argument("--force", action="store_true", help="Delete and rebuild collection")
    p_build.add_argument("--batch-size", type=int, default=500)
    p_build.add_argument("--max-rows", type=int, default=None, help="Optional cap for faster local demos")

    p_query = sub.add_parser("query", help="Run vector retrieval query")
    p_query.add_argument("--text", required=True, help="Natural-language query")
    p_query.add_argument("--top-k", type=int, default=3)

    p_decide = sub.add_parser("decide", help="Run retrieval with decision-layer threshold/fallback")
    p_decide.add_argument("--text", required=True, help="Natural-language query")
    p_decide.add_argument("--top-k", type=int, default=3)
    p_decide.add_argument("--threshold", type=float, default=ChromaManifestRetriever.DEFAULT_CONFIDENCE_THRESHOLD)

    args = parser.parse_args()

    retriever = ChromaManifestRetriever()

    if args.cmd == "build":
        n = retriever.build_collection(
            force_rebuild=args.force,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
        )
        print(f"Indexed rows: {n}")
        print(f"Collection count: {retriever.count()}")
        return

    if retriever.count() == 0:
        n = retriever.build_collection(force_rebuild=False)
        print(f"Collection was empty, indexed rows: {n}")

    if args.cmd == "decide":
        decision = retriever.decide(
            args.text,
            top_k=args.top_k,
            confidence_threshold=args.threshold,
        )
        print(f"Query: {decision['query']}")
        print(
            f"Decision: {decision['decision']} branch={decision['branch']} "
            f"confidence={decision['confidence']:.3f} threshold={decision['confidence_threshold']:.3f}"
        )
        print(f"Reason: {decision['reason']}")
        print("Top results:")
        for r in decision["top_results"]:
            print(
                f"rank={r['rank']} sample_index={r['sample_index']} "
                f"score={r['similarity_score']:.2f} align={r['support_alignment']:.2f} "
                f"feas={r['feasibility_status']} stage={r['filter_stage']}"
            )
        return

    rows = retriever.retrieve(args.text, top_k=args.top_k)
    print(f"Query: {args.text}")
    print("Top results:")
    for r in rows:
        print(
            f"rank={r['rank']} sample_index={r['sample_index']} "
            f"distance={r['distance']:.4f} score={r['similarity_score']:.2f} "
            f"file={r['filename']} stage={r['filter_stage']} candidates={r['candidate_count']}"
        )
        if r["unsupported_constraints"]:
            print(f"  unsupported: {r['unsupported_constraints']}")


if __name__ == "__main__":
    main()
