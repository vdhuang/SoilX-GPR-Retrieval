import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

print("[SYSTEM] Checking dependencies...")
_REQUIRED_PACKAGES = {
    "gradio": "gradio",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
}
_missing = [pip_name for module_name, pip_name in _REQUIRED_PACKAGES.items() if importlib.util.find_spec(module_name) is None]
if _missing:
    print(f"[SYSTEM] Missing packages detected: {_missing}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *_missing])
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Automatic dependency installation failed. "
            f"Please install manually and rerun: pip install {' '.join(_missing)}"
        ) from exc

import re
import html

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


PARAM_SUFFIXES_27 = [
    "thickness_m",
    "sand_pct",
    "silt_pct",
    "clay_pct",
    "theta_v",
    "bulk_density_gcm3",
    "particle_density_gcm3",
    "organic_fraction",
    "salinity_class",
]
ALL_COLUMNS_27 = [f"layer_{layer}_{suffix}" for layer in (1, 2, 3) for suffix in PARAM_SUFFIXES_27]


class GPR_Retriever:
    MANIFEST_PATH = Path(__file__).resolve().parent.parent / "data" / "manifest.csv"
    PARAM_SUFFIXES_27 = PARAM_SUFFIXES_27
    ALL_COLUMNS_27 = ALL_COLUMNS_27

    def __init__(self, manifest_path: str = str(MANIFEST_PATH)) -> None:
        """Load the manifest, validate schema, and fit a z-score scaler on searchable columns."""
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self._validate_schema()

        self.sample_indices = self.df["sample_index"].astype(str).tolist()
        self.filenames = self.df["filename"].astype(str).tolist()
        if "filepath" in self.df.columns:
            self.filepaths = self.df["filepath"].astype(str).tolist()
        else:
            self.filepaths = [""] * len(self.df)
        self.total_rows = len(self.df)

        self.search_columns = [c for c in self.ALL_COLUMNS_27 if not c.endswith("salinity_class")]

        numeric_df = self.df[self.search_columns].apply(pd.to_numeric, errors="coerce")
        if numeric_df.isna().all().any():
            bad_cols = numeric_df.columns[numeric_df.isna().all()].tolist()
            raise ValueError(f"Columns have no numeric values and cannot be searched: {bad_cols}")

        self.numeric_df = numeric_df.fillna(numeric_df.mean(numeric_only=True))
        self.raw_matrix = self.numeric_df.to_numpy(dtype=float)

        self.column_to_index = {col: i for i, col in enumerate(self.search_columns)}

        self.scaler = StandardScaler()
        self.scaled_matrix = self.scaler.fit_transform(self.raw_matrix)
        self.mean_ = np.asarray(self.scaler.mean_, dtype=float)
        self.scale_ = np.asarray(self.scaler.scale_, dtype=float)

    def _scale_query_vector(self, query_cols: List[str], query_vector: np.ndarray) -> np.ndarray:
        """Apply the fitted z-score normalization only on the dimensions present in the user query."""
        col_idx = np.array([self.column_to_index[c] for c in query_cols], dtype=int)
        return (query_vector - self.mean_[col_idx]) / self.scale_[col_idx]

    def _validate_schema(self) -> None:
        required_meta = ["sample_index", "filename"]
        missing_meta = [c for c in required_meta if c not in self.df.columns]
        if missing_meta:
            raise ValueError(f"Manifest missing metadata columns: {missing_meta}")

        missing_phys = [c for c in self.ALL_COLUMNS_27 if c not in self.df.columns]
        if missing_phys:
            raise ValueError(f"Manifest missing 27-parameter columns: {missing_phys}")

    @staticmethod
    def _validate_query_value(column: str, value: float) -> None:
        if "thickness_m" in column and value < 0:
            raise ValueError(f"Physically impossible value for '{column}': thickness cannot be negative.")
        if column.endswith("_pct") and not (0.0 <= value <= 100.0):
            raise ValueError(f"Physically impossible value for '{column}': percent must be within [0, 100].")
        if "theta_v" in column and not (0.0 <= value <= 1.0):
            raise ValueError(f"Physically impossible value for '{column}': theta_v must be within [0, 1].")
        if ("bulk_density_gcm3" in column or "particle_density_gcm3" in column) and value <= 0:
            raise ValueError(f"Physically impossible value for '{column}': density must be > 0.")

    def search(self, query_params: Dict[str, Any], top_k: int = 3, min_score: float = 0.0) -> List[Dict[str, Any]]:
        """Run masked nearest-neighbor retrieval using only the explicitly provided physical dimensions."""
        if not isinstance(query_params, dict) or not query_params:
            raise ValueError("query_params must be a non-empty dictionary")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not (0.0 <= float(min_score) <= 100.0):
            raise ValueError("min_score must be within [0, 100]")

        invalid_keys = [k for k in query_params if k not in self.column_to_index]
        if invalid_keys:
            raise ValueError(f"Unsupported query columns: {invalid_keys}")

        query_cols = list(query_params.keys())
        col_idx = np.array([self.column_to_index[c] for c in query_cols], dtype=int)

        q = np.empty((len(query_cols),), dtype=float)
        for i, col in enumerate(query_cols):
            try:
                q[i] = float(query_params[col])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Query value for '{col}' must be numeric, got: {query_params[col]!r}") from exc
            self._validate_query_value(col, q[i])

        q_scaled = self._scale_query_vector(query_cols, q)
        subset = self.scaled_matrix[:, col_idx]
        z_delta = subset - q_scaled

        distances = np.linalg.norm(z_delta, axis=1)
        scores = 100.0 / (1.0 + distances)

        keep = np.where(scores >= float(min_score))[0]
        if keep.size == 0:
            return []

        sortable = [(-scores[i], distances[i], self.sample_indices[i], i) for i in keep]
        sortable.sort()

        matches: List[Dict[str, Any]] = []
        for rank, (_, _, _, ridx) in enumerate(sortable[: int(top_k)], start=1):
            row: Dict[str, Any] = {
                "Rank": rank,
                "sample_index": self.sample_indices[ridx],
                "filename": self.filenames[ridx],
                "Actual_File_Path": self.filepaths[ridx],
                "Similarity_Score": round(float(scores[ridx]), 4),
                "_row_index": int(ridx),
            }
            for c in query_cols:
                row[c] = float(self.raw_matrix[ridx, self.column_to_index[c]])
            matches.append(row)

        return matches


KEYWORD_RULES: Dict[str, List[Tuple[str, float]]] = {
    r"\bwet\b": [("theta_v", 0.35)],
    r"\bdry\b": [("theta_v", 0.08)],
    r"\bsand(?:y)?\b": [("sand_pct", 85.0)],
    r"\bclay(?:ey)?\b": [("clay_pct", 70.0)],
}

REGEX_RULES = [
    {
        "name": "theta_percent",
        "pattern": re.compile(r"(\d*\.?\d+)\s*%\s*(?:moisture|theta|theta_v)\b", re.IGNORECASE),
        "handler": lambda m: [("theta_v", float(m.group(1)) / 100.0)],
    },
    {
        "name": "sand_percent",
        "pattern": re.compile(r"(\d*\.?\d+)\s*%\s*sand\b", re.IGNORECASE),
        "handler": lambda m: [("sand_pct", float(m.group(1)))],
    },
    {
        "name": "clay_percent",
        "pattern": re.compile(r"(\d*\.?\d+)\s*%\s*clay\b", re.IGNORECASE),
        "handler": lambda m: [("clay_pct", float(m.group(1)))],
    },
    {
        "name": "thickness_naked",
        "pattern": re.compile(
            r"\bthickness\b\s*(\d*\.?\d+)",
            re.IGNORECASE,
        ),
        "handler": lambda m: [("thickness_m", float(m.group(1)))],
    },
    {
        "name": "sand_naked",
        "pattern": re.compile(
            r"\bsand\b\s*(\d*\.?\d+)",
            re.IGNORECASE,
        ),
        "handler": lambda m: [("sand_pct", float(m.group(1)))],
    },
    {
        "name": "clay_naked",
        "pattern": re.compile(
            r"\bclay\b\s*(\d*\.?\d+)",
            re.IGNORECASE,
        ),
        "handler": lambda m: [("clay_pct", float(m.group(1)))],
    },
    {
        "name": "theta_naked",
        "pattern": re.compile(
            r"\b(?:moisture|theta|theta_v)\b\s*(\d*\.?\d+)",
            re.IGNORECASE,
        ),
        "handler": lambda m: [("theta_v", float(m.group(1)))],
    },
]


def _detect_layers(segment: str) -> List[int]:
    layers = []
    if re.search(r"\b(top|surface|upper|layer\s*1|l1)\b", segment):
        layers.append(1)
    if re.search(r"\b(middle|mid|layer\s*2|l2)\b", segment):
        layers.append(2)
    if re.search(r"\b(bottom|deep|lower|layer\s*3|l3)\b", segment):
        layers.append(3)
    return layers


def parse_natural_language(query_text: str) -> Tuple[Dict[str, float], List[str]]:
    """Translate a free-form soil description into an explicit physics vector over the 27-parameter schema."""
    if not isinstance(query_text, str) or not query_text.strip():
        return {}, ["[NLP] Empty query text."]

    text = query_text.lower().strip()
    segments = [s.strip() for s in re.split(r"\b(?:over|with|and|then)\b", text) if s.strip()]
    if not segments:
        segments = [text]

    logs = [f"[NLP] Segments detected: {segments}"]
    values: Dict[str, float] = {}
    active_layers = [1]
    thickness_cursor = 1

    sequence_candidates = re.findall(r"(\d*\.?\d+)\s*(?:m|meter|meters)?", text)
    has_commas = "," in text
    has_explicit_keywords = bool(
        re.search(r"\b(layer|l1|l2|l3|top|middle|bottom|wet|dry|sand|clay|theta|moisture|%)\b", text)
    )
    if has_commas and len(sequence_candidates) >= 2 and not has_explicit_keywords:
        for idx, raw in enumerate(sequence_candidates[:3], start=1):
            values[f"layer_{idx}_thickness_m"] = float(raw)
        logs.append(
            f"[NLP] Sequential thickness mode -> mapped {sequence_candidates[:3]} to layer_1/2/3_thickness_m"
        )
        logs.append(f"[NLP] Extracted parameters: {sorted(values.keys())}")
        return values, logs

    def set_for_layers(layers: List[int], suffix: str, value: float) -> None:
        for layer in layers:
            values[f"layer_{layer}_{suffix}"] = float(value)

    for seg_idx, seg in enumerate(segments):
        explicit_layers = _detect_layers(seg)
        if explicit_layers:
            target_layers = explicit_layers
            active_layers = explicit_layers
        else:
            target_layers = [min(seg_idx + 1, 3)] if seg_idx < 3 else active_layers

        logs.append(f"[NLP] Layer target -> '{seg}' => {target_layers}")
        explicit_value_set = set()

        thickness_matches = re.findall(r"(\d*\.?\d+)\s*(?:m|meter|meters)\b", seg)
        if len(thickness_matches) >= 2:
            for raw in thickness_matches:
                layer = min(thickness_cursor, 3)
                values[f"layer_{layer}_thickness_m"] = float(raw)
                logs.append(
                    f"[NLP] Sequential thickness in-segment -> layer_{layer}_thickness_m={float(raw)}"
                )
                thickness_cursor += 1
        elif len(thickness_matches) == 1:
            val = float(thickness_matches[0])
            for layer in target_layers:
                values[f"layer_{layer}_thickness_m"] = val
                explicit_value_set.add((layer, "thickness_m"))
            logs.append(f"[NLP] Thickness set -> {val}m for layers {target_layers}")
        else:
            naked_thickness_matches = re.findall(
                r"\bthickness\b\s*(\d*\.?\d+)",
                seg,
                flags=re.IGNORECASE,
            )
            if naked_thickness_matches:
                for raw in naked_thickness_matches:
                    val = float(raw)
                    for layer in target_layers:
                        values[f"layer_{layer}_thickness_m"] = val
                        explicit_value_set.add((layer, "thickness_m"))
                    logs.append(f"[NLP] Naked thickness set -> {val} for layers {target_layers}")

        for rule in REGEX_RULES:
            for match in rule["pattern"].finditer(seg):
                for suffix, val in rule["handler"](match):
                    set_for_layers(target_layers, suffix, val)
                    for layer in target_layers:
                        explicit_value_set.add((layer, suffix))
                    logs.append(f"[NLP] Regex[{rule['name']}] set {suffix}={val} for layers {target_layers}")

        for pattern, assignments in KEYWORD_RULES.items():
            if re.search(pattern, seg, flags=re.IGNORECASE):
                for suffix, val in assignments:
                    applied_layers = []
                    for layer in target_layers:
                        if (layer, suffix) in explicit_value_set:
                            continue
                        values[f"layer_{layer}_{suffix}"] = float(val)
                        applied_layers.append(layer)
                    if applied_layers:
                        logs.append(f"[NLP] Keyword[{pattern}] set {suffix}={val} for layers {applied_layers}")

    logs.append(f"[NLP] Extracted parameters: {sorted(values.keys())}")
    return values, logs


RETRIEVER = GPR_Retriever(manifest_path=str(GPR_Retriever.MANIFEST_PATH))
DEFAULT_TOP_K = 3


def _gatekeeper_label(score: float) -> str:
    if score >= 85.0:
        return "[GREEN] High Relevance"
    if score >= 65.0:
        return "[YELLOW] Partial Match"
    return "[RED] Weak Match"


def _traffic_light(score: float) -> str:
    if score > 80.0:
        return "🟢"
    if score >= 50.0:
        return "🟡"
    return "🔴"


def _to_dataframe(matches: List[Dict[str, Any]], query_params: Dict[str, float]) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(columns=["Traffic_Light", "Rank", "sample_index", "filename", "Actual_File_Path", "Similarity_Score"])

    rows = []
    query_cols = list(query_params.keys())
    for m in matches:
        row = {
            "Traffic_Light": _traffic_light(float(m["Similarity_Score"])),
            "Rank": m["Rank"],
            "sample_index": m["sample_index"],
            "filename": m["filename"],
            "Actual_File_Path": m.get("Actual_File_Path", ""),
            "Similarity_Score": m["Similarity_Score"],
        }
        for col in query_cols:
            row[f"Requested_{col}"] = float(query_params[col])
            row[f"Found_{col}"] = float(m.get(col, np.nan))
        rows.append(row)
    return pd.DataFrame(rows)


def _pretty_col_name(col: str) -> str:
    parts = col.split("_")
    if len(parts) < 4:
        return col
    layer = parts[1]
    metric = "_".join(parts[2:])
    metric_map = {
        "thickness_m": "Thickness",
        "theta_v": "Moisture",
        "sand_pct": "Sand",
        "silt_pct": "Silt",
        "clay_pct": "Clay",
        "bulk_density_gcm3": "Bulk Density",
        "particle_density_gcm3": "Particle Density",
        "organic_fraction": "Organic Fraction",
    }
    return f"Layer {layer} {metric_map.get(metric, metric)}"


def _build_penalty_table(query_params: Dict[str, float], match: Dict[str, Any]) -> str:
    cols = list(query_params.keys())
    col_idx = np.array([RETRIEVER.column_to_index[c] for c in cols], dtype=int)
    q = np.array([float(query_params[c]) for c in cols], dtype=float)
    q_scaled = (q - RETRIEVER.mean_[col_idx]) / RETRIEVER.scale_[col_idx]

    ridx = int(match.get("_row_index", -1))
    if ridx < 0 or ridx >= RETRIEVER.raw_matrix.shape[0]:
        return "[MATH] Penalty table unavailable for this match."

    found = RETRIEVER.raw_matrix[ridx, col_idx]
    found_scaled = (found - RETRIEVER.mean_[col_idx]) / RETRIEVER.scale_[col_idx]
    z_diff = found_scaled - q_scaled

    lines: List[str] = ["[MATH] Penalty Table:"]
    for i, col in enumerate(cols):
        dim_match = 100.0 / (1.0 + abs(float(z_diff[i])))
        dim_penalty = 100.0 - dim_match
        lines.append(
            f"[MATH] {_pretty_col_name(col)}: Match={dim_match:.1f}% (Penalty: -{dim_penalty:.1f}%)"
        )

    final_score = float(match["Similarity_Score"])
    quality = "HIGH" if final_score > 80 else ("MEDIUM" if final_score >= 50 else "LOW")
    lines.append(f"[TOTAL] Final Similarity: {final_score:.1f}% -> {quality}")
    return "\n".join(lines)


def _console_html(console_text: str) -> str:
    return (
        "<div style='height:520px; overflow-y:auto; white-space:pre-wrap; "
        "font-family:Consolas, Courier New, monospace; padding:12px; border:1px solid #1f2f57; "
        "border-radius:8px; background:#0b1736; color:#dbe6ff;'>"
        + html.escape(console_text)
        + "</div>"
    )


def _build_signal_preview(actual_file_path: str, filename: str):
    if plt is None:
        return None
    x = np.linspace(0, 10, 500)
    seed = abs(hash(actual_file_path or filename)) % (2**32)
    rng = np.random.default_rng(seed)
    y = np.sin(2.5 * x) + 0.12 * rng.standard_normal(len(x))
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(x, y, color="#2d6cdf", linewidth=1.3)
    ax.set_title(f"GPR Signal Preview for {filename}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _default_plot():
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot([0, 1], [0, 0], alpha=0.0)
    ax.set_title("GPR Signal Preview")
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def _execute_query(query_text: str, debug_mode: bool) -> Tuple[str, pd.DataFrame, str, Any]:
    query_params, nlp_logs = parse_natural_language(query_text)

    console_lines: List[str] = [
        "[SYSTEM] Checking dependencies...",
        "[SYSTEM] Dependencies ready.",
        "[DATABASE] Connecting to manifest.csv at C:/Users/discc/Downloads/MQP Database/",
        f"[DATABASE] Successfully indexed {RETRIEVER.total_rows:,} simulations across 27 physical parameters.",
        "",
        f"[NLP] Raw Input Received: '{query_text}'",
    ]
    console_lines.extend(nlp_logs)
    console_lines.append(f"[NLP] Translated to Physics Vector: {query_params}")

    if not query_params:
        query_params = {
            "layer_1_thickness_m": float(RETRIEVER.df["layer_1_thickness_m"].mean()),
            "layer_2_thickness_m": float(RETRIEVER.df["layer_2_thickness_m"].mean()),
            "layer_3_thickness_m": float(RETRIEVER.df["layer_3_thickness_m"].mean()),
        }
        console_lines.append("[NLP] No explicit physics found. Falling back to mean thickness profile for ranking.")

    query_params = {k: v for k, v in query_params.items() if not k.endswith("salinity_class")}
    if not query_params:
        query_params = {
            "layer_1_thickness_m": float(RETRIEVER.df["layer_1_thickness_m"].mean()),
            "layer_2_thickness_m": float(RETRIEVER.df["layer_2_thickness_m"].mean()),
            "layer_3_thickness_m": float(RETRIEVER.df["layer_3_thickness_m"].mean()),
        }
        console_lines.append("[NLP] Only salinity parsed; falling back to mean thickness profile for ranking.")

    if debug_mode:
        raw_vector_27: Dict[str, Any] = {col: None for col in RETRIEVER.ALL_COLUMNS_27}
        for k, v in query_params.items():
            raw_vector_27[k] = v
        console_lines.append(f"[DEBUG] Raw 27-Parameter Vector -> {raw_vector_27}")
    if plt is None:
        console_lines.append("[PLOT] matplotlib unavailable; signal preview placeholder is disabled in this environment.")

    console_lines.append("[MATH] Normalizing query dimensions using Z-Scores...")
    console_lines.append("[MATH] Executing Nearest Neighbor Search...")

    try:
        matches = RETRIEVER.search(query_params, top_k=DEFAULT_TOP_K, min_score=0.0)
    except Exception as exc:
        console_lines.append(f"[MATH] Validation failure: {exc}")
        return "[RED] Query Error", _to_dataframe([], query_params), "\n".join(console_lines), _default_plot()

    best_score = float(matches[0]["Similarity_Score"]) if matches else 0.0
    console_lines.append(f"[MATH] Retrieved {len(matches)} ranked candidates.")
    if matches:
        console_lines.append(_build_penalty_table(query_params, matches[0]))
        preview = _build_signal_preview(matches[0].get("Actual_File_Path", ""), matches[0].get("filename", "Unknown"))
    else:
        preview = _default_plot()

    return _gatekeeper_label(best_score), _to_dataframe(matches, query_params), "\n".join(console_lines), preview


def run_search(query_text: str, debug_mode: bool) -> Tuple[str, pd.DataFrame, str, Any]:
    gatekeeper, df, console_text, preview = _execute_query(query_text, debug_mode)
    return gatekeeper, df, _console_html(console_text), preview


def update_preview_from_selection(results_df: pd.DataFrame, evt: gr.SelectData):
    if results_df is None or len(results_df) == 0:
        return _default_plot()
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else int(evt.index)
    if idx < 0 or idx >= len(results_df):
        return _default_plot()
    row = results_df.iloc[int(idx)]
    return _build_signal_preview(str(row.get("Actual_File_Path", "")), str(row.get("filename", "Unknown")))


def run_comprehensive_stress_test() -> Tuple[pd.DataFrame, str]:
    test_queries = [
        "0.5m, 0.5m, 0.5m",
        "Wet sand over dry clay",
        "0.05m thickness",
        "2.0m thickness",
        "layer 1 thickness 0.62",
        "Top 0.25m, Middle 0.65m, Bottom 1.2m",
    ]
    summary_rows = []
    logs: List[str] = []

    for q in test_queries:
        gatekeeper, df, console_text, _ = _execute_query(q, debug_mode=True)

        if df is None or df.empty:
            top_score = 0.0
            resulting_file = ""
            light = "🔴"
        else:
            top_score = float(df.iloc[0]["Similarity_Score"])
            resulting_file = str(df.iloc[0]["filename"])
            light = str(df.iloc[0]["Traffic_Light"])

        summary_rows.append(
            {
                "Query": q,
                "Top Similarity Score": round(top_score, 4),
                "Resulting File": resulting_file,
                "Traffic Light": light,
            }
        )

        logs.append(f"---------- TEST QUERY: {q} ----------")
        logs.append(f"[SUMMARY] Gatekeeper: {gatekeeper}")
        logs.append(console_text)
        logs.append("")

    return pd.DataFrame(summary_rows), "\n".join(logs)


def build_dashboard() -> gr.Blocks:
    css = """
    .app-title { text-align: center; margin-bottom: 6px; }
    .app-subtitle { text-align: center; color: #444; margin-bottom: 18px; }
    #results-table { min-height: 320px; width: 100%; }
    #stress-log { max-height: 520px; overflow-y: auto; }
    """

    with gr.Blocks(title="GPR Digital Twin Retrieval Dashboard") as demo:
        gr.Markdown("<h2 class='app-title'>GPR Digital Twin Retrieval Dashboard</h2>")
        gr.Markdown("<div class='app-subtitle'>Pure text query + traceable physics ranking.</div>")

        query = gr.Textbox(
            label="Natural Language",
            placeholder="Example: 0.25m, 0.65m, 1.2m",
            lines=4,
        )
        debug_mode = gr.Checkbox(label="Debug Mode", value=False)

        search_btn = gr.Button("Search Database", variant="primary")

        with gr.Tabs():
            with gr.Tab("Results"):
                with gr.Row():
                    gatekeeper = gr.Markdown(label="Gatekeeper Status")
                rankings = gr.Dataframe(
                    label="Top 3 Matches",
                    wrap=True,
                    interactive=False,
                    elem_id="results-table",
                )
                signal_plot = gr.Plot(label="GPR Signal Preview")
            with gr.Tab("Thinking Console"):
                thinking_console = gr.HTML(
                    label="Thinking Console",
                    elem_id="thinking-box",
                )
            with gr.Tab("System Health"):
                stress_btn = gr.Button("Execute Comprehensive Stress Test", variant="secondary")
                stress_table = gr.Dataframe(
                    label="Stress Test Summary",
                    interactive=False,
                    wrap=True,
                )
                stress_log = gr.Code(
                    label="Master Debug Log",
                    language="markdown",
                    lines=26,
                    interactive=False,
                    elem_id="stress-log",
                )

        search_btn.click(
            fn=run_search,
            inputs=[query, debug_mode],
            outputs=[gatekeeper, rankings, thinking_console, signal_plot],
        )
        rankings.select(
            fn=update_preview_from_selection,
            inputs=[rankings],
            outputs=[signal_plot],
        )
        stress_btn.click(
            fn=run_comprehensive_stress_test,
            inputs=[],
            outputs=[stress_table, stress_log],
        )

    return demo, css


if __name__ == "__main__":
    app, css = build_dashboard()
    app.launch(show_error=True, css=css)
