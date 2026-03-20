# SoilX: GPR Digital Twin Retrieval Engine

SoilX is a physics-aware search engine for Ground Penetrating Radar simulations. It combines natural-language parsing with vector similarity search so a user can describe a soil profile, retrieve the closest simulation twins, and inspect exactly how the retrieval math behaved.

## Key Features

- Natural language parsing of soil profiles
- Dimensional masking for partial-query accuracy
- Z-score normalization across 27 physical parameters
- Thinking Console for retrieval transparency
- System Health tab for batch stress testing and internal QA
- Signal preview placeholder linked to retrieved simulation files

## Repository Layout

- `src/VectorDB.py`: main Gradio dashboard and retrieval engine
- `data/manifest.csv`: simulation manifest used by the retriever
- `docs/`: presentation materials, diagrams, and project documentation

Note: `SoilX_MQP_Update.pptx` was not present in the source directory during cleanup, so `docs/` was created but the presentation file could not be moved automatically.

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the dashboard:

```bash
python src/VectorDB.py
```

## Dependencies

- Gradio
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Usage

Enter a free-form soil query such as:

- `0.25m, 0.65m, 1.2m`
- `Wet sand over dry clay`
- `layer 1 thickness 0.62`

The dashboard returns the top three simulation matches, shows requested versus found values directly in the results table, and exposes a Thinking Console with the retrieval audit.

## Development Notes

- The retriever only scores dimensions explicitly present in the parsed query.
- Similarity uses a forgiving normalized score: `100 / (1 + distance)`.
- The System Health tab runs a fixed stress suite and captures a full debug log for regression checking.
