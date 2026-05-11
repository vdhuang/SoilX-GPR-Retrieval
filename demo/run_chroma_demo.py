"""Minimal demo: query -> embedding -> Chroma -> top-3."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from chroma_retriever import ChromaManifestRetriever  # noqa: E402


def main() -> None:
    retriever = ChromaManifestRetriever()

    if retriever.count() == 0:
        n = retriever.build_collection(force_rebuild=False, max_rows=500)
        print(f"Built Chroma collection with {n} rows.")

    query_text = "Top layer 0.8m with moderate moisture over denser middle soil"
    rows = retriever.retrieve(query_text, top_k=3)

    print("Chroma Top-3 Demo")
    print(f"Query: {query_text}\n")
    for r in rows:
        print(
            f"Rank {r['rank']}: sample_index={r['sample_index']} "
            f"score={r['similarity_score']:.2f} distance={r['distance']:.4f}"
        )
        print(f"  filename: {r['filename']}")
        print(f"  filepath: {r['filepath']}")
        print("-")


if __name__ == "__main__":
    main()

