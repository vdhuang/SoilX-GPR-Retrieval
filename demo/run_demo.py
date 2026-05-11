"""Minimal end-to-end demo: simulated chatbot output -> retrieval backend -> ranked results."""

from pathlib import Path
import sys


# Allow imports from the project src directory when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from VectorDB import GPR_Retriever  # noqa: E402


def main() -> None:
    retriever = GPR_Retriever()

    # Simulated chatbot output: structured numeric constraints only.
    chatbot_output = {
        "layer_1_thickness_m": 0.8246026101549886,
        "layer_1_theta_v": 0.1279936482574776,
        "layer_2_thickness_m": 1.4849785295158628,
    }

    matches = retriever.search(chatbot_output, top_k=3, min_score=0.0)

    print("SoilX End-to-End Demo")
    print("Chatbot structured output:")
    print(chatbot_output)
    print("\nTop retrieval results:\n")

    for row in matches:
        print(
            f"Rank {row['Rank']}: sample_index={row['sample_index']} | "
            f"score={row['Similarity_Score']:.4f} | file={row['filename']}"
        )
        print(f"  path: {row.get('Actual_File_Path', '')}")
        for key in chatbot_output:
            found_val = row.get(key)
            print(f"  {key}: requested={chatbot_output[key]} | found={found_val}")
        print("-")


if __name__ == "__main__":
    main()
