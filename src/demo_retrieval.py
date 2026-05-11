from __future__ import annotations

from chroma_retriever import ChromaManifestRetriever


def main() -> None:
    retriever = ChromaManifestRetriever()
    if retriever.count() == 0:
        indexed = retriever.build_collection(force_rebuild=False)
        print(f"Built Chroma collection with {indexed} rows.")

    query = "layer 2 silt 30 percent"
    decision = retriever.decide(query, top_k=3)

    print(f"Query: {query}")
    print(
        f"Decision: {decision['decision']} "
        f"branch={decision['branch']} "
        f"confidence={decision['confidence']:.3f}"
    )
    print("Top results:")
    for row in decision["top_results"]:
        print(
            f"rank={row['rank']} sample_index={row['sample_index']} "
            f"score={row['similarity_score']:.2f} "
            f"support={row['support_alignment']:.2f}"
        )


if __name__ == "__main__":
    main()
