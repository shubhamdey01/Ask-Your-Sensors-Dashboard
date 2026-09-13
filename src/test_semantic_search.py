from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_PATH = "./data/qdrant"
COLLECTION_NAME = "sensor_events"


def main():

    print("=" * 70)
    print("SEMANTIC SENSOR SEARCH TEST")
    print("=" * 70)

    # ---------------------------------------------------------
    # Load embedding model
    # ---------------------------------------------------------
    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda"
    )

    print(f"Model: {MODEL_NAME}")
    print(f"Device: {model.device}")

    # ---------------------------------------------------------
    # Open Qdrant
    # ---------------------------------------------------------
    print("\nOpening Qdrant...")

    client = QdrantClient(path=QDRANT_PATH)

    # ---------------------------------------------------------
    # Test queries
    # ---------------------------------------------------------
    queries = [
        "Which rooms had unusually high temperature?",
        "When was there significant human activity in a room?",
        "Which rooms experienced unusually high CO2?",
        "Which rooms had very bright lighting?",
        "Find periods where the room appeared occupied.",
    ]

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------
    for query in queries:

        print("\n" + "-" * 70)
        print(f"QUERY: {query}")
        print("-" * 70)

        query_vector = model.encode(
            query,
            normalize_embeddings=True
        )

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector.tolist(),
            limit=5,
        ).points

        for rank, result in enumerate(results, start=1):

            payload = result.payload

            print(
                f"\n{rank}. "
                f"Score: {result.score:.4f} | "
                f"Event ID: {payload.get('event_id')} | "
                f"Room: {payload.get('room_id')}"
            )

            print(
                f"   Time: "
                f"{payload.get('event_start')} → "
                f"{payload.get('event_end')}"
            )

            print(
                f"   Reasons: "
                f"{payload.get('reasons')}"
            )

            print(
                f"   Description: "
                f"{payload.get('semantic_text')}"
            )

    client.close()

    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()