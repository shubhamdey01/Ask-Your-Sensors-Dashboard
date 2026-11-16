from pathlib import Path

import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

EVENTS_PATH = DATA_DIR / "semantic_events_text.csv"
QDRANT_PATH = DATA_DIR / "qdrant"

COLLECTION_NAME = "sensor_events"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384
BATCH_SIZE = 64


PAYLOAD_COLUMNS = [
    "event_id",
    "room_id",
    "event_start",
    "event_end",
    "duration_minutes",
    "window_count",
    "temperature_mean",
    "temperature_min",
    "temperature_max",
    "humidity_mean",
    "co2_mean",
    "co2_min",
    "co2_max",
    "light_mean",
    "pir_active_ratio",
    "reasons",
    "semantic_text",
]


def load_events():
    """Load semantic event descriptions."""

    if not EVENTS_PATH.exists():
        raise FileNotFoundError(
            f"Semantic event text file not found: {EVENTS_PATH}"
        )

    return pd.read_csv(
        EVENTS_PATH,
        parse_dates=["event_start", "event_end"],
    )


def create_collection(client):
    """Recreate the Qdrant collection."""

    collections = {
        collection.name
        for collection in client.get_collections().collections
    }

    if COLLECTION_NAME in collections:
        print(
            f"Deleting existing collection: {COLLECTION_NAME}"
        )

        client.delete_collection(
            collection_name=COLLECTION_NAME
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: {COLLECTION_NAME}"
    )


def optional_float(value):
    """Convert a value to float while preserving missing values."""

    if pd.isna(value):
        return None

    return float(value)


def create_payload(row):
    """Create the Qdrant payload for one semantic event."""

    return {
        "event_id": int(row.event_id),
        "room_id": str(row.room_id),
        "event_start": str(row.event_start),
        "event_end": str(row.event_end),
        "duration_minutes": optional_float(
            row.duration_minutes
        ),
        "window_count": int(row.window_count),

        "temperature_mean": optional_float(
            row.temperature_mean
        ),
        "temperature_min": optional_float(
            row.temperature_min
        ),
        "temperature_max": optional_float(
            row.temperature_max
        ),

        "humidity_mean": optional_float(
            row.humidity_mean
        ),

        "co2_mean": optional_float(
            row.co2_mean
        ),
        "co2_min": optional_float(
            row.co2_min
        ),
        "co2_max": optional_float(
            row.co2_max
        ),

        "light_mean": optional_float(
            row.light_mean
        ),

        "pir_active_ratio": optional_float(
            row.pir_active_ratio
        ),

        "reasons": str(row.reasons),
        "semantic_text": str(row.semantic_text),
    }


def create_points(df, embeddings):
    """Create Qdrant points from events and embeddings."""

    points = []

    for row, embedding in zip(
        df.itertuples(index=False),
        embeddings,
    ):
        points.append(
            PointStruct(
                id=int(row.event_id),
                vector=embedding.tolist(),
                payload=create_payload(row),
            )
        )

    return points


def main():
    """Generate embeddings and store semantic events in Qdrant."""

    print("=" * 70)
    print("SENSOR EVENT EMBEDDING PIPELINE")
    print("=" * 70)

    df = load_events()

    print(
        f"\nEvents loaded: {len(df):,}"
    )

    required_columns = set(PAYLOAD_COLUMNS)

    missing_columns = sorted(
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "semantic_events_text.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cuda",
    )

    print(
        f"Model: {MODEL_NAME}"
    )
    print(
        f"Device: {model.device}"
    )

    QDRANT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nOpening Qdrant...")

    client = QdrantClient(
        path=str(QDRANT_PATH)
    )

    try:
        create_collection(client)

        print("\nGenerating embeddings...")

        embeddings = model.encode(
            df["semantic_text"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        print(
            f"Embedding shape: {embeddings.shape}"
        )

        print("\nPreparing Qdrant points...")

        points = create_points(
            df,
            embeddings,
        )

        print("\nInserting vectors into Qdrant...")

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

        info = client.get_collection(
            collection_name=COLLECTION_NAME
        )

        print("\nQdrant collection information:")
        print(
            f"Points: {info.points_count}"
        )
        print(
            "Vectors: "
            f"{info.config.params.vectors.size}-dimensional"
        )

    finally:
        client.close()

    print("\n" + "=" * 70)
    print("EMBEDDING PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()