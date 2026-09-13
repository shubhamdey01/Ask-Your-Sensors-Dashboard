from pathlib import Path
import sqlite3

import pandas as pd
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

EVENTS_PATH = DATA_DIR / "semantic_events_text.csv"

QDRANT_PATH = DATA_DIR / "qdrant"

DB_PATH = DATA_DIR / "sensor.db"


# ============================================================
# CONFIGURATION
# ============================================================

COLLECTION_NAME = "sensor_events"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

VECTOR_SIZE = 384

BATCH_SIZE = 64


# ============================================================
# LOAD EVENTS
# ============================================================

print("=" * 70)
print("SENSOR EVENT EMBEDDING PIPELINE")
print("=" * 70)

print("\nLoading semantic events...")

df = pd.read_csv(
    EVENTS_PATH,
    parse_dates=[
        "event_start",
        "event_end",
    ],
)

print(f"Events loaded: {len(df):,}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda",
)

print(f"Model: {MODEL_NAME}")
print(f"Device: {model.device}")


# ============================================================
# CREATE QDRANT CLIENT
# ============================================================

print("\nOpening Qdrant...")

client = QdrantClient(
    path=str(QDRANT_PATH)
)


# ============================================================
# RESET COLLECTION
# ============================================================

existing_collections = [
    collection.name
    for collection in client.get_collections().collections
]

if COLLECTION_NAME in existing_collections:

    print(
        f"Deleting existing collection: "
        f"{COLLECTION_NAME}"
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


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

texts = df["semantic_text"].tolist()

embeddings = model.encode(
    texts,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True,
)

print(
    f"Embedding shape: {embeddings.shape}"
)


# ============================================================
# PREPARE QDRANT POINTS
# ============================================================

print("\nPreparing Qdrant points...")

points = []

for i, row in df.iterrows():

    payload = {
        "event_id": int(row["event_id"]),
        
        "room_id": str(row["room_id"]),

        "event_start": str(row["event_start"]),

        "event_end": str(row["event_end"]),

        "duration_minutes": float(
            row["duration_minutes"]
        ),

        "window_count": int(
            row["window_count"]
        ),

        "temperature_mean": (
            None
            if pd.isna(row["temperature_mean"])
            else float(row["temperature_mean"])
        ),

        "temperature_min": (
            None
            if pd.isna(row["temperature_min"])
            else float(row["temperature_min"])
        ),

        "temperature_max": (
            None
            if pd.isna(row["temperature_max"])
            else float(row["temperature_max"])
        ),

        "humidity_mean": (
            None
            if pd.isna(row["humidity_mean"])
            else float(row["humidity_mean"])
        ),

        "co2_mean": (
            None
            if pd.isna(row["co2_mean"])
            else float(row["co2_mean"])
        ),

        "co2_min": (
            None
            if pd.isna(row["co2_min"])
            else float(row["co2_min"])
        ),

        "co2_max": (
            None
            if pd.isna(row["co2_max"])
            else float(row["co2_max"])
        ),

        "light_mean": (
            None
            if pd.isna(row["light_mean"])
            else float(row["light_mean"])
        ),

        "light_min": (
            None
            if pd.isna(row["light_min"])
            else float(row["light_min"])
        ),

        "light_max": (
            None
            if pd.isna(row["light_max"])
            else float(row["light_max"])
        ),

        "pir_active_ratio": (
            None
            if pd.isna(row["pir_active_ratio"])
            else float(row["pir_active_ratio"])
        ),

        "reasons": str(row["reasons"]),

        "semantic_text": str(
            row["semantic_text"]
        ),
    }

    points.append(
        PointStruct(
            id=int(i),
            vector=embeddings[i].tolist(),
            payload=payload,
        )
    )


# ============================================================
# INSERT INTO QDRANT
# ============================================================

print("\nInserting vectors into Qdrant...")

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points,
)


# ============================================================
# VERIFY
# ============================================================

info = client.get_collection(
    collection_name=COLLECTION_NAME
)

print("\nQdrant collection information:")

print(
    f"Points: "
    f"{info.points_count}"
)

print(
    f"Vectors: "
    f"{info.config.params.vectors.size}-dimensional"
)


# ============================================================
# CLOSE
# ============================================================

client.close()


print("\n" + "=" * 70)
print("EMBEDDING PIPELINE COMPLETE")
print("=" * 70)