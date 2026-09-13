from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

QDRANT_PATH = (
    PROJECT_ROOT
    / "data"
    / "qdrant"
)


COLLECTION_NAME = "sensor_events"


# ------------------------------------------------------------
# Create local Qdrant client
# ------------------------------------------------------------

client = QdrantClient(
    path=str(QDRANT_PATH)
)

print("Qdrant client created.")


# ------------------------------------------------------------
# Create collection
# ------------------------------------------------------------

existing = [
    collection.name
    for collection in client.get_collections().collections
]

if COLLECTION_NAME not in existing:

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE,
        ),
    )

    print(
        f"Created collection: {COLLECTION_NAME}"
    )

else:

    print(
        f"Collection already exists: {COLLECTION_NAME}"
    )


# ------------------------------------------------------------
# Insert one test vector
# ------------------------------------------------------------

test_vector = [0.0] * 384

test_vector[0] = 1.0


client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        PointStruct(
            id=1,
            vector=test_vector,
            payload={
                "room_id": "TEST",
                "description": "Qdrant test event",
            },
        )
    ],
)


print("Test vector inserted.")


# ------------------------------------------------------------
# Search
# ------------------------------------------------------------

result = client.query_points(
    collection_name=COLLECTION_NAME,
    query=test_vector,
    limit=1,
)


print("\nSearch result:")

for point in result.points:

    print(
        f"ID: {point.id}"
    )

    print(
        f"Score: {point.score}"
    )

    print(
        f"Payload: {point.payload}"
    )


# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------

client.close()

print("\nQdrant test completed successfully.")