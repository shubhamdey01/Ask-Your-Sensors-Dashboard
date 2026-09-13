import torch
from sentence_transformers import SentenceTransformer

print("PyTorch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"
)

print("Embedding model loaded successfully")

text = """
Temperature increased gradually from 27.1°C to 28.4°C
between 10:00 and 10:01. The temperature variation
was moderate and the sensor behaved normally.
"""

embedding = model.encode(text)

print("Embedding type:", type(embedding))
print("Embedding dimensions:", embedding.shape)

print(model.device)

from qdrant_client import QdrantClient

client = QdrantClient(path="./data/qdrant_test")

print("Qdrant initialized successfully")