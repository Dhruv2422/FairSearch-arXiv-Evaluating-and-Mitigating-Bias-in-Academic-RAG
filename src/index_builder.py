import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def load_data(path):
    return pd.read_parquet(path)

def embed_texts(texts):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    print("Using device:", device)
    vectors = model.encode(texts, normalize_embeddings=True)
    return model, vectors

def create_collection(client, name, vector_size):
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )

def build_points(df, vectors):
    points = []

    for i, row in df.iterrows():
        points.append(
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "paper_id": row["paper_id"],
                    "title": row["title"],
                    "abstract": row["abstract"],
                    "authors": row["authors"],
                    "category": row["category"],
                    "year": int(row["year"])
                }
            )
        )

    return points

def upload(client, collection_name, points):
    return client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

def build_index(
    parquet_path="../data/processed/papers.parquet",
    qdrant_path="../data/indices/qdrant",
    collection_name="fairsearch_arxiv"):

    # 1. Load dataset
    df = load_data(parquet_path)
    print(f"Loaded {len(df)} papers")

    # 2. Create embedding input
    texts = (df["title"].fillna("") + " " + df["abstract"].fillna("")).tolist()

    # 3. Embeddings
    model, vectors = embed_texts(texts)

    # 4. Qdrant setup
    client = QdrantClient(path=qdrant_path)

    create_collection(client, collection_name, len(vectors[0]))

    # 5. Build Qdrant points (like their quickstart style)
    points = build_points(df, vectors)

    # 6. Upload
    result = upload(client, collection_name, points)

    print(result)


if __name__ == "__main__":
    build_index()