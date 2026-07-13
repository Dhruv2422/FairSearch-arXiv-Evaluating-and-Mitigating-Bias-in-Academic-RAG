from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

QDRANT_PATH = str(Path(__file__).resolve().parent.parent / "data" / "indices" / "qdrant")
COLLECTION_NAME = "fairsearch_arxiv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def connect_qdrant():
    return QdrantClient(path=QDRANT_PATH)


def embed_query(query, model):
    return model.encode(query, normalize_embeddings=True).tolist()


def search(query, model, client, k=10, with_vectors=False):
    query_vector = embed_query(query, model)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        with_payload=True,
        with_vectors=with_vectors,
    )

    return results.points


if __name__ == "__main__":
    query = "Is AI going to bring about the end of the world?"

    print(f"\nQuery: {query}\n")

    model = load_model()
    client = connect_qdrant()
    results = search(query, model, client, k=10)

    for rank, hit in enumerate(results, start=1):
        payload = hit.payload or {}
        print(f"Rank {rank}")
        print(f"Score: {hit.score:.4f}")
        print(f"Paper ID: {payload.get('paper_id')}")
        print(f"Title: {payload.get('title')}")
        print(f"Category: {payload.get('category')}")
        print(f"Year: {payload.get('year')}")
        print(f"Institution label: {payload.get('institution_label')}")
        print("-" * 60)