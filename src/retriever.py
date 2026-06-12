from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

def load_model():
    """
    Load the same embedding model used during indexing.
    """
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def connect_qdrant():
    """
    Connect to the local Qdrant database.
    """
    return QdrantClient(path="../data/indices/qdrant")


def search(query, model, client, k=10):
    """
    Embed the query and retrieve the top-k most similar papers.
    """
    query_vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name="fairsearch_arxiv",
        query=query_vector,
        limit=k
    )

    return results.points


if __name__ == "__main__":

    query = "Is AI going to bring about the end of the world?"
    # query = "Recent advances in graph neural networks"

    print(f"\nQuery: {query}\n")

    model = load_model()
    client = connect_qdrant()

    results = search(query, model, client)

    for rank, hit in enumerate(results, start=1):
        print(f"Rank {rank}")
        print(f"Score: {hit.score:.4f}")
        print(f"Title: {hit.payload['title']}")
        print(f"Category: {hit.payload['category']}")
        print(f"Year: {hit.payload['year']}")
        print("-" * 60)