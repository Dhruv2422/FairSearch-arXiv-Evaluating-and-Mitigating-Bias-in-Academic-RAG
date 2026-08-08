import numpy as np
import numpy as np
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


def search_mmr(query, model, client, k=10, fetch_k=50, lambda_mult=0.5):
    """
    Diversity-aware retrieval via MMR.

    Over retrieves fetch_k candidates by cosine similarity, then uses greedy approach to 
    re rank them

    MMR = lambda_mult * sim(query, doc) - (1 - lambda_mult) * max_{s in selected} sim(doc, s)

    Returns the top-k points in MMR order with same shape as search()
    """
    query_vector = model.encode(query, normalize_embeddings=True)

    candidates = client.query_points(
        collection_name="fairsearch_arxiv",
        query=query_vector.tolist(),
        limit=fetch_k,
        with_vectors=True,
    ).points

    if not candidates:
        return []

    # embeddings are L2-normalized, so cosine similarity is just the dot product
    doc_vectors = np.array([hit.vector for hit in candidates])
    query_sims = doc_vectors @ query_vector
    doc_sims = doc_vectors @ doc_vectors.T

    selected: list[int] = []
    remaining = list(range(len(candidates)))
    k = min(k, len(candidates))

    while len(selected) < k:
        if not selected:
            # first pick is the most query-relevant candidate
            best_idx = int(remaining[int(np.argmax(query_sims[remaining]))])
        else:
            best_idx, best_score = None, None
            for idx in remaining:
                redundancy = max(doc_sims[idx][j] for j in selected)
                score = lambda_mult * query_sims[idx] - (1 - lambda_mult) * redundancy
                if best_score is None or score > best_score:
                    best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return [candidates[i] for i in selected]


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