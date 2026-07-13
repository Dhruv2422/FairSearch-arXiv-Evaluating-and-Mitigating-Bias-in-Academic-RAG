import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


COLLECTION_NAME = "fairsearch_arxiv"
QDRANT_PATH = "/Users/dhruvdolas/Information Retrival project/FairSearch-arXiv-Evaluating-and-Mitigating-Bias-in-Academic-RAG-master/data/indices/qdrant"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model():
    return SentenceTransformer(MODEL_NAME)


def connect_qdrant():
    return QdrantClient(path=QDRANT_PATH)


def embed_query(query, model):
    return model.encode(query, normalize_embeddings=True)


def search(query, model, client, k=10, with_vectors=False):
    query_vector = embed_query(query, model)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=k,
        with_payload=True,
        with_vectors=with_vectors,
    )

    return results.points


def search_mmr(query, model, client, k=10, fetch_k=50, lambda_mult=0.5):
    query_vector = embed_query(query, model)

    candidates = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=fetch_k,
        with_payload=True,
        with_vectors=True,
    ).points

    if not candidates:
        return []

    doc_vectors = np.array([hit.vector for hit in candidates], dtype=float)
    query_sims = doc_vectors @ query_vector
    doc_sims = doc_vectors @ doc_vectors.T

    selected = []
    remaining = list(range(len(candidates)))
    k = min(k, len(candidates))

    while len(selected) < k:
        if not selected:
            best_idx = max(remaining, key=lambda i: query_sims[i])
        else:
            best_idx = None
            best_score = None

            for idx in remaining:
                redundancy = max(doc_sims[idx][j] for j in selected)
                mmr_score = lambda_mult * query_sims[idx] - (1 - lambda_mult) * redundancy

                if best_score is None or mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    reranked = []
    for rank, idx in enumerate(selected, start=1):
        hit = candidates[idx]
        hit.payload = hit.payload or {}
        hit.payload["mmr_rank"] = rank
        hit.payload["mmr_lambda"] = lambda_mult
        hit.payload["query_similarity"] = round(float(query_sims[idx]), 6)
        reranked.append(hit)

    return reranked


if __name__ == "__main__":
    query = "Is AI going to bring about the end of the world?"

    print(f"\nQuery: {query}\n")

    model = load_model()
    client = connect_qdrant()

    print("Baseline results:\n")
    baseline = search(query, model, client, k=10)
    for rank, hit in enumerate(baseline, start=1):
        payload = hit.payload or {}
        print(f"Rank {rank}")
        print(f"Score: {hit.score:.4f}")
        print(f"Paper ID: {payload.get('paper_id')}")
        print(f"Title: {payload.get('title')}")
        print(f"Category: {payload.get('category')}")
        print(f"Year: {payload.get('year')}")
        print(f"Institution label: {payload.get('institution_label')}")
        print("-" * 60)

    print("\nMMR results:\n")
    mmr_results = search_mmr(query, model, client, k=10, fetch_k=50, lambda_mult=0.5)
    for rank, hit in enumerate(mmr_results, start=1):
        payload = hit.payload or {}
        print(f"Rank {rank}")
        print(f"Original score: {hit.score:.4f}")
        print(f"Paper ID: {payload.get('paper_id')}")
        print(f"Title: {payload.get('title')}")
        print(f"Category: {payload.get('category')}")
        print(f"Year: {payload.get('year')}")
        print(f"Institution label: {payload.get('institution_label')}")
        print(f"MMR rank: {payload.get('mmr_rank')}")
        print(f"MMR lambda: {payload.get('mmr_lambda')}")
        print(f"Query similarity: {payload.get('query_similarity')}")
        print("-" * 60)