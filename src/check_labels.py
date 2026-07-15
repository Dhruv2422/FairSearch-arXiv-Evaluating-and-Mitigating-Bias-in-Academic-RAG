from collections import Counter
from qdrant_client import QdrantClient

client = QdrantClient(path="../data/indices/qdrant")

counts, offset = Counter(), None
while True:
    batch, offset = client.scroll(
        collection_name="fairsearch_arxiv",
        limit=1000,
        offset=offset,
        with_payload=["institution_label"],
        with_vectors=False,
    )
    counts.update(p.payload.get("institution_label") for p in batch)
    if offset is None:
        break

print(counts)