from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient

# Anchored to the file, not the working directory, so this runs from anywhere.
QDRANT_PATH = Path(__file__).resolve().parent.parent / "data" / "indices" / "qdrant"

client = QdrantClient(path=str(QDRANT_PATH))

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