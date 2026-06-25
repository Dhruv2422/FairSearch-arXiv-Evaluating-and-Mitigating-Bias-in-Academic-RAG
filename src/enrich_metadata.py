"""
enrich_metadata.py

Fetches author institutional affiliations from OpenAlex for every paper in
the Qdrant collection, classifies each as 'privileged' or 'underrepresented'
based on QS World University Rankings 2024 Top 20, and upserts the label
back into the Qdrant payload.

No API key required — OpenAlex is fully open. Adding your email to requests
is courteous and gets you faster responses.

Run from inside src/:
    python enrich_metadata.py
"""

import os
import time
import json
import logging
from pathlib import Path

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BATCH_SIZE = 50           # OpenAlex pipe-filter limit per request
RETRY_LIMIT = 5
BACKOFF_BASE = 2.0
REQUEST_DELAY = 0.15      # OpenAlex allows up to 10 req/s — stay conservative

OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = os.environ.get("CONTACT_EMAIL", "ceal.j@northeastern.edu")

CACHE_PATH = Path("../data/processed/affiliation_cache.json")
COLLECTION = "fairsearch_arxiv"

# QS World University Rankings 2024 — Top 20
TOP_20_INSTITUTIONS = {
    "massachusetts institute of technology",
    "mit",
    "imperial college london",
    "university of oxford",
    "harvard university",
    "university of cambridge",
    "stanford university",
    "eth zurich",
    "national university of singapore",
    "ucl",
    "university college london",
    "university of california berkeley",
    "uc berkeley",
    "university of chicago",
    "university of pennsylvania",
    "cornell university",
    "california institute of technology",
    "caltech",
    "yale university",
    "princeton university",
    "columbia university",
    "university of michigan",
    "johns hopkins university",
    "university of edinburgh",
    "peking university",
    "tsinghua university",
}

# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------

def fetch_affiliations_batch(arxiv_ids: list[str]) -> dict[str, list[str]]:
    """
    Fetch institutional affiliations for a batch of arXiv IDs from OpenAlex.
    Returns {arxiv_id: [institution_name, ...]}
    """
    url_filter = "|".join(f"http://arxiv.org/abs/{aid}" for aid in arxiv_ids)

    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(
                OPENALEX_URL,
                params={
                    "filter": f"locations.landing_page_url:{url_filter}",
                    "select": "authorships,locations",
                    "per-page": BATCH_SIZE,
                    "mailto": MAILTO,
                },
                timeout=30,
            )
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                log.warning(f"Rate limited — waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** attempt
            log.warning(f"Request error ({e}) — retrying in {wait:.0f}s")
            time.sleep(wait)
    else:
        log.error(f"Failed batch after {RETRY_LIMIT} attempts, skipping.")
        return {}

    results: dict[str, list[str]] = {}
    for work in resp.json().get("results", []):
        # Extract the arXiv ID from locations
        arxiv_url = next(
            (loc["landing_page_url"] for loc in work.get("locations", [])
             if loc.get("landing_page_url") and loc["landing_page_url"].startswith("http://arxiv.org/abs/")),
            None,
        )
        if not arxiv_url:
            continue
        arxiv_id = arxiv_url.replace("http://arxiv.org/abs/", "")

        institutions = [
            inst["display_name"]
            for authorship in work.get("authorships", [])
            for inst in authorship.get("institutions", [])
            if inst.get("display_name")
        ]
        results[arxiv_id] = institutions

    return results


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(institutions: list[str]) -> str:
    """
    Return 'privileged' if any institution fuzzy-matches a Top-20 entry,
    'underrepresented' if institutions are found but none match,
    'unknown' if no institution data at all.
    """
    if not institutions:
        return "unknown"
    for name in institutions:
        name_lower = name.lower()
        if any(top in name_lower for top in TOP_20_INSTITUTIONS):
            return "privileged"
    return "underrepresented"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

def connect_qdrant():
    repo_root = Path(__file__).resolve().parent.parent
    return QdrantClient(path=str(repo_root / "data" / "indices" / "qdrant"))


def get_all_points(client: QdrantClient) -> list:
    points, offset = [], None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("Connecting to Qdrant...")
    client = connect_qdrant()

    log.info("Loading all points from collection...")
    points = get_all_points(client)
    log.info(f"Found {len(points)} points")

    cache = load_cache()
    log.info(f"Cache has {len(cache)} entries from previous runs")

    to_fetch = [
        p.payload["paper_id"]
        for p in points
        if p.payload.get("paper_id") and p.payload["paper_id"] not in cache
    ]
    log.info(f"{len(to_fetch)} papers need affiliation lookup")

    total_batches = -(-len(to_fetch) // BATCH_SIZE)
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch_ids = to_fetch[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log.info(f"Fetching batch {batch_num}/{total_batches} ({len(batch_ids)} papers)")

        result = fetch_affiliations_batch(batch_ids)
        cache.update(result)

        # Mark papers with no OpenAlex record so we don't retry them
        for pid in batch_ids:
            if pid not in cache:
                cache[pid] = []

        save_cache(cache)
        time.sleep(REQUEST_DELAY)

    log.info("Classifying and upserting labels into Qdrant...")
    counts = {"privileged": 0, "underrepresented": 0, "unknown": 0}

    for point in points:
        pid = point.payload.get("paper_id")
        if not pid:
            continue
        institutions = cache.get(pid, [])
        label = classify(institutions)
        counts[label] += 1

        client.set_payload(
            collection_name=COLLECTION,
            payload={
                "institution_label": label,
                "affiliations": institutions,
            },
            points=PointIdsList(points=[point.id]),
        )

    log.info(
        f"Done. privileged={counts['privileged']} | "
        f"underrepresented={counts['underrepresented']} | "
        f"unknown={counts['unknown']}"
    )
    log.info("Payload fields written: institution_label, affiliations")


if __name__ == "__main__":
    main()
