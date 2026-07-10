"""
enrich_metadata.py

Fetches author institutional affiliations from OpenAlex for every paper in
the Qdrant collection, classifies each as 'privileged' or 'underrepresented'
based on QS World University Rankings 2024 Top 20, and upserts the label
back into the Qdrant payload.

Lookup strategy (in order):
  1. Batch URL lookup — http:// and https:// arXiv URL variants (50 per request)
  2. Title-based fallback — for papers still unmatched after URL lookup

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

BATCH_SIZE = 50
RETRY_LIMIT = 5
BACKOFF_BASE = 2.0
REQUEST_DELAY = 0.5  # delay between batch requests — OpenAlex 429s at faster rates in practice

OPENALEX_URL = "https://api.openalex.org/works"
MAILTO = os.environ.get("CONTACT_EMAIL", "ceal.j@northeastern.edu")

CACHE_PATH = Path("../data/processed/affiliation_cache.json")
COLLECTION = "fairsearch_arxiv"

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
# Shared request helper
# ---------------------------------------------------------------------------

def _get(params: dict) -> list:
    """GET OPENALEX_URL with retry/backoff. Returns results list or []."""
    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(OPENALEX_URL, params={**params, "mailto": MAILTO}, timeout=30)
            if resp.status_code == 429:
                wait = BACKOFF_BASE ** attempt
                log.warning(f"Rate limited — waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                return []
            resp.raise_for_status()
            return resp.json().get("results", [])
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** attempt
            log.warning(f"Request error ({e}) — retrying in {wait:.0f}s")
            time.sleep(wait)
    log.error("Failed after max retries, skipping.")
    return []


def _extract_institutions(work: dict) -> list[str]:
    """
    Pull institution names from every place OpenAlex records them:
      1. authorships[].institutions — structured affiliation data
      2. authorships[].raw_affiliation_strings — unstructured affiliation text
      3. locations[].source.host_organization_name — institutional repositories
         (a paper deposited in e.g. Apollo implies a Cambridge author)
    """
    names = []
    for authorship in work.get("authorships", []):
        for inst in authorship.get("institutions", []):
            if inst.get("display_name"):
                names.append(inst["display_name"])
        names.extend(authorship.get("raw_affiliation_strings", []))
    for loc in work.get("locations", []):
        source = loc.get("source") or {}
        # arXiv itself is hosted by Cornell — counting it would label every
        # paper as Cornell. Skip preprint servers, keep true institutional repos.
        display = (source.get("display_name") or "").lower()
        if any(preprint in display for preprint in ("arxiv", "biorxiv", "medrxiv", "ssrn")):
            continue
        if source.get("type") == "repository" and source.get("host_organization_name"):
            names.append(source["host_organization_name"])
    # dedupe, preserve order
    return list(dict.fromkeys(names))


# ---------------------------------------------------------------------------
# Strategy 1: batch URL lookup (http + https variants)
# ---------------------------------------------------------------------------

def fetch_affiliations_batch(arxiv_ids: list[str]) -> dict[str, list[str]]:
    """
    Try both http:// and https:// arXiv URL variants in one filter.
    Returns {arxiv_id: [institution_name, ...]}
    """
    urls = []
    for aid in arxiv_ids:
        urls.append(f"http://arxiv.org/abs/{aid}")
        urls.append(f"https://arxiv.org/abs/{aid}")
    url_filter = "|".join(urls)

    works = _get({
        "filter": f"locations.landing_page_url:{url_filter}",
        "select": "authorships,locations",
        "per-page": BATCH_SIZE * 2,
    })

    results: dict[str, list[str]] = {}
    for work in works:
        arxiv_id = None
        for loc in work.get("locations", []):
            url = loc.get("landing_page_url") or ""
            for prefix in ("http://arxiv.org/abs/", "https://arxiv.org/abs/"):
                if url.startswith(prefix):
                    arxiv_id = url.replace(prefix, "").split("v")[0]
                    break
            if arxiv_id:
                break
        if arxiv_id and arxiv_id in arxiv_ids:
            results[arxiv_id] = _extract_institutions(work)

    return results


# ---------------------------------------------------------------------------
# Strategy 2: title-based fallback (one request per paper)
# ---------------------------------------------------------------------------

def _title_similarity(a: str, b: str) -> float:
    """Jaccard similarity on word sets — good enough to catch wrong-paper matches."""
    wa = set(a.lower().split())
    wb = set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _sanitize_title(title: str) -> str:
    """Strip characters that break OpenAlex filter queries."""
    import re
    title = title.replace("\n", " ").replace("\r", " ")
    title = re.sub(r"[\\$%&|<>{}()\[\]]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title[:200]  # OpenAlex search degrades on very long titles


def fetch_by_title(title: str) -> list[str] | None:
    """
    Search OpenAlex by title and take the best-affiliated close match.
    OpenAlex often splits a paper into multiple records (arXiv preprint vs.
    journal/repository version) where only one carries institution data, so
    check the top few results, not just the first. Returns None if no result
    matches the title closely (Jaccard >= 0.5).
    """
    clean = _sanitize_title(title)
    if not clean:
        return None
    works = _get({
        "filter": f"title.search:{clean}",
        "select": "authorships,title,locations",
        "per-page": 5,
    })
    matched = False
    for work in works:
        if _title_similarity(title, work.get("title") or "") < 0.5:
            continue
        matched = True
        institutions = _extract_institutions(work)
        if institutions:
            return institutions
    return [] if matched else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(institutions: list[str]) -> str:
    import re
    if not institutions:
        return "unknown"
    for name in institutions:
        lower = name.lower()
        for top in TOP_20_INSTITUTIONS:
            # short acronyms need word boundaries: "mit" is inside "smith",
            # "ucl" is inside "ucla"
            if len(top) <= 4:
                if re.search(rf"\b{top}\b", lower):
                    return "privileged"
            elif top in lower:
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

    # Build a lookup from paper_id → point for title fallback
    id_to_point = {p.payload["paper_id"]: p for p in points if p.payload.get("paper_id")}

    # --- Stage 1: batch URL lookup for anything not already in cache ---
    to_fetch = [pid for pid in id_to_point if pid not in cache]
    log.info(f"Stage 1 — URL batch lookup: {len(to_fetch)} papers")

    total_batches = -(-len(to_fetch) // BATCH_SIZE)
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch_ids = to_fetch[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        log.info(f"  Batch {batch_num}/{total_batches} ({len(batch_ids)} papers)")

        result = fetch_affiliations_batch(batch_ids)
        cache.update(result)
        for pid in batch_ids:
            if pid not in cache:
                cache[pid] = []
        save_cache(cache)
        time.sleep(REQUEST_DELAY)

    # --- Stage 2: batched URL re-lookup with the wider institution extraction ---
    # Stage 1's cached extractions only read authorships; re-fetching the same
    # batches now also captures raw_affiliation_strings and institutional
    # repository hosts. ~660 requests total, no per-paper title searches.
    still_empty = [pid for pid, affs in cache.items() if not affs and pid in id_to_point]
    log.info(f"Stage 2 — batch URL re-lookup: {len(still_empty)} papers with no affiliation data")

    total_batches = -(-len(still_empty) // BATCH_SIZE)
    recovered = 0
    for i in range(0, len(still_empty), BATCH_SIZE):
        batch_ids = still_empty[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1

        result = fetch_affiliations_batch(batch_ids)
        for pid, insts in result.items():
            if insts:
                cache[pid] = insts
                recovered += 1

        if batch_num % 10 == 0:
            save_cache(cache)
            log.info(f"  Batch {batch_num}/{total_batches} (recovered {recovered} so far)")
        time.sleep(REQUEST_DELAY)

    save_cache(cache)
    log.info(f"Stage 2 complete — recovered affiliations for {recovered} additional papers")

    # --- Classify and upsert ---
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
            payload={"institution_label": label, "affiliations": institutions},
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
