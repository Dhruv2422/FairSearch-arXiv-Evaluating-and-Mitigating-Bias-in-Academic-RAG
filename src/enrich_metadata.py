"""
enrich_metadata.py

Fetches author institutional affiliations from OpenAlex for every paper in
the Qdrant collection, classifies each as 'privileged' or 'underrepresented',
and upserts the label back into the Qdrant payload.

'Privileged' is the union of four tiers — QS World University Rankings 2027
Top 20, leading CS departments by CSRankings, the Ivy League, and major
industry research labs. See PRIVILEGED_INSTITUTIONS below for why a single
global university ranking is not a sound proxy for a cs.* corpus.

Lookup strategy (in order):
  1. Batch URL lookup — OpenAlex, http:// and https:// arXiv URL variants
     (50 per request)
  2. Batch URL re-lookup — same query, wider institution extraction
  3. Semantic Scholar batch — 500 arXiv IDs per request, free. ~16% recovery
     on papers OpenAlex missed.
  4. Crossref by DOI — free and unmetered, ~44% recovery, but only applies to
     the papers that carry a DOI (see arxiv_dois.json).
  5. OpenAlex title search — one request per paper, ~44% recovery but METERED
     BY SPEND (~$0.001 each). Off unless --title-search is passed.

Stages 3-5 record which papers they have already tried, so re-runs resume
rather than repeating a completed pass.

Run from inside src/:
    python enrich_metadata.py                 # stages 1-4, free
    python enrich_metadata.py --title-search  # adds stage 5, costs money
"""

import os
import time
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Stage 5 runs one request per paper, so it is the long pole. OpenAlex's
# polite pool allows ~10 req/s: 4 workers each pausing 0.2s around a ~0.2s
# request holds the aggregate at roughly that.
TITLE_WORKERS = 4
TITLE_DELAY = 0.2

CACHE_PATH = Path("../data/processed/affiliation_cache.json")
# Per-paper stages record what they have already tried here, so re-runs resume
# instead of repeating the pass. See load_attempts().
S2_ATTEMPTS_PATH = Path("../data/processed/s2_attempted.json")
CROSSREF_ATTEMPTS_PATH = Path("../data/processed/crossref_attempted.json")
TITLE_ATTEMPTS_PATH = Path("../data/processed/title_fallback_attempted.json")
COLLECTION = "fairsearch_arxiv"

# ---------------------------------------------------------------------------
# What counts as an "elite" (privileged) institution
# ---------------------------------------------------------------------------
# The privileged group is the union of four tiers. Names are matched after
# normalization (see _normalize) — no need to enumerate punctuation variants,
# just distinct name forms (full name + common acronym).
#
# Rationale for using four tiers rather than a single global university
# ranking: this corpus is entirely cs.*, and a general-purpose all-disciplines
# ranking is a poor instrument for prestige *within computer science*. Under
# QS Top 20 alone, Carnegie Mellon, Princeton, UIUC, Georgia Tech and Google
# Research all landed in the "underrepresented" group, which is not a
# defensible description of institutional advantage in CS publishing.

# Tier 1 — QS World University Rankings 2027, rank <= 20 (incl. ties at
# #2, #8, #16, #20). Source: https://www.topuniversities.com/world-university-rankings
QS_TOP_20 = {
    "massachusetts institute of technology",
    "mit",
    "imperial college london",
    "stanford university",
    "university of oxford",
    "harvard university",
    "university of cambridge",
    "california institute of technology",
    "caltech",
    "eth zurich",
    "ucl",
    "university college london",
    "national university of singapore",
    "nus",
    "university of hong kong",
    "hku",
    "nanyang technological university",
    "ntu",
    "peking university",
    "tsinghua university",
    "university of pennsylvania",
    "cornell university",
    "yale university",
    "chinese university of hong kong",
    "cuhk",
    "university of new south wales",
    "unsw sydney",
    "unsw",
    "johns hopkins university",
    "university of california berkeley",
    "uc berkeley",
}

# Tier 2 — CSRankings (all areas, ranked by publication count), the union of
# the global Top 20 and the US Top 20, excluding institutions already covered
# by QS_TOP_20 (Tsinghua, Peking, ETH Zurich, MIT, UC Berkeley, Cornell, NUS,
# Stanford, NTU, Penn) or IVY_LEAGUE (Princeton, Columbia).
# Source: https://csrankings.org, retrieved 2026-08-09.
#
# Taking the union of both lists rather than the global list alone keeps strong
# US departments that the global ranking pushes past rank 20 (UT Austin,
# Wisconsin, UCLA) while leaving the cutoff as CSRankings' own top-20 boundary
# rather than a depth we picked. Ranks in comments: W = world, US = US-only.
CS_ELITE = {
    "carnegie mellon university",                       # W1  US1
    "cmu",
    "university of illinois urbana-champaign",           # W3  US2
    "university of illinois at urbana-champaign",
    "uiuc",
    "shanghai jiao tong university",                     # W4
    "sjtu",
    "university of california san diego",                # W5  US3
    "uc san diego",
    "ucsd",
    "georgia institute of technology",                   # W6  US4
    "georgia tech",
    "university of washington",                          # W10 US6
    "zhejiang university",                               # W13
    "university of maryland college park",               # W14 US9
    "university of michigan",                            # W15 US10
    "hong kong university of science and technology",    # W16
    "hkust",
    # NOTE: CSRankings' "Northeastern University" is the Boston institution.
    # Northeastern University (Shenyang, China) is a distinct university whose
    # name normalizes identically and will also match here. Flagged as a
    # labeling limitation in the README rather than silently conflated.
    "northeastern university",                           # W20 US12
    "purdue university",                                 #     US13
    "new york university",                               #     US14
    "nyu",
    "university of texas at austin",                     #     US14
    "ut austin",
    "university of wisconsin-madison",                   #     US14
    "university of wisconsin madison",
    "university of california los angeles",              #     US20
    "ucla",
}

# Tier 3 — Ivy League members not already covered by Tier 1
# (Harvard, Yale, Penn and Cornell are already in QS_TOP_20).
IVY_LEAGUE = {
    "brown university",
    "dartmouth college",
    "columbia university",
    "princeton university",
}

# Tier 4 — major industry research labs. In CS these are a genuine locus of
# institutional advantage that university rankings are blind to.
# NOTE: "meta" is deliberately absent as a bare token — it matches
# "meta-analysis" in raw affiliation strings. Match the company by its
# qualified names instead.
INDUSTRY_LABS = {
    "google",
    "deepmind",
    "microsoft",
    "meta platforms",
    "meta ai",
    "facebook",
    "amazon",
    "apple",
    "openai",
    "anthropic",
    "nvidia",
    "ibm",
    "adobe",
    "salesforce",
    "baidu",
    "alibaba",
    "tencent",
    "huawei",
    "bell labs",
    "allen institute for ai",
    "allen institute for artificial intelligence",
}

PRIVILEGED_INSTITUTIONS = QS_TOP_20 | CS_ELITE | IVY_LEAGUE | INDUSTRY_LABS

# ---------------------------------------------------------------------------
# Shared request helper
# ---------------------------------------------------------------------------

# Latched once OpenAlex reports the daily spend budget is gone, so the rest of
# the run stops issuing OpenAlex requests that cannot possibly succeed.
_budget_exhausted = False

def _get(params: dict) -> list:
    """
    GET OPENALEX_URL with retry/backoff. Returns results list or [].

    OpenAlex returns 429 for two very different conditions, and they need
    opposite handling:
      * per-second rate limiting — transient, back off and retry.
      * daily spend budget exhausted — retryAfter is measured in *hours*, so
        retrying is pointless. Every subsequent call in this run will fail the
        same way, so latch a flag and fail fast instead of burning 31s of
        backoff per request across hundreds of batches.
    """
    global _budget_exhausted
    if _budget_exhausted:
        return []

    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(OPENALEX_URL, params={**params, "mailto": MAILTO}, timeout=30)
            if resp.status_code == 429:
                if "insufficient budget" in resp.text.lower():
                    if not _budget_exhausted:
                        _budget_exhausted = True
                        log.error(
                            "OpenAlex daily spend budget exhausted — skipping all "
                            "remaining OpenAlex lookups this run. It resets at "
                            "midnight UTC, or add funds at https://openalex.org/pricing"
                        )
                    return []
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


# Hosts that indicate where a paper was *deposited*, not who wrote it.
# Preprint servers, generic data repositories, aggregators and publishers.
# Deliberately conservative: every entry is a distinctive multi-character token
# that will not appear inside a legitimate institution name. Short or ambiguous
# publisher tokens ("acm", "ieee", "sage") are omitted — "Sage Bionetworks" is a
# real research institute, and the cost of a false strip is a lost affiliation.
NON_INSTITUTIONAL_HOSTS = (
    "arxiv", "biorxiv", "medrxiv", "ssrn", "figshare", "zenodo", "datacite",
    "dryad", "dataverse", "researchgate", "mendeley", "open science framework",
    "semantic scholar", "openaire", "federal reserve bank",
    "springer", "elsevier", "wiley", "taylor & francis", "mdpi",
    "hindawi", "preprints.org",
)


def _is_institutional(name: str) -> bool:
    """False for repository/publisher hosts that describe where a paper was
    deposited rather than who wrote it."""
    low = name.lower()
    return not any(bad in low for bad in NON_INSTITUTIONAL_HOSTS)


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
        display = source.get("display_name") or ""
        host = source.get("host_organization_name") or ""
        # Generic repositories and publishers are not affiliations. Figshare and
        # the Federal Reserve Bank of St. Louis (which hosts RePEc) between them
        # accounted for ~660 papers being scored as underrepresented on the
        # strength of where the file was deposited, not who wrote it.
        if not _is_institutional(display) or not _is_institutional(host):
            continue
        if source.get("type") == "repository" and host:
            names.append(host)
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
# Strategy 3: Semantic Scholar batch lookup
# ---------------------------------------------------------------------------
# Free, and takes 500 arXiv IDs per request rather than one — the whole
# remaining corpus is ~59 requests. Measured recovery on papers OpenAlex could
# not match: 16% (245 of 1,500 sampled). Lower yield than an OpenAlex title
# search, but it costs nothing, where title search is metered (see fetch_by_title).
# The trade is that S2 records affiliations as author-entered free text
# ("UMass Amherst", "TU Munich") rather than normalized institution names.

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_BATCH_SIZE = 500
S2_RETRY_LIMIT = 6
S2_BACKOFF_BASE = 3.0
S2_DELAY = 4.0  # unauthenticated S2 throttles hard; a free API key raises this ceiling


def fetch_affiliations_s2(arxiv_ids: list[str]) -> dict[str, list[str]] | None:
    """
    Look up up to S2_BATCH_SIZE papers in one request.
    Returns {arxiv_id: [affiliation, ...]} for papers that have any, or None if
    the chunk exhausted its retries — the caller must distinguish "S2 has no
    affiliations for these" from "we never got an answer".
    """
    for attempt in range(S2_RETRY_LIMIT):
        try:
            resp = requests.post(
                S2_BATCH_URL,
                params={"fields": "authors.affiliations"},
                json={"ids": [f"ARXIV:{aid}" for aid in arxiv_ids]},
                timeout=60,
            )
        except requests.RequestException as e:
            log.warning(f"S2 request error ({e})")
            time.sleep(S2_BACKOFF_BASE * (2 ** attempt))
            continue
        if resp.status_code == 200:
            results: dict[str, list[str]] = {}
            for aid, record in zip(arxiv_ids, resp.json()):
                if not record:
                    continue
                names = [
                    a for author in (record.get("authors") or [])
                    for a in (author.get("affiliations") or [])
                ]
                names = [n for n in dict.fromkeys(names) if _is_institutional(n)]
                if names:
                    results[aid] = names
            return results
        wait = S2_BACKOFF_BASE * (2 ** attempt)
        log.warning(f"S2 HTTP {resp.status_code} — waiting {wait:.0f}s")
        time.sleep(wait)
    log.error("S2 batch failed after max retries, skipping.")
    return None


# ---------------------------------------------------------------------------
# Strategy 4: Crossref lookup by DOI
# ---------------------------------------------------------------------------
# Free and unmetered (unlike OpenAlex), but only applicable to the ~25% of the
# corpus that carries a DOI in the arXiv metadata. Measured recovery on 100
# sampled DOI-bearing papers that OpenAlex and S2 both missed: 44%.
# DOIs are not in papers.parquet — they are lifted out of the raw arXiv
# snapshot into arxiv_dois.json, which is committed so this stage works
# without re-downloading the (gitignored) raw dataset.

CROSSREF_URL = "https://api.crossref.org/works/"
DOI_MAP_PATH = Path("../data/processed/arxiv_dois.json")
CROSSREF_DELAY = 0.1


def fetch_affiliations_crossref(doi: str) -> list[str]:
    """Return affiliation names recorded on a Crossref work, or []."""
    try:
        resp = requests.get(
            CROSSREF_URL + doi, params={"mailto": MAILTO}, timeout=20
        )
    except requests.RequestException:
        return []
    if resp.status_code != 200:
        return []
    try:
        message = resp.json()["message"]
    except (ValueError, KeyError):
        return []
    names = [
        aff["name"]
        for author in message.get("author", [])
        for aff in author.get("affiliation", [])
        if aff.get("name")
    ]
    return [n for n in dict.fromkeys(names) if _is_institutional(n)]


# ---------------------------------------------------------------------------
# Strategy 5: title-based fallback (one request per paper)
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

def _normalize(name: str) -> str:
    """
    Strip punctuation and diacritics OpenAlex names commonly include
    ("University of California, Berkeley", "ETH Zürich") so substring
    matching against TOP_20_INSTITUTIONS isn't broken by characters that
    aren't in our hand-typed name variants.
    """
    import re
    import unicodedata
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[,.]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


# Distinct institutions whose names CONTAIN a privileged name as a substring.
# These are blanked out of the string before matching so they can't
# trigger a false privileged label.
EXCLUDED_LOOKALIKES = [
    "city university of hong kong",       # not University of Hong Kong
    "open university of hong kong",
    "education university of hong kong",
    "university of michigan-dearborn",    # satellite campuses, not Ann Arbor
    "university of michigan-flint",
    "google scholar",                     # a search product, not an affiliation
    # Distinct Chinese universities whose names contain a CSRankings top-20
    # name as a prefix. Zhejiang fronts several unrelated institutions, none of
    # which are the ranked one. The Nanjing entries are kept because they guard
    # the same prefix pattern even though Nanjing University itself is no
    # longer in the privileged tier.
    "nanjing university of aeronautics and astronautics",
    "nanjing university of science and technology",
    "nanjing university of posts and telecommunications",
    "nanjing university of information science and technology",
    "nanjing university of chinese medicine",
    "nanjing university of finance and economics",
    "nanjing university of the arts",
    "zhejiang university of technology",
    "zhejiang university of science and technology",
    "zhejiang university of finance and economics",
    "zhejiang university of media and communications",
    # PASSHE state schools, not UPenn
    "california university of pennsylvania",
    "indiana university of pennsylvania",
    "bloomsburg university of pennsylvania",
    "clarion university of pennsylvania",
    "east stroudsburg university of pennsylvania",
    "edinboro university of pennsylvania",
    "kutztown university of pennsylvania",
    "lock haven university of pennsylvania",
    "mansfield university of pennsylvania",
    "millersville university of pennsylvania",
    "shippensburg university of pennsylvania",
    "slippery rock university of pennsylvania",
    "west chester university of pennsylvania",
]


def matched_trigger(name: str) -> str | None:
    """Return the PRIVILEGED_INSTITUTIONS entry this name matches, or None."""
    import re
    lower = _normalize(name)
    for exc in EXCLUDED_LOOKALIKES:
        lower = lower.replace(exc, " ")
    for top in PRIVILEGED_INSTITUTIONS:
        # Every entry is matched on word boundaries, not as a bare substring.
        # Acronyms need it ("mit" is inside "smith", "ucl" inside "ucla",
        # "nus" inside "campus") and so do short company names now that the
        # industry tier exists ("apple" inside "pineapple", "amazon" inside
        # "amazonas", which is a Brazilian university, not AWS).
        if re.search(rf"\b{re.escape(top)}\b", lower):
            return top
    return None


def classify(institutions: list[str]) -> str:
    if not institutions:
        return "unknown"
    for name in institutions:
        if matched_trigger(name):
            return "privileged"
    return "underrepresented"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    """
    Load the affiliation cache, dropping any non-institutional names an earlier
    run recorded. Applying the filter on read (rather than only at extraction)
    means an existing cache self-heals instead of needing to be rebuilt, and a
    paper left with no real affiliation falls back to 'unknown' — and becomes
    eligible for the Stage 3 title search.
    """
    if not CACHE_PATH.exists():
        return {}
    raw = json.loads(CACHE_PATH.read_text())
    return {pid: [n for n in affs if _is_institutional(n)] for pid, affs in raw.items()}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def load_attempts(path: Path) -> set[str]:
    """
    Papers a per-paper lookup stage has already tried. A stage that finds
    nothing leaves an empty cache entry, indistinguishable from "not yet
    tried" — without this marker every re-run would redo the whole pass.
    """
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_attempts(path: Path, attempted: set[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(attempted)))


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

def classify_and_upsert(client, points: list, cache: dict):
    """Label every point from its cached affiliations and write it back."""
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


def main(title_search_enabled: bool = False, reclassify_only: bool = False):
    log.info("Connecting to Qdrant...")
    client = connect_qdrant()

    log.info("Loading all points from collection...")
    points = get_all_points(client)
    log.info(f"Found {len(points)} points")

    cache = load_cache()
    log.info(f"Cache has {len(cache)} entries from previous runs")

    # Build a lookup from paper_id → point for title fallback
    id_to_point = {p.payload["paper_id"]: p for p in points if p.payload.get("paper_id")}

    if reclassify_only:
        # Changing PRIVILEGED_INSTITUTIONS does not require re-fetching
        # anything — the affiliations are already cached, only the labels
        # derived from them are stale.
        log.info("Reclassify-only: skipping all lookup stages.")
        classify_and_upsert(client, points, cache)
        return

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

    # --- Stage 3: Semantic Scholar batch lookup ---
    s2_attempted = load_attempts(S2_ATTEMPTS_PATH)
    still_empty = [
        pid for pid, affs in cache.items()
        if not affs and pid in id_to_point and pid not in s2_attempted
    ]
    log.info(
        f"Stage 3 — Semantic Scholar batch: {len(still_empty)} papers "
        f"({len(s2_attempted)} already attempted in a previous run)"
    )

    recovered = 0
    total_chunks = -(-len(still_empty) // S2_BATCH_SIZE)
    for i in range(0, len(still_empty), S2_BATCH_SIZE):
        chunk = still_empty[i: i + S2_BATCH_SIZE]
        chunk_num = i // S2_BATCH_SIZE + 1
        result = fetch_affiliations_s2(chunk)
        if result is None:
            # Chunk exhausted its retries — leave it unmarked so a later run
            # picks it up rather than writing it off as a genuine miss.
            log.warning(f"  Chunk {chunk_num}/{total_chunks} — failed, will retry next run")
            continue
        for pid, insts in result.items():
            cache[pid] = insts
            recovered += 1
        s2_attempted.update(chunk)
        save_cache(cache)
        save_attempts(S2_ATTEMPTS_PATH, s2_attempted)
        log.info(f"  Chunk {chunk_num}/{total_chunks} — recovered {recovered} so far")
        time.sleep(S2_DELAY)

    save_cache(cache)
    save_attempts(S2_ATTEMPTS_PATH, s2_attempted)
    log.info(f"Stage 3 complete — recovered affiliations for {recovered} additional papers")

    # --- Stage 4: Crossref lookup for papers that carry a DOI ---
    if not DOI_MAP_PATH.exists():
        log.info(f"Stage 4 — Crossref: SKIPPED ({DOI_MAP_PATH} not found)")
    else:
        doi_map = json.loads(DOI_MAP_PATH.read_text())
        cr_attempted = load_attempts(CROSSREF_ATTEMPTS_PATH)
        candidates = [
            pid for pid, affs in cache.items()
            if not affs and pid in id_to_point and pid in doi_map and pid not in cr_attempted
        ]
        log.info(
            f"Stage 4 — Crossref by DOI: {len(candidates)} papers "
            f"({len(cr_attempted)} already attempted in a previous run)"
        )

        recovered = 0
        for n, pid in enumerate(candidates, start=1):
            names = fetch_affiliations_crossref(doi_map[pid])
            cr_attempted.add(pid)
            if names:
                cache[pid] = names
                recovered += 1
            if n % 250 == 0:
                save_cache(cache)
                save_attempts(CROSSREF_ATTEMPTS_PATH, cr_attempted)
                log.info(f"  {n}/{len(candidates)} — recovered {recovered} ({recovered / n:.0%})")
            time.sleep(CROSSREF_DELAY)

        save_cache(cache)
        save_attempts(CROSSREF_ATTEMPTS_PATH, cr_attempted)
        log.info(f"Stage 4 complete — recovered affiliations for {recovered} additional papers")

    # --- Stage 5 (opt-in): per-paper OpenAlex title search ---
    # Stages 1 and 2 both ask OpenAlex the same question — "which work has this
    # arXiv landing page URL?" — so a paper OpenAlex indexes without that URL
    # stays invisible to both. Searching by title finds those records: measured
    # recovery on 50 previously-unmatched papers was 22/50 (95% CI ~30-58%).
    #
    # OFF BY DEFAULT because OpenAlex meters this by spend, not request rate.
    # A title search costs $0.001 against a daily budget that is ~$0.10 on the
    # free tier, so a full pass over this corpus runs to roughly $29 and will
    # otherwise die partway through with HTTP 429 "Insufficient budget".
    # Enable deliberately, with funded credit:  python enrich_metadata.py --title-search
    remaining = sum(1 for pid, affs in cache.items() if not affs and pid in id_to_point)
    if not title_search_enabled:
        log.info(
            f"Stage 5 — OpenAlex title search: SKIPPED ({remaining} papers still "
            f"unmatched). Re-run with --title-search to enable; note the cost "
            f"(~${remaining * 0.001:,.2f} at $0.001/search) before you do."
        )
    else:
        attempted = load_attempts(TITLE_ATTEMPTS_PATH)
        candidates = [
            pid for pid, affs in cache.items()
            if not affs and pid in id_to_point and pid not in attempted
        ]
        log.warning(
            f"Stage 5 — OpenAlex title search: {len(candidates)} papers "
            f"(~${len(candidates) * 0.001:,.2f}); "
            f"{len(attempted)} already attempted in a previous run"
        )

        recovered = 0
        if candidates:
            def lookup(pid: str):
                title = id_to_point[pid].payload.get("title") or ""
                if not title:
                    return pid, None
                result = fetch_by_title(title)
                # TITLE_WORKERS in flight, each pausing TITLE_DELAY, keeps the
                # aggregate near OpenAlex's ~10 req/s ceiling. Note that staying
                # under the rate limit does not help with the spend limit.
                time.sleep(TITLE_DELAY)
                return pid, result

            with ThreadPoolExecutor(max_workers=TITLE_WORKERS) as pool:
                futures = [pool.submit(lookup, pid) for pid in candidates]
                for n, future in enumerate(as_completed(futures), start=1):
                    pid, result = future.result()
                    attempted.add(pid)
                    if result:
                        cache[pid] = result
                        recovered += 1
                    if n % 500 == 0:
                        save_cache(cache)
                        save_attempts(TITLE_ATTEMPTS_PATH, attempted)
                        log.info(
                            f"  {n}/{len(candidates)} searched "
                            f"(recovered {recovered}, {recovered / n:.0%})"
                        )

        save_cache(cache)
        save_attempts(TITLE_ATTEMPTS_PATH, attempted)
        log.info(f"Stage 5 complete — recovered affiliations for {recovered} additional papers")

    classify_and_upsert(client, points, cache)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--title-search",
        action="store_true",
        help="Enable Stage 5, the per-paper OpenAlex title search. This is "
             "metered by spend (~$0.001/paper) and needs funded OpenAlex "
             "credit; without it the stage dies partway with HTTP 429.",
    )
    parser.add_argument(
        "--reclassify-only",
        action="store_true",
        help="Re-label from the existing affiliation cache without contacting "
             "any API. Use after editing the privileged institution tiers.",
    )
    args = parser.parse_args()
    main(
        title_search_enabled=args.title_search,
        reclassify_only=args.reclassify_only,
    )
