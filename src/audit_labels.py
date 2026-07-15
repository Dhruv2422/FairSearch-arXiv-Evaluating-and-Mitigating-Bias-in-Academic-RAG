"""
audit_labels.py — offline audit of institution classification quality.

Reads affiliation_cache.json (no API calls, no Qdrant) and reports:
  1. Label distribution
  2. Which TOP_20 keyword triggered each privileged classification
     (catches false positives — a bad trigger shows up immediately)
  3. Near-misses: institution names classified underrepresented that are
     highly similar to a TOP_20 entry (catches Berkeley-comma-style bugs)
  4. Most common institutions among underrepresented papers
     (eyeball check: nothing top-20 should appear here)

Run from inside src/:
    python audit_labels.py
"""

import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from enrich_metadata import TOP_20_INSTITUTIONS, _normalize, classify, matched_trigger

CACHE_PATH = Path("../data/processed/affiliation_cache.json")

NEAR_MISS_THRESHOLD = 0.75


def main():
    cache = json.loads(CACHE_PATH.read_text())

    labels = Counter()
    trigger_hits = Counter()          # top-20 keyword -> count of papers it flagged
    trigger_examples = {}             # top-20 keyword -> set of matched name strings
    under_names = Counter()           # institution name -> count among underrepresented

    for pid, institutions in cache.items():
        label = classify(institutions)
        labels[label] += 1

        if label == "privileged":
            for name in institutions:
                trig = matched_trigger(name)
                if trig:
                    trigger_hits[trig] += 1
                    trigger_examples.setdefault(trig, set()).add(name)
                    break
        elif label == "underrepresented":
            for name in institutions:
                under_names[_normalize(name)] += 1

    # --- 1. Label distribution ---
    total = sum(labels.values())
    print("=" * 70)
    print("1. LABEL DISTRIBUTION")
    print("=" * 70)
    for label, n in labels.most_common():
        print(f"  {label:18} {n:>7,}  ({n / total:.1%})")

    # --- 2. Privileged triggers ---
    print()
    print("=" * 70)
    print("2. PRIVILEGED TRIGGERS (which keyword matched, how often)")
    print("=" * 70)
    for trig, n in trigger_hits.most_common():
        print(f"  {trig:45} {n:>6,} papers")
        for example in sorted(trigger_examples[trig])[:3]:
            print(f"      e.g. {example[:70]}")

    # --- 3. Near-misses among underrepresented ---
    print()
    print("=" * 70)
    print(f"3. NEAR-MISSES (underrepresented names >= {NEAR_MISS_THRESHOLD} similar to a TOP_20 entry)")
    print("=" * 70)
    long_tops = [t for t in TOP_20_INSTITUTIONS if len(t) > 4]
    near_misses = []
    for name, count in under_names.items():
        best_ratio, best_top = 0.0, None
        for top in long_tops:
            ratio = SequenceMatcher(None, name, top).ratio()
            if ratio > best_ratio:
                best_ratio, best_top = ratio, top
        if best_ratio >= NEAR_MISS_THRESHOLD:
            near_misses.append((best_ratio, name, best_top, count))

    if not near_misses:
        print("  (none found)")
    for ratio, name, top, count in sorted(near_misses, reverse=True):
        print(f"  {ratio:.2f}  {name[:50]:50} ~ {top}  ({count:,} papers)")

    # --- 4. Top underrepresented institutions ---
    print()
    print("=" * 70)
    print("4. MOST COMMON UNDERREPRESENTED INSTITUTIONS (top 40 — sanity check)")
    print("=" * 70)
    for name, n in under_names.most_common(40):
        print(f"  {n:>6,}  {name[:60]}")


if __name__ == "__main__":
    main()
