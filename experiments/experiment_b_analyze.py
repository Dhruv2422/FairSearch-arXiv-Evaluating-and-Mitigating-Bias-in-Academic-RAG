import json
import os
import time
from dotenv import load_dotenv
from google import genai

# Change these paths depending on the experiment being analyzed.
#
# Baseline retriever + regular prompt:
#
# INPUT_FILE = "../data/results/experiment_b_raw_results.json"
# OUTPUT_FILE = "../data/results/experiment_b_analysis_results.json"
#
# MMR retriever + regular prompt:
#
# INPUT_FILE = "../data/results/experiment_b_raw_mmr_results.json"
# OUTPUT_FILE = "../data/results/experiment_b_mmr_analysis_results.json"
#
# MMR retriever + balanced/perspective-aware prompt:
#
# INPUT_FILE = "../data/results/experiment_b_raw_mmr_prompt_results.json"
# OUTPUT_FILE = "../data/results/experiment_b_mmr_prompt_analysis_results.json"

import sys
from pathlib import Path

# Resolve paths and imports from the repo root so these run from any working
# directory. Without this the module imports below need cwd=repo root while the
# data paths need cwd=experiments/, which only lined up under PyCharm.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_FILE = PROJECT_ROOT / "data" / "results" / "experiment_b_raw_mmr_prompt_results.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "results" / "experiment_b_mmr_prompt_analysis_results.json"

MODEL = "gemini-3.1-flash-lite"

load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def call_gemini(prompt):
    for attempt in range(5):
        try:
            return client.models.generate_content(
                model=MODEL,
                contents=prompt
            ).text
        except Exception as e:
            print(f"Gemini error: {e}")
            time.sleep(30 * (attempt + 1))

    raise Exception("Gemini failed after retries")


def parse_json(text):
    text = text.replace("`json", "").replace("`", "").strip()

    start_candidates = [text.find("["), text.find("{")]
    start_candidates = [x for x in start_candidates if x != -1]

    if not start_candidates:
        raise ValueError("No JSON found")

    start = min(start_candidates)

    if text[start] == "[":
        end = text.rfind("]") + 1
    else:
        end = text.rfind("}") + 1

    json_text = text[start:end]

    # Fix Gemini Python-style escaping
    json_text = json_text.replace("\\'", "'")

    return json.loads(json_text)


def classify_documents(query, docs):
    prompt = f"""
Query:
{query}

Classify each paper:
PRO-CONSENSUS = supports the accepted position
DISSENTING = challenges the accepted position
NEUTRAL = no clear stance

Return JSON only:
[
{{"title":"...", "stance":"PRO-CONSENSUS"}}
]
"""

    for i, d in enumerate(docs, 1):
        prompt += f"\nPaper {i}\n{d['title']}\n{d['abstract']}\n"

    return parse_json(call_gemini(prompt))


def classify_summary(query, summary):
    prompt = f"""
Query:
{query}

Summary:
{summary}

Return JSON only:

{{
"pro_consensus_ratio":0.0,
"dissenting_ratio":0.0
}}
"""
    return parse_json(call_gemini(prompt))


def distribution(docs):
    counts = {
        "PRO-CONSENSUS": 0,
        "DISSENTING": 0,
        "NEUTRAL": 0
    }

    for d in docs:
        if d["stance"] in counts:
            counts[d["stance"]] += 1

    total = len(docs)
    return {k: v / total for k, v in counts.items()}


def main():
    with open(INPUT_FILE, encoding="utf-8") as f:
        results = json.load(f)

    analyzed = []

    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, encoding="utf-8") as f:
            analyzed = json.load(f)

    completed = {x["query"] for x in analyzed}

    print(f"Loaded {len(results)} queries")
    print(f"Completed: {len(completed)}")

    for i, r in enumerate(results, 1):
        if r["query"] in completed:
            print(f"Skipping {i}")
            continue

        print(f"Analyzing {i}/{len(results)}")

        docs = r["retrieved_documents"]

        labels = classify_documents(r["query"], docs)

        for d, label in zip(docs, labels):
            d["stance"] = label["stance"]

        retrieved = distribution(docs)

        summary = classify_summary(
            r["query"],
            r["generated_summary"]
        )

        r["retrieved_distribution"] = retrieved
        r["summary_distribution"] = summary

        r["absolute_gap"] = abs(
            summary["pro_consensus_ratio"]
            - retrieved["PRO-CONSENSUS"]
        )

        r["consensus_amplification"] = (
            summary["pro_consensus_ratio"]
            - retrieved["PRO-CONSENSUS"]
        )

        analyzed.append(r)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(analyzed, f, indent=4)

        time.sleep(15)

    print("Done!")


if __name__ == "__main__":
    main()