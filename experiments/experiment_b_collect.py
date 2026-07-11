"""
Experiment B: Generative Faithfulness - Data Collection

Purpose:
    This script runs the baseline RAG pipeline on a set of contradictory
    research queries to collect data for the synthesis neutrality analysis.

Pipeline:
    1. Load the predefined contradictory query set.
    2. Retrieve the top-k relevant arXiv papers from Qdrant.
    3. Generate a Gemini-based summary using the retrieved abstracts.
    4. Save the retrieved documents and generated summaries for downstream
       analysis.

Output:
    A JSON file containing:
        - query metadata
        - retrieved paper titles and abstracts
        - generated Gemini summaries

Note:
    This script only collects raw experiment data. Stance classification,
    token analysis, and fairness metrics are performed separately in
    experiment_b_analyze.py.
"""

import json
import time

from src.retriever import load_model, connect_qdrant, search
from src.generator import generate


QUERY_FILE = "../data/eval/contradictory_queries.json"
OUTPUT_FILE = "../data/results/experiment_b_raw_results.json"


def load_queries(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(results, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)


def main():

    print("Loading queries...")
    queries = load_queries(QUERY_FILE)

    print(f"Loaded {len(queries)} queries")

    print("Loading embedding model...")
    embedding_model = load_model()

    print("Connecting to Qdrant...")
    client = connect_qdrant()

    experiment_results = []

    for q in queries:

        print("\n==============================")
        print(f"Query {q['id']}: {q['query']}")

        try:
            # 1. Retrieve top-10 papers
            hits = search(
                q["query"],
                embedding_model,
                client,
                k=10
            )

            print(f"Retrieved {len(hits)} papers")

            # 2. Generate Gemini summary
            print("Generating Gemini response...")
            answer = generate(
                q["query"],
                hits
            )

            # 3. Store results
            result = {
                "id": q["id"],
                "category": q["category"],
                "query": q["query"],

                "retrieved_documents": [
                    {
                        "title": hit.payload["title"],
                        "abstract": hit.payload["abstract"],
                        "score": hit.score
                    }
                    for hit in hits
                ],

                "generated_summary": answer
            }

            experiment_results.append(result)

            # Save after every successful query
            save_results(
                experiment_results,
                OUTPUT_FILE
            )

            print("Saved result successfully")

            # Gemini free-tier rate limit protection
            time.sleep(5)


        except Exception as e:
            print(f"ERROR processing query {q['id']}: {e}")

            # Save progress even after failure
            save_results(
                experiment_results,
                OUTPUT_FILE
            )

            # Wait before continuing
            time.sleep(30)


    print("\n==============================")
    print("Experiment B retrieval + generation complete!")
    print(f"Saved {len(experiment_results)} successful queries")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()