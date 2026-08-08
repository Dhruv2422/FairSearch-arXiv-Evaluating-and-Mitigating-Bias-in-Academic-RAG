import json
import pandas as pd

# Input files for the three Experiment B conditions.
BASELINE_FILE = "../data/results/experiment_b_analysis_results.json"
MMR_FILE = "../data/results/experiment_b_mmr_analysis_results.json"
MMR_PROMPT_FILE = "../data/results/experiment_b_mmr_prompt_analysis_results.json"

# Output comparison table.
OUTPUT_FILE = "../data/results/experiment_b_summary_table.csv"


def load_metrics(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        results = json.load(f)

    retrieved = [r["retrieved_distribution"]["PRO-CONSENSUS"] for r in results]
    generated = [r["summary_distribution"]["pro_consensus_ratio"] for r in results]
    amplification = [r["consensus_amplification"] for r in results]
    gap = [r["absolute_gap"] for r in results]

    return {
        "queries": len(results),
        "retrieved": sum(retrieved) / len(retrieved),
        "generated": sum(generated) / len(generated),
        "amplification": sum(amplification) / len(amplification),
        "gap": sum(gap) / len(gap),
    }


baseline = load_metrics(BASELINE_FILE)
mmr = load_metrics(MMR_FILE)
mmr_prompt = load_metrics(MMR_PROMPT_FILE)

summary = pd.DataFrame({
    "Metric": [
        "Queries Analysed",
        "Average Retrieved Consensus Ratio",
        "Average Generated Consensus Ratio",
        "Average Consensus Amplification",
        "Average Absolute Perspective Gap"
    ],
    "Baseline": [
        baseline["queries"],
        round(baseline["retrieved"], 3),
        round(baseline["generated"], 3),
        round(baseline["amplification"], 3),
        round(baseline["gap"], 3)
    ],
    "MMR": [
        mmr["queries"],
        round(mmr["retrieved"], 3),
        round(mmr["generated"], 3),
        round(mmr["amplification"], 3),
        round(mmr["gap"], 3)
    ],
    "MMR + Prompt": [
        mmr_prompt["queries"],
        round(mmr_prompt["retrieved"], 3),
        round(mmr_prompt["generated"], 3),
        round(mmr_prompt["amplification"], 3),
        round(mmr_prompt["gap"], 3)
    ]
})

print("\n" + "=" * 80)
print("EXPERIMENT B — GENERATIVE FAITHFULNESS SUMMARY")
print("=" * 80)
print(summary.to_string(index=False))
print("=" * 80)

summary.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved summary table to: {OUTPUT_FILE}")