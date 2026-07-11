import json
import pandas as pd

INPUT_FILE = "../data/results/experiment_b_analysis_results.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

retrieved = []
generated = []
amplification = []
gap = []

for r in results:
    retrieved.append(r["retrieved_distribution"]["PRO-CONSENSUS"])
    generated.append(r["summary_distribution"]["pro_consensus_ratio"])
    amplification.append(r["consensus_amplification"])
    gap.append(r["absolute_gap"])

# Calculate averages
avg_retrieved = sum(retrieved) / len(retrieved)
avg_generated = sum(generated) / len(generated)
avg_amplification = sum(amplification) / len(amplification)
avg_gap = sum(gap) / len(gap)

# Count outcomes
positive = sum(1 for x in amplification if x > 0)
negative = sum(1 for x in amplification if x < 0)
neutral = sum(1 for x in amplification if abs(x) <= 0.05)

# Create summary table
summary = pd.DataFrame({
    "Metric": [
        "Queries Analysed",
        "Average Retrieved Consensus Ratio",
        "Average Generated Consensus Ratio",
        "Average Consensus Amplification",
        "Average Absolute Perspective Gap",
        "Consensus Amplified",
        "Dissent Amplified",
        "Approximately Neutral"
    ],
    "Value": [
        len(results),
        round(avg_retrieved, 3),
        round(avg_generated, 3),
        round(avg_amplification, 3),
        round(avg_gap, 3),
        f"{positive}/{len(results)}",
        f"{negative}/{len(results)}",
        f"{neutral}/{len(results)}"
    ]
})

print(summary)

summary.to_csv(
    "../data/results/experiment_b_summary_table.csv",
    index=False
)