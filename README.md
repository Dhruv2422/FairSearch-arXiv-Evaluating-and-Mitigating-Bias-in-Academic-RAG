# FairSearch-arXiv

**Evaluating and Mitigating Bias in Academic RAG**

FairSearch-arXiv is a retrieval-augmented generation system built over a 50,000-paper
sample of the arXiv computer science (`cs.*`) corpus, together with the instrumentation
needed to measure whether it is fair. It embeds papers with a sentence-transformer,
indexes them in a local [Qdrant](https://qdrant.tech/) database, retrieves semantically
similar papers for a query, and synthesizes an answer with the Gemini API.

## Research questions

The project investigates three questions, addressed by two experiments:

- **RQ1 — Retrieval Parity.** Does semantic vector search exhibit *institutional
  homophily*, systematically ranking papers from elite institutions higher than
  relevant work from regional or emerging research hubs?
- **RQ2 — Synthesis Neutrality.** To what extent does the LLM generator prioritize
  consensus views over dissenting evidence when synthesizing context from multiple
  retrieved documents?
- **RQ3 — The Fairness–Utility Tradeoff.** What is the quantitative impact on NDCG@10
  and MRR when re-ranking strategies are applied to improve demographic parity?

| | RQ1 | RQ2 | RQ3 |
|---|:-:|:-:|:-:|
| **Experiment A** — retrieval bias audit + MMR re-ranking | ● | | ● |
| **Experiment B** — synthesis neutrality across three conditions | | ● | ● |

**Experiment A** answers RQ1 by measuring whether the baseline retriever over-selects
elite institutions relative to their corpus share, then answers RQ3 on the retrieval
side by sweeping the MMR λ parameter to trace what each increment of fairness costs in
ranking quality.

**Experiment B** answers RQ2 by measuring how much more one-sided a generated summary
is than the evidence it was given, and contributes the generation-side half of RQ3:
whether the intervention that fixes synthesis neutrality carries a comparable utility
penalty, measured with RAGAS. It does not — which is the more interesting half of the
answer.

The two experiments are deliberately separable. A operates on institutional metadata and
never looks at generated text; B operates on generated text and never looks at
institution labels. They share the corpus, the index, and the retrievers, and nothing
else.

## Headline results

**Experiment A** — 100 queries, top-10 labeled results per query. Privileged
institutions hold 28.6% of the labeled corpus but take 38.8% of baseline top-10 slots.

| Metric | Baseline | MMR (λ=0.5) | Direction |
|---|---:|---:|---|
| Privileged share of top-10 | 38.8% | 35.2% | lower is fairer |
| Selection rate ratio (corpus-normalized) | 1.5864 | 1.3593 | 1.0 is parity |
| Equalized Odds difference | 0.0579 | 0.0402 | 0.0 is parity |
| NDCG@10 | 0.7312 | 0.5526 | higher is better |
| MRR | 0.9633 | 0.8798 | higher is better |

A λ sweep (`sweep_mmr_lambda.py`) shows the tradeoff is **not monotonic** and that the
two fairness metrics disagree about the best operating point: the selection rate ratio
is best at λ=0.5, while Equalized Odds is best at λ=0.8 — which costs only 0.05 NDCG
instead of 0.18.

| λ | Selection ratio | Equalized Odds | NDCG@10 |
|---:|---:|---:|---:|
| 0.5 | **1.3593** | 0.0402 | 0.5526 |
| 0.6 | 1.3833 | 0.0357 | 0.5847 |
| 0.7 | 1.4445 | 0.0302 | 0.6375 |
| 0.8 | 1.5272 | **0.0286** | 0.6807 |
| 0.9 | 1.5014 | 0.0348 | 0.7146 |
| 1.0 (= baseline) | 1.5864 | 0.0579 | **0.7312** |

**Experiment B** — 50 contradictory queries, three conditions. Consensus amplification
is the generated consensus ratio minus the retrieved consensus ratio: how much more
one-sided the summary is than the evidence it was given.

| Condition | Consensus amplification | Absolute gap | Answer relevancy | Faithfulness |
|---|---:|---:|---:|---:|
| Baseline retrieval + standard prompt | 0.2000 | 0.2920 | 0.7798 | 0.9853 |
| MMR retrieval + standard prompt | 0.1950 | 0.2930 | 0.7807 | 0.9846 |
| MMR + balanced prompt | **0.1230** | **0.2090** | 0.7756 | 0.9834 |

The two experiments reach complementary conclusions. **Diversifying retrieval barely
moves synthesis neutrality** (0.2000 → 0.1950), but **changing the prompt cuts
consensus amplification by 38.5%** (0.2000 → 0.1230) at essentially no cost in RAGAS
answer relevancy or faithfulness. Retrieval-side and generation-side bias are distinct
failure modes and need distinct interventions.

## Repository layout

```
.
├── data/
│   ├── raw/          # Raw arXiv JSON snapshot (gitignored; only to rebuild from scratch)
│   ├── processed/    # papers.parquet + affiliation/judgment caches — committed
│   ├── indices/      # Generated Qdrant index (gitignored, regenerated locally)
│   ├── eval/         # Query sets and relevance judgments — committed
│   └── results/      # Metric outputs — committed (except results/sweep/)
├── src/                            # The RAG system: reusable components
│   ├── preprocess.py               # Filter cs.*, sample, clean → papers.parquet
│   ├── index_builder.py            # Embed papers and build the Qdrant index
│   ├── retriever.py                # Baseline dense retrieval
│   ├── retriever_mmr.py            # MMR (Maximal Marginal Relevance) re-ranked retrieval
│   ├── generator.py                # Gemini synthesis + BALANCED_SYSTEM_PROMPT
│   ├── enrich_metadata.py          # Fetch affiliations, assign institution labels
│   ├── audit_labels.py             # Offline audit of label-matching quality
│   ├── check_labels.py             # Print the institution-label census
│   ├── build_eval_queries.py       # Interactive hand-labeling of relevant_ids
│   ├── build_qrels.py              # Score-threshold auto-judged relevance labels
│   ├── metrics.py                  # Precision@k / Recall@k over hand-labeled queries
│   └── test_pipeline.py            # End-to-end smoke test
├── experiments/                    # Both experiments and all evaluators
│   ├── build_qrels_llm.py                  # A: LLM-judged pooled relevance labels
│   ├── experiment_a.py                     # A: bias audit, baseline retriever
│   ├── experiment_a_mmr.py                 # A: bias audit, MMR retriever
│   ├── evaluate_equalized_odds.py          # A: Equalized Odds (TPR/FPR gaps)
│   ├── evaluate_ndcg_mrr.py                # A: NDCG@10 / MRR
│   ├── evaluate_DP_ED.py                   # A: Demographic Parity + Exposure Diversity
│   ├── sweep_mmr_lambda.py                 # A: fairness–utility tradeoff across λ
│   ├── experiment_b_collect.py             # B: baseline retrieval + standard prompt
│   ├── experiment_b_mmr_collect.py         # B: MMR retrieval + standard prompt
│   ├── experiment_b_mmr_prompt_collect.py  # B: MMR + balanced prompt
│   ├── experiment_b_analyze.py             # B: LLM-as-judge consensus/dissent labeling
│   ├── experiment_b_ragas.py               # B: RAGAS relevancy + faithfulness
│   └── experiment_b_results.py             # B: aggregate the three conditions
├── app/
│   └── app.py        # Streamlit UI for exploring the retrievers interactively
├── LICENSE
└── requirements.txt
```

`src/` holds things you would import; `experiments/` holds things you would run once
to produce a number. Every script in `experiments/` resolves its paths from the
repository root, so all of them run from anywhere — the commands below use the repo
root for consistency.

## Prerequisites

- **Python 3.10+** (the codebase uses `X | None` type syntax).
- **A [Google AI Studio](https://aistudio.google.com/) API key** — for generation, the
  LLM-judged relevance labels in Experiment A, and the consensus/dissent judge in
  Experiment B. Enable billing: the free tier's daily quota is too low for a full
  100-query judging pass. Total cost is a few dollars.
- **An [OpenAI](https://platform.openai.com/) API key** — only for
  `experiment_b_ragas.py`, which uses `text-embedding-3-small` for answer relevancy.
- **A [Kaggle](https://www.kaggle.com/) account** — only if rebuilding the corpus from
  scratch, which is not recommended (see below).

## Setup

```bash
git clone <repo-url>
cd FairSearch-arXiv-Evaluating-and-Mitigating-Bias-in-Academic-RAG

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root (gitignored, never committed):

```
GEMINI_API_KEY=your-key-here
CONTACT_EMAIL=your-email-here    # OpenAlex polite-pool "mailto" parameter
OPENAI_API_KEY=your-key-here     # only needed for experiment_b_ragas.py
```

Then build the index from the committed corpus:

```bash
python src/index_builder.py      # → data/indices/qdrant/
python src/enrich_metadata.py    # adds institution_label to each point
python src/check_labels.py       # verify
```

`check_labels.py` must print exactly:

```
Counter({'unknown': 24181, 'underrepresented': 18447, 'privileged': 7372})
```

If it does, your index matches the one every number in this README was computed from.
`enrich_metadata.py` reads the committed affiliation cache and makes no API calls
unless it encounters papers the cache does not cover.

### Reproducibility: use the committed corpus

The 50,000-paper sample is committed at `data/processed/papers.parquet`, with its
affiliation lookups in `data/processed/affiliation_cache.json`. **All results are keyed
to that specific sample.**

You *can* rebuild from scratch by downloading the raw arXiv snapshot from the
[arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv) to
`data/raw/arxiv-metadata-oai-snapshot.json` and running `python src/preprocess.py`. But
the Kaggle snapshot is refreshed regularly, so a fresh download produces a *different*
50,000-paper sample even with the same seed — the seed makes sampling reproducible only
for an identical input file. Rebuilt corpora will not reproduce the numbers here.

## Trying it interactively

The fastest way to see the system work is the Streamlit app:

```bash
streamlit run app/app.py
```

It runs a query through either retriever, shows the institution-label mix of the
results, and optionally synthesizes an answer with Gemini. The sidebar exposes `k`,
the MMR candidate pool (`fetch_k`), and λ.

> **Reading the app against this README.** By default the app shows the raw top-k,
> *including* `unknown`-labeled papers — which are 48.4% of the corpus and so dominate
> any unfiltered slice. The experiments exclude them. Tick **"Exclude unknown-labeled
> papers"** to make the app's mix comparable to the numbers above.

Or from the command line:

```bash
python src/retriever.py       # baseline retrieval, prints top-k
python src/retriever_mmr.py   # baseline and MMR side by side
python src/generator.py       # retrieve + synthesize an answer
python src/test_pipeline.py   # five queries end to end
```

Each runs a hardcoded example query; edit the `query` variable in the `__main__` block
at the bottom of the file to try your own.

## How institution labels are assigned

Everything in Experiment A rests on this step, so it is worth understanding before
trusting any fairness number.

`enrich_metadata.py` fetches author affiliations and labels each paper `privileged`,
`underrepresented`, or `unknown` (no affiliation data found anywhere). Institution
names are matched on word boundaries after normalization, with explicit guards against
lookalikes — City University of Hong Kong must not match University of Hong Kong, and
"Universidade Federal do Amazonas" must not match Amazon.

**What counts as privileged.** The corpus is entirely `cs.*`, and an all-disciplines
university ranking is a poor instrument for prestige *within* computer science: under
QS Top 20 alone, Carnegie Mellon, Princeton, UIUC, Georgia Tech and Google Research all
came out `underrepresented`. The privileged set is therefore the union of four groups
of institutions, each capturing a different kind of research prestige:

| Group | Definition | Papers |
|---|---|---:|
| Global universities | QS World University Rankings 2027, Top 20 (incl. ties) | 3,775 |
| CS departments | [CSRankings](https://csrankings.org) all-areas world Top 20 ∪ US Top 20 (retrieved 2026-08-09) | 2,225 |
| Ivy League | Ivy League institutions not already captured by the QS group | 326 |
| Industry labs | Major corporate research labs | 1,053 |

The CS-departments group unions the world and US lists so that strong US departments
the global ranking pushes past rank 20 (UT Austin, Wisconsin, UCLA) are still captured,
with CSRankings' own top-20 boundary as the cutoff rather than an arbitrary depth.
National research agencies (CNRS, the Chinese Academy of Sciences, INRIA) are
deliberately **excluded** — they are umbrella organizations spanning labs of widely
varying prestige, so labeling them uniformly elite is indefensible.

The four groups are not ranked against one another; membership in any one of them is
sufficient, and a paper counted once regardless of how many it matches. Two limitations
of this operationalization are worth stating outright. Including industry labs means
"privileged" spans global corporate research (Tencent, Huawei, Alibaba, Baidu among
them), not only Western academia. And CSRankings' "Northeastern University" is the
Boston institution, while Northeastern University (Shenyang) is a distinct university
whose name normalizes identically — both currently match the CS-departments group.

**Coverage.** Lookup runs in five stages: an OpenAlex batch pass, a re-lookup with
wider extraction, a free Semantic Scholar batch pass (500 arXiv IDs per request, ~16%
recovery on OpenAlex's misses), a free Crossref pass by DOI (~44% recovery, but only
for papers that have one), and an optional per-paper OpenAlex title search. Stages 3–5
record what they have already attempted, so a re-run resumes rather than repeating a
finished pass.

The last stage is **off by default**. OpenAlex meters it by spend rather than request
rate (~$0.001/paper against a free daily budget of roughly $0.10), so a full pass over
this corpus costs about $25 and will otherwise die partway with HTTP 429 "Insufficient
budget". Enable it deliberately, with funded credit:

```bash
python src/enrich_metadata.py --title-search
```

Coverage stands at **51.6%** (25,819 of 50,000 papers). The free sources are
effectively exhausted: 92.9% of the remaining unlabeled papers carry no DOI,
journal reference or report number, so there is no identifier left to look them up by.

**Missingness is not random, and this bounds the audit.** Affiliation data comes from
*published* versions, so coverage splits sharply on publication status:

| | Papers | Labeled | Privileged share of labeled |
|---|---:|---:|---:|
| Has a DOI or journal reference | 14,288 | 88.2% | 24.1% |
| Preprint-only | 35,712 | 37.0% | 32.8% |

Among preprint-only papers, the ones that *are* labeled skew more privileged (32.8%)
than published ones (24.1%), which suggests the unlabeled remainder skews
underrepresented. Every fairness metric here is therefore computed over a subsample
that over-represents published work and, probably, elite institutions. No additional
lookup source can fix this: the upstream data does not exist.

`audit_labels.py` is an offline diagnostic (no API calls) that reports which institution
triggered each `privileged` label and flags `underrepresented` names that closely
resemble a privileged-list entry — useful for validating matching before the labels
feed into metrics.

```bash
python src/audit_labels.py
```

---

# Experiment A — retrieval bias

*Addresses **RQ1** (retrieval parity) and the retrieval half of **RQ3** (fairness–utility
tradeoff).*

**Question.** Does dense retrieval over-select papers from elite institutions relative
to their share of the corpus, and does MMR re-ranking reduce that bias at an acceptable
cost in ranking quality?

**Design.** 100 standardized queries in `data/eval/retrieval_eval_queries.json` —
queries 1–50 are neutral topical queries, 51–100 are contradictory and debate-framed.
They were generated with an LLM and then reviewed by the authors, and are committed
rather than regenerated so that every result here is keyed to the exact query set used.
Each retriever is run over all 100, and the **top-10 labeled** results are scored — both
arms oversample to depth 100 and discard `unknown` papers before taking ten, so neither
is scored over a smaller result set than the other.

> This symmetry matters. An earlier version filtered MMR's top-10 *after* retrieval
> rather than oversampling first, leaving a mean of 5.2 labeled results against the
> baseline's 10. Every MMR metric was then computed over half as many results as the
> baseline it was compared to, mechanically depressing NDCG and both groups' true
> positive rates. `λ=1.0` now reproduces the baseline exactly, which is the check that
> this is wired up correctly.

**Relevance judgments.** `build_qrels_llm.py` builds `data/eval/qrels_auto.json`. For
each query, candidates are pooled from the baseline and MMR top-30 (TREC-style pooling,
so neither retriever's misses are invisible to the other), and Gemini grades each
candidate 0–3 using only title and abstract — independent of the retrieval score being
evaluated. Judgments are cached in `data/processed/llm_judgment_cache.json` and the run
is safe to interrupt and resume.

### Running it

Order matters: the evaluators read the two audit files, so run those first.

```bash
# 1. Build relevance judgments (only needed once; qrels_auto.json is committed)
python experiments/build_qrels_llm.py

# 2. The two retrieval audits
python experiments/experiment_a.py           # → data/results/experiment_a_results.json
python experiments/experiment_a_mmr.py       # → data/results/experiment_a_mmr_results.json

# 3. The evaluators (each reads the two files above)
python experiments/evaluate_equalized_odds.py   # → equalized_odds_results.json
python experiments/evaluate_ndcg_mrr.py         # → ndcg_mrr_results.json
python experiments/evaluate_DP_ED.py            # → fairness_metrics_from_results.json
python experiments/evaluate_DP_ED.py \
    --input data/results/experiment_a_mmr_results.json \
    --output data/results/fairness_metrics_mmr.json

# 4. The fairness–utility tradeoff curve
python experiments/sweep_mmr_lambda.py
```

Steps 2–4 need only the local index — no API keys, no network. Step 1 calls Gemini.

### Metrics

| Metric | What it measures |
|---|---|
| **Selection rate ratio** | Corpus-normalized (Fairlearn): the chance a labeled corpus paper reaches a top-10, privileged over underrepresented. 1.0 is parity. |
| **Statistical Parity Difference** | Raw gap in top-10 shares. Dominated by corpus base rates — report alongside the ratio, never alone. |
| **Equalized Odds difference** | Larger of the TPR / FPR gaps, treating retrieval as a classifier over judged-relevant papers. This is the only metric here *conditioned on relevance*. |
| **NDCG@10 / MRR** | Ranking quality against the LLM-judged qrels. |
| **Exposure Diversity** | Rank-weighted (log-discounted) share of attention each group receives. |

> **Two different ratios, pointing opposite ways.** In the result JSONs,
> `fairlearn_aggregate.selection_rate_ratio` is **privileged / underrepresented,
> corpus-normalized** (baseline 1.5864) — this is the bias figure. The separate
> `mean_selection_rate_ratio` field is **underrepresented / privileged** as a raw
> per-query share (baseline 2.1281). They are not comparable, and a value above 1 means
> opposite things in each. Quote the first.

The relevance-conditioned picture is worth separating from the raw one. Baseline TPR is
0.5659 for privileged versus 0.5080 for underrepresented — a ratio of **1.11**, well
below the unconditioned 1.59. Most of the 1.59× is that elite papers are judged relevant
more often in the first place (52.7% vs 47.7% of judged candidates); the residual
ranking preference among equally relevant papers is much smaller. Under MMR the TPR
ratio is 0.959, i.e. slightly favoring underrepresented papers — though both TPRs fall,
so MMR closes the gap partly by retrieving fewer relevant papers overall.

### The λ sweep

`sweep_mmr_lambda.py` re-runs the whole audit across λ. It guards against a subtle
evaluation artifact: `qrels_auto.json` was pooled from the baseline and MMR(λ=0.5)
top-30, so a λ whose results fall outside that pool is penalized by the *evaluation*
rather than by its own ranking quality. The script measures judged-pool coverage at
each point and refuses to report utility below λ=0.5 unless `--allow-unjudged` is
passed. Coverage across the reported range is 96.8–99.6%, so those numbers are sound.

```bash
python experiments/sweep_mmr_lambda.py
python experiments/sweep_mmr_lambda.py --lambdas 0.1,0.3,0.5,0.7,0.9 --allow-unjudged
```

---

# Experiment B — synthesis neutrality

*Addresses **RQ2** (synthesis neutrality) and the generation half of **RQ3** (fairness–utility
tradeoff).*

**Question.** When retrieved evidence genuinely conflicts, does the generator report
the consensus position more strongly than its sources warrant, and can that be
corrected on the retrieval side or the prompt side?

**Design.** 50 contradictory queries in `data/eval/contradictory_queries.json`, ten in
each of five areas: AI Ethics/Security/Society, Autonomous Systems, Large Language
Models & NLP, Machine Learning, and Software Engineering. Each was chosen so that the
retrievable literature contains genuine disagreement. As with the Experiment A set,
these were generated with an LLM and then reviewed by the authors, and are committed
rather than regenerated. The set is deliberately separate from Experiment A's — the two
are tuned for different questions. Experiment B never touches `institution_label`.

Three conditions isolate the prompt's effect from retrieval's:

| Condition | Retrieval | Prompt |
|---|---|---|
| `experiment_b_collect.py` | baseline dense | standard |
| `experiment_b_mmr_collect.py` | MMR (λ=0.5) | standard |
| `experiment_b_mmr_prompt_collect.py` | MMR (λ=0.5) | `BALANCED_SYSTEM_PROMPT` |

**Measurement.** `experiment_b_analyze.py` uses an LLM judge to classify each retrieved
abstract and each claim in the generated summary as `PRO-CONSENSUS`, `DISSENTING`, or
`NEUTRAL`. Two quantities follow:

- **Consensus amplification** = generated consensus ratio − retrieved consensus ratio.
  Positive means the summary is more one-sided than the evidence it was handed. This is
  the primary measure; it is a *difference*, so it isolates the generator's contribution
  from whatever skew retrieval already introduced.
- **Absolute gap** = total divergence between the retrieved and generated distributions.

`experiment_b_ragas.py` then scores every condition on RAGAS **answer relevancy** and
**faithfulness**, so that any neutrality gain can be checked against a possible quality
loss.

### Running it

```bash
# 1. Collect — each calls Gemini once per query
python experiments/experiment_b_collect.py             # → experiment_b_raw_results.json
python experiments/experiment_b_mmr_collect.py         # → experiment_b_raw_mmr_results.json
python experiments/experiment_b_mmr_prompt_collect.py  # → experiment_b_raw_mmr_prompt_results.json

# 2. Judge each condition (--condition selects which raw file to read)
python experiments/experiment_b_analyze.py --condition baseline
python experiments/experiment_b_analyze.py --condition mmr
python experiments/experiment_b_analyze.py --condition mmr_prompt

# 3. RAGAS across all three (needs OPENAI_API_KEY)
python experiments/experiment_b_ragas.py               # → experiment_b_ragas_comparison.json

# 4. Aggregate into the summary table
python experiments/experiment_b_results.py             # → experiment_b_summary_table.csv
```

The three collection scripts **resume**: each writes after every successful query and
skips queries already present in its output file, so a run interrupted by a Gemini 503
picks up where it stopped. `experiment_b_analyze.py` and `experiment_b_ragas.py` do
**not** cache — an interrupted run restarts from the beginning. A full pass across all
stages is roughly 2,400 API requests, so run these detached rather than in a foreground
shell.

`experiment_b_analyze.py` defaults to the judge model Experiment B's published results
used; override with `FAIRSEARCH_JUDGE_MODEL` rather than editing the file.

---

## Evaluation against hand-labeled data

Separately from the LLM-judged qrels, `build_eval_queries.py` is an interactive tool for
hand-labeling `relevant_ids`, which `metrics.py` scores for Precision@k / Recall@k. This
gives a small human-labeled check on the automated judgments.

```bash
python src/build_eval_queries.py
python src/metrics.py
```

## Limitations

Stated plainly, because several of them bound what the numbers can support:

1. **Relevance judgments are LLM-generated**, not human. Pooling and score-independent
   grading reduce the obvious failure modes, but this is not human ground truth, and
   Experiment B's consensus/dissent labels are LLM-generated too.
2. **48.4% of the corpus is unlabeled**, and the missingness is non-random in a
   direction that likely understates bias (see the coverage table above).
3. **"Privileged" is a constructed category.** Four ranking sources, one snapshot in
   time, and the Northeastern collision above are all judgment calls that a different
   defensible choice would change.
4. **The fairness metrics disagree with each other**, which is a finding rather than a
   defect — but it means "MMR is fairer" is only meaningful once you say which metric.
5. **NDCG falls materially at low λ.** MMR buys fairness with relevance; the sweep
   quantifies the exchange rate rather than pretending it is free.
6. **50,000 papers is a sample** of `cs.*`, not the full arXiv corpus.

## Notes

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim) for both
  indexing and retrieval. `index_builder.py` uses CUDA when available, else CPU.
- **Generation model:** `gemini-2.5-flash-lite`. Override with `FAIRSEARCH_GEN_MODEL`.
  (`gemini-1.5-flash`, originally specified, is no longer available via the API.)
- **Collection name:** `fairsearch_arxiv`. Override with `FAIRSEARCH_COLLECTION` to
  point the retrievers at a different index.
- **Committed outputs:** `data/results/` is committed so results can be inspected
  without re-running anything; only `data/results/sweep/` (per-λ intermediates) is
  gitignored.

## License

MIT — see [LICENSE](LICENSE). The licence covers the code. Paper metadata under `data/`
derives from the arXiv dataset (CC0 1.0) and from affiliation records retrieved via the
OpenAlex, Semantic Scholar, and Crossref APIs, each carrying its own terms.
