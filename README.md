# FairSearch-arXiv

**Evaluating and Mitigating Bias in Academic RAG**

FairSearch-arXiv is a retrieval-augmented search system built over a sample of the
arXiv computer science (`cs.*`) corpus. It preprocesses raw arXiv metadata, embeds
papers with a sentence-transformer model, indexes them in a local
[Qdrant](https://qdrant.tech/) vector database, retrieves semantically similar papers
for a given query, and synthesizes answers using Google's Gemini model.

## Project structure

```
.
├── data/
│   ├── raw/          # Raw arXiv JSON snapshot (downloaded from Kaggle — see below)
│   ├── processed/    # Cleaned/sampled dataset (papers.parquet, generated)
│   ├── indices/      # Generated Qdrant index files (qdrant/, generated)
│   ├── eval/         # Evaluation query sets (retrieval_eval_queries.json)
│   └── results/      # Metric outputs (retrieval_metrics.json, generated)
├── src/
│   ├── preprocess.py          # Filter cs.* papers, sample, clean → papers.parquet
│   ├── index_builder.py       # Embed papers and build the Qdrant index
│   ├── retriever.py           # Run a semantic search query against the index
│   ├── generator.py           # Synthesize answers from retrieved papers via Gemini
│   ├── build_eval_queries.py  # Interactive tool for building the evaluation query set
│   ├── metrics.py             # Compute Precision@k and Recall@k over the eval set
│   └── test_pipeline.py       # End-to-end smoke test across multiple queries
├── app/              # (Reserved) Streamlit application
├── experiments/      # (Reserved) Experiment outputs and evaluation results
└── requirements.txt
```

## Prerequisites

- Python 3.9+
- A [Kaggle](https://www.kaggle.com/) account (to download the dataset)
- A [Google AI Studio](https://aistudio.google.com/) API key (for generation)

## Setup

1. **Clone the repository and enter it**

   ```bash
   git clone <repo-url>
   cd FairSearch-arXiv-Evaluating-and-Mitigating-Bias-in-Academic-RAG
   ```

2. **Create and activate a virtual environment** (recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**

   Download the raw arXiv metadata JSON from the
   [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv).
   Unzip it and place the file in the `data/raw/` folder. The preprocessing script
   expects it at:

   ```
   data/raw/arxiv-metadata-oai-snapshot.json
   ```

   > If your downloaded file has a different name, either rename it to
   > `arxiv-metadata-oai-snapshot.json` or update `input_path` at the bottom of
   > `src/preprocess.py`.

5. **Add your Gemini API key**

   Create a `.env` file in the repo root:

   ```
   GEMINI_API_KEY=your-key-here
   ```

   This file is gitignored and will never be committed.

## Running the pipeline

The scripts use paths relative to the `src/` directory, so **run them from inside
`src/`** and in the following order:

```bash
cd src

# 1. Preprocess: filter cs.* papers, sample 50,000, clean, and save as parquet
python preprocess.py
#    → writes data/processed/papers.parquet

# 2. Build the index: embed papers and store them in a local Qdrant database
python index_builder.py
#    → writes data/indices/qdrant/

# 3. Retrieve: run a semantic search query against the index
python retriever.py

# 4. Generate: retrieve + synthesize an answer via Gemini
python generator.py
```

### What each step does

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1 | `preprocess.py` | `data/raw/arxiv-metadata-oai-snapshot.json` | `data/processed/papers.parquet` |
| 2 | `index_builder.py` | `data/processed/papers.parquet` | `data/indices/qdrant/` (Qdrant collection `fairsearch_arxiv`) |
| 3 | `retriever.py` | `data/indices/qdrant/` | Top-k search results printed to the console |
| 4 | `generator.py` | `data/indices/qdrant/` + Gemini API | Synthesized answer with citations printed to the console |

### Customizing the search query

`retriever.py` and `generator.py` each run a hardcoded example query. To search for
something else, edit the `query` variable in the `__main__` block at the bottom of
the respective file:

```python
query = "Recent advances in graph neural networks"
```

### Running the end-to-end test

`test_pipeline.py` runs five diverse queries through the full retrieval and generation
pipeline and prints retrieved papers and synthesized answers for each:

```bash
python test_pipeline.py
```

## Evaluation

### Building the evaluation query set

`build_eval_queries.py` is an interactive command-line tool for constructing
`data/eval/retrieval_eval_queries.json` — a set of queries with human-labeled
relevant document IDs used to score retrieval quality.

```bash
python build_eval_queries.py
```

For each query you enter, the tool retrieves the top results, displays them, and
asks you to mark which ones are relevant. Entries are appended to the eval file
incrementally, so you can build it up across multiple sessions.

### Computing retrieval metrics

`metrics.py` runs every query in the eval file through the retriever and computes
Precision@k and Recall@k against the labeled relevant documents:

```bash
python metrics.py
```

Results are printed to the console as a per-query table and also saved to
`data/results/retrieval_metrics.json`.

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of the top-k retrieved papers that are relevant |
| Recall@k | Fraction of all relevant papers that appear in the top-k results |

## Notes

- **Embedding model:** Both indexing and retrieval use
  `sentence-transformers/all-MiniLM-L6-v2`. The same model must be used for both,
  which the scripts handle automatically.
- **GPU acceleration:** `index_builder.py` automatically uses CUDA if a compatible
  GPU is available; otherwise it falls back to CPU. Building the index over 50,000
  papers on CPU can take a while.
- **Sampling is deterministic:** `preprocess.py` samples 50,000 papers with a fixed
  seed (`42`), so the dataset is reproducible across runs.
- **Gemini model:** `generator.py` uses `gemini-2.5-flash-lite`. `gemini-1.5-flash`
  (originally specified) is no longer available via the Google AI API.

