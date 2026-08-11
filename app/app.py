"""
streamlit demo UI for manually testing the FairSearch-arXiv pipeline.

lets the user run a query against the baseline retriever or the fairness-aware
MMR retriever, inspect the institution-label mix of the results, and
optionally generate a Gemini-synthesized answer without touching the CLI.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from retriever_mmr import QDRANT_PATH, connect_qdrant, load_model, search, search_mmr  # noqa: E402
from generator import generate  # noqa: E402

LABEL_EMOJI = {
    "privileged": "🏛️ privileged",
    "underrepresented": "🌍 underrepresented",
    "unknown": "❔ unknown",
}

st.set_page_config(page_title="FairSearch-arXiv", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Connecting to Qdrant index...")
def get_client():
    return connect_qdrant()


def render_results(hits, mmr_mode):
    if not hits:
        st.warning("No results.")
        return

    labels = [h.payload.get("institution_label", "unknown") for h in hits]
    counts = Counter(labels)
    st.caption("Institution-label mix among these results")
    st.bar_chart(pd.Series(counts, name="count"))

    for rank, hit in enumerate(hits, start=1):
        p = hit.payload or {}
        label = LABEL_EMOJI.get(p.get("institution_label"), p.get("institution_label", "unknown"))
        header = f"**#{rank}** — {p.get('title', 'Untitled')}  \n`{label}` · {p.get('category')} · {p.get('year')}"
        if mmr_mode:
            header += f" · similarity {p.get('query_similarity', hit.score):.4f}"
        else:
            header += f" · score {hit.score:.4f}"

        with st.expander(header):
            st.write(f"**Paper ID:** {p.get('paper_id')}")
            st.write(f"**Authors:** {p.get('authors')}")
            st.write(p.get("abstract", ""))


def main():
    st.title("FairSearch-arXiv")
    st.caption(
        "Test harness for the retrieval pipeline: search the arXiv CS index with the "
        "baseline retriever or the fairness-aware MMR re-ranker, and optionally "
        "synthesize an answer with Gemini."
    )

    if not Path(QDRANT_PATH).exists():
        st.error(
            f"No Qdrant index found at `{QDRANT_PATH}`.\n\n"
            "Run `python index_builder.py` (and `python enrich_metadata.py`) from "
            "inside `src/` first see the README"
        )
        return

    with st.sidebar:
        st.header("Settings")
        mode = st.radio("Retriever", ["Baseline", "MMR (fairness-aware)"])
        k = st.slider("Results to show (k)", min_value=1, max_value=20, value=10)

        fetch_k, lambda_mult = 50, 0.5
        if mode.startswith("MMR"):
            fetch_k = st.slider("Candidate pool size (fetch_k)", min_value=k, max_value=200, value=50)
            lambda_mult = st.slider(
                "Relevance vs. diversity (lambda)", min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                help="1.0 = pure relevance (like baseline). 0.0 = maximize diversity.",
            )

        st.divider()
        generate_answer = st.checkbox("Generate answer with Gemini", value=False)
        api_key = None
        if generate_answer:
            api_key = st.text_input(
                "Gemini API key (optional)",
                type="password",
                help="Leave blank to use GEMINI_API_KEY from the repo's .env file.",
            )

    query = st.text_input("Query", value="Recent advances in graph neural networks")
    run = st.button("Search", type="primary")

    if not run:
        return

    if not query.strip():
        st.warning("Enter a query first.")
        return

    model = get_model()
    client = get_client()

    with st.spinner("Searching..."):
        if mode.startswith("MMR"):
            hits = search_mmr(query, model, client, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        else:
            hits = search(query, model, client, k=k)

    if generate_answer:
        with st.spinner("Generating answer with Gemini..."):
            try:
                answer = generate(query, hits, api_key=api_key or None)
                st.subheader("Generated answer")
                st.markdown(answer)
            except ValueError as e:
                st.error(str(e))

    st.subheader(f"Retrieved papers ({mode})")
    render_results(hits, mmr_mode=mode.startswith("MMR"))


if __name__ == "__main__":
    main()
