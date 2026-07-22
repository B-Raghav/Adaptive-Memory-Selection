"""Interactive demo: adaptive memory selection for a long-term chat agent.

Shows, for a real Multi-Session Chat conversation, how the trained classifier
scores each past memory and how the four retrieval strategies (no-memory,
store-all, top-K similarity, ML-selected) change the LLM's reply and the number
of context tokens spent.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from llm_integration import GEN_MODEL, build_prompt, generate, get_demo_episode, judge  # noqa: E402
from scorer import MemoryScorer  # noqa: E402

st.set_page_config(page_title="Adaptive Memory Selection", layout="wide")


@st.cache_resource
def load_scorer():
    return MemoryScorer()


@st.cache_data
def load_episode(session: int, index: int):
    return get_demo_episode(session, index=index)


st.title("🧠 Adaptive Memory Selection for Long-Term AI Agents")
st.caption(
    "A trained classifier predicts which past conversational memories to feed the "
    "LLM — compared against no-memory, store-all, and top-K similarity retrieval."
)

scorer = load_scorer()

with st.sidebar:
    st.header("Conversation")
    session = st.selectbox("MSC session", [5, 4, 3], index=0)
    index = st.number_input("Episode index", min_value=0, max_value=400, value=0, step=1)
    st.divider()
    st.header("Selection controls")
    threshold = st.slider("ML retain threshold", 0.1, 0.9, 0.5, 0.05)
    k = st.slider("top-K (similarity baseline)", 1, 15, 5)
    st.divider()
    st.caption(f"Classifier: **{scorer.name}**  ·  LLM: `{GEN_MODEL}`")

memories, personas, qa = load_episode(session, index)
mem_texts = [m["text"] for m in memories]

default_q = qa[0][0] if qa else "What have we talked about before?"
query = st.text_input("User message", value=default_q)

# score memories for the current query
feats = scorer.features(memories, query, personas)
probs = scorer.pipeline.predict_proba(feats)[:, 1]
sims = feats["sim_to_current_query"].values

bank = pd.DataFrame({
    "memory": mem_texts,
    "retain_prob": probs.round(3),
    "similarity": sims.round(3),
    "session": [m["memory_session"] for m in memories],
}).sort_values("retain_prob", ascending=False)

top_k_idx = set(sims.argsort()[::-1][:k])
ml_idx = [i for i in range(len(mem_texts)) if probs[i] >= threshold]
if not ml_idx:
    ml_idx = list(probs.argsort()[::-1][:3])
ml_idx = sorted(ml_idx, key=lambda i: -probs[i])[:8]

left, right = st.columns([1, 1])

with left:
    st.subheader(f"Memory bank ({len(mem_texts)} memories)")
    st.caption("Sorted by predicted retain-probability. Bars show the classifier score.")
    st.dataframe(
        bank, width="stretch", height=420, hide_index=True,
        column_config={
            "retain_prob": st.column_config.ProgressColumn(
                "retain_prob", min_value=0.0, max_value=1.0, format="%.2f"),
            "similarity": st.column_config.NumberColumn("similarity", format="%.2f"),
        },
    )
    st.metric("ML-selected memories", f"{len(ml_idx)}", help="predicted retain ≥ threshold")

with right:
    st.subheader("Selected context")
    st.markdown(f"**ML-selected ({len(ml_idx)} memories):**")
    for i in ml_idx:
        st.markdown(f"- {mem_texts[i]}  \n  <small>retain={probs[i]:.2f}</small>",
                    unsafe_allow_html=True)
    st.markdown(f"**Top-{k} by similarity:**")
    for i in sorted(top_k_idx, key=lambda i: -sims[i]):
        st.markdown(f"- {mem_texts[i]}  \n  <small>sim={sims[i]:.2f}</small>",
                    unsafe_allow_html=True)

st.divider()

if st.button("Generate responses (all 4 modes)", type="primary"):
    modes = {
        "No memory": [],
        "Store-all": mem_texts,
        f"Top-{k} similarity": [mem_texts[i] for i in sorted(top_k_idx, key=lambda i: -sims[i])],
        "ML-selected": [mem_texts[i] for i in ml_idx],
    }
    cols = st.columns(4)
    for col, (mode, mems) in zip(cols, modes.items()):
        with col:
            st.markdown(f"#### {mode}")
            tok = sum(len(m.split()) for m in mems)
            st.caption(f"{len(mems)} memories · {tok} context tokens")
            with st.spinner("generating…"):
                try:
                    resp = generate(build_prompt(query, mems))
                    score = judge(query, resp)
                except Exception as exc:  # noqa: BLE001
                    resp, score = f"⚠️ Ollama error: {exc}", 0
            st.info(resp)
            st.metric("Judge score", f"{score}/5")
else:
    st.info("Set the query and controls, then click **Generate responses** to compare the four modes.")
