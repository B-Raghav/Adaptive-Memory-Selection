"""Inference-time memory scorer used by the LLM demo and the Streamlit app.

Rebuilds the exact 16-feature vector from :mod:`features` for an arbitrary set of
memories against a live query, then applies the trained classifier to produce a
retain-probability per memory. Also assembles memory records from a raw MSC
episode so the demo can operate on a real conversation.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

import msc_embed
from features import FIRST_PERSON, PROPER_NOUN as PROPER, WORD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
from features import FEATURE_COLUMNS


def episode_memories(episode: dict, prev_dialogs: list[dict], target_session: int) -> list[dict]:
    """Flatten an MSC episode's previous_dialogs into memory records with metadata."""
    flat = []
    for j, pdlg in enumerate(prev_dialogs):
        for i, turn in enumerate(pdlg.get("dialog", [])):
            text = turn.get("text", "")
            if text:
                flat.append((j + 1, i, text))
    total = len(flat)
    records = []
    for gi, (mem_session, turn_idx, text) in enumerate(flat):
        records.append({
            "text": text,
            "memory_session": mem_session,
            "recency_sessions": target_session - mem_session,
            "turns_since_recall": total - gi,
            "position_in_session": turn_idx,
            "role_speaker1": int(turn_idx % 2 == 0),
        })
    return records


class MemoryScorer:
    def __init__(self):
        with open(MODELS / "best_model.pkl", "rb") as fh:
            blob = pickle.load(fh)
        self.name = blob["name"]
        self.pipeline = blob["pipeline"]
        with open(MODELS / "tfidf_vectorizer.pkl", "rb") as fh:
            self.tfidf = pickle.load(fh)

    def _surface(self, text: str) -> dict:
        toks = WORD.findall(text.lower())
        sw = sum(w in ENGLISH_STOP_WORDS for w in toks) / len(toks) if toks else 0.0
        return {
            "sentence_length": len(text.split()),
            "char_length": len(text),
            "has_number": int(bool(re.search(r"\d", text))),
            "is_question": int("?" in text),
            "has_proper_noun": int(bool(PROPER.search(text))),
            "first_person_count": len(FIRST_PERSON.findall(text)),
            "stopword_ratio": sw,
        }

    def features(self, memories: list[dict], query: str, personas: list[str]) -> pd.DataFrame:
        texts = [m["text"] for m in memories]
        mem_emb = msc_embed.embed(texts)
        q_emb = msc_embed.embed([query])
        sim_q = msc_embed.cosine(mem_emb, np.repeat(q_emb, len(texts), axis=0))

        mem_n = mem_emb / (np.linalg.norm(mem_emb, axis=1, keepdims=True) + 1e-8)
        if personas:
            p_emb = msc_embed.embed(personas)
            p_emb = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-8)
            persona_sal = (mem_n @ p_emb.T).max(axis=1)
        else:
            persona_sal = np.zeros(len(texts))

        sim = mem_n @ mem_n.T
        access = np.array([int((sim[i, :i] > 0.6).sum()) for i in range(len(texts))])

        tfidf = self.tfidf.transform(texts)
        tfidf_max = np.asarray(tfidf.max(axis=1).todense()).ravel()
        nnz = np.diff(tfidf.tocsr().indptr)
        tfidf_sum = np.asarray(tfidf.sum(axis=1)).ravel()
        tfidf_mean = np.divide(tfidf_sum, nnz, out=np.zeros_like(tfidf_sum), where=nnz > 0)

        rows = []
        for i, m in enumerate(memories):
            row = {
                "recency_sessions": m["recency_sessions"],
                "turns_since_recall": m["turns_since_recall"],
                "position_in_session": m["position_in_session"],
                "role_speaker1": m["role_speaker1"],
                "sim_to_current_query": float(sim_q[i]),
                "persona_salience": float(persona_sal[i]),
                "access_frequency": float(access[i]),
                "tfidf_max": float(tfidf_max[i]),
                "tfidf_mean": float(tfidf_mean[i]),
            }
            row.update(self._surface(m["text"]))
            rows.append(row)
        return pd.DataFrame(rows)[FEATURE_COLUMNS]

    def score(self, memories: list[dict], query: str, personas: list[str]) -> np.ndarray:
        X = self.features(memories, query, personas)
        return self.pipeline.predict_proba(X)[:, 1]
