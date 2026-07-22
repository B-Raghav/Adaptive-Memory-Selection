"""Feature engineering for the memory-importance classifier.

Turns the raw memory records from :mod:`parse_msc` into a numeric feature matrix
(``data/processed/features.parquet``) and caches the memory embeddings
(``data/processed/mem_emb.npy``) for reuse by labeling and the demo.

Every feature is computable at *recall time* -- the moment a new session begins --
so nothing here peeks at future turns (that is reserved for the label).

Feature groups
--------------
Temporal / structural : recency_sessions, turns_since_recall, position_in_session
Speaker               : role_speaker1
Surface / lexical      : sentence_length, char_length, has_number, has_proper_noun,
                         is_question, first_person_count, stopword_ratio,
                         tfidf_max, tfidf_mean
Semantic               : sim_to_current_query, persona_salience
Recurrence             : access_frequency
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

import msc_embed

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

FIRST_PERSON = re.compile(r"\b(i|i'm|im|my|me|mine|myself|i've|i'll|i'd|we|our)\b", re.I)
PROPER_NOUN = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")
WORD = re.compile(r"[a-z']+")


def _load_memories() -> pd.DataFrame:
    rows = [json.loads(l) for l in open(PROC / "memories.jsonl", encoding="utf-8")]
    df = pd.DataFrame(rows).sort_values("memory_id").reset_index(drop=True)
    assert (df["memory_id"].values == np.arange(len(df))).all(), "memory_id must be contiguous"
    return df


def _surface_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].astype(str)
    df["sentence_length"] = text.str.split().apply(len)
    df["char_length"] = text.str.len()
    df["has_number"] = text.str.contains(r"\d").astype(int)
    df["is_question"] = text.str.contains(r"\?").astype(int)
    df["has_proper_noun"] = text.apply(lambda t: int(bool(PROPER_NOUN.search(t))))
    df["first_person_count"] = text.apply(lambda t: len(FIRST_PERSON.findall(t)))

    def _stopword_ratio(t: str) -> float:
        toks = WORD.findall(t.lower())
        if not toks:
            return 0.0
        return sum(w in ENGLISH_STOP_WORDS for w in toks) / len(toks)

    df["stopword_ratio"] = text.apply(_stopword_ratio)
    return df


def _tfidf_features(df: pd.DataFrame) -> pd.DataFrame:
    import pickle

    vec = TfidfVectorizer(stop_words="english", min_df=2, max_features=8000)
    mat = vec.fit_transform(df["text"].astype(str))
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "tfidf_vectorizer.pkl", "wb") as fh:
        pickle.dump(vec, fh)  # reused at inference time by the demo
    df["tfidf_max"] = np.asarray(mat.max(axis=1).todense()).ravel()
    sums = np.asarray(mat.sum(axis=1)).ravel()
    nnz = np.diff(mat.tocsr().indptr)
    df["tfidf_mean"] = np.divide(sums, nnz, out=np.zeros_like(sums), where=nnz > 0)
    return df


def _semantic_and_recurrence(df: pd.DataFrame, mem_emb: np.ndarray) -> pd.DataFrame:
    # sim_to_current_query: memory vs the opening turn of the new session.
    queries = df["current_query"].astype(str).tolist()
    q_emb = msc_embed.embed(queries)
    df["sim_to_current_query"] = msc_embed.cosine(mem_emb, q_emb)

    # persona_salience: max similarity to any carried-forward persona sentence.
    episodes = {json.loads(l)["episode_id"]: json.loads(l)
                for l in open(PROC / "episodes.jsonl", encoding="utf-8")}
    persona_sal = np.zeros(len(df), dtype=np.float32)
    access_freq = np.zeros(len(df), dtype=np.float32)

    for ep_id, idx in df.groupby("episode_id").groups.items():
        idx = np.asarray(idx)
        emb = mem_emb[idx]
        emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

        # persona salience
        persona = episodes[ep_id].get("persona_sentences") or []
        if persona:
            p_emb = msc_embed.embed(persona)
            p_emb = p_emb / (np.linalg.norm(p_emb, axis=1, keepdims=True) + 1e-8)
            persona_sal[idx] = (emb_n @ p_emb.T).max(axis=1)

        # access_frequency: how many *earlier* memories in the episode are
        # near-duplicates (cosine > 0.6) -- captures a recurring topic.
        sim = emb_n @ emb_n.T
        order = np.argsort(idx)  # chronological (memory_id increasing)
        for local, gi in enumerate(order):
            earlier = order[:local]
            if len(earlier):
                access_freq[idx[gi]] = int((sim[gi, earlier] > 0.6).sum())

    df["persona_salience"] = persona_sal
    df["access_frequency"] = access_freq
    return df


FEATURE_COLUMNS = [
    "recency_sessions", "turns_since_recall", "position_in_session", "role_speaker1",
    "sentence_length", "char_length", "has_number", "has_proper_noun", "is_question",
    "first_person_count", "stopword_ratio", "tfidf_max", "tfidf_mean",
    "sim_to_current_query", "persona_salience", "access_frequency",
]


def main() -> None:
    df = _load_memories()
    print(f"Loaded {len(df)} memories.")

    print("Embedding memory texts ...")
    mem_emb = msc_embed.embed(df["text"].astype(str).tolist())
    np.save(PROC / "mem_emb.npy", mem_emb)

    df = _surface_features(df)
    df = _tfidf_features(df)
    df = _semantic_and_recurrence(df, mem_emb)

    keep = ["memory_id", "episode_id", "target_session", "memory_session",
            "persona_worthy"] + FEATURE_COLUMNS
    out = df[keep].copy()
    out.to_parquet(PROC / "features.parquet", index=False)

    print(f"Wrote features for {len(out)} memories -> {PROC / 'features.parquet'}")
    print("Feature summary:")
    print(out[FEATURE_COLUMNS].describe().T[["mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
