"""Generate retain/discard labels via a future-reference heuristic.

A memory is labelled ``retain = 1`` when it is *referenced or paraphrased* later
in the conversation -- specifically in the turns that follow the opening query of
the new session (``future_texts[1:]``). "Reference" is detected without using the
embedding-similarity-to-query feature, so the label is not a trivial restatement
of that feature and a similarity-retrieval baseline cannot win by construction.

Three triggers (union):
  1. **Named reference** -- a proper noun or number from the memory reappears
     later (strong, unambiguous reference).
  2. **Content reference** -- at least ``MIN_CONTENT`` specific content words
     (length >= 5, non-stopword) are shared with a later turn.
  3. **Paraphrase** -- a later turn is highly semantically similar
     (cosine > ``TAU_PARA``) to the memory.

Validation: MSC's human ``persona_worthy`` annotation is an *independent* notion
of memory importance. We report the agreement between our automatic label and
that annotation, plus a 120-row sample for manual inspection.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import cohen_kappa_score

import msc_embed

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

WORD = re.compile(r"[A-Za-z']+")
NUMBER = re.compile(r"\b\d+\b")
PROPER = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}\b")

MIN_CONTENT = 2
TAU_PARA = 0.60
MAX_DF = 0.03  # a content word counts as "specific" if it appears in < 3% of memories


def _content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) >= 5 and w not in ENGLISH_STOP_WORDS}


def _named_tokens(text: str) -> set[str]:
    # Proper nouns, minus title-cased stopwords ("The", "One", ...), plus numbers.
    proper = {p.lower() for p in PROPER.findall(text) if p.lower() not in ENGLISH_STOP_WORDS}
    return proper | set(NUMBER.findall(text))


def build_specific_vocab(all_texts: list[str]) -> set[str]:
    """Words rare enough (document frequency < MAX_DF) to signal a real topic."""
    n = len(all_texts)
    df: dict[str, int] = {}
    for t in all_texts:
        for w in _content_words(t):
            df[w] = df.get(w, 0) + 1
    cutoff = MAX_DF * n
    return {w for w, c in df.items() if c <= cutoff}


def label_episode(mem_texts, mem_emb, future_texts, specific_vocab):
    """Return (labels, trigger) arrays for one episode's memories."""
    future = future_texts[1:]  # exclude the opening query (that feeds a feature)
    n = len(mem_texts)
    labels = np.zeros(n, dtype=int)
    trigger = np.array(["none"] * n, dtype=object)
    if not future:
        return labels, trigger

    fut_content: set[str] = set()
    fut_named: set[str] = set()
    for t in future:
        fut_content |= _content_words(t)
        fut_named |= _named_tokens(t)

    fut_emb = msc_embed.embed(future)
    fut_emb = fut_emb / (np.linalg.norm(fut_emb, axis=1, keepdims=True) + 1e-8)
    mem_n = mem_emb / (np.linalg.norm(mem_emb, axis=1, keepdims=True) + 1e-8)
    max_para = (mem_n @ fut_emb.T).max(axis=1)

    for i, text in enumerate(mem_texts):
        named_shared = _named_tokens(text) & fut_named
        content_shared = (_content_words(text) & fut_content) & specific_vocab
        if named_shared:
            labels[i], trigger[i] = 1, "named"
        elif len(content_shared) >= MIN_CONTENT:
            labels[i], trigger[i] = 1, "content"
        elif max_para[i] > TAU_PARA:
            labels[i], trigger[i] = 1, "paraphrase"
    return labels, trigger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    mems = pd.DataFrame(json.loads(l) for l in open(PROC / "memories.jsonl", encoding="utf-8"))
    mems = mems.sort_values("memory_id").reset_index(drop=True)
    episodes = {json.loads(l)["episode_id"]: json.loads(l)
                for l in open(PROC / "episodes.jsonl", encoding="utf-8")}
    mem_emb = np.load(PROC / "mem_emb.npy")
    specific_vocab = build_specific_vocab(mems["text"].tolist())
    print(f"Specific-vocabulary size (df < {MAX_DF:.0%}): {len(specific_vocab)}")

    labels = np.zeros(len(mems), dtype=int)
    triggers = np.array(["none"] * len(mems), dtype=object)
    for ep_id, idx in mems.groupby("episode_id").groups.items():
        idx = np.asarray(idx)
        lab, trig = label_episode(
            mems.loc[idx, "text"].tolist(), mem_emb[idx],
            episodes[ep_id]["future_texts"], specific_vocab,
        )
        labels[idx] = lab
        triggers[idx] = trig

    mems["label_retain"] = labels
    mems["trigger"] = triggers

    pos = labels.mean()
    print(f"Positive (retain) rate: {pos:.3f}  ({labels.sum()}/{len(labels)})")
    print("Trigger breakdown:")
    print(mems.loc[mems.label_retain == 1, "trigger"].value_counts().to_string())

    # ---- validation against the human persona-worthiness annotation ----
    val = mems.dropna(subset=["persona_worthy"]).copy()
    val["persona_worthy"] = val["persona_worthy"].astype(int)
    agree = (val["label_retain"] == val["persona_worthy"]).mean()
    kappa = cohen_kappa_score(val["persona_worthy"], val["label_retain"])
    # P(referenced later | persona-worthy) vs P(referenced later | not worthy)
    p_ref_worthy = val.loc[val.persona_worthy == 1, "label_retain"].mean()
    p_ref_not = val.loc[val.persona_worthy == 0, "label_retain"].mean()
    # chi-square test of association between our label and the human annotation
    from scipy.stats import chi2_contingency

    ct = pd.crosstab(val["persona_worthy"], val["label_retain"]).values
    chi2, p_val, _, _ = chi2_contingency(ct)
    odds = (p_ref_worthy / (1 - p_ref_worthy)) / (p_ref_not / (1 - p_ref_not))
    print(f"\nValidation vs human persona_worthy (n={len(val)}):")
    print(f"  raw agreement = {agree:.3f}   Cohen's kappa = {kappa:.3f}")
    print(f"  P(retain | persona-worthy)     = {p_ref_worthy:.3f}")
    print(f"  P(retain | NOT persona-worthy) = {p_ref_not:.3f}   (odds ratio {odds:.2f})")
    print(f"  chi-square = {chi2:.1f}, p = {p_val:.2e}  -> "
          f"{'significant' if p_val < 0.01 else 'not significant'} positive association")

    # merge label into feature table
    feats = pd.read_parquet(PROC / "features.parquet")
    feats = feats.merge(mems[["memory_id", "label_retain", "trigger"]], on="memory_id")
    feats.to_parquet(PROC / "features_labeled.parquet", index=False)
    print(f"\nWrote labelled features -> {PROC / 'features_labeled.parquet'}")

    # manual-validation sample
    RESULTS.mkdir(exist_ok=True)
    sample = mems.sample(n=min(120, len(mems)), random_state=args.seed)[
        ["memory_id", "text", "label_retain", "trigger", "persona_worthy", "persona_text"]
    ]
    sample.to_csv(RESULTS / "label_validation_sample.csv", index=False)
    # persist label stats for the report
    with open(RESULTS / "label_stats.json", "w") as fh:
        json.dump(
            {
                "positive_rate": float(pos),
                "n": int(len(labels)),
                "agreement_persona_worthy": float(agree),
                "cohen_kappa": float(kappa),
                "p_retain_given_worthy": float(p_ref_worthy),
                "p_retain_given_not_worthy": float(p_ref_not),
                "odds_ratio": float(odds),
                "chi2": float(chi2),
                "chi2_pvalue": float(p_val),
                "trigger_counts": mems.loc[mems.label_retain == 1, "trigger"].value_counts().to_dict(),
                "tau_paraphrase": TAU_PARA,
                "min_content_overlap": MIN_CONTENT,
            },
            fh,
            indent=2,
        )
    print(f"Wrote validation sample + stats -> {RESULTS}")


if __name__ == "__main__":
    main()
