"""Significance testing for the LLM judge scores.

The four memory regimes are evaluated on the *same* queries, so their judge scores
are a repeated-measures design. We run a Friedman test (the non-parametric
equivalent of a repeated-measures ANOVA) across the four regimes, and report a
bootstrap 95% CI on each regime's mean score. Written to results/llm_stats.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"


def bootstrap_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    df = pd.read_csv(RES / "llm_comparison.csv")
    wide = df.pivot_table(index=["episode", "query_id"], columns="mode",
                          values="judge_score", aggfunc="first").dropna()
    modes = ["no_memory", "top_k", "ml_selected", "store_all"]
    cols = [m for m in modes if m in wide.columns]
    stat, p = friedmanchisquare(*[wide[m].values for m in cols])

    per_mode = {}
    for m in cols:
        lo, hi = bootstrap_ci(wide[m].values)
        per_mode[m] = {"mean": float(wide[m].mean()), "ci95": [round(lo, 3), round(hi, 3)]}

    out = {
        "n_queries": int(len(wide)),
        "friedman_chi2": float(stat),
        "friedman_pvalue": float(p),
        "significant_at_0.05": bool(p < 0.05),
        "per_mode": per_mode,
    }
    (RES / "llm_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
