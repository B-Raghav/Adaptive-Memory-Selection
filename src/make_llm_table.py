"""Emit reports/llm_table.tex from results/llm_summary.csv."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
s = pd.read_csv(ROOT / "results" / "llm_summary.csv", index_col=0)
order = ["no_memory", "top_k", "store_all", "ml_selected"]
pretty = {"no_memory": "No memory", "top_k": "Top-$K$ similarity",
          "store_all": "Store-all", "ml_selected": "ML-selected (ours)"}
s = s.reindex([m for m in order if m in s.index])
n_q = None
try:
    df = pd.read_csv(ROOT / "results" / "llm_comparison.csv")
    n_q = df[["episode", "query_id"]].drop_duplicates().shape[0] if "episode" in df else df["query_id"].nunique()
except Exception:
    pass

rows = []
for m, r in s.iterrows():
    label = f"\\textbf{{{pretty[m]}}}" if m == "ml_selected" else pretty[m]
    tok = f"\\textbf{{{r.avg_context_tokens:.0f}}}" if m == "ml_selected" else f"{r.avg_context_tokens:.0f}"
    rows.append(f"{label} & {r.avg_memories:.1f} & {tok} & {r.avg_judge:.2f} \\\\")

cap = (f"Four-condition LLM comparison over {n_q} queries "
       if n_q else "Four-condition LLM comparison ")
tex = r"""\begin{table}[t]
\centering\small
\caption{%s(Llama\,3.1 8B answers; judged 1--5 by an independent model). Judge
scores are a statistical tie across all regimes. ML-selection is far cheaper than
store-all, but costs more than top-$K$ for no measurable quality gain; its
distinguishing behaviour is adapting the memory count per query rather than a fixed
$K$. Quality parity here means the classification metrics, not this demo, carry the
thesis.}
\label{tab:llm}
\begin{tabular}{lccc}
\toprule
Memory regime & Avg.\ memories & Avg.\ context tokens & Avg.\ judge (1--5) \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (cap, "\n".join(rows))

(ROOT / "reports" / "llm_table.tex").write_text(tex)
print("wrote reports/llm_table.tex")
print(s.round(2).to_string())
