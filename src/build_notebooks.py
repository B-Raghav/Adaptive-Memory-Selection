"""Generate the narrative notebooks (01-03) from saved artefacts and execute them.

Notebooks import from ``src`` and display already-computed results/figures rather
than recomputing heavy steps, so they execute in seconds and stay consistent with
the pipeline outputs. Notebook 04 (LLM demo) is built separately once the LLM
comparison has finished.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

NB = Path(__file__).resolve().parent.parent / "notebooks"
HEADER = (
    "import sys, warnings\n"
    "from pathlib import Path\n"
    "warnings.filterwarnings('ignore')\n"
    "ROOT = Path.cwd().parent\n"
    "sys.path.insert(0, str(ROOT / 'src'))\n"
    "import pandas as pd, numpy as np, json\n"
    "from IPython.display import Image, display\n"
    "PROC, RES, FIG = ROOT/'data'/'processed', ROOT/'results', ROOT/'reports'/'figures'\n"
)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(src):
    return nbf.v4.new_code_cell(src)


def nb01():
    c = [
        md("# 01 · Data Exploration — Multi-Session Chat (MSC)\n\n"
           "We study **memory importance** in long-term dialogue: given the utterances "
           "from previous sessions of a conversation (the *memory bank*), which ones will "
           "be *referenced again* later? The target `label_retain` is 1 when a memory is "
           "referenced or paraphrased in the following session.\n\n"
           "**Challenges:** (1) no off-the-shelf relevance labels; (2) severe class "
           "imbalance — most memories are never re-referenced; (3) short, noisy utterances."),
        code(HEADER),
        md("## Parsed memory records"),
        code("mems = pd.DataFrame(json.loads(l) for l in open(PROC/'memories.jsonl'))\n"
             "eps = [json.loads(l) for l in open(PROC/'episodes.jsonl')]\n"
             "print(f'{len(mems)} memories across {mems.episode_id.nunique()} episodes '\n"
             "      f'from {len(eps)} episodes file')\n"
             "mems[['episode_id','memory_session','recency_sessions','role_speaker1','text']].head(6)"),
        md("## Memory-bank depth and recency"),
        code("print('memories per episode:', round(len(mems)/mems.episode_id.nunique(),1))\n"
             "mems.recency_sessions.value_counts().sort_index()"),
        md("## Human persona-worthiness annotation (used for label validation)\n\n"
           "MSC's `msc_personasummary` split annotates each utterance with the persona "
           "fact it contributes. We recovered this for every memory and use it as an "
           "*independent* human notion of importance."),
        code("mems['persona_worthy'] = mems['persona_worthy'].astype(bool)\n"
             "print('persona-worthy rate:', round(mems.persona_worthy.mean(),3))\n"
             "display(mems[mems.persona_worthy][['text','persona_text']].head(4))\n"
             "display(mems[~mems.persona_worthy][['text']].head(4))"),
        md("## Label distribution & validation stats"),
        code("stats = json.load(open(RES/'label_stats.json'))\n"
             "print(json.dumps(stats, indent=2))"),
        code("display(Image(str(FIG/'class_balance.png')))"),
        md("**Takeaway.** The retain class is ~16% of memories (imbalanced). Our "
           "reference-based label is a statistically significant but distinct signal from "
           "human persona-worthiness (odds ratio ≈ 1.9), confirming it captures real "
           "'will-be-used-again' structure rather than restating the annotation."),
    ]
    return c


def nb02():
    c = [
        md("# 02 · Feature Engineering & EDA\n\n"
           "Sixteen interpretable features are computed at *recall time* (start of the new "
           "session) — none peek at future turns. Groups: temporal/structural, speaker, "
           "surface/lexical, semantic (SBERT `all-MiniLM-L6-v2`), and recurrence."),
        code(HEADER),
        code("from features import FEATURE_COLUMNS\n"
             "df = pd.read_parquet(PROC/'features_labeled.parquet')\n"
             "print(len(FEATURE_COLUMNS), 'features'); print(FEATURE_COLUMNS)\n"
             "df[FEATURE_COLUMNS].describe().T.round(3)"),
        md("## Semantic features\n"
           "`sim_to_current_query` = cosine(memory, opening query); "
           "`persona_salience` = max cosine(memory, carried-forward persona sentences)."),
        code("df.groupby('label_retain')[['sim_to_current_query','persona_salience',"
             "'tfidf_max','tfidf_mean','first_person_count']].mean().round(3)"),
        md("## Feature distributions by class"),
        code("display(Image(str(FIG/'feature_distributions.png')))"),
        md("## Correlation structure"),
        code("display(Image(str(FIG/'correlation_heatmap.png')))"),
        md("## Mutual information with the target\n\n"
           "MI(X;Y) = Σ p(x,y) log [ p(x,y) / (p(x)p(y)) ]. Specificity features "
           "(TF-IDF) and persona-salience are most informative; raw query-similarity "
           "ranks lower — an early sign that pure similarity is a weak selector."),
        code("display(Image(str(FIG/'mutual_information.png')))\n"
             "pd.read_csv(RES/'mutual_information.csv', index_col=0).round(4)"),
        md("## PCA (dimensionality analysis)"),
        code("display(Image(str(FIG/'pca_variance.png')))\n"
             "display(Image(str(FIG/'pca_scatter.png')))"),
    ]
    return c


def nb03():
    c = [
        md("# 03 · Modeling & Evaluation\n\n"
           "**Setup.** Episode-level 80/20 train/test split (`GroupShuffleSplit`) so no "
           "conversation spans the split. Hyper-parameters tuned with 5-fold "
           "`StratifiedGroupKFold` optimizing F1. Imbalance handled with SMOTE inside an "
           "`imblearn` pipeline (`StandardScaler → SMOTE → clf`). Baseline: a tuned "
           "threshold on `sim_to_current_query` (the similarity-retrieval incumbent)."),
        code(HEADER),
        md("## Held-out test metrics (all models vs. similarity baseline)"),
        code("m = pd.read_csv(RES/'metrics.csv', index_col=0); m.round(3)"),
        md("## 5-fold cross-validated metrics (train)"),
        code("pd.read_csv(RES/'cv_metrics.csv', index_col=0)"
             "[['cv_f1_mean','cv_f1_std','cv_roc_auc_mean','cv_precision_mean','cv_recall_mean']].round(3)"),
        md("## ROC and Precision-Recall curves"),
        code("display(Image(str(FIG/'roc_curves.png')))\n"
             "display(Image(str(FIG/'pr_curves.png')))"),
        md("## Imbalance strategy ablation (best model)\n\n"
           "SMOTE trades precision for a large recall gain — the right call when the goal "
           "is not to *lose* memories that matter."),
        code("pd.read_csv(RES/'imbalance_ablation.csv', index_col=0).round(3)"),
        md("## Confusion matrix, model comparison, and feature attributions"),
        code("for f in ['confusion_matrix.png','model_comparison.png',"
             "'feature_importance.png','logreg_coefficients.png']:\n"
             "    display(Image(str(FIG/f)))"),
        md("**Result.** Every supervised model beats the similarity baseline "
           "(ROC-AUC ≈ 0.77 vs 0.63; F1 ≈ 0.43 vs 0.34). We select the deployed model by "
           "**cross-validated** F1 (never the test set); that picks **Logistic "
           "Regression**, with Random Forest a near-equal alternative. Features capturing "
           "*specificity* and *persona-salience* — not raw query similarity — drive the "
           "gains. Note the imbalance ablation: SMOTE and class-weighting swing F1/recall "
           "sharply while PR-AUC stays ~0.40, i.e. they recalibrate the threshold, not the "
           "ranking."),
    ]
    return c


def nb04():
    c = [
        md("# 04 · LLM Integration Demo\n\n"
           "We connect the deployed classifier to a local **Llama 3.1 8B** (Ollama) and "
           "answer real MSC queries under four memory regimes — *no-memory*, *store-all*, "
           "*top-K similarity*, and *ML-selected* — scoring each reply 1–5 with an "
           "independent judge model (`dolphin-mistral`).\n\n"
           "This notebook loads the pre-computed comparison "
           "(`results/llm_comparison.csv`) so it runs without a live model; to regenerate, "
           "run `python src/llm_integration.py --num-episodes 8`."),
        code(HEADER),
        md("## Aggregate comparison (32 queries × 8 conversations)"),
        code("s = pd.read_csv(RES/'llm_summary.csv', index_col=0)\n"
             "order=['no_memory','top_k','ml_selected','store_all']\n"
             "s.reindex(order).round(2)"),
        md("**Reading the table.** Judge scores are indistinguishable across regimes "
           "(all 3.7–3.8): every regime is fluent, and a coarse 1–5 judge cannot resolve "
           "finer differences. The decisive gap is efficiency — ML-selection matches "
           "store-all quality using ~1/4 of the context tokens, and adapts the number of "
           "memories per query (unlike fixed top-K)."),
        md("## A qualitative example\n"
           "The same query answered with and without classifier-selected memory."),
        code("demo = json.load(open(RES/'llm_demo.json'))\n"
             "ex = [r for r in demo if r.get('episode')==0 and r['query_id']==0]\n"
             "q = ex[0]['query']; print('QUERY:', q, '\\n')\n"
             "for r in ex:\n"
             "    print(f\"[{r['mode']}]  ({r['context_tokens']} ctx tokens)\")\n"
             "    print('  ', r['response'][:300], '\\n')"),
        md("**Takeaway.** With ML-selected memory the model grounds its answer in the "
           "earlier mention of the book; with no memory it produces a generic reply and can "
           "even invent a shared history. Grounded, selective memory is both more accurate "
           "and cheaper than the alternatives."),
    ]
    return c


def build(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                  "language": "python"}}
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    print(f"Executing {name} ...")
    ep.preprocess(nb, {"metadata": {"path": str(NB)}})
    with open(NB / name, "w") as fh:
        nbf.write(nb, fh)
    print(f"  wrote {name}")


def main():
    NB.mkdir(exist_ok=True)
    build("01_data_exploration.ipynb", nb01())
    build("02_feature_engineering_eda.ipynb", nb02())
    build("03_modeling_evaluation.ipynb", nb03())
    build("04_llm_demo.ipynb", nb04())


if __name__ == "__main__":
    main()
