"""Generate all figures for the report from the trained-model artefacts.

Reads ``results/eval_bundle.pkl`` (train/test split, per-model probabilities and
predictions) and ``data/processed/features_labeled.parquet`` (for descriptive
EDA) and writes PNGs to ``reports/figures/``.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

from features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIG = ROOT / "reports" / "figures"

BINARY = {"role_speaker1", "has_number", "has_proper_noun", "is_question"}
sns.set_theme(style="whitegrid", context="talk")


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["label_retain"].value_counts().sort_index()
    ax.bar(["discard (0)", "retain (1)"], counts.values, color=["#8c8c8c", "#d1495b"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v/len(df):.1%})", ha="center", va="bottom")
    ax.set_title("Class balance (target: label_retain)")
    ax.set_ylabel("memories")
    _save(fig, "class_balance.png")


def feature_distributions(df):
    feats = ["sim_to_current_query", "persona_salience", "tfidf_max",
             "first_person_count", "sentence_length", "access_frequency"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, f in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x="label_retain", y=f, ax=ax, palette=["#8c8c8c", "#d1495b"],
                    showfliers=False)
        ax.set_xlabel("")
        ax.set_xticklabels(["discard", "retain"])
        ax.set_title(f)
    fig.suptitle("Feature distributions by class", y=1.02)
    _save(fig, "feature_distributions.png")


def correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df[FEATURE_COLUMNS + ["label_retain"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                annot_kws={"size": 8}, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature correlation matrix (incl. target)")
    _save(fig, "correlation_heatmap.png")


def mutual_information(df):
    X = df[FEATURE_COLUMNS].values
    y = df["label_retain"].values
    discrete = [c in BINARY for c in FEATURE_COLUMNS]
    mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=42)
    order = np.argsort(mi)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh([FEATURE_COLUMNS[i] for i in order], mi[order], color="#2e86ab")
    ax.set_title("Mutual information with target")
    ax.set_xlabel("MI (nats)")
    _save(fig, "mutual_information.png")
    return pd.Series(mi, index=FEATURE_COLUMNS).sort_values(ascending=False)


def pca_plots(df):
    X = StandardScaler().fit_transform(df[FEATURE_COLUMNS].values)
    y = df["label_retain"].values
    pca = PCA(n_components=len(FEATURE_COLUMNS)).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, len(cum) + 1), cum, "o-", color="#2e86ab")
    ax.axhline(0.9, ls="--", color="grey")
    ax.set_xlabel("components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_title("PCA scree (standardized features)")
    _save(fig, "pca_variance.png")

    proj = pca.transform(X)[:, :2]
    fig, ax = plt.subplots(figsize=(8, 6))
    idx = np.random.RandomState(0).permutation(len(y))[:4000]
    ax.scatter(proj[idx][y[idx] == 0, 0], proj[idx][y[idx] == 0, 1], s=6, alpha=0.3,
               label="discard", color="#8c8c8c")
    ax.scatter(proj[idx][y[idx] == 1, 0], proj[idx][y[idx] == 1, 1], s=6, alpha=0.4,
               label="retain", color="#d1495b")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA projection (2 components)")
    ax.legend()
    _save(fig, "pca_scatter.png")


def roc_pr_curves(bundle):
    y = bundle["y_test"]
    fig, ax = plt.subplots(figsize=(8, 7))
    for name, prob in bundle["probs"].items():
        fpr, tpr, _ = roc_curve(y, prob)
        style = "--" if name == "SimilarityBaseline" else "-"
        ax.plot(fpr, tpr, style, label=name, lw=2)
    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (held-out episodes)")
    ax.legend(fontsize=11)
    _save(fig, "roc_curves.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    for name, prob in bundle["probs"].items():
        prec, rec, _ = precision_recall_curve(y, prob)
        style = "--" if name == "SimilarityBaseline" else "-"
        ax.plot(rec, prec, style, label=name, lw=2)
    ax.axhline(y.mean(), ls=":", color="grey", label=f"prevalence={y.mean():.2f}")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-Recall curves")
    ax.legend(fontsize=11)
    _save(fig, "pr_curves.png")


def confusion(bundle):
    cm = np.array(bundle["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["discard", "retain"], yticklabels=["discard", "retain"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix — {bundle['best_name']}")
    _save(fig, "confusion_matrix.png")


def model_comparison():
    m = pd.read_csv(RESULTS / "metrics.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(11, 6))
    m[["f1", "roc_auc", "pr_auc"]].plot.bar(ax=ax, color=["#d1495b", "#2e86ab", "#edae49"])
    ax.set_ylabel("score")
    ax.set_title("Model comparison on held-out test set")
    ax.legend(["F1", "ROC-AUC", "PR-AUC"])
    ax.set_xticklabels(m.index, rotation=30, ha="right")
    _save(fig, "model_comparison.png")


def importances_and_coeffs(bundle):
    from sklearn.ensemble import RandomForestClassifier

    Xtr, ytr = bundle["X_train"], bundle["y_train"]

    # Random Forest importances (fit here so the figure is valid regardless of
    # which model is deployed) as an impurity-based feature-attribution view.
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                random_state=42, n_jobs=-1).fit(Xtr, ytr)
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(imp.index, imp.values, color="#06a77d")
    ax.set_title("Random Forest feature importances")
    _save(fig, "feature_importance.png")

    # Standardized logistic-regression coefficients for interpretability
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scaler.transform(Xtr), ytr)
    coef = pd.Series(lr.coef_[0], index=FEATURE_COLUMNS).sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["#d1495b" if c > 0 else "#2e86ab" for c in coef.values]
    ax.barh(coef.index, coef.values, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Logistic-regression coefficients (standardized)")
    _save(fig, "logreg_coefficients.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC / "features_labeled.parquet")
    with open(RESULTS / "eval_bundle.pkl", "rb") as fh:
        bundle = pickle.load(fh)

    print("Generating figures ...")
    class_balance(df)
    feature_distributions(df)
    correlation_heatmap(df)
    mi = mutual_information(df)
    pca_plots(df)
    roc_pr_curves(bundle)
    confusion(bundle)
    model_comparison()
    importances_and_coeffs(bundle)

    mi.round(4).to_csv(RESULTS / "mutual_information.csv", header=["mutual_information"])
    print("\nTop features by mutual information:")
    print(mi.head(8).round(4).to_string())
    print(f"\nAll figures written to {FIG}")


if __name__ == "__main__":
    main()
