"""Train and compare five classifiers for memory-importance prediction.

Experimental setup
------------------
* **Split**: episodes (not memories) are split 80/20 into train/test with
  :class:`GroupShuffleSplit`, so all memories from one conversation stay on the
  same side -- preventing context leakage across the split.
* **Model selection**: hyper-parameters are tuned with
  :class:`StratifiedGroupKFold` (5-fold, grouped by episode) optimising F1.
* **Imbalance**: the minority (retain) class is ~16%. Each model is trained
  inside an imbalanced-learn pipeline ``StandardScaler -> SMOTE -> clf``. SMOTE
  is fit on training folds only. A class-weight / no-resampling ablation is run
  on the best model.
* **Baseline**: a similarity-threshold classifier (predict retain when
  ``sim_to_current_query`` exceeds a tuned threshold) stands in for the
  incumbent "similarity-based retrieval" approach.

Outputs: fitted best model + scaler to ``models/``, metric tables to
``results/``, and an evaluation bundle consumed by ``evaluate.py``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SEED = 42

SCORERS = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]

MODELS_AND_GRIDS = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, random_state=SEED),
        {"clf__C": [0.1, 1.0, 10.0]},
    ),
    "kNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors": [15, 31, 51], "clf__weights": ["uniform", "distance"]},
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__max_depth": [4, 8, None], "clf__min_samples_leaf": [1, 20]},
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        {"clf__max_depth": [None, 12], "clf__min_samples_leaf": [1, 5]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(algorithm="SAMME", random_state=SEED),
        {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.5, 1.0]},
    ),
}


def _metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
    }


def similarity_baseline(Xtr, ytr, Xte, yte, col="sim_to_current_query") -> dict:
    """Tune a threshold on the similarity feature to maximise train F1."""
    s_tr, s_te = Xtr[col].values, Xte[col].values
    best_t, best_f1 = 0.5, -1.0
    for t in np.quantile(s_tr, np.linspace(0.5, 0.99, 50)):
        f1 = f1_score(ytr, (s_tr >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    y_pred = (s_te >= best_t).astype(int)
    m = _metrics(yte, y_pred, s_te)
    m["threshold"] = float(best_t)
    return m


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    df = pd.read_parquet(PROC / "features_labeled.parquet")
    X = df[FEATURE_COLUMNS]
    y = df["label_retain"].astype(int).values
    groups = df["episode_id"].values
    print(f"Dataset: {len(df)} memories, {df['episode_id'].nunique()} episodes, "
          f"positive rate {y.mean():.3f}")

    # episode-level train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, te_idx = next(gss.split(X, y, groups))
    Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    gtr = groups[tr_idx]
    print(f"Train {len(tr_idx)} / Test {len(te_idx)} memories "
          f"({pd.Series(gtr).nunique()} / {pd.Series(groups[te_idx]).nunique()} episodes)")

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows, cv_rows, bundle_probs, bundle_preds = [], [], {}, {}
    fitted = {}

    # similarity-retrieval baseline
    base = similarity_baseline(Xtr, ytr, Xte, yte)
    rows.append({"model": "SimilarityBaseline", **{k: base[k] for k in
                 ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]}})
    bundle_probs["SimilarityBaseline"] = Xte["sim_to_current_query"].values
    bundle_preds["SimilarityBaseline"] = (Xte["sim_to_current_query"].values >= base["threshold"]).astype(int)
    print(f"\nSimilarity baseline (thr={base['threshold']:.3f}): "
          f"F1={base['f1']:.3f} ROC-AUC={base['roc_auc']:.3f}")

    for name, (clf, grid) in MODELS_AND_GRIDS.items():
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", clf),
        ])
        gs = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr, groups=gtr)
        best = gs.best_estimator_
        fitted[name] = best

        # grouped CV metrics on the training data (mean over folds)
        cvres = cross_validate(best, Xtr, ytr, groups=gtr, cv=cv, scoring=SCORERS, n_jobs=-1)
        cv_row = {"model": name}
        for s in SCORERS:
            cv_row[f"cv_{s}_mean"] = cvres[f"test_{s}"].mean()
            cv_row[f"cv_{s}_std"] = cvres[f"test_{s}"].std()
        cv_rows.append(cv_row)

        # held-out test metrics
        y_prob = best.predict_proba(Xte)[:, 1]
        y_pred = best.predict(Xte)
        m = _metrics(yte, y_pred, y_prob)
        rows.append({"model": name, **m})
        bundle_probs[name] = y_prob
        bundle_preds[name] = y_pred
        print(f"{name:20s} best={gs.best_params_}  test F1={m['f1']:.3f} "
              f"ROC-AUC={m['roc_auc']:.3f} PR-AUC={m['pr_auc']:.3f}")

    metrics = pd.DataFrame(rows).set_index("model").round(4)
    cv_metrics = pd.DataFrame(cv_rows).set_index("model").round(4)
    metrics.to_csv(RESULTS / "metrics.csv")
    cv_metrics.to_csv(RESULTS / "cv_metrics.csv")

    # Select the best model by CROSS-VALIDATED F1 (never the test set). The test
    # set is touched only once, to report the chosen model's performance.
    best_name = cv_metrics["cv_f1_mean"].idxmax()
    best_model = fitted[best_name]
    print(f"\nBest model (by CV F1): {best_name} "
          f"(CV F1={cv_metrics.loc[best_name,'cv_f1_mean']:.3f}, "
          f"test F1={metrics.loc[best_name,'f1']:.3f})")

    # imbalance ablation on the best model type
    ablation = imbalance_ablation(MODELS_AND_GRIDS[best_name][0], Xtr, ytr, Xte, yte, cv, gtr)
    ablation.to_csv(RESULTS / "imbalance_ablation.csv")
    print("\nImbalance ablation (best model type):")
    print(ablation.round(4).to_string())

    # persist best model (full pipeline incl. scaler) + metadata
    with open(MODELS / "best_model.pkl", "wb") as fh:
        pickle.dump({"name": best_name, "pipeline": best_model,
                     "features": FEATURE_COLUMNS}, fh)

    cm = confusion_matrix(yte, bundle_preds[best_name])
    bundle = {
        "feature_columns": FEATURE_COLUMNS,
        "X_train": Xtr, "y_train": ytr, "groups_train": gtr,
        "X_test": Xte, "y_test": yte,
        "probs": bundle_probs, "preds": bundle_preds,
        "best_name": best_name, "confusion_matrix": cm.tolist(),
        "similarity_threshold": base["threshold"],
    }
    with open(RESULTS / "eval_bundle.pkl", "wb") as fh:
        pickle.dump(bundle, fh)

    with open(RESULTS / "run_summary.json", "w") as fh:
        json.dump({
            "n_memories": int(len(df)),
            "n_episodes": int(df["episode_id"].nunique()),
            "positive_rate": float(y.mean()),
            "best_model": best_name,
            "test_metrics": metrics.loc[best_name].to_dict(),
            "similarity_baseline": {k: base[k] for k in ["f1", "roc_auc", "pr_auc", "threshold"]},
        }, fh, indent=2)
    print(f"\nSaved models/ and results/ artefacts.")


def imbalance_ablation(base_clf, Xtr, ytr, Xte, yte, cv, gtr) -> pd.DataFrame:
    from sklearn.base import clone

    out = []
    strategies = {"none": None, "smote": "smote"}
    if "class_weight" in base_clf.get_params():
        strategies["class_weight"] = "cw"
    for label, strat in strategies.items():
        clf = clone(base_clf)
        steps = [("scaler", StandardScaler())]
        if strat == "smote":
            steps.append(("smote", SMOTE(random_state=SEED)))
        if strat == "cw":
            clf.set_params(class_weight="balanced")
        steps.append(("clf", clf))
        pipe = ImbPipeline(steps)
        pipe.fit(Xtr, ytr)
        y_prob = pipe.predict_proba(Xte)[:, 1]
        y_pred = pipe.predict(Xte)
        out.append({"strategy": label, **_metrics(yte, y_pred, y_prob)})
    return pd.DataFrame(out).set_index("strategy")


if __name__ == "__main__":
    main()
