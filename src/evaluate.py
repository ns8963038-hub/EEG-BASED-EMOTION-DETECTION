"""Cross-validated evaluation of every model on every emotion dimension.

Two evaluation schemes are reported:

* **Stratified k-fold** – the standard trial-level split. Class balance is kept
  in every fold. This is the number usually quoted in coursework.
* **Subject-independent (GroupKFold by subject)** – no participant appears in
  both train and test. This is the honest measure of how well the model would
  generalise to a *new person*, and it is almost always lower. Reporting both
  shows the gap instead of hiding it.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    GroupKFold,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from . import config
from .model import build_models, make_pipeline

SCORING = ["accuracy", "f1", "roc_auc"]


def _summarise(scores):
    """Mean +/- std for each metric from a ``cross_validate`` result."""
    out = {}
    for metric in SCORING:
        values = scores[f"test_{metric}"]
        out[metric] = (float(np.mean(values)), float(np.std(values)))
    return out


def evaluate_dimension(X, y, groups, dimension):
    """Benchmark every model on one emotion dimension under both CV schemes.

    Returns a list of result rows (one per model x scheme).
    """
    rows = []
    strat = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)
    group = GroupKFold(n_splits=config.N_FOLDS)

    for model_name, classifier in build_models().items():
        pipe = make_pipeline(classifier)

        strat_scores = cross_validate(pipe, X, y, cv=strat, scoring=SCORING, n_jobs=-1)
        group_scores = cross_validate(
            pipe, X, y, groups=groups, cv=group, scoring=SCORING, n_jobs=-1
        )

        for scheme, scores in [("stratified", strat_scores), ("subject_independent", group_scores)]:
            summary = _summarise(scores)
            rows.append(
                {
                    "dimension": dimension,
                    "model": model_name,
                    "scheme": scheme,
                    "accuracy": summary["accuracy"][0],
                    "accuracy_std": summary["accuracy"][1],
                    "f1": summary["f1"][0],
                    "f1_std": summary["f1"][1],
                    "roc_auc": summary["roc_auc"][0],
                    "roc_auc_std": summary["roc_auc"][1],
                }
            )
    return rows


def best_model_confusion(X, y, dimension, rows):
    """Confusion matrix for the best stratified model on this dimension.

    Uses ``cross_val_predict`` so every prediction comes from a fold where that
    sample was held out (no leakage), then aggregates into one matrix.
    """
    candidates = [r for r in rows if r["dimension"] == dimension and r["scheme"] == "stratified"]
    best = max(candidates, key=lambda r: r["accuracy"])
    classifier = build_models()[best["model"]]
    pipe = make_pipeline(classifier)

    strat = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True,
                            random_state=config.RANDOM_STATE)
    y_pred = cross_val_predict(pipe, X, y, cv=strat, n_jobs=-1)
    cm = confusion_matrix(y, y_pred)
    return best["model"], cm
