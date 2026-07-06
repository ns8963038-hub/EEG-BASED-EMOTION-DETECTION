"""DREAMER EEG Emotion Recognition — main entry point.

Pipeline
--------
1. Load all 414 EEG trials from ``DREAMER.mat``.
2. Extract time-domain + frequency-band-power features (98 per trial).
3. Cross-validate four classifiers on Valence / Arousal / Dominance under both
   a stratified and a subject-independent scheme, with SMOTE applied safely
   inside each training fold.
4. Save a metrics table and result figures to ``results/``.

Run with::

    python main.py                 # full run (uses cached features if present)
    python main.py --no-cache      # recompute features from the .mat file
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np

from src import config, evaluate, visualize
from src.data_loader import load_trials
from src.features import build_feature_matrix


def _feature_signature():
    """A short string that changes whenever feature settings change, so a stale
    cache is never silently reused after editing config/features."""
    return (f"seg={config.SEGMENT_SECONDS};fs={config.SAMPLING_RATE};"
            f"thr={config.LABEL_THRESHOLD};bands={sorted(config.FREQUENCY_BANDS.items())}")


def get_features(use_cache=True):
    """Load features from cache if available, otherwise extract and cache them."""
    if use_cache and config.FEATURE_CACHE.exists():
        cache = np.load(config.FEATURE_CACHE, allow_pickle=True)
        cached_sig = str(cache["signature"]) if "signature" in cache else ""
        if cached_sig == _feature_signature():
            print(f"Loading cached features from {config.FEATURE_CACHE.name} ...")
            X = cache["X"]
            groups = cache["groups"]
            labels = {dim: cache[f"y_{dim}"] for dim in config.EMOTION_DIMENSIONS}
            return X, labels, groups
        print("Feature settings changed since the cache was built — recomputing.")

    print("Loading DREAMER dataset (this reads a ~432 MB .mat file)...")
    trials = load_trials()
    print(f"  Extracted {len(trials)} trials "
          f"from {len({t.subject for t in trials})} participants.")

    print("Extracting features (time-domain + band power)...")
    X, labels, groups = build_feature_matrix(trials)
    print(f"  Feature matrix: {X.shape[0]} trials x {X.shape[1]} features.")

    config.RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(
        config.FEATURE_CACHE,
        X=X,
        groups=groups,
        signature=_feature_signature(),
        **{f"y_{dim}": labels[dim] for dim in labels},
    )
    return X, labels, groups


def write_metrics_csv(rows):
    path = config.RESULTS_DIR / "metrics.csv"
    fieldnames = ["dimension", "model", "scheme", "accuracy", "accuracy_std",
                  "f1", "f1_std", "roc_auc", "roc_auc_std"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"\nSaved detailed metrics to {path}")


def print_summary(rows):
    print("\n" + "=" * 74)
    print("SUMMARY — best model per dimension (accuracy, mean over 5 folds)")
    print("=" * 74)
    print(f"{'Dimension':<12}{'Scheme':<22}{'Best model':<22}{'Accuracy':>10}")
    print("-" * 74)
    for dim in config.EMOTION_DIMENSIONS:
        for scheme in ("stratified", "subject_independent"):
            subset = [r for r in rows if r["dimension"] == dim and r["scheme"] == scheme]
            best = max(subset, key=lambda r: r["accuracy"])
            print(f"{dim:<12}{scheme:<22}{best['model']:<22}"
                  f"{best['accuracy'] * 100:>8.2f} %")
    print("=" * 74)


def main():
    parser = argparse.ArgumentParser(description="DREAMER EEG emotion recognition")
    parser.add_argument("--no-cache", action="store_true",
                        help="recompute features instead of using the cache")
    args = parser.parse_args()

    start = time.time()
    config.RESULTS_DIR.mkdir(exist_ok=True)

    X, labels, groups = get_features(use_cache=not args.no_cache)

    all_rows = []
    for dim in config.EMOTION_DIMENSIONS:
        print(f"\nEvaluating {dim} ...")
        y = labels[dim]
        n_high = int(y.sum())
        print(f"  Class balance: Low={len(y) - n_high}, High={n_high}")
        rows = evaluate.evaluate_dimension(X, y, groups, dim)
        all_rows.extend(rows)

        # Figures for this dimension.
        model_name, cm = evaluate.best_model_confusion(X, y, dim, rows)
        visualize.plot_confusion(dim, model_name, cm)
        visualize.plot_feature_importance(X, y, dim)

    # Dataset / cross-model figures.
    visualize.plot_label_distribution(labels)
    visualize.plot_model_comparison(all_rows)

    write_metrics_csv(all_rows)
    print_summary(all_rows)

    print(f"\nAll figures written to {config.RESULTS_DIR}")
    print(f"Done in {time.time() - start:.1f} s.")


if __name__ == "__main__":
    main()
