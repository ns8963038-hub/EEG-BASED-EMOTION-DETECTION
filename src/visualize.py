"""All plotting. Figures are saved to ``results/`` instead of shown on screen so
the project runs unattended (e.g. on a server or in CI) and the images can go
straight into a report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from . import config
from .features import feature_names
from .model import build_models, make_pipeline

RESULTS = config.RESULTS_DIR


def plot_label_distribution(labels):
    fig, ax = plt.subplots(figsize=(7, 4))
    dims = list(labels.keys())
    low = [int((labels[d] == 0).sum()) for d in dims]
    high = [int((labels[d] == 1).sum()) for d in dims]
    x = np.arange(len(dims))
    ax.bar(x - 0.2, low, width=0.4, label="Low (<=3)", color="#4C72B0")
    ax.bar(x + 0.2, high, width=0.4, label="High (>3)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Number of trials")
    ax.set_title("Class distribution per emotion dimension (414 trials)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "label_distribution.png", dpi=150)
    plt.close(fig)


def plot_confusion(dimension, model_name, cm):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[f"Low {dimension}", f"High {dimension}"],
    )
    disp.plot(cmap="Blues", ax=ax, colorbar=True)
    fig.suptitle(f"Confusion Matrix - {dimension}", y=0.98)
    ax.set_title(f"best model: {model_name}, stratified 5-fold CV", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / f"confusion_{dimension.lower()}.png", dpi=150)
    plt.close(fig)


def plot_model_comparison(rows):
    """Grouped bar chart: stratified accuracy for every model x dimension."""
    models = list(build_models().keys())
    dims = config.EMOTION_DIMENSIONS
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(models)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for i, model in enumerate(models):
        accs, errs = [], []
        for dim in dims:
            row = next(r for r in rows if r["model"] == model
                       and r["dimension"] == dim and r["scheme"] == "stratified")
            accs.append(row["accuracy"] * 100)
            errs.append(row["accuracy_std"] * 100)
        x = np.arange(len(dims)) + i * width
        ax.bar(x, accs, width=width, yerr=errs, capsize=3,
               label=model, color=colors[i % len(colors)])

    ax.set_xticks(np.arange(len(dims)) + width * (len(models) - 1) / 2)
    ax.set_xticklabels(dims)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(50, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_title("Model comparison - stratified 5-fold accuracy")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(X, y, dimension, top_n=20):
    """Top feature importances from a RandomForest fit on the full data."""
    pipe = make_pipeline(build_models()["RandomForest"])
    pipe.fit(X, y)
    importances = pipe.named_steps["clf"].feature_importances_
    names = np.array(feature_names())
    order = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(order))[::-1], importances[order], color="#55A868")
    ax.set_yticks(range(len(order))[::-1])
    ax.set_yticklabels(names[order], fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features (RandomForest, {dimension})")
    fig.tight_layout()
    fig.savefig(RESULTS / f"feature_importance_{dimension.lower()}.png", dpi=150)
    plt.close(fig)
