"""Model definitions and the leakage-safe training pipeline.

The single most important fix over the original script is *where* SMOTE runs.
The original oversampled the whole dataset and only then split into train/test,
so synthetic copies of test samples leaked into training and inflated the
accuracy. Here SMOTE lives inside an imbalanced-learn ``Pipeline``; when that
pipeline is handed to cross-validation, SMOTE and the scaler are re-fit on the
training fold *only*, and the held-out fold stays untouched.
"""

from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config


def build_models():
    """Return the dictionary of classifiers we benchmark against each other."""
    rs = config.RANDOM_STATE
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=rs, n_jobs=-1
        ),
        "SVM-RBF": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", random_state=rs,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=rs),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=rs
        ),
    }


def make_pipeline(classifier):
    """Wrap a classifier so SMOTE + scaling happen only on training folds."""
    return Pipeline(
        steps=[
            ("smote", SMOTE(random_state=config.RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )
