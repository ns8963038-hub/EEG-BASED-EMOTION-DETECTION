# EEG-Based Emotion Recognition on the DREAMER Dataset

Classifying human emotions from raw EEG signals along three affective
dimensions — **Valence**, **Arousal**, and **Dominance** — using classical
machine learning. Built on the public
[DREAMER](https://zenodo.org/record/546113) dataset (23 participants, 18 film
clips each, 14-channel Emotiv EPOC headset @ 128 Hz).

Each dimension is framed as a binary problem: a self-assessment score of
**4–5 → High**, **1–3 → Low**.

---

## Results

Five-fold cross-validated accuracy (mean over folds). Two evaluation schemes are
reported:

- **Stratified** — trials are split at random with class balance preserved.
- **Subject-independent** — no participant appears in both train and test
  (`GroupKFold` by subject). This is the honest measure of generalisation to a
  *new person*, and is expected to be lower.

Accuracy is the mean over 5 folds; chance ≈ 50%. Best model shown per cell;
full per-model metrics (accuracy, F1, ROC-AUC ± std) are in `results/metrics.csv`.

| Emotion   | Stratified 5-fold        | Subject-independent           |
|-----------|--------------------------|-------------------------------|
| Valence   | **60.4 %** (RandomForest) | **61.4 %** (HistGradientBoosting) |
| Arousal   | **55.8 %** (RandomForest) | **51.2 %** (RandomForest)     |
| Dominance | **59.4 %** (RandomForest) | **58.2 %** (RandomForest)     |

`RandomForest` was the most consistent model across dimensions. These are
*cross-subject* numbers on binary High/Low labels — modest, but honest.
Valence and Dominance generalise to unseen participants (58–61 %); Arousal is
close to chance, which is a genuine and well-documented difficulty of the
DREAMER dataset rather than a bug. The previous version of this project reported
much higher accuracy only because oversampling leaked test data into training
and it evaluated on 6 held-out samples.

Figures generated in `results/`:

| File | Shows |
|------|-------|
| `model_comparison.png`        | Accuracy of all four models across the three dimensions |
| `confusion_valence.png` etc.  | Confusion matrix for the best model per dimension |
| `feature_importance_*.png`    | Which channels/bands the RandomForest relied on |
| `label_distribution.png`      | Class balance across the 414 trials |
| `metrics.csv`                 | Full metric table (accuracy, F1, ROC-AUC ± std) |

---

## What this project improves over a naive baseline

This started as a single 150-line script. The rewrite fixes two correctness
problems that quietly inflate results, and adds real EEG feature engineering:

1. **Uses the whole dataset.** The original indexing bug read only the *first*
   film clip per participant, training on **23 trials instead of 414**. The
   loader now walks every trial in the `18 × 1` stimulus cell-array.

2. **No data leakage.** The original applied SMOTE oversampling to the *entire*
   dataset **before** the train/test split, so synthetic copies of test samples
   leaked into training. SMOTE now lives inside an `imblearn` pipeline and is
   re-fit on the training fold only, within cross-validation.

3. **Frequency-domain features.** Emotion-related EEG information is mostly
   spectral. On top of the per-channel mean/std, the pipeline computes log band
   power in five classic bands (delta, theta, alpha, beta, gamma) via Welch's
   method — 14 channels × 5 bands = 70 extra features (98 total).

4. **Honest evaluation.** Results come from 5-fold cross-validation (not one
   arbitrary split), across four models, reporting accuracy, F1 and ROC-AUC —
   plus the subject-independent scheme that most coursework omits.

5. **Reproducible & shareable.** Modular `src/` package, pinned
   `requirements.txt`, cached features, headless figures saved to disk, and a
   `.gitignore` that keeps the 432 MB dataset out of version control.

---

## Project structure

```
.
├── main.py                 # entry point: load → features → evaluate → plots
├── requirements.txt
├── src/
│   ├── config.py           # all constants (bands, sampling rate, folds, ...)
│   ├── data_loader.py      # correct DREAMER .mat extraction (all 414 trials)
│   ├── features.py         # time-domain + band-power feature extraction
│   ├── model.py            # model zoo + leakage-safe SMOTE pipeline
│   ├── evaluate.py         # stratified & subject-independent cross-validation
│   └── visualize.py        # confusion matrices, comparisons, importances
├── results/                # generated metrics.csv + figures
└── data_preprocessed_python/
    └── DREAMER.mat         # NOT in git — download separately (see below)
```

## Feature vector (98 features per trial)

| Group | Per channel | Count |
|-------|-------------|-------|
| Time-domain mean       | 14 | 14 |
| Time-domain std. dev.  | 14 | 14 |
| Log band power × 5 bands | 14 × 5 | 70 |

Features are computed on the **last 60 seconds** of each clip (the most
emotionally salient part) after DC-offset removal.

## Models compared

RandomForest · SVM (RBF) · HistGradientBoosting · LogisticRegression — each
wrapped in `SMOTE → StandardScaler → classifier` so preprocessing never sees the
held-out fold.

---

## Getting started — run it on your own machine

Follow these steps in order. Anyone who clones the repo can reproduce every
result on the table above.

### Prerequisites

- **Python 3.10 or newer** (developed and tested on 3.14) — check with
  `python3 --version`
- **git**
- **~1 GB free disk** (the dataset is ~432 MB)
- No GPU needed — it runs on any CPU in about 10 seconds once features are cached.

### Step 1 — Clone the repository

```bash
git clone https://github.com/ns8963038-hub/EEG-BASED-EMOTION-DETECTION.git
cd EEG-BASED-EMOTION-DETECTION
```

### Step 2 — Download the DREAMER dataset (required)

The dataset is **not** in this repo (it is ~432 MB — above GitHub's limit — and
has its own license), so you download it once yourself:

1. Open the official source: <https://zenodo.org/record/546113>
2. Agree to the license and download **`DREAMER.mat`**.
3. Put the file inside a `data_preprocessed_python/` folder so the final path is
   exactly `data_preprocessed_python/DREAMER.mat`:

```bash
mkdir -p data_preprocessed_python
mv ~/Downloads/DREAMER.mat data_preprocessed_python/   # adjust path as needed
```

> Citation: S. Katsigiannis and N. Ramzan, "DREAMER: A Database for Emotion
> Recognition Through EEG and ECG Signals from Wireless Low-cost Off-the-Shelf
> Devices," *IEEE Journal of Biomedical and Health Informatics*, 2018.

### Step 3 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

### Step 4 — Install the dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5 — Run the project

```bash
python main.py
```

The **first** run reads the `.mat` file and extracts features (~1–2 minutes,
mostly loading the 432 MB file), then caches them. **Later** runs reuse the cache
and finish in ~10 seconds. To force a fresh feature extraction:

```bash
python main.py --no-cache
```

### Step 6 — View the results

Everything is written to the **`results/`** folder, and a summary table is also
printed in the terminal:

- `metrics.csv` — accuracy / F1 / ROC-AUC for every model and scheme
- `model_comparison.png` — accuracy of all four models across the three dimensions
- `confusion_valence.png`, `confusion_arousal.png`, `confusion_dominance.png`
- `feature_importance_*.png` — the channels/bands the model relied on
- `label_distribution.png` — class balance across the 414 trials

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: ...DREAMER.mat` | The dataset isn't in `data_preprocessed_python/`. Recheck Step 2 — the filename must be exactly `DREAMER.mat`. |
| `ModuleNotFoundError` | The virtual environment isn't active, or Step 4 didn't finish. Re-activate it and re-run `pip install -r requirements.txt`. |
| `python3: command not found` (Windows) | Use `py` instead of `python3`. |
| No plot windows pop up | Intentional — figures are **saved** to `results/` instead of shown, so it works on any machine or server. Open the PNGs from there. |

---

## Limitations & future work

- Trial-level cross-validation lets the same subject appear in train and test;
  the **subject-independent** numbers are the ones to trust for real-world use.
- Only classical ML is used. Deeper features (differential entropy, connectivity
  measures) or a compact CNN/LSTM on the raw signal are natural next steps.
- Binary High/Low labels simplify the 1–5 scale; a 3-class or regression
  formulation would preserve more information.
- ECG signals in DREAMER are unused and could be fused with EEG.
