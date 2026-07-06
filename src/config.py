"""Central configuration for the DREAMER emotion-recognition pipeline.

Keeping every "magic number" in one place makes the experiment easy to tweak
and easy to explain in a report.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data_preprocessed_python" / "DREAMER.mat"
RESULTS_DIR = PROJECT_ROOT / "results"
FEATURE_CACHE = RESULTS_DIR / "features_cache.npz"

# ---------------------------------------------------------------------------
# Signal properties (fixed by the DREAMER / Emotiv EPOC hardware)
# ---------------------------------------------------------------------------
SAMPLING_RATE = 128          # Hz
N_CHANNELS = 14
CHANNEL_NAMES = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

# Only the final N seconds of every film clip are used. The emotional response
# builds up over the clip, so the last segment is the most informative part and
# this also keeps every trial the same length.
SEGMENT_SECONDS = 60

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
# Classic EEG frequency bands (Hz). Gamma is capped below the 64 Hz Nyquist.
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Self-assessment scores run from 1..5. Anything strictly above the threshold
# is treated as the "High" class, everything else as "Low".
LABEL_THRESHOLD = 3
EMOTION_DIMENSIONS = ["Valence", "Arousal", "Dominance"]

# ---------------------------------------------------------------------------
# Modelling / evaluation
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_FOLDS = 5
