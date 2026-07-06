"""Turn a raw EEG recording into a fixed-length feature vector.

The original project used only the per-channel mean and standard deviation
(28 time-domain numbers). Emotion-related information in EEG lives mostly in the
*frequency* domain, so here we also compute the power in five classic bands
(delta/theta/alpha/beta/gamma) for every channel.

Crucially, band power is expressed **relative to the subject's own baseline**
recording (the neutral clip DREAMER records before each film). Taking
``log(stimulus_power / baseline_power)`` cancels out the large, person-specific
differences in absolute EEG amplitude, which is what makes a model trained on
some people work on a new person. That gives ``14 channels x 5 bands = 70``
frequency features, for 98 features in total.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import welch

from . import config

_EPS = 1e-8


def feature_names():
    """Human-readable name for every column (used by importance plots)."""
    names = [f"{ch}_mean" for ch in config.CHANNEL_NAMES]
    names += [f"{ch}_std" for ch in config.CHANNEL_NAMES]
    for band in config.FREQUENCY_BANDS:
        names += [f"{ch}_{band}" for ch in config.CHANNEL_NAMES]
    return names


def _last_segment(signal):
    """Keep only the final ``SEGMENT_SECONDS`` of a recording."""
    n_keep = config.SEGMENT_SECONDS * config.SAMPLING_RATE
    if signal.shape[0] > n_keep:
        return signal[-n_keep:]
    return signal


def _band_powers(signal):
    """Absolute power in each frequency band for every channel.

    Uses Welch's method (which detrends each segment internally, so no manual
    DC removal is needed). ``signal`` is ``[n_samples, n_channels]``; the result
    is ``[n_bands, n_channels]``.
    """
    nperseg = min(256, signal.shape[0])  # ~2 s windows at 128 Hz
    freqs, psd = welch(signal, fs=config.SAMPLING_RATE, nperseg=nperseg, axis=0)

    powers = []
    for low, high in config.FREQUENCY_BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        powers.append(np.trapezoid(psd[mask], freqs[mask], axis=0))
    return np.asarray(powers)  # [n_bands, n_channels]


def extract_features(eeg, baseline):
    """Feature vector for one trial.

    Parameters
    ----------
    eeg : ndarray [n_samples, 14]   stimulus (film-clip) recording
    baseline : ndarray [n_samples, 14]   neutral baseline recording
    """
    stim = _last_segment(np.asarray(eeg, dtype=np.float64))
    base = _last_segment(np.asarray(baseline, dtype=np.float64))

    # Time-domain statistics on the raw stimulus signal.
    mean_features = stim.mean(axis=0)
    std_features = stim.std(axis=0)

    # Baseline-relative log band power: how much each band changed from the
    # subject's neutral state. Shape [n_bands, n_channels] -> flat [70].
    stim_bp = _band_powers(stim)
    base_bp = _band_powers(base)
    rel_band = np.log((stim_bp + _EPS) / (base_bp + _EPS)).ravel()

    return np.concatenate([mean_features, std_features, rel_band])


def build_feature_matrix(trials):
    """Vectorise a list of trials into ``X`` and the three label vectors.

    Returns
    -------
    X : ndarray [n_trials, n_features]
    labels : dict of {dimension_name: binary ndarray}
    groups : ndarray [n_trials]   subject id of each trial (for subject-independent CV)
    """
    X, groups = [], []
    valence, arousal, dominance = [], [], []

    for trial in trials:
        X.append(extract_features(trial.eeg, trial.baseline))
        groups.append(trial.subject)
        valence.append(1 if trial.valence > config.LABEL_THRESHOLD else 0)
        arousal.append(1 if trial.arousal > config.LABEL_THRESHOLD else 0)
        dominance.append(1 if trial.dominance > config.LABEL_THRESHOLD else 0)

    X = np.asarray(X)
    labels = {
        "Valence": np.asarray(valence),
        "Arousal": np.asarray(arousal),
        "Dominance": np.asarray(dominance),
    }
    return X, labels, np.asarray(groups)
