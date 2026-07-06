"""Load raw EEG trials from the DREAMER ``.mat`` file.

DREAMER stores its data as nested MATLAB structs/cell-arrays. The important
detail is that every one of the 23 participants watched 18 film clips, so the
dataset contains ``23 x 18 = 414`` trials in total. The stimulus recordings for
a participant live in an ``18 x 1`` cell array and must be indexed row by row.
"""

from __future__ import annotations

import numpy as np
from scipy.io import loadmat

from . import config


class Trial:
    """A single film-clip recording plus its self-assessment labels."""

    __slots__ = ("subject", "video", "eeg", "baseline",
                 "valence", "arousal", "dominance")

    def __init__(self, subject, video, eeg, baseline, valence, arousal, dominance):
        self.subject = subject          # 0-based participant index
        self.video = video              # 0-based clip index
        self.eeg = eeg                  # [n_samples, 14] stimulus recording
        self.baseline = baseline        # [n_samples, 14] neutral baseline
        self.valence = int(valence)     # 1..5
        self.arousal = int(arousal)     # 1..5
        self.dominance = int(dominance)  # 1..5


def load_trials(mat_path=config.DATA_PATH):
    """Return a list of :class:`Trial` objects for every subject/clip.

    This is the function that fixes the main bug in the original script: the old
    code read ``stimuli[0, 0][0]`` which is only the *first row* of the trial
    cell-array, so it kept a single trial per subject (23 samples). Here we walk
    every row and recover all 414 trials.
    """
    mat_path = str(mat_path)
    data = loadmat(mat_path)
    dreamer = data["DREAMER"][0, 0]
    participants = dreamer["Data"][0]

    trials: list[Trial] = []
    for subject_idx, participant in enumerate(participants):
        eeg_struct = participant["EEG"][0, 0]
        stimuli = eeg_struct["stimuli"][0, 0]      # shape (18, 1), object cells
        baseline = eeg_struct["baseline"][0, 0]    # shape (18, 1), object cells

        valence = participant["ScoreValence"][0, 0].ravel()
        arousal = participant["ScoreArousal"][0, 0].ravel()
        dominance = participant["ScoreDominance"][0, 0].ravel()

        n_videos = stimuli.shape[0]
        for video_idx in range(n_videos):
            eeg = np.asarray(stimuli[video_idx, 0], dtype=np.float64)
            base = np.asarray(baseline[video_idx, 0], dtype=np.float64)

            # Sanity check: recordings must be [samples, 14 channels].
            if eeg.ndim != 2 or eeg.shape[1] != config.N_CHANNELS:
                continue

            trials.append(
                Trial(
                    subject=subject_idx,
                    video=video_idx,
                    eeg=eeg,
                    baseline=base,
                    valence=valence[video_idx],
                    arousal=arousal[video_idx],
                    dominance=dominance[video_idx],
                )
            )

    if not trials:
        raise ValueError(
            f"No EEG trials extracted from {mat_path!r}. "
            "Check that the DREAMER.mat file is present and uncorrupted."
        )
    return trials
