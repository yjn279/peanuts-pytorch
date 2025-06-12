from typing import List

import numpy as np
import torch
from scipy.signal import find_peaks as _find_peaks


def find_peaks(x: np.ndarray, mph: float = 0.6, mpd: int = 10) -> np.ndarray:
    return _find_peaks(x, height=mph, distance=mpd)[0]


def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def get_diffs(
    pred: np.ndarray, true: np.ndarray, mph: float = 0.6, mpd: int = 10
) -> List[int]:
    pred = find_peaks(pred, mph=mph, mpd=mpd)
    true = find_peaks(true, mph=mph, mpd=mpd)

    diffs = pred - true[:, np.newaxis]
    if diffs.size == 0:
        return []

    diffs = np.abs(diffs).min(axis=1)
    diffs = diffs.tolist()
    return diffs
