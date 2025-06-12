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


class Metrics:
    def __init__(self, mph=0.6, mpd=10, tol=300):  # 0.3, 50
        self.mph = mph
        self.mpd = mpd
        self.tol = tol

        self.positives = 0
        self.trues = 0
        self.true_positives = 0

    def count_up(self, pred, y):
        pred = find_peaks(pred, mph=self.mph, mpd=self.mpd)
        y = find_peaks(y, mph=self.mph, mpd=self.mpd)

        positives = len(pred)
        trues = len(y)
        axis = 0 if trues > positives else 1  # metricsが1を超さないようにする

        diff = pred - y[:, np.newaxis]
        true_positives = np.abs(diff) < self.tol / dt
        true_positives = true_positives.any(axis=axis)
        true_positives = true_positives.sum()

        self.positives += positives
        self.trues += trues
        self.true_positives += true_positives

    def precision(self) -> float:
        if self.positives == 0:
            return 0.0
        return self.true_positives / self.positives

    def recall(self) -> float:
        if self.trues == 0:
            return 0.0
        return self.true_positives / self.trues

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
