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
    pred: np.ndarray,
    true: np.ndarray,
    mph: float = 0.6,
    mpd: int = 10,
    sampling_rate: int = 100,
) -> List[int]:
    pred = find_peaks(pred, mph=mph, mpd=mpd)
    true = find_peaks(true, mph=mph, mpd=mpd)

    diffs = (pred - true[:, np.newaxis]) / sampling_rate
    if diffs.size == 0:
        return []

    # ラベルに最も近いピックを取得する
    index = np.abs(diffs).argmin(axis=1)
    diffs = diffs[np.arange(diffs.shape[0]), index]
    diffs = diffs.tolist()
    return diffs


class Metrics:
    def __init__(self, mph=0.3, mpd=50, tol=3.0, sampling_rate=100):
        self.mph = mph
        self.mpd = mpd
        self.tol = tol
        self.sampling_rate = sampling_rate

        self.positives = 0
        self.trues = 0
        self.tp_positive = 0
        self.tp_true = 0

    def count_up(self, pred, y):
        pred = find_peaks(pred, mph=self.mph, mpd=self.mpd)
        y = find_peaks(y, mph=self.mph, mpd=self.mpd)

        diff = (pred - y[:, np.newaxis]) / self.sampling_rate
        true_positives = np.abs(diff) < self.tol

        positives = len(pred)
        trues = len(y)
        tp_positive = true_positives.any(axis=1).sum()  # ピックから見たTrue Positive
        tp_true = true_positives.any(axis=0).sum()  # ラベルから見たTrue Positive

        self.positives += positives
        self.trues += trues
        self.tp_positive += tp_positive
        self.tp_true += tp_true

    def precision(self) -> float:
        if self.positives == 0:
            return 0.0
        return self.tp_positive / self.positives

    def recall(self) -> float:
        if self.trues == 0:
            return 0.0
        return self.tp_true / self.trues

    def f1(self) -> float:
        precision = self.precision()
        recall = self.recall()
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
