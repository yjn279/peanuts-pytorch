import numpy as np

from .misc import find_peaks


class Metrics:
    def __init__(self, mph=0.6, mpd=10, tol=300):
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
        true_positives = np.abs(diff) < self.tol
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
