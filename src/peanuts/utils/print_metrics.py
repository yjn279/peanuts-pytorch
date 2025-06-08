from scipy.signal import find_peaks
import numpy as np


class Metrics:
    def __init__(self, mph=0.6, mpd=10, tol=300):
        self.mph = mph
        self.mpd = mpd
        self.tol = tol

        self.positives = 0
        self.trues = 0
        self.true_positives = 0

    def count_up(self, pred, y):
        pred = find_peaks(pred, height=self.mph, distance=self.mpd)[0]
        y = find_peaks(y, height=self.mph, distance=self.mpd)[0]

        positives = len(pred)
        trues = len(y)
        axis = 0 if trues > positives else 1  # metricsが1を超さないようにする

        diff = pred - y[:, np.newaxis]
        # print(diff)
        true_positives = np.abs(diff) < self.tol
        true_positives = true_positives.any(axis=axis)
        true_positives = true_positives.sum()

        self.positives += positives
        self.trues += trues
        self.true_positives += true_positives

    def print(self, **kwargs):
        precision = self.true_positives / self.positives
        recall = self.true_positives / self.trues
        f1_score = 2 * precision * recall / (precision + recall)

        print(
            f"Precision: {precision:.4f}",
            f"Recall: {recall:.4f}",
            f"F1-score: {f1_score:.4f}",
            **kwargs,
        )


def print_metrics(loss: float, p_metrics: Metrics, s_metrics: Metrics) -> None:
    print(f"Loss: {loss:.4f}", end=" ")
    p_metrics.print(end=" ")
    s_metrics.print()
