from scipy.signal import find_peaks

from metrics import Precision, Recall, F1Score


class MetricsHelper:
    def __init__(self, mph=0.6, mpd=10, tol=300):
        self.mph = mph
        self.mpd = mpd
        self.tol = tol

        self.p_precision = Precision(tol)
        self.s_precision = Precision(tol)
        self.p_recall = Recall(tol)
        self.s_recall = Recall(tol)
        self.p_f1 = F1Score()
        self.s_f1 = F1Score()

    @property
    def value(self):
        self.p_f1.update(self.p_precision.value, self.p_recall.value)
        self.s_f1.update(self.s_precision.value, self.s_recall.value)

        return {
            "p_precision": self.p_precision.value,
            "s_precision": self.s_precision.value,
            "p_recall": self.p_recall.value,
            "s_recall": self.s_recall.value,
            "p_f1": self.p_f1.value,
            "s_f1": self.s_f1.value,
        }

    def update(self, pred, y):
        p_pred = find_peaks(pred[1], height=self.mph, distance=self.mpd)[0]
        s_pred = find_peaks(pred[2], height=self.mph, distance=self.mpd)[0]
        p_y = find_peaks(y[1], height=self.mph, distance=self.mpd)[0]
        s_y = find_peaks(y[2], height=self.mph, distance=self.mpd)[0]

        self.p_precision.update(p_pred, p_y)
        self.s_precision.update(s_pred, s_y)
        self.p_recall.update(p_pred, p_y)
        self.s_recall.update(s_pred, s_y)

    def print(self, **kwargs):
        self.p_precision.print(**kwargs)
        self.p_recall.print(**kwargs)
        self.p_f1.print(**kwargs)
        self.s_precision.print(**kwargs)
        self.s_recall.print(**kwargs)
        self.s_f1.print(**kwargs)
