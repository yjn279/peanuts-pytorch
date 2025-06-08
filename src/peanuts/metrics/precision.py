import numpy as np


class Precision:
    def __init__(self, tol=300):
        self.tol = tol
        self.positives = 0
        self.true_positives = 0

    @property
    def value(self):
        return self.true_positives / self.positives

    def update(self, pred, y):
        # 差分がtol未満のインデックスの数をカウント
        diff = pred - y[:, np.newaxis]
        true_positives = np.abs(diff) < self.tol
        true_positives = true_positives.any(axis=0)  # pred方向に足し合わせる
        true_positives = true_positives.sum()

        # インデックスの数をアップデート
        self.positives += len(pred)
        self.true_positives += true_positives

    def print(self, **kwargs):
        print(f"Precision: {self.value:.4f}", **kwargs)  # 小数第4位まで表示
