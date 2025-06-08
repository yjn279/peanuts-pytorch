import numpy as np

class Recall:
    def __init__(self, tol=300):
        self.tol = tol
        self.trues = 0
        self.true_positives = 0


    @property
    def value(self):
        return self.true_positives / self.trues


    def update(self, pred, y):
        # 差分がtol未満のインデックスの数をカウント
        diff = pred - y[:, np.newaxis]
        true_positives = np.abs(diff) < self.tol
        true_positives = true_positives.any(axis=1)  # y方向に足し合わせる
        true_positives = true_positives.sum()

        # インデックスの数をアップデート
        self.trues += len(y)
        self.true_positives += true_positives

    
    def print(self, **kwargs):
        print(f"Recall: {self.value:.4f}", **kwargs)  # 小数第4位まで表示
