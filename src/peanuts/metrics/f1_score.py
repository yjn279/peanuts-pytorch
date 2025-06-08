class F1Score:
    def __init__(self):
        self.__value = 0

    @property
    def value(self):
        return self.__value

    def update(self, precision, recall):
        self.precision = precision
        self.recall = recall

        numerator = 2 * self.precision * self.recall
        denominator = self.precision + self.recall
        self.value = numerator / denominator

    def print(self, **kwargs):
        print(f"F1 Score: {self.value:.4f}", **kwargs)  # 小数第4位まで表示
