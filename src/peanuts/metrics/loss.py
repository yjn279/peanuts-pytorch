class Loss:
    def __init__(self, dataloader):
        self.__value = 0
        self.batch_count = len(dataloader)

    @property
    def value(self):
        return self.__value / self.batch_count

    @value.setter
    def value(self, value):
        self.__value = value

    def update(self, loss):
        self.__value += loss.item()

    def print(self, **kwargs):
        print(f"Loss: {self.value:.4f}", **kwargs)  # 小数第4位まで表示
