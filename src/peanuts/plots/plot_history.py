import matplotlib.pyplot as plt
import pandas as pd


def plot_logs(path):
    df = pd.read_csv(path)

    epochs = df["epoch"]
    loss = df["loss"]
    precision = df["precision"]
    recall = df["recall"]
    f1 = 2 * precision * recall / (precision + recall)

    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    plt.plot(epochs, loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(222)
    plt.plot(epochs, precision)
    plt.xlabel("Epochs")
    plt.ylabel("Precision")

    plt.subplot(223)
    plt.plot(epochs, recall)
    plt.xlabel("Epochs")
    plt.ylabel("Recall")

    plt.subplot(224)
    plt.plot(epochs, f1)
    plt.xlabel("Epochs")
    plt.ylabel("F1")

    plt.show()


if __name__ == "__main__":
    plot_logs(input("path: "))
