from scipy.signal import find_peaks
import torch
import numpy as np

from utils import get_device


def evaluate(dataloader, model, loss_fn):
    model.eval()
    device = get_device()

    loss = 0
    p_metrics = Metrics()
    s_metrics = Metrics()
    batch_count = len(dataloader)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred = torch.nn.Softmax2d()(pred)
            loss += loss_fn(pred, y).item()

            for pred_event, y_event in zip(pred, y):
                pred_event = pred_event[..., 0].cpu().numpy()
                y_event = y_event[..., 0].cpu().numpy()

                p_metrics.count_up(pred_event[1], y_event[1])
                s_metrics.count_up(pred_event[2], y_event[2])

    print(f"Loss: {loss / batch_count:.4f}", end=" ")
    p_metrics.print(end=" ")
    s_metrics.print()


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


def plot(waveforms, labels, hoge):
    import matplotlib.pyplot as plt
    ymax = waveforms.max()
    ymin = waveforms.min()

    plt.figure()

    plt.subplot(511)
    plt.plot(waveforms[0, :, 0], "k", label="E", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(512)
    plt.plot(waveforms[1, :, 0], "k", label="N", linewidth=1)
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(513)
    plt.plot(waveforms[2, :, 0], "k", label="U", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(514)
    plt.plot(labels[1, :, 0], "orange", label="P-Wave", linewidth=1)
    plt.plot(labels[2, :, 0], "blue", label="S-Wave", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(515)
    plt.plot(hoge[1, :, 0], "orange", label="P-Wave", linewidth=1)
    plt.plot(hoge[2, :, 0], "blue", label="S-Wave", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.show()

