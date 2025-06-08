from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


def detect_peaks(ndarray, mph=0.6, mpd=100):
    # TODO: mpd=0.5 * sampling_rate
    return find_peaks(ndarray, height=mph, distance=mpd)[0]


def count_true_positives(preds, trues, tol=300):
    diff = preds - trues[:, np.newaxis]
    true_positives = np.abs(diff) < tol
    true_positives = true_positives.any(axis=0)
    true_positives = true_positives.sum()
    return true_positives


def precision(true_positives, trues):
    return true_positives / trues


def recall(true_positives, positives):
    return true_positives / positives


def f1_score(precision, recall):
    return 2 * precision * recall / (precision + recall)


def plot(waveforms, labels):
    ymax = waveforms.max()
    ymin = waveforms.min()

    plt.figure()

    plt.subplot(411)
    plt.plot(waveforms[0, :, 0], "k", label="E", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(412)
    plt.plot(waveforms[1, :, 0], "k", label="N", linewidth=1)
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(413)
    plt.plot(waveforms[2, :, 0], "k", label="U", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(414)
    plt.plot(labels[1, :, 0], "orange", label="P-Wave", linewidth=1)
    plt.plot(labels[2, :, 0], "blue", label="S-Wave", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.show()
