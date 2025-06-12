import os

import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def plot_event(x, y, pred, path=None):
    y_min, y_max = x.min(axis=1), x.max(axis=1)

    itp = find_peaks(y[1], height=0.6, distance=10)[0]
    its = find_peaks(y[2], height=0.6, distance=10)[0]

    plt.figure()

    plt.subplot(411)
    plt.plot(x[0], "black", label="E", linewidth=1)
    plt.vlines(itp, y_min[0], y_max[0], colors="blue", linewidth=1)
    plt.vlines(its, y_min[0], y_max[0], colors="red", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(412)
    plt.plot(x[1], "black", label="N", linewidth=1)
    plt.vlines(itp, y_min[1], y_max[1], colors="blue", linewidth=1)
    plt.vlines(its, y_min[1], y_max[1], colors="red", linewidth=1)
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(413)
    plt.plot(x[2], "black", label="U", linewidth=1)
    plt.vlines(itp, y_min[2], y_max[2], colors="blue", linewidth=1)
    plt.vlines(its, y_min[2], y_max[2], colors="red", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(414)
    plt.plot(pred[1], "blue", label="P-wave", linewidth=1)
    plt.plot(pred[2], "red", label="S-wave", linewidth=1)
    plt.ylabel("Probability")
    plt.legend(loc="upper right", fontsize="small")

    if path is not None:
        dirname = os.path.dirname(path)

        os.makedirs(dirname, exist_ok=True)
        plt.savefig(path)

    plt.close()
