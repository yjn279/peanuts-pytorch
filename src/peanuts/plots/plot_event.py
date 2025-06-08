import matplotlib.pyplot as plt


def plot_event(x, y, pred, path=None):
    plt.figure()

    plt.subplot(511)
    plt.plot(x[0], "k", label="E", linewidth=1)
    plt.vlines(y[0], 0, 1, colors="k", linestyles="dashed", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(512)
    plt.plot(x[1], "k", label="N", linewidth=1)
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(513)
    plt.plot(x[2], "k", label="U", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(514)
    plt.plot(y[1], "orange", label="P-Wave", linewidth=1)
    plt.plot(y[2], "blue", label="S-Wave", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(515)
    plt.plot(pred[1], "orange", label="P-Wave", linewidth=1)
    plt.plot(pred[2], "blue", label="S-Wave", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    if path is not None:
        plt.savefig(path)

    else:
        plt.show()

    plt.close()
