import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_diffs(
    diffs: List[int],
    filepath: str,
    title: str | None = None,
) -> None:
    if not diffs:
        return

    os.makedirs("plots", exist_ok=True)

    plt.figure()
    plt.hist(diffs, bins=30, range=(-0.5, 0.5), color="blue")
    plt.xlabel("Travel time residual (s)")
    plt.ylabel("# of events")

    if title is not None:
        plt.title(title)

    mean = np.mean(diffs)
    std = np.std(diffs)

    plt.text(
        x=0.98,
        y=0.98,
        s=f"Mean: {mean:.2f} s\nSD: {std:.2f} s",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )

    plt.savefig(filepath)
    plt.close()
