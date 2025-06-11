import matplotlib.pyplot as plt
import os
import numpy as np
from typing import List


def plot_diffs(
    diffs: List[int],
    filepath: str,
    title: str | None = None,
    bins: int = 50,
) -> None:
    if not diffs:
        return

    os.makedirs("plots", exist_ok=True)
    
    plt.figure()
    plt.hist(diffs, bins=bins, color="blue")
    plt.xlabel("Travel time residual (s)")
    plt.ylabel("# of events")
    
    if title is not None:
        plt.title(title)
        
    mean = np.mean(diffs)
    std = np.std(diffs)
    
    plt.text(
        x=0.98, 
        y=0.95,
        s=f"Mean: {mean:.2f} s\nSD: {std:.2f} s",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
    )
        
    plt.savefig(filepath)
    plt.close()
