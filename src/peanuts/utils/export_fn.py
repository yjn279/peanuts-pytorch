from scipy.signal import find_peaks
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import os

from ..dataset import *  # noqa: F403
from ..models import *  # noqa: F403
from .get_device import get_device
from ..plots.plot_event import plot_event


def export_fn(
    dataloader: DataLoader,
    model: nn.Module,
    mph: float = 0.6,
    mpd: int = 10,
) -> None:
    device = get_device()

    # Accumulate all diffs for histogram plotting
    all_diffs_p = []
    all_diffs_s = []

    # Ensure output directories exist
    os.makedirs("plots", exist_ok=True)

    # Initialize CSV file (overwrite if exists)
    csv_path = "pick_diffs.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)

    with torch.no_grad():
        model.eval()

        for x, y, path in tqdm(dataloader, desc="Exporting"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)
            pred = torch.nn.Softmax2d()(pred)

            # Batch data for CSV output
            batch_data = []

            # Calculate metrics and generate plots
            for x_event, y_event, pred_event, path_event in zip(x, y, pred, path):
                x_event = x_event.squeeze().cpu().numpy()
                y_event = y_event.squeeze().cpu().numpy()
                pred_event = pred_event.squeeze().cpu().numpy()

                y_peaks_p = find_peaks(y_event[1], height=mph, distance=mpd)[0]
                y_peaks_s = find_peaks(y_event[2], height=mph, distance=mpd)[0]
                pred_peaks_p = find_peaks(pred_event[1], height=mph, distance=mpd)[0]
                pred_peaks_s = find_peaks(pred_event[2], height=mph, distance=mpd)[0]

                # Calculate differences for each predicted peak to closest true peak
                diffs_p = []
                diffs_s = []

                # P-wave differences
                if len(pred_peaks_p) > 0 and len(y_peaks_p) > 0:
                    for pred_p in pred_peaks_p:
                        # Find closest true peak and calculate signed difference
                        closest_y = min(y_peaks_p, key=lambda y_p: abs(pred_p - y_p))
                        signed_diff = pred_p - closest_y
                        diffs_p.append(signed_diff)
                        all_diffs_p.append(signed_diff)

                # S-wave differences
                if len(pred_peaks_s) > 0 and len(y_peaks_s) > 0:
                    for pred_s in pred_peaks_s:
                        # Find closest true peak and calculate signed difference
                        closest_y = min(y_peaks_s, key=lambda y_s: abs(pred_s - y_s))
                        signed_diff = pred_s - closest_y
                        diffs_s.append(signed_diff)
                        all_diffs_s.append(signed_diff)

                # Prepare CSV row data
                row_data = {
                    "fname": path_event,
                    "itp": str(pred_peaks_p.tolist()),
                    "itp_diffs": str(diffs_p),
                    "its": str(pred_peaks_s.tolist()),
                    "its_diffs": str(diffs_s),
                }
                batch_data.append(row_data)

                plot_event(
                    x=x_event,
                    y=y_event,
                    pred=pred_event,
                    path=f"plots/waveforms/{path_event}.png",
                )

            # Output batch data to CSV
            batch_df = pd.DataFrame(batch_data)

            # Append to CSV file (create header if first batch)
            header = not os.path.exists(csv_path)
            batch_df.to_csv(csv_path, mode="a", header=header, index=False)

            # Clear batch data for memory management
            del batch_data
            del batch_df

    # Generate histogram plots for all accumulated differences
    if all_diffs_p:
        plt.figure(figsize=(10, 6))
        plt.hist(all_diffs_p, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("P-wave Peak Difference (samples)")
        plt.ylabel("Frequency")
        plt.title("P-wave Peak Prediction Differences")
        plt.grid(True, alpha=0.3)
        plt.savefig("plots/itp_diffs.png", dpi=150, bbox_inches="tight")
        plt.close()

    if all_diffs_s:
        plt.figure(figsize=(10, 6))
        plt.hist(all_diffs_s, bins=50, alpha=0.7, edgecolor="black")
        plt.xlabel("S-wave Peak Difference (samples)")
        plt.ylabel("Frequency")
        plt.title("S-wave Peak Prediction Differences")
        plt.grid(True, alpha=0.3)
        plt.savefig("plots/its_diffs.png", dpi=150, bbox_inches="tight")
        plt.close()
