import matplotlib.pyplot as plt


def plot_residual(diff_p, diff_s, diff_ps, tol, dt):
    box = dict(boxstyle="round", facecolor="white", alpha=1)
    text_loc = [0.07, 0.95]
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 3, 1)
    plt.hist(
        diff_p,
        range=(-tol, tol),
        bins=int(2 * tol / dt) + 1,
        facecolor="b",
        edgecolor="black",
        linewidth=1,
    )
    plt.ylabel("Number of picks")
    plt.xlabel("Residual (s)")
    plt.text(
        text_loc[0],
        text_loc[1],
        "(i)",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        fontsize="small",
        fontweight="normal",
        bbox=box,
    )
    plt.title("P-phase")

    plt.subplot(1, 3, 2)
    plt.hist(
        diff_s,
        range=(-tol, tol),
        bins=int(2 * tol / dt) + 1,
        facecolor="b",
        edgecolor="black",
        linewidth=1,
    )
    plt.xlabel("Residual (s)")
    plt.text(
        text_loc[0],
        text_loc[1],
        "(ii)",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        fontsize="small",
        fontweight="normal",
        bbox=box,
    )
    plt.title("S-phase")

    plt.subplot(1, 3, 3)
    plt.hist(
        diff_ps,
        range=(-tol, tol),
        bins=int(2 * tol / dt) + 1,
        facecolor="b",
        edgecolor="black",
        linewidth=1,
    )
    plt.xlabel("Residual (s)")
    plt.text(
        text_loc[0],
        text_loc[1],
        "(iii)",
        horizontalalignment="left",
        verticalalignment="top",
        transform=plt.gca().transAxes,
        fontsize="small",
        fontweight="normal",
        bbox=box,
    )
    plt.title("PS-phase")
    plt.tight_layout()
    plt.savefig("residuals.png", dpi=300)


def main(label_csv, prediction_csv):
    import numpy as np
    import pandas as pd

    label_df = pd.read_csv(label_csv, index_col=0)
    label_p = label_df["itp"].str.strip("[]")
    label_p = label_p.str.split(" ", expand=True)
    label_p = label_p.map(lambda x: int(x) if len(x) else np.nan)
    label_p = label_p - 2000
    label_s = label_df["its"].str.strip("[]")
    label_s = label_s.str.split(" ", expand=True)
    label_s = label_s.map(lambda x: int(x) if len(x) else np.nan)
    label_s = label_s - 2000

    prediction_df = pd.read_csv(prediction_csv, index_col=0)
    prediction_p = prediction_df["itp"].str.strip("[]")
    prediction_p = prediction_p.str.split(" ", expand=True)
    prediction_p = prediction_p.fillna("")
    prediction_p = prediction_p.map(lambda x: int(x) if len(x) else np.nan)
    prediction_s = prediction_df["its"].str.strip("[]")
    prediction_s = prediction_s.str.split(" ", expand=True)
    prediction_s = prediction_s.fillna("")
    prediction_s = prediction_s.map(lambda x: int(x) if len(x) else np.nan)

    diff_itp = prediction_p - label_p
    diff_itp = diff_itp.values.flatten()
    diff_itp = diff_itp / 100
    diff_its = prediction_s - label_s
    diff_its = diff_its.values.flatten()
    diff_its = diff_its / 100
    diff_itps = np.append(diff_itp, diff_its)

    diff_itp = np.array(diff_itp)
    diff_itp = diff_itp[np.abs(diff_itp) < 0.5]  # mpd?
    diff_its = np.array(diff_its)
    diff_its = diff_its[np.abs(diff_its) < 0.5]  # mpd?
    diff_itps = np.append(diff_itp, diff_its)

    print(np.nanmean(diff_itp), np.nanstd(diff_itp))
    print(np.nanmean(diff_its), np.nanstd(diff_its))
    print(np.nanmean(diff_itps), np.nanstd(diff_itps))

    plot_residual(diff_itp, diff_its, diff_itps, 0.5, 0.04)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("label_csv", type=str)
    parser.add_argument("prediction_csv", type=str)
    parser.add_argument("--fname", type=str, default="residual.png")
    args = parser.parse_args()

    main(args.label_csv, args.prediction_csv)
