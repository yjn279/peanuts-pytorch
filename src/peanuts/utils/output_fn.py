import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..output.export_predictions import export_predictions
from ..output.plot_diffs import plot_diffs
from ..output.plot_event import plot_event
from .misc import get_device, get_diffs


def output_fn(
    dataloader: DataLoader,
    model: nn.Module,
    mph: float = 0.6,
    mpd: int = 10,
) -> None:
    device = get_device()

    diffs_p = []
    diffs_s = []

    with torch.no_grad():
        model.eval()

        for x, y, path in tqdm(dataloader, desc="Exporting"):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred = torch.nn.Softmax2d()(pred)

            for x_event, y_event, pred_event, path_event in zip(x, y, pred, path):
                x_event = x_event.squeeze().cpu().numpy()
                y_event = y_event.squeeze().cpu().numpy()
                pred_event = pred_event.squeeze().cpu().numpy()

                plot_event(
                    x=x_event,
                    y=y_event,
                    pred=pred_event,
                    path=f"plots/waveforms/{path_event}.eps",
                )

                export_predictions(path_event, pred_event)

                diffs_p += get_diffs(pred_event[1], y_event[1], mph, mpd)
                diffs_s += get_diffs(pred_event[2], y_event[2], mph, mpd)

    plot_diffs(diffs_p, "plots/diffs_p.eps")
    plot_diffs(diffs_s, "plots/diffs_s.eps")
    plot_diffs(diffs_p + diffs_s, "plots/diffs.eps")
