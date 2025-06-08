import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .utils.get_device import get_device
from .utils.print_metrics import print_metrics, Metrics
from .plots.plot_event import plot_event


def test_fn(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
) -> None:
    device = get_device()
    model.eval()

    loss_value = 0
    p_metrics = Metrics()
    s_metrics = Metrics()
    batch_count = len(dataloader)

    with torch.no_grad():
        for x, y, paths in tqdm(dataloader, desc="Validation"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)
            loss_value += loss_fn(pred, y).item()
            pred = torch.nn.Softmax2d()(pred)

            # Calculate metrics and generate plots
            for x_event, y_event, pred_event, path in zip(x, y, pred, paths):
                pred_event = pred_event[..., 0].cpu().numpy()
                y_event = y_event[..., 0].cpu().numpy()

                p_metrics.count_up(pred_event[1], y_event[1])
                s_metrics.count_up(pred_event[2], y_event[2])
                
                plot_event(
                    x=x_event.squeeze().cpu().numpy(),
                    y=y_event.squeeze().cpu().numpy(),
                    pred=pred_event.squeeze().cpu().numpy(),
                    path=f"{path}.png",
                )

    loss_value = loss_value / batch_count
    print_metrics(loss_value, p_metrics, s_metrics)
