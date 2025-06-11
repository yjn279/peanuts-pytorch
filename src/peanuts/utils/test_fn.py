import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .export_history import export_history

from ..dataset import *  # noqa: F403
from ..models import *  # noqa: F403
from .get_device import get_device
from .metrics import Metrics


def test_fn(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    epoch: int | None = None,
) -> None:
    device = get_device()

    loss_value = 0
    p_metrics = Metrics()
    s_metrics = Metrics()
    batch_count = len(dataloader)

    with torch.no_grad():
        model.eval()
        
        for x, y, path in tqdm(dataloader, desc="Validation"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)
            loss_value += loss_fn(pred, y).item()
            pred = torch.nn.Softmax2d()(pred)
            
            # Calculate metrics and generate plots
            for x_event, y_event, pred_event, path_event in zip(x, y, pred, path):
                x_event = x_event.squeeze().cpu().numpy()
                y_event = y_event.squeeze().cpu().numpy()
                pred_event = pred_event.squeeze().cpu().numpy()

                p_metrics.count_up(pred_event[1], y_event[1])
                s_metrics.count_up(pred_event[2], y_event[2])
                
    # Save metrics
    if epoch is not None:
        export_history(
            filepath="history_valid.csv",
            epoch=epoch,
            loss=loss_value / batch_count,
            precision_p=p_metrics.precision(),
            recall_p=p_metrics.recall(),
            f1score_p=p_metrics.f1(),
            precision_s=s_metrics.precision(),
            recall_s=s_metrics.recall(),
            f1score_s=s_metrics.f1(),
        )
