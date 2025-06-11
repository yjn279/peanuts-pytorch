import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import *  # noqa: F403
from ..models import *  # noqa: F403
from .export_history import export_history
from .get_device import get_device
from .metrics import Metrics


def train_fn(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int = 0,
) -> None:
    device = get_device()

    loss_value = 0
    p_metrics = Metrics()
    s_metrics = Metrics()
    batch_count = len(dataloader)

    for x, y, _ in tqdm(dataloader, desc="Training"):
        model.train()
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update metrics
        with torch.no_grad():
            model.eval()
            pred = torch.nn.Softmax2d()(pred)
            loss_value += loss.item()

            for y_event, pred_event in zip(y, pred):
                y_event = y_event.squeeze().cpu().numpy()
                pred_event = pred_event.squeeze().cpu().numpy()

                p_metrics.count_up(pred_event[1], y_event[1])
                s_metrics.count_up(pred_event[2], y_event[2])

    # Save metrics
    export_history(
        filepath="history_train.csv",
        epoch=epoch,
        loss=loss_value / batch_count,
        precision_p=p_metrics.precision(),
        recall_p=p_metrics.recall(),
        f1score_p=p_metrics.f1(),
        precision_s=s_metrics.precision(),
        recall_s=s_metrics.recall(),
        f1score_s=s_metrics.f1(),
    )
