import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset import *  # noqa: F403
from ..models import *  # noqa: F403
from ..evaluate import Metrics
from ..plots.plot_event import plot_event
from .get_device import get_device


def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
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

            for pred_event, y_event in zip(pred, y):
                pred_event_np = pred_event[..., 0].cpu().numpy()
                y_event_np = y_event[..., 0].cpu().numpy()

                p_metrics.count_up(pred_event_np[1], y_event_np[1])
                s_metrics.count_up(pred_event_np[2], y_event_np[2])

    average_loss = loss_value / batch_count
    print_metrics(average_loss, p_metrics, s_metrics)


def test(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    generate_plots: bool = True,
    plot_prefix: str = "eval",
) -> None:
    device = next(model.parameters()).device
    model.eval()

    loss_value = 0
    p_metrics = Metrics()
    s_metrics = Metrics()
    batch_count = len(dataloader)

    with torch.no_grad():
        for x, y, paths in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)
            loss_value += loss_fn(pred, y).item()
            pred = torch.nn.Softmax2d()(pred)

            # Calculate metrics for each sample in the batch
            for x_event, y_event, pred_event, path in zip(x, y, pred, paths):
                pred_event_np = pred_event[..., 0].cpu().numpy()
                y_event_np = y_event[..., 0].cpu().numpy()

                # Update metrics for P and S phases
                p_metrics.count_up(pred_event_np[1], y_event_np[1])
                s_metrics.count_up(pred_event_np[2], y_event_np[2])

                # Generate plot
                if generate_plots:
                    plot_event(
                        x=x_event.squeeze().cpu().numpy(),
                        y=y_event.squeeze().cpu().numpy(),
                        pred=pred_event.squeeze().cpu().numpy(),
                        path=f"{plot_prefix}_{path}.png",
                    )

    average_loss = loss_value / batch_count
    print_metrics(average_loss, p_metrics, s_metrics)


def print_metrics(loss: float, p_metrics: Metrics, s_metrics: Metrics) -> None:
    print(f"Loss: {loss:.4f}", end=" ")
    p_metrics.print(end=" ")
    s_metrics.print()
