import os

import torch
from tqdm import tqdm

from .get_device import get_device
from .metrics_helper import MetricsHelper
from metrics import Loss
from plots import plot_event

def test_fn(dataloader, model, loss_fn, mph=0.6, mpd=10, tol=300):
    model.eval()
    device = get_device()

    loss = Loss(dataloader)
    metrics = MetricsHelper(mph, mpd, tol)

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(device), y.to(device)

            # Loss
            pred = model(x)
            loss_batch = loss_fn(pred, y)
            loss.update(loss_batch)

            # Metrics
            pred = torch.nn.Softmax2d()(pred)
            pred = pred.squeeze().cpu().numpy()
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            metrics.update(pred, y)

            # Plot
            path = dataloader.dataset.df["fname"][i]
            path = os.path.join("figures", path)
            path = f"{path}.png"

            os.makedirs(os.path.dirname(path), exist_ok=True)
            plot_event(x, y, pred, path)

    loss.print(end=" ")
    metrics.print(end=" ")
