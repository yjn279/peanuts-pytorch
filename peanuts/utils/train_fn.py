import torch
from tqdm import tqdm

from .get_device import get_device
from .metrics_helper import MetricsHelper
from metrics import Loss


def train_fn(dataloader, model, loss_fn, optimizer, mph=0.6, mpd=10, tol=300):
    model.train()
    device = get_device()

    loss = Loss(dataloader)
    metrics = MetricsHelper(mph, mpd, tol)

    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss_batch = loss_fn(pred, y)
        loss_batch.backward()
        loss.update(loss_batch)

        # Backpropagation
        optimizer.step()
        optimizer.zero_grad()

        with torch.no_grad():
            pred = torch.nn.Softmax2d()(pred)
            for pred_event, y_event in zip(pred, y):
                pred_event = pred_event.squeeze().cpu().numpy()
                y_event = y_event.squeeze().cpu().numpy()
                metrics.update(pred_event, y_event)

            # TODO:
            #   - picks.csv
            #   -

    loss.print(end=" ")
    metrics.print(end=" ")
