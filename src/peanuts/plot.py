import torch

from .plots import plot_event
from .utils import get_device


def plot(dataloader, model):
    model.eval()
    device = get_device()

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            pred = torch.nn.Softmax2d()(pred)

            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            pred = pred.squeeze().cpu().numpy()
            path = dataloader.dataset.df["fname"][i]
            path = f"{path.replace('/', '-')}.png"

            plot_event(x, y, pred, path)
