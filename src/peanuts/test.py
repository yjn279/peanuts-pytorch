import hydra
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .evaluate import Metrics
from .plots.plot_event import plot_event
from .utils import get_device


@hydra.main(version_base=None, config_path="../../config", config_name="test")
def test(config: DictConfig) -> None:
    device = get_device()

    # Test Dataloader
    test_config = config.data.test
    test_dataset = eval(test_config.dataset)(
        path_dir=test_config.event_dir,
        path_csv=test_config.csv_path,
    )
    
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
    )

    # Model
    model = eval(config.model.name)()
    model = model.to(device)

    # Optimizer
    loss_fn = nn.CrossEntropyLoss()

    # Evaluation
    model.eval()
    loss_value = 0
    p_metrics = Metrics()
    s_metrics = Metrics()
    batch_count = len(test_dataloader)

    with torch.no_grad():
        for x, y, paths in tqdm(test_dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)

            # Forward pass
            pred = model(x)
            loss_value += loss_fn(pred, y).item()
            pred = torch.nn.Softmax2d()(pred)

            # Calculate metrics for each sample in the batch
            for x_event, y_event, pred_event, path in zip(x, y, pred, paths):
                pred_event = pred_event[..., 0].cpu().numpy()
                y_event = y_event[..., 0].cpu().numpy()

                # Update metrics for P and S phases
                p_metrics.count_up(pred_event[1], y_event[1])
                s_metrics.count_up(pred_event[2], y_event[2])

                plot_event(
                    x=x_event.squeeze().cpu().numpy(),
                    y=y_event.squeeze().cpu().numpy(),
                    pred=pred_event.squeeze().cpu().numpy(),
                    path=f"{path}.png",
                )
                
    print(f"Loss: {loss_value / batch_count:.4f}", end=" ")
    p_metrics.print(end=" ")
    s_metrics.print()


if __name__ == "__main__":
    test()
