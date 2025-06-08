import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .evaluate import Metrics
from .plots.plot_event import plot_event
from .utils import get_device


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def train(config: DictConfig) -> None:
    device = get_device()

    # Train Dataloader
    train_config = config.data.train
    train_dataset = eval(train_config.dataset)(
        path_dir=train_config.event_dir,
        path_csv=train_config.csv_path,
    )
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    # Validation Dataloader
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
    optimizer = optim.Adam(model.parameters(), lr=config.optimizer.learning_rate)
    scheduler = ExponentialLR(optimizer, gamma=config.optimizer.gamma)

    # Training
    epochs = config.model.epochs
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")
        
        loss_value = 0
        p_metrics = Metrics()
        s_metrics = Metrics()
        batch_count = len(train_dataloader)

        for x, y, _ in tqdm(train_dataloader):
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
                    pred_event = pred_event[..., 0].cpu().numpy()
                    y_event = y_event[..., 0].cpu().numpy()

                    p_metrics.count_up(pred_event[1], y_event[1])
                    s_metrics.count_up(pred_event[2], y_event[2])

        print(f"Loss: {loss_value / batch_count:.4f}", end=" ")
        p_metrics.print(end=" ")
        s_metrics.print()
            
        loss_value = 0
        p_metrics = Metrics()
        s_metrics = Metrics()
        batch_count = len(test_dataloader)

        with torch.no_grad():
            model.eval()

            for x, y, paths in test_dataloader:
                x, y = x.to(device), y.to(device)

                pred = model(x)
                loss_value += loss_fn(pred, y).item()
                pred = torch.nn.Softmax2d()(pred)

                for x, y, pred, path in zip(x, y, pred, paths):
                    pred_event = pred_event[..., 0].cpu().numpy()
                    y_event = y_event[..., 0].cpu().numpy()

                    p_metrics.count_up(pred_event[1], y_event[1])
                    s_metrics.count_up(pred_event[2], y_event[2])
                    
                    plot_event(
                        x=x.squeeze().cpu().numpy(),
                        y=y.squeeze().cpu().numpy(),
                        pred=pred.squeeze().cpu().numpy(),
                        path=f"{path}.png",
                    )

                    break
                break
            
        print(f"Loss: {loss_value / batch_count:.4f}", end=" ")
        p_metrics.print(end=" ")
        s_metrics.print()

        scheduler.step()


if __name__ == "__main__":
    train()
