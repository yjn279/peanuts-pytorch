import hydra
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .utils import get_device
from .utils.test_fn import test_fn
from .utils.train_fn import train_fn


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def main(config: DictConfig) -> None:
    device = get_device()
    print(f"Using device: {device}")

    # Train DataLoader
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

    # Test DataLoader
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

        train_fn(train_dataloader, model, loss_fn, optimizer, epoch)
        test_fn(test_dataloader, model, loss_fn, epoch)

        torch.save(model.state_dict(), "model_weights.pth")
        scheduler.step()


if __name__ == "__main__":
    main()
