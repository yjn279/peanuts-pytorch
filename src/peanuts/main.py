import hydra
from omegaconf import DictConfig
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .utils import get_device


@hydra.main(version_base=None, config_path="../../config", config_name="train")
def train(config: DictConfig) -> None:
    device = get_device()

    # Train DataLoader
    train_config = config.data.train
    train_dataset = eval(train_config.dataset)(**train_config)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
    )

    # Test DataLoader
    test_config = config.data.test
    test_dataset = eval(test_config.dataset)(**test_config)
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

    epochs = config.model.epochs
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}/{epochs}")

        # Training
        train(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

        # Validation
        test(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
        )

        scheduler.step()
        
        
@hydra.main(version_base=None, config_path="../../config", config_name="test")
def test(config: DictConfig) -> None:
    device = get_device()

    # Test DataLoader
    test_config = config.data.test
    test_dataset = eval(test_config.dataset)(**test_config)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=test_config.batch_size,
        shuffle=False,
    )

    # Model
    model = eval(config.model.name)()
    model = model.to(device)

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Test
    test(
        dataloader=test_dataloader,
        model=model,
        loss_fn=loss_fn,
    )

# plot_event(
#     x=x_event.squeeze().cpu().numpy(),
#     y=y_event.squeeze().cpu().numpy(),
#     pred=pred_event.squeeze().cpu().numpy(),
#     path=f"{plot_prefix}_{path}.png",
# )
