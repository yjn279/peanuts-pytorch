import hydra
from omegaconf import DictConfig
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from dataset import *
from models import *
from evaluate import evaluate
from plots.plot_event import plot_event
from utils import get_device


@hydra.main(version_base=None, config_path="../config", config_name="train")
def train(config: DictConfig) -> None:
    device = get_device()

    # Train Dataloader
    train_config = config.data.train
    train_dataset = eval(train_config.dataset)(
        train_config.event_dir, train_config.csv_path
    )
    train_dataloader = DataLoader(train_dataset, train_config.batch_size, shuffle=True)

    # Validation Dataloader
    test_config = config.data.test
    test_dataset = eval(test_config.dataset)(
        test_config.event_dir, test_config.csv_path
    )
    test_dataloader = DataLoader(test_dataset, test_config.batch_size)

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
        model.train()

        for x, y in tqdm(train_dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        evaluate(train_dataloader, model, loss_fn)
        evaluate(test_dataloader, model, loss_fn)

        with torch.no_grad():
            model.eval()

            for x_batch, y_batch in test_dataloader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                pred_batch = model(x_batch)
                pred_batch = torch.nn.Softmax2d()(pred_batch)

                for x, y, pred in zip(x_batch, y_batch, pred_batch):
                    plot_event(
                        x=x.squeeze().cpu().numpy(),
                        y=y.squeeze().cpu().numpy(),
                        pred=pred.squeeze().cpu().numpy(),
                        path=f"{epoch}.png",
                    )

                    break
                break

        scheduler.step()


if __name__ == "__main__":
    train()
