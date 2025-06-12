import os

import hydra
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .utils.misc import get_device
from .utils.output_fn import output_fn
from .utils.test_fn import test_fn


@hydra.main(version_base=None, config_path="../../config", config_name="test")
def main(config: DictConfig) -> None:
    device = get_device()
    print(f"Using device: {device}")

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

    # Load model weights
    path = os.path.join("../../..", config.model.path)
    if len(path) > 0:
        state_dict = torch.load(path, weights_only=True)
        model.load_state_dict(state_dict)

    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    # Test
    test_fn(test_dataloader, model, loss_fn)
    output_fn(test_dataloader, model)


if __name__ == "__main__":
    main()
