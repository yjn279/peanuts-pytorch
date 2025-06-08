import hydra
from omegaconf import DictConfig
from torch import nn

from .dataset import *  # noqa: F403
from .models import *  # noqa: F403
from .utils import get_device
from .utils.common import (
    create_dataloader,
    test,
    print_metrics,
)


@hydra.main(version_base=None, config_path="../../config", config_name="test")
def main(config: DictConfig) -> None:
    device = get_device()

    # Dataloaders
    test_dataloader = create_dataloader(
        config.data.test,
        shuffle=False,
    )

    # Model
    model = eval(config.model.name)()
    model = model.to(device)

    # Loss function
    loss_fn = nn.CrossEntropyLoss()

    # Evaluation
    (
        eval_loss,
        eval_p_metrics,
        eval_s_metrics,
    ) = test(
        model=model,
        dataloader=test_dataloader,
        loss_fn=loss_fn,
        generate_plots=True,
        plot_prefix="test",
    )

    print_metrics(eval_loss, eval_p_metrics, eval_s_metrics)


if __name__ == "__main__":
    main()
