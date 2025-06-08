import torch


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


if __name__ == "__main__":
    device = get_device()
    print(f"Using device: {device}")
