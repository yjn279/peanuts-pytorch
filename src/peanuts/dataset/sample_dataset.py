import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(
        self,
        path_dir=None,
        path_csv=None,
        transform=None,
        target_transform=None,
    ):
        pass

    def __len__(self):
        return 6

    def __getitem__(self, idx):
        # itp = random.randint(900, 1100)
        itp = 1000
        p_left = itp - 2
        p_right = itp + 3

        s_left = p_left + 498
        s_right = p_right + 502

        x = torch.zeros(3, 3001, 1)
        x[:, p_left:p_right] = 0.8
        x[:, s_left:s_right] = 1.2

        y = torch.zeros(3, 3001, 1)
        y[0] = 1
        y[0, p_left:p_right] = torch.Tensor([0.7, 0.3, 0, 0.3, 0.7]).reshape(5, 1)
        y[1, p_left:p_right] = torch.Tensor([0.3, 0.7, 1, 0.7, 0.3]).reshape(5, 1)
        y[0, s_left:s_right] = torch.Tensor(
            [0.8, 0.6, 0.4, 0.2, 0, 0.2, 0.4, 0.6, 0.8]
        ).reshape(9, 1)
        y[2, s_left:s_right] = torch.Tensor(
            [0.2, 0.4, 0.6, 0.8, 1, 0.8, 0.6, 0.4, 0.2]
        ).reshape(9, 1)
        return x, y, idx
