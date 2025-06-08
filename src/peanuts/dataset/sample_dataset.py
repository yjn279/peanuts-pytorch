import torch
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(
        self,
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
        return x, y


def plot(waveforms, labels):
    import matplotlib.pyplot as plt

    # ymax = waveforms.max()
    # ymin = waveforms.min()

    plt.figure()

    plt.subplot(411)
    plt.plot(waveforms[0, :, 0], "k", label="E", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(412)
    plt.plot(waveforms[1, :, 0], "k", label="N", linewidth=1)
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(413)
    plt.plot(waveforms[2, :, 0], "k", label="U", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.subplot(414)
    plt.plot(labels[1, :, 0], "orange", label="P-Wave", linewidth=1)
    plt.plot(labels[2, :, 0], "blue", label="S-Wave", linewidth=1)
    plt.legend(loc="upper right", fontsize="small")

    plt.show()


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = SampleDataset()
    dataloader = DataLoader(dataset, batch_size=2)

    for x, y in dataloader:
        for waveforms, labels in zip(x, y):
            plot(waveforms, labels)
