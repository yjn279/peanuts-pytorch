import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GeoSciAIDataset(Dataset):
    def __init__(
        self,
        path_dir,
        path_csv,
        transform=None,
        target_transform=None,
    ):
        self.path = path_dir
        self.df = pd.read_csv(path_csv)
        self.df = self.df.fillna(-1)
        self.df["itp"] = self.df["itp"].astype(int)
        self.df["its"] = self.df["its"].astype(int)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        series = self.df.iloc[idx]

        # 波形データを取得
        path = os.path.join(self.path, series["fname"])
        waveforms = np.load(path)["data"]

        # 検測データを取得
        itp = series["itp"]
        itp = [] if itp < 0 else [itp]
        itp = np.array(itp)
        itp = itp * 200  # TODO: waveforms.csvを修正する。

        its = series["its"]
        its = [] if its < 0 else [its]
        its = np.array(its)
        its = its * 200  # TODO: waveforms.csvを修正する。

        shift = np.min(np.append(itp, its))
        waveforms, itp, its = self.random_shift(
            waveforms, itp, its, (500 - shift, 2500 - shift)
        )
        waveforms, itp, its = self.trim(waveforms, itp, its, (0, 6001))
        waveforms = self.normalize(waveforms)
        labels = self.generate_labels(waveforms, itp, its)

        # if self.transform:
        #     image = self.transform(image)

        # if self.target_transform:
        #     label = self.target_transform(label)

        waveforms = torch.from_numpy(waveforms).float()
        labels = torch.from_numpy(labels).float()
        return waveforms, labels

    def random_shift(self, waveforms, itp, its, range=None):
        length = waveforms.shape[1]

        if range is None:
            indices = np.append(itp, its)
            min = indices.min() if len(indices) else length
            max = indices.max() if len(indices) else 0
            shift = np.random.randint(low=max - length, high=min + 1)

        else:
            min, max = range
            shift = np.random.randint(low=min, high=max + 1)

        waveforms = np.roll(waveforms, shift=shift, axis=1)
        itp = (itp + shift) % length
        its = (its + shift) % length
        return waveforms, itp, its

    def trim(self, waveforms, itp, its, range=None):
        if range is None:
            raise ValueError("range parameter must be provided")

        # 波形をトリミング
        start, end = range
        length = end - start
        waveforms = waveforms[:, start:end]

        # インデックスを修正
        itp -= start
        its -= start

        # 範囲外のインデックスを削除
        itp = itp[0 <= itp < length]
        its = its[0 <= its < length]

        return waveforms, itp, its

    def normalize(self, waveforms):
        mean = waveforms.mean(axis=1, keepdims=True)
        std = waveforms.std(axis=1, keepdims=True)
        std = np.where(std, std, 1)  # ゼロ除算を回避
        return (waveforms - mean) / std

    def generate_labels(self, waveforms, itp, its, filter_width=60):
        # 正解データ
        labels = np.zeros_like(waveforms)
        length = waveforms.shape[1]

        # ガウシアンフィルタ
        harf_width = filter_width // 2
        filtered = np.exp(
            -((np.arange(-harf_width, harf_width + 1)) ** 2)
            / (2 * (filter_width / 5) ** 2)
        )

        # itp
        for index in itp:
            start = index - harf_width
            end = index + harf_width + 1

            if 0 <= start and end <= length:
                labels[1, start:end, 0] = filtered

        # its
        for index in its:
            start = index - harf_width
            end = index + harf_width + 1

            if 0 <= start and end <= length:
                labels[2, start:end, 0] = filtered

        # ノイズ
        labels[0, ...] = 1 - np.sum(labels[1:, ...], axis=0)
        return labels


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    path_dir = "data/npz/train/"
    path_csv = "data/npz/train.csv"

    dataset = GeoSciAIDataset(path_dir, path_csv)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for x, y in dataloader:
        for waveforms, labels in zip(x, y):
            # plot(waveforms, labels)
            pass
