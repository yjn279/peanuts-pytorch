import os

import numpy as np


def export_predictions(path, pred):
    dirname = os.path.dirname(f"predictions/{path}.npz")
    os.makedirs(dirname, exist_ok=True)

    pred = pred.reshape(3, 3001, 1)
    pred = pred.transpose(1, 2, 0)

    np.savez(f"predictions/{path}", pred=pred)
