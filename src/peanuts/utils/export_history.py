import os
import pandas as pd


def export_history(
    filepath: str,
    epoch: int,
    loss: float,
    precision_p: float,
    recall_p: float,
    f1score_p: float,
    precision_s: float,
    recall_s: float,
    f1score_s: float,
) -> None:
    row = {
        "epoch": epoch,
        "loss": loss,
        "precision_p": precision_p,
        "recall_p": recall_p,
        "f1score_p": f1score_p,
        "precision_s": precision_s,
        "recall_s": recall_s,
        "f1score_s": f1score_s,
    }
    df = pd.DataFrame([row], columns=row.keys())
    path = os.path.join(os.getcwd(), filepath)
    
    if os.path.exists(path):
        df.to_csv(path, header=False, index=False, mode="a")
        
    else:
        df.to_csv(path, index=False, mode="w")
