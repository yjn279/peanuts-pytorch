import numpy as np
from typing import List

from .find_peaks import find_peaks

def get_diffs(pred, true, mph=0.6, mpd=10) -> List[int]:
    pred = find_peaks(pred, mph=mph, mpd=mpd)
    true = find_peaks(true, mph=mph, mpd=mpd)
    
    diffs = pred - true[:, np.newaxis]
    if diffs.size == 0:
        return []
        
    diffs = np.abs(diffs).min(axis=1)
    diffs = diffs.tolist()
    return diffs
