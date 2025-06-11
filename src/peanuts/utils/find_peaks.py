from scipy.signal import find_peaks as _find_peaks


def find_peaks(x, mph=0.6, mpd=10):
    return _find_peaks(x, height=mph, distance=mpd)[0]
