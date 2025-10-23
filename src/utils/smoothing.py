import numpy as np
from scipy.signal import savgol_filter

def smooth(x, win=5, poly=2):
    x = np.asarray(x, float)
    if len(x) < win or win < 3:
        return x
    if win % 2 == 0: win += 1
    return savgol_filter(x, window_length=win, polyorder=poly, mode="interp")
