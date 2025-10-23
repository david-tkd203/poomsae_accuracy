import numpy as np

def angle_deg(a, b, c):
    """Ángulo ABC en grados. a,b,c: (x,y) o (x,y,z)."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    nba = ba / (np.linalg.norm(ba) + 1e-8)
    nbc = bc / (np.linalg.norm(bc) + 1e-8)
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def diff_series(x):
    x = np.asarray(x, float)
    d = np.diff(x, prepend=x[:1])
    return d

def moving_avg(x, w=5):
    if w <= 1: return x
    x = np.asarray(x, float)
    c = np.convolve(x, np.ones(w)/w, mode="same")
    return c
