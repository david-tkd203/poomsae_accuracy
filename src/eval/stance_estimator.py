# src/eval/stance_estimator.py
from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import pandas as pd

LMK = dict(L_SH=11, R_SH=12, L_ANK=27, R_ANK=28)

def _series_xy(df: pd.DataFrame, lmk_id: int, nframes: int) -> np.ndarray:
    sub = df[df["lmk_id"] == lmk_id][["frame","x","y"]]
    arr = np.full((nframes, 2), np.nan, dtype=np.float32)
    if len(sub) > 0:
        idx = sub["frame"].to_numpy(int)
        xy = sub[["x","y"]].to_numpy(np.float32)
        idx = np.clip(idx, 0, nframes-1)
        arr[idx] = xy
    arr = np.nan_to_num(arr, nan=0.5)  # centro si faltan
    # ffill/bfill simple
    for k in range(2):
        v = arr[:,k]
        last = np.nan
        for i in range(len(v)):
            if not np.isnan(v[i]): last = v[i]
            elif not np.isnan(last): v[i] = last
        last = np.nan
        for i in range(len(v)-1,-1,-1):
            if not np.isnan(v[i]): last = v[i]
            elif not np.isnan(last): v[i] = last
        arr[:,k] = v
    return arr

def estimate_stance_code(df: pd.DataFrame, frame_idx: int, nframes: int) -> str:
    """Clasifica grosera: ap_kubi / dwit_kubi / beom_seogi usando tobillos y heading."""
    lsh = _series_xy(df, LMK["L_SH"], nframes)
    rsh = _series_xy(df, LMK["R_SH"], nframes)
    lank = _series_xy(df, LMK["L_ANK"], nframes)
    rank = _series_xy(df, LMK["R_ANK"], nframes)

    f = max(0, min(frame_idx, nframes-1))
    sh = rsh[f] - lsh[f]
    theta = np.arctan2(sh[1], sh[0])  # rad

    # rotar los tobillos al marco del tronco
    def rot(p):
        c, s = np.cos(-theta), np.sin(-theta)
        x = p[0] - 0.5; y = p[1] - 0.5
        xr = c*x - s*y; yr = s*x + c*y
        return np.array([xr, yr], dtype=np.float32)

    LA = rot(lank[f]); RA = rot(rank[f])
    d = RA - LA
    fwd = abs(d[0])     # separación “adelante-atrás”
    lat = abs(d[1])     # separación lateral

    # heurísticas (ajusta si hace falta):
    if fwd >= 0.18 and lat <= 0.10:
        return "ap_kubi"
    if fwd <= 0.12 and lat <= 0.12:
        return "beom_seogi"
    return "dwit_kubi"
