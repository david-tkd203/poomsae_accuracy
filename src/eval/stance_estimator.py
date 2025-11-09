# src/eval/stance_estimator.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

LMK = dict(L_SH=11, R_SH=12, L_ANK=27, R_ANK=28, L_HIP=23, R_HIP=24)

def _series_xy(df: pd.DataFrame, lmk_id: int, nframes: int) -> np.ndarray:
    """
    Extrae serie temporal de coordenadas (x,y) para un landmark.
    Usa forward-fill y backward-fill para interpolar NaN.
    """
    sub = df[df["lmk_id"] == lmk_id][["frame","x","y"]]
    arr = np.full((nframes, 2), np.nan, dtype=np.float32)
    
    if len(sub) > 0:
        idx = sub["frame"].to_numpy(int)
        xy = sub[["x","y"]].to_numpy(np.float32)
        idx = np.clip(idx, 0, nframes-1)
        arr[idx] = xy
    
    # Forward-fill y backward-fill para interpolar gaps
    for k in range(2):
        v = arr[:,k]
        # Forward fill
        last_valid = None
        for i in range(len(v)):
            if not np.isnan(v[i]):
                last_valid = v[i]
            elif last_valid is not None:
                v[i] = last_valid
        
        # Backward fill
        last_valid = None
        for i in range(len(v)-1, -1, -1):
            if not np.isnan(v[i]):
                last_valid = v[i]
            elif last_valid is not None:
                v[i] = last_valid
        
        arr[:,k] = v
    
    # Si todavía quedan NaN (todo el video sin ese landmark), usar centro
    arr = np.nan_to_num(arr, nan=0.5)
    
    return arr

def validate_frame_quality(df: pd.DataFrame, frame_idx: int, nframes: int) -> Tuple[bool, float]:
    """
    Valida calidad de landmarks en un frame específico.
    
    Returns:
        (is_valid, confidence): True si el frame es usable, score de confianza [0-1]
    """
    required_landmarks = [LMK["L_SH"], LMK["R_SH"], LMK["L_ANK"], LMK["R_ANK"], 
                          LMK["L_HIP"], LMK["R_HIP"]]
    
    frame_data = df[df["frame"] == frame_idx]
    
    if len(frame_data) == 0:
        logger.debug(f"Frame {frame_idx}: Sin datos de landmarks")
        return False, 0.0
    
    present_landmarks = set(frame_data["lmk_id"].values)
    missing = [lmk for lmk in required_landmarks if lmk not in present_landmarks]
    
    if missing:
        logger.debug(f"Frame {frame_idx}: Faltan landmarks {missing}")
        confidence = 1.0 - (len(missing) / len(required_landmarks))
        return len(missing) <= 2, confidence  # Tolerar hasta 2 faltantes
    
    # Check si los valores son razonables (dentro de imagen normalizada [0,1])
    coords = frame_data[["x", "y"]].values
    if np.any(coords < 0) or np.any(coords > 1):
        logger.debug(f"Frame {frame_idx}: Coordenadas fuera de rango [0,1]")
        return False, 0.5
    
    # Check shoulder width mínimo (evita divisiones por cero)
    lsh_data = frame_data[frame_data["lmk_id"] == LMK["L_SH"]][["x", "y"]]
    rsh_data = frame_data[frame_data["lmk_id"] == LMK["R_SH"]][["x", "y"]]
    
    if len(lsh_data) > 0 and len(rsh_data) > 0:
        shoulder_width = np.linalg.norm(rsh_data.values[0] - lsh_data.values[0])
        if shoulder_width < 0.02:  # Muy estrecho, probablemente corrupto
            logger.debug(f"Frame {frame_idx}: Shoulder width muy pequeño ({shoulder_width:.4f})")
            return False, 0.3
    
    return True, 1.0

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
