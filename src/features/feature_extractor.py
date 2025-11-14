
from __future__ import annotations
# --- UTILIDAD PARA CREAR CARPETA DE SALIDA ---
from pathlib import Path
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Genera features biomecánicos desde landmarks")
    parser.add_argument('--in-root', required=True, help='Directorio de landmarks')
    parser.add_argument('--out-root', required=True, help='Directorio de salida de features')
    parser.add_argument('--alias', default='8yang', help='Alias del poomsae')
    parser.add_argument('--overwrite', action='store_true', help='Sobrescribir archivos existentes')
    args = parser.parse_args()

    # Crear carpeta de salida si no existe
    ensure_dir(args.out_root)

    # Aquí iría el resto del procesamiento real de features
    # Por ejemplo, cargar landmarks, procesar y guardar features
    # ...
# src/features/feature_extractor.py
import numpy as np, pandas as pd
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.features.angles import angles_from_landmarks
from src.utils.smoothing import smooth
from math import atan2, cos, sin

# Índices MediaPipe Pose
LMK = {"L_SH":11, "R_SH":12, "L_HIP":23, "R_HIP":24}

STAT_FUNCS = {
    "mean": np.nanmean,
    "std":  np.nanstd,
    "p10":  lambda x: np.nanpercentile(x, 10),
    "p50":  lambda x: np.nanpercentile(x, 50),
    "p90":  lambda x: np.nanpercentile(x, 90),
    "range": lambda x: np.nanmax(x) - np.nanmin(x),
    "max_speed": lambda v: np.nanmax(np.abs(np.gradient(v))),
    "max_acc":   lambda v: np.nanmax(np.abs(np.gradient(np.gradient(v)))),
}

def _rot2d(theta: float) -> np.ndarray:
    c, s = cos(theta), sin(theta)
    return np.array([[c, -s],[s, c]], dtype=np.float32)

def _ffill_bfill(arr: np.ndarray) -> np.ndarray:
    """Rellena NaN hacia adelante y atrás por columna (in-place safe)."""
    out = arr.copy()
    for k in range(out.shape[1]):
        v = out[:, k]
        if np.isnan(v).any():
            # ffill
            for i in range(1, len(v)):
                if np.isnan(v[i]) and not np.isnan(v[i-1]):
                    v[i] = v[i-1]
            # bfill
            for i in range(len(v)-2, -1, -1):
                if np.isnan(v[i]) and not np.isnan(v[i+1]):
                    v[i] = v[i+1]
            out[:, k] = v
    return out

def canonicalize_xy(xy33: np.ndarray) -> np.ndarray:
    """
    xy33: (33,2) en [0..1]. Devuelve (33,2) canónico:
      - origen = centro de caderas
      - eje Y = vector cadera->hombros
      - escala = ancho de hombros
      - refleja si es necesario para que L_SH.x < R_SH.x
    Si la entrada no tiene forma válida, devuelve el xy original.
    """
    if not isinstance(xy33, np.ndarray) or xy33.ndim != 2 or xy33.shape[1] < 2 or xy33.shape[0] < 25:
        return xy33

    xy = xy33[:, :2].astype(np.float32)
    if np.isnan(xy).any():
        xy = _ffill_bfill(xy)
        xy = np.nan_to_num(xy, nan=0.0)

    c_hip = 0.5 * (xy[LMK["L_HIP"]] + xy[LMK["R_HIP"]])
    c_sh  = 0.5 * (xy[LMK["L_SH"]]  + xy[LMK["R_SH"]])

    # trasladar
    xy = xy - c_hip

    # rotar a +Y
    v = c_sh - np.array([0.0, 0.0], dtype=np.float32)
    theta = atan2(v[1], v[0])               # ángulo vs +X
    R = _rot2d((np.pi/2) - theta)           # llevarlo a +Y
    xy = (R @ xy.T).T

    # escalar por ancho de hombros
    shw = float(np.linalg.norm(xy[LMK["R_SH"]] - xy[LMK["L_SH"]])) + 1e-6
    xy = xy / shw

    # reflejar si L_SH quedó a la derecha
    if xy[LMK["L_SH"], 0] > xy[LMK["R_SH"], 0]:
        xy[:, 0] *= -1.0

    return xy

def _apply_canon_to_lmks(lmks_one_frame):
    """
    Aplica canonicalización a la estructura de landmarks del frame,
    conservando z/visibility si existen. Soporta:
      - np.ndarray shape (33,2/3/4)
      - list/tuple de 33 items con [x,y,(z),(v)]
    Si no puede, retorna el original.
    """
    try:
        arr = np.asarray(lmks_one_frame)
        if arr.ndim != 2 or arr.shape[0] < 25 or arr.shape[1] < 2:
            return lmks_one_frame

        xy_canon = canonicalize_xy(arr[:, :2])

        # reconstruir manteniendo columnas extra (z, vis)
        out = arr.copy()
        out[:, 0:2] = xy_canon

        if isinstance(lmks_one_frame, np.ndarray):
            return out
        else:
            # convertimos a la misma “forma” de entrada (lista de tuplas)
            out_list = []
            for row in out:
                row = row.tolist()
                out_list.append(tuple(row))
            return out_list
    except Exception:
        return lmks_one_frame

class FeatureExtractor:
    def __init__(self, angle_triplets, smoothing_window=5, canonicalize=True):
        """
        angle_triplets: dict -> ver angles_from_landmarks()
        smoothing_window: ventana para suavizar las series angulares
        canonicalize: si True, aplica canonalización 2D antes de computar ángulos
        """
        self.triplets = angle_triplets
        self.win = smoothing_window
        self.use_canon = canonicalize

    def segment_features(self, lmks_seq, frames_idx, meta=None):
        """
        lmks_seq: lista de landmarks por frame (cada item = 33×(x,y[,z,vis])).
        frames_idx: índices absolutos de frame.
        meta: dict opcional (poom, movimiento, atleta, etc.)
        Return: dict de features + metadatos
        """
        # 1) canonalizar por frame para robustez a cámara
        if self.use_canon:
            lmks_seq_proc = [_apply_canon_to_lmks(l) for l in lmks_seq]
        else:
            lmks_seq_proc = lmks_seq

        # 2) series de ángulos (rotación-invariantes)
        series = {k: [] for k in self.triplets.keys()}
        for lmks in lmks_seq_proc:
            angs = angles_from_landmarks(lmks, self.triplets)
            for k, v in angs.items():
                series[k].append(v)

        # 3) suavizado + estadísticas (tus métricas actuales)
        feats = {}
        for name, vals in series.items():
            vals = np.asarray(vals, float)
            vals = smooth(vals, win=self.win, poly=2)
            feats[f"{name}_mean"]  = STAT_FUNCS["mean"](vals)
            feats[f"{name}_std"]   = STAT_FUNCS["std"](vals)
            feats[f"{name}_p10"]   = STAT_FUNCS["p10"](vals)
            feats[f"{name}_p50"]   = STAT_FUNCS["p50"](vals)
            feats[f"{name}_p90"]   = STAT_FUNCS["p90"](vals)
            feats[f"{name}_range"] = STAT_FUNCS["range"](vals)
            feats[f"{name}_max_speed"] = STAT_FUNCS["max_speed"](vals)
            feats[f"{name}_max_acc"]   = STAT_FUNCS["max_acc"](vals)

        # 4) metadatos mínimos
        if meta:
            feats.update({f"meta_{k}": v for k, v in meta.items()})
        feats["meta_frame_start"] = int(frames_idx[0])
        feats["meta_frame_end"]   = int(frames_idx[-1])
        feats["meta_len"]         = int(len(frames_idx))
        return feats
