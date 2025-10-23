# src/pose/csv_backend.py
from __future__ import annotations
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from .base import PoseBackend

POSE_LANDMARKS = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

class CSVBackend(PoseBackend):
    """
    Soporta dos formatos:
    (A) ANCHO (exportado por MediaPipe): frame,time_s,<name>_{x,y,z,v}...
    (B) LARGO (tidy): video_id,frame,lmk_id,x,y,z,visibility
    """

    def __init__(self, video_path, csv_path, resize_w=960):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError("No se pudo abrir video")

        self.resize_w = resize_w
        self.vid = Path(video_path).stem

        df = pd.read_csv(csv_path)
        cols = set(df.columns.astype(str))

        # Detectar formato
        wide = any((name + "_x") in cols for name in POSE_LANDMARKS)
        long = {"frame", "lmk_id"}.issubset(cols)

        if wide:
            self.mode = "wide"
            # asegurar columnas mínimas
            if "frame" not in df.columns:
                raise RuntimeError("CSV ancho sin columna 'frame'")
            self.df = df
            self.max_frame = int(df["frame"].max())
        elif long:
            self.mode = "long"
            if "video_id" in df.columns:
                df = df[df["video_id"].astype(str) == self.vid]
            # sanity
            needed = {"frame","lmk_id","x","y"}
            if not needed.issubset(set(df.columns)):
                raise RuntimeError("CSV largo sin columnas requeridas: frame,lmk_id,x,y")
            self.df = df
            self.max_frame = int(df["frame"].max()) if not df.empty else -1
        else:
            raise RuntimeError("Formato de CSV no reconocido (ni ancho ni largo).")

    def iter_frames(self):
        idx = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            if w != self.resize_w:
                img = cv2.resize(img, (self.resize_w, int(h * self.resize_w / w)))
            yield {"frame_idx": idx, "image": img}
            idx += 1

    def landmarks_for_frame(self, frame_idx):
        if self.mode == "wide":
            row = self.df[self.df["frame"] == frame_idx]
            if row.empty:
                return None
            row = row.iloc[0]
            out = []
            for name in POSE_LANDMARKS:
                x = row.get(f"{name}_x", np.nan)
                y = row.get(f"{name}_y", np.nan)
                z = row.get(f"{name}_z", np.nan)
                v = row.get(f"{name}_v", 0.0)
                out.append((x, y, z, v))
            return out

        # modo largo
        sub = self.df[self.df["frame"] == frame_idx]
        if sub.empty:
            return None
        sub = sub.sort_values("lmk_id")
        out = []
        for _, r in sub.iterrows():
            x = r["x"]; y = r["y"]
            z = r["z"] if "z" in sub.columns else np.nan
            v = r["visibility"] if "visibility" in sub.columns else 0.0
            out.append((x, y, z, v))
        return out
