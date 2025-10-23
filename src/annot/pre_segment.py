# src/tools/pre_segment.py
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
from typing import List, Tuple, Iterable
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")

def iter_videos(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    out: List[Path] = []
    for ext in VIDEO_EXTS:
        out += list(path.rglob(f"*{ext}"))
    return sorted(out)

def motion_series(cap: cv2.VideoCapture, down_w: int, stride: int) -> Tuple[np.ndarray, float]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=False)
    vals: List[float] = []

    pbar = tqdm(total=total if total>0 else None, desc="Frames", unit="f", leave=False)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if stride > 1 and (i % stride) != 0:
            i += 1; pbar.update(1); continue
        i += 1; pbar.update(1)

        h, w = frame.shape[:2]
        if down_w and w > down_w:
            nh = int(h * (down_w / w))
            frame = cv2.resize(frame, (down_w, nh))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg = bg.apply(gray)
        # limpia ruido
        fg = cv2.medianBlur(fg, 3)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
        vals.append((fg > 0).mean())
    pbar.close()
    return np.asarray(vals, dtype=np.float32), float(fps/stride)

def smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or w >= len(x):
        return x
    # media móvil
    c = np.cumsum(np.insert(x, 0, 0.0))
    y = (c[w:] - c[:-w]) / float(w)
    # alinear tamaño
    pad_left = w//2
    pad_right = len(x) - len(y) - pad_left
    return np.pad(y, (pad_left, pad_right), mode="edge")

def segments_from_activity(a: np.ndarray, fps_eff: float, thr: float,
                           min_dur: float, merge_gap: float) -> List[Tuple[float,float,float]]:
    active = a >= thr
    segs: List[Tuple[int,int]] = []
    s = None
    for i, on in enumerate(active):
        if on and s is None:
            s = i
        elif not on and s is not None:
            segs.append((s, i-1)); s = None
    if s is not None:
        segs.append((s, len(active)-1))

    # filtrar por duración mínima y fusionar huecos cortos
    def to_time(idx: int) -> float:
        return max(0.0, idx / fps_eff)

    out: List[Tuple[float,float,float]] = []
    last = None
    for f0, f1 in segs:
        t0, t1 = to_time(f0), to_time(f1)
        if (t1 - t0) < min_dur:
            continue
        if last and (t0 - last[1]) <= merge_gap:
            # fusiona con anterior
            last = (last[0], t1)
        else:
            if last: out.append((last[0], last[1], 0.0))
            last = (t0, t1)
    if last:
        out.append((last[0], last[1], 0.0))

    # score = media de actividad dentro del segmento
    results: List[Tuple[float,float,float]] = []
    for t0, t1, _ in out:
        i0, i1 = int(t0 * fps_eff), int(t1 * fps_eff)
        i1 = min(len(a)-1, max(i1, i0+1))
        score = float(a[i0:i1+1].mean())
        results.append((t0, t1, score))
    return results

def process_video(vpath: Path, down_w: int, stride: int, smooth_win_s: float,
                  thr: float, min_dur: float, merge_gap: float) -> List[Tuple[float,float,float]]:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[WARN] No se puede abrir: {vpath}")
        return []
    series, fps_eff = motion_series(cap, down_w=down_w, stride=stride)
    cap.release()

    sw = max(1, int(round(smooth_win_s * fps_eff)))
    sseries = smooth(series, sw)
    segs = segments_from_activity(sseries, fps_eff, thr=thr, min_dur=min_dur, merge_gap=merge_gap)
    return segs

def main():
    ap = argparse.ArgumentParser(description="Pre-segmentación por movimiento (sugerencias de in/out).")
    ap.add_argument("--in", dest="inp", required=True, help="Video o carpeta con videos")
    ap.add_argument("--out-csv", required=True, help="Ruta CSV de salida")
    ap.add_argument("--alias", default="", help="Alias opcional (p.ej. 8yang) para la columna 'alias'")
    ap.add_argument("--down-w", type=int, default=640, help="Redimensionar ancho para análisis")
    ap.add_argument("--stride", type=int, default=1, help="Procesar cada N frames (>=1)")
    ap.add_argument("--smooth-win", type=float, default=0.25, help="Ventana de suavizado en segundos")
    ap.add_argument("--thr", type=float, default=0.015, help="Umbral de actividad [0..1]")
    ap.add_argument("--min-dur", type=float, default=0.80, help="Duración mínima del segmento (s)")
    ap.add_argument("--merge-gap", type=float, default=0.40, help="Fusión de huecos menores a (s)")
    args = ap.parse_args()

    base = Path(args.inp)
    vids = iter_videos(base)
    if not vids:
        sys.exit("No se encontraron videos.")

    rows = []
    for vp in tqdm(vids, desc="Videos"):
        segs = process_video(vp, args.down_w, args.stride, args.smooth_win, args.thr, args.min_dur, args.merge_gap)
        for (t0, t1, score) in segs:
            rows.append({
                "alias": args.alias,
                "video": str(vp),
                "start_s": round(float(t0), 3),
                "end_s": round(float(t1), 3),
                "score": round(float(score), 5)
            })

    df = pd.DataFrame(rows, columns=["alias","video","start_s","end_s","score"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")
    print(f"[OK] Sugerencias: {args.out_csv}  ({len(df)} segmentos)")

if __name__ == "__main__":
    main()
