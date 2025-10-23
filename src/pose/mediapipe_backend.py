# src/pose/mediapipe_backend.py
from __future__ import annotations
import argparse, sys, os, csv
from pathlib import Path
from typing import Iterable, List, Tuple, Optional
import cv2
import numpy as np
from tqdm import tqdm

# --------- listado estable de 33 landmarks de MediaPipe Pose ---------
POSE_LANDMARKS = [
    "nose","left_eye_inner","left_eye","left_eye_outer","right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right","left_shoulder","right_shoulder",
    "left_elbow","right_elbow","left_wrist","right_wrist","left_pinky","right_pinky",
    "left_index","right_index","left_thumb","right_thumb","left_hip","right_hip",
    "left_knee","right_knee","left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

VIDEO_EXTS = (".mp4",".mkv",".webm",".mov",".m4v",".avi",".mpg",".mpeg")

def iter_videos(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    vids: List[Path] = []
    for ext in VIDEO_EXTS:
        vids += list(path.rglob(f"*{ext}"))
    return sorted(vids)

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def export_landmarks_mp(video_path: Path, out_csv: Path, *, stride: int = 1,
                        model_complexity: int = 1,
                        min_det: float = 0.5, min_track: float = 0.5,
                        smooth: bool = True) -> None:
    """
    Escribe CSV con columnas:
      frame,time_s, <name>_x,<name>_y,<name>_z,<name>_v  (para los 33 landmarks)
    Coordenadas normalizadas [0..1], z en metros relativos, visibility∈[0..1].
    """
    import mediapipe as mp  # importa aquí para fallar rápido si no está disponible
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ensure_dir(out_csv.parent)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # encabezado
        header = ["frame","time_s"]
        for name in POSE_LANDMARKS:
            header += [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_v"]
        writer.writerow(header)

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            smooth_landmarks=smooth,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track
        ) as pose:
            idx = 0
            pbar = tqdm(total=total if total>0 else None, desc=video_path.name, unit="f")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pbar.update(1)
                if stride > 1 and (idx % stride) != 0:
                    idx += 1
                    continue
                idx += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)

                row = [idx-1, (idx-1)/fps]
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    for j in range(33):
                        row += [lm[j].x, lm[j].y, lm[j].z, lm[j].visibility]
                else:
                    # sin detección: NaNs
                    for _ in range(33):
                        row += [np.nan, np.nan, np.nan, 0.0]
                writer.writerow(row)
            pbar.close()
    cap.release()

def main():
    ap = argparse.ArgumentParser(description="Exporta landmarks MediaPipe Pose a CSV.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in-video", type=str, help="Ruta a un video")
    g.add_argument("--in-dir", type=str, help="Carpeta con videos (recursivo)")
    ap.add_argument("--out-csv", type=str, help="CSV de salida (modo --in-video)")
    ap.add_argument("--out-dir", type=str, help="Carpeta para CSV (modo --in-dir)")
    ap.add_argument("--stride", type=int, default=1, help="Procesar cada N frames")
    ap.add_argument("--model-complexity", type=int, default=1, choices=(0,1,2))
    ap.add_argument("--min-det", type=float, default=0.5)
    ap.add_argument("--min-track", type=float, default=0.5)
    ap.add_argument("--no-smooth", action="store_true")
    ap.add_argument("--alias", type=str, help="Si se usa --in-dir, subcarpeta destino: data/landmarks/<alias>/[...].csv")
    args = ap.parse_args()

    if args.in_video:
        if not args.out_csv:
            sys.exit("--out-csv requerido con --in-video")
        vp = Path(args.in_video)
        export_landmarks_mp(
            vp, Path(args.out_csv),
            stride=args.stride,
            model_complexity=args.model_complexity,
            min_det=args.min_det, min_track=args.min_track,
            smooth=not args.no_smooth
        )
        print(f"[OK] {args.out_csv}")
        return

    # carpeta
    in_dir = Path(args.in_dir)
    if not in_dir.exists():
        sys.exit(f"No existe: {in_dir}")
    if not args.out_dir and not args.alias:
        sys.exit("Use --out-dir o --alias para dirigir la salida.")
    out_root = Path(args.out_dir) if args.out_dir else (Path("data/landmarks") / args.alias)
    ensure_dir(out_root)

    videos = iter_videos(in_dir)
    if not videos:
        sys.exit("Sin videos en la carpeta indicada.")

    for vp in videos:
        rel = vp.relative_to(in_dir) if vp.is_relative_to(in_dir) else Path(vp.name)
        csv_path = out_root / rel.with_suffix(".csv")
        ensure_dir(csv_path.parent)
        try:
            export_landmarks_mp(
                vp, csv_path,
                stride=args.stride,
                model_complexity=args.model_complexity,
                min_det=args.min_det, min_track=args.min_track,
                smooth=not args.no_smooth
            )
            print(f"[OK] {csv_path}")
        except Exception as e:
            print(f"[FAIL] {vp} -> {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
