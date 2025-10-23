# src/tools/extract_landmarks.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2
import pandas as pd
import numpy as np

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")

def iter_videos(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    out: list[Path] = []
    for ext in VIDEO_EXTS:
        out += list(path.rglob(f"*{ext}"))
    return sorted(out)

def extract_for_video(
    vpath: Path,
    out_csv: Path,
    resize_w: int,
    overwrite: bool,
    model_complexity: int = 2,
    det_conf: float = 0.5,
    track_conf: float = 0.5,
    vis_min: float = 0.15,          # NUEVO: bajo este visibility → NaN (forzamos ffill/bfill luego)
    clamp01: bool = True            # NUEVO: normalizados a [0,1] robusto
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists() and not overwrite:
        print(f"[SKIP] {out_csv} (existe, use --overwrite para regenerar)")
        return

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir: {vpath}")
        return

    try:
        import mediapipe as mp
    except ImportError:
        sys.exit("Falta mediapipe. Instala: pip install mediapipe opencv-python pandas numpy tqdm")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=det_conf,
        min_tracking_confidence=track_conf,
    )

    rows = []
    vid = vpath.stem
    idx = 0
    n_landmarks = 33

    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if resize_w and w != resize_w:
            nh = int(round(h * (resize_w / float(w))))
            img = cv2.resize(img, (resize_w, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            for j in range(n_landmarks):
                lm = lms[j]
                x = float(getattr(lm, "x", np.nan))
                y = float(getattr(lm, "y", np.nan))
                z = float(getattr(lm, "z", 0.0))
                v = float(getattr(lm, "visibility", 0.0))

                if not np.isfinite(x) or not np.isfinite(y) or v < vis_min:
                    x, y = np.nan, np.nan
                elif clamp01:
                    # MP puede salirse ligeramente de [0,1]; clamp para estabilidad de etapas posteriores
                    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                    y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)

                rows.append((vid, idx, j, x, y, z, v))
        else:
            for j in range(n_landmarks):
                rows.append((vid, idx, j, np.nan, np.nan, 0.0, 0.0))

        idx += 1

    cap.release()
    pose.close()

    if not rows:
        print(f"[WARN] Sin landmarks: {vpath}")
        return

    df = pd.DataFrame(rows, columns=["video_id","frame","lmk_id","x","y","z","visibility"])
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[OK] {out_csv}  ({len(df)} filas, frames={idx})")

def main():
    ap = argparse.ArgumentParser(description="Extracción de landmarks con MediaPipe Pose (salida normalizada 0..1, NaN si visibility baja).")
    ap.add_argument("--in", dest="inp", required=True, help="Video o carpeta con videos")
    ap.add_argument("--out-root", required=True, help="Raíz de salida p.ej. data/landmarks")
    ap.add_argument("--alias", required=True, help="Alias p.ej. 8yang")
    ap.add_argument("--subset", default="", help="train/val/test (opcional)")
    ap.add_argument("--resize-w", type=int, default=960, help="Ancho de procesamiento")
    ap.add_argument("--overwrite", action="store_true", help="Regenerar CSV si existe")

    # Calidad
    ap.add_argument("--model-complexity", type=int, default=2)
    ap.add_argument("--det-conf", type=float, default=0.5)
    ap.add_argument("--track-conf", type=float, default=0.5)

    # Robustez NUEVO
    ap.add_argument("--vis-min", type=float, default=0.15, help="Umbral de visibility para aceptar x,y (NaN si menor)")
    ap.add_argument("--no-clamp01", action="store_true", help="No forzar x,y a [0,1]")

    args = ap.parse_args()

    base = Path(args.inp)
    vids = iter_videos(base)
    if not vids:
        sys.exit("No se encontraron videos.")

    out_root = Path(args.out_root) / args.alias
    if args.subset:
        out_root = out_root / args.subset

    for vp in vids:
        out_csv = out_root / f"{vp.stem}.csv"
        extract_for_video(
            vp, out_csv,
            resize_w=args.resize_w, overwrite=args.overwrite,
            model_complexity=args.model_complexity,
            det_conf=args.det_conf, track_conf=args.track_conf,
            vis_min=args.vis_min, clamp01=(not args.no_clamp01)
        )

if __name__ == "__main__":
    main()
