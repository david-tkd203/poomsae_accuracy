# src/main_offline.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

from src.pose.csv_backend import CSVBackend

# Conexiones básicas (MediaPipe Pose 33 pts)
CONNECTIONS = [
    (11,12),(11,23),(12,24),(23,24),
    (11,13),(13,15),(12,14),(14,16),
    (23,25),(25,27),(24,26),(26,28),
    (27,31),(28,32),(29,31),(30,32)
]

def resolve_csv_for_video(video: Path, landmarks_csv: Path | None,
                          alias: str | None, landmarks_root: Path | None) -> Path:
    if landmarks_csv:
        return landmarks_csv
    parts = video.resolve().parts
    try:
        i = parts.index("raw_videos")
        eff_alias = alias or parts[i+1]
        rel = Path(*parts[i+2:]).with_suffix(".csv")
        root = Path("data/landmarks") if landmarks_root is None else Path(landmarks_root)
        return (root / eff_alias / rel)
    except ValueError:
        return video.with_suffix(".csv")

def to_pixels(landmarks, w, h):
    pts = []
    for tup in landmarks:
        if tup is None or len(tup) < 2:
            pts.append(None); continue
        x, y = tup[0], tup[1]
        if x is None or y is None or (isinstance(x, float) and np.isnan(x)) or (isinstance(y, float) and np.isnan(y)):
            pts.append(None); continue
        px = int(x * w) if x <= 2.0 else int(x)
        py = int(y * h) if y <= 2.0 else int(y)
        pts.append((px, py))
    return pts

def draw_pose(img, pts):
    for a,b in CONNECTIONS:
        if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
            cv2.line(img, pts[a], pts[b], (0,255,255), 2)
    for p in pts:
        if p: cv2.circle(img, p, 3, (0,255,0), -1)

def overlay_header(img, video_name: str, t_sec: float, fps: float, idx: int, total_frames: int):
    h, w = img.shape[:2]
    t_txt = f"{int(t_sec//60):02d}:{(t_sec%60):05.2f}"
    txt = f"Video: {video_name} | t={t_txt}  f={idx+1}/{total_frames}  fps={fps:.2f}"
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

def run(video_path: str,
        csv_landmarks_path: str,
        model_path: str | None = None,
        render_out: str | None = None,
        save_json: str | None = None,
        video_id: str | None = None):

    vid = Path(video_path)
    csvp = Path(csv_landmarks_path)
    if not vid.exists():
        raise SystemExit(f"No existe el video: {vid}")
    if not csvp.exists():
        print(f"[WARN] CSV no encontrado: {csvp}")

    be = CSVBackend(video_path=str(vid), csv_path=str(csvp), resize_w=960)

    # meta del video
    try:
        cap_meta = cv2.VideoCapture(str(vid))
        fps_native = float(cap_meta.get(cv2.CAP_PROP_FPS) or 30.0)
        n_frames_native = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap_meta.release()
    except Exception:
        fps_native, n_frames_native = 30.0, 0

    meta = {
        "video_id": video_id or vid.stem,
        "video_path": str(vid.resolve()),
        "landmarks_csv": str(csvp.resolve()),
        "alias": None,  # opcional; el reporte puede rellenar si aplica
        "fps": fps_native,
        "n_frames": n_frames_native
    }

    # writer si se pide render
    writer = None
    fps_cap = fps_native
    frame_size = None
    video_name = vid.name

    # warm-up para tamaño
    first_frame_image = None
    for f in be.iter_frames():
        first_frame_image = f["image"]
        break
    if first_frame_image is None:
        raise SystemExit("Video sin frames leíbles.")

    h0, w0 = first_frame_image.shape[:2]
    frame_size = (w0, h0)
    if render_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        Path(render_out).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(render_out, fourcc, fps_cap, frame_size)

    # re-iterar desde el inicio
    be.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n_frames = 0
    first_ts = None
    last_ts = None
    frame_preds = []   # por-frame (placeholder si no hay modelo)
    # intento de cargar modelo; si no existe, se etiqueta "desconocido"
    clf = None
    fe = None
    if model_path:
        try:
            from joblib import load as joblib_load
            clf = joblib_load(model_path)
            try:
                # extractor opcional para reproducir features de entrenamiento
                from src.features.feature_extractor import FeatureExtractor  # noqa: F401
                fe = FeatureExtractor()
            except Exception:
                fe = None
        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo: {e}")
            clf = None

    def basic_features(lms):
        # features mínimos y estables si no hay extractor (no deben romper)
        # devuelve dict “flat”; el pipeline de entrenamiento debe haber usado claves compatibles
        # aquí sólo generamos un puñado de ángulos normalizados por si acaso
        def angle(a,b,c):
            if any(v is None for v in (a,b,c)): return np.nan
            ax,ay = a[0], a[1]
            bx,by = b[0], b[1]
            cx,cy = c[0], c[1]
            v1 = np.array([ax-bx, ay-by], dtype=np.float32)
            v2 = np.array([cx-bx, cy-by], dtype=np.float32)
            n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
            if n1<1e-6 or n2<1e-6: return np.nan
            cosang = float(np.clip(np.dot(v1,v2)/(n1*n2), -1.0, 1.0))
            return np.degrees(np.arccos(cosang))
        # índices MediaPipe
        def pick(i):
            if i < len(lms) and lms[i] is not None:
                return (lms[i][0], lms[i][1])
            return None
        LSH, LEL, LWR = pick(11), pick(13), pick(15)
        RSH, REL, RWR = pick(12), pick(14), pick(16)
        LHP, LKN, LNK = pick(23), pick(25), pick(27)
        RHP, RKN, RNK = pick(24), pick(26), pick(28)
        return {
            "ang_left_elbow": angle(LSH, LEL, LWR),
            "ang_right_elbow": angle(RSH, REL, RWR),
            "ang_left_knee": angle(LHP, LKN, LNK),
            "ang_right_knee": angle(RHP, RKN, RNK),
        }

    # recorrido
    idx = 0
    while True:
        ok, img = be.cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if w != 960:
            img = cv2.resize(img, (960, int(h*960/w)))
            h, w = img.shape[:2]
        lms = be.landmarks_for_frame(idx)
        t = float(idx) / float(fps_cap)
        if first_ts is None: first_ts = t
        last_ts = t

        label = "desconocido"
        proba = None

        if lms is not None:
            pts = to_pixels(lms, w, h)
            draw_pose(img, pts)

            if clf is not None:
                # extraer features
                if fe is not None:
                    try:
                        feats = fe.from_landmarks(lms)  # tu extractor debe exponer este método
                    except Exception:
                        feats = basic_features(pts)
                else:
                    feats = basic_features(pts)
                try:
                    import pandas as pd
                    X = pd.DataFrame([feats])
                    y_hat = clf.predict(X)[0]
                    label = str(y_hat)
                    if hasattr(clf, "predict_proba"):
                        proba = float(np.max(clf.predict_proba(X)))
                except Exception as e:
                    # si algo falla, mantenemos “desconocido”
                    pass
        else:
            cv2.putText(img, "Sin landmarks", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(img, "Sin landmarks", (12, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        overlay_header(img, video_name, t, fps_cap, idx, n_frames_native if n_frames_native>0 else idx+1)

        frame_preds.append({
            "frame": int(idx),
            "t": round(t, 3),
            "label": label,
            "proba": None if proba is None else round(proba, 4)
        })

        if writer is not None:
            writer.write(img)

        idx += 1
        n_frames += 1

    if writer is not None:
        writer.release()

    # segmentación por runs de etiqueta (simple)
    segments = []
    if frame_preds:
        cur_lab = frame_preds[0]["label"]
        s = frame_preds[0]["t"]
        for i in range(1, len(frame_preds)):
            if frame_preds[i]["label"] != cur_lab:
                e = frame_preds[i-1]["t"]
                segments.append({"start_s": round(s,3), "end_s": round(e,3), "label": cur_lab})
                cur_lab = frame_preds[i]["label"]
                s = frame_preds[i]["t"]
        segments.append({"start_s": round(s,3), "end_s": round(frame_preds[-1]["t"],3), "label": cur_lab})
    else:
        duration = (last_ts - (first_ts or 0.0)) if last_ts is not None else 0.0
        segments = [{"start_s": 0.0, "end_s": round(float(duration), 3), "label": "desconocido"}]

    payload = {
        "meta": meta,
        "model_used": str(model_path) if model_path else None,
        "preds": frame_preds,
        "segments": segments
    }

    if save_json:
        Path(save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[OK] JSON -> {save_json}")

def main():
    ap = argparse.ArgumentParser(description="Ejecución offline con landmarks CSV (JSON y render opcionales).")
    ap.add_argument("--model", required=False, help="Modelo .joblib a usar (opcional)")
    ap.add_argument("--video", required=True, help="Ruta al video")
    ap.add_argument("--landmarks-csv", help="CSV de landmarks; si no, se resuelve automáticamente")
    ap.add_argument("--landmarks-root", help="Raíz de landmarks (default data/landmarks)")
    ap.add_argument("--alias", help="Alias para resolver rutas espejo (p.ej. 8yang)")
    ap.add_argument("--render-out", help="Salida de video renderizado", default=None)
    ap.add_argument("--save-json", help="Salida JSON", default=None)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"No existe el video: {video_path}")

    csv_path = resolve_csv_for_video(
        video=video_path,
        landmarks_csv=Path(args.landmarks_csv) if args.landmarks_csv else None,
        alias=args.alias,
        landmarks_root=Path(args.landmarks_root) if args.landmarks_root else None
    )

    run(video_path=str(video_path),
        csv_landmarks_path=str(csv_path),
        model_path=args.model,
        render_out=args.render_out,
        save_json=args.save_json,
        video_id=video_path.stem)

if __name__ == "__main__":
    main()
