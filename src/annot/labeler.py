# src/annot/labeler.py
from __future__ import annotations
import argparse, sys, os, time, platform
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import cv2

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")
LABELS = ["correcto", "error_leve", "error_grave"]

# ---------- util UI ----------

def time_str(t: float) -> str:
    m = int(t // 60)
    s = t - 60*m
    return f"{m:02d}:{s:06.3f}"

def draw_help_panel(img, show_help: bool):
    if not show_help:
        return
    H, W = img.shape[:2]
    pad = 12
    box_w = min(900, W - 2*pad)
    box_h = 220
    x1, y1 = pad, H - box_h - pad
    x2, y2 = x1 + box_w, y1 + box_h

    # fondo semitransparente
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (32, 32, 32), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 2)

    lines = [
        "CONTROLES:",
        "ESPACIO: play/pause   |   A/D: -/+0.1s   |   Shift+A/D: -/+1s   |   I/O: marcar IN/OUT",
        "1/2/3: correcto / error_leve / error_grave   |   M: siguiente movimiento",
        "ENTER o N: guardar segmento   |   P: sugerencia previa   |   Borrar: limpiar   |   Q/Esc: salir",
        "H: mostrar/ocultar ayuda",
    ]
    y = y1 + 32
    for i, txt in enumerate(lines):
        cv2.putText(img, txt, (x1 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x1 + 16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)
        y += 34

    # fila de "botones" visuales
    btns = ["[I] IN", "[O] OUT", "[1] Correcto", "[2] Error leve", "[3] Error grave", "[ENTER] Guardar"]
    bx = x1 + 16
    by = y1 + box_h - 48
    for b in btns:
        (tw, th), _ = cv2.getTextSize(b, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        bw, bh = tw + 24, th + 18
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (180, 180, 180), 2)
        cv2.putText(img, b, (bx + 12, by + bh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(img, b, (bx + 12, by + bh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        bx += bw + 10

# ---------- decodificador unificado (decord -> OpenCV) ----------

class VideoReader:
    def __init__(self, path: str, backend: str = "auto"):
        """
        backend: 'auto' | 'decord' | 'opencv'
        """
        self.path = path
        self.backend = backend
        self.use_decord = False
        self._fps = 0.0
        self._n = 0
        self._vr = None
        self.cap = None

        if platform.system() == "Windows":
            # fuerza preferencia FFmpeg en OpenCV para evitar MSMF en .webm
            os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

        if backend in ("auto", "decord"):
            try:
                from decord import VideoReader as DVR, cpu, gpu
                try:
                    self._vr = DVR(path, ctx=gpu(0))
                except Exception:
                    self._vr = DVR(path, ctx=cpu(0))
                self.use_decord = True
                self._fps = float(self._vr.get_avg_fps())
                self._n = int(len(self._vr))
                self.backend = "decord"
                print(f"[INFO] VideoReader: usando decord ({self._fps:.2f} FPS, {self._n} frames)")
                return
            except Exception as e:
                if backend == "decord":
                    raise RuntimeError(f"No se pudo usar decord: {e}")

        # Fallback OpenCV
        self._open_opencv()

    def _open_opencv(self):
        self.use_decord = False
        # probar preferencia FFmpeg si está disponible
        try:
            self.cap = cv2.VideoCapture(self.path, cv2.CAP_FFMPEG)
        except Exception:
            self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir: {self.path}")
        self._fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 60.0)
        self._n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.backend = "opencv"
        print(f"[INFO] VideoReader: usando OpenCV/FFmpeg ({self._fps:.2f} FPS, {self._n} frames)")

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def nframes(self) -> int:
        return self._n

    def _frame_at_time_decord(self, t_seconds: float) -> Tuple[np.ndarray, int]:
        # decord indexa por frame
        idx = int(round(t_seconds * self._fps))
        idx = max(0, min(idx, self._n - 1))
        try:
            frame = self._vr[idx].asnumpy()
        except Exception as e:
            # Fallback a OpenCV si decord falla
            print(f"[WARN] decord fallo en seek/index ({e}), cambiando a OpenCV…")
            self._vr = None
            self._open_opencv()
            return self._frame_at_time_opencv(t_seconds)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame, idx

    def _frame_at_time_opencv(self, t_seconds: float) -> Tuple[np.ndarray, int]:
        idx = int(round(t_seconds * self._fps))
        idx = max(0, min(idx, self._n - 1))
        # seek por índice
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            # reabrir y reintentar
            self.cap.release()
            self._open_opencv()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok or frame is None:
                # intentar leer secuencial
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 1))
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError("No se pudo leer el frame solicitado")
        return frame, idx

    def frame_at_time(self, t_seconds: float) -> Tuple[np.ndarray, int]:
        if self.use_decord:
            return self._frame_at_time_decord(t_seconds)
        return self._frame_at_time_opencv(t_seconds)

    def close(self):
        if self._vr is not None:
            self._vr = None
        if self.cap is not None:
            self.cap.release()

# ---------- CSV I/O ----------

def load_suggestions(csv_path: Optional[Path]) -> pd.DataFrame:
    if csv_path and csv_path.exists():
        df = pd.read_csv(csv_path)
        required = {"video","start_s","end_s"}
        if not required.issubset(set(df.columns)):
            raise SystemExit("CSV de sugerencias debe contener: video,start_s,end_s")
        if "label" not in df.columns:
            df["label"] = ""
        if "move_id" not in df.columns:
            df["move_id"] = ""
        return df
    return pd.DataFrame(columns=["video","start_s","end_s","label","move_id"])

def write_row(out_csv: Path, row: Dict):
    header = not out_csv.exists()
    with open(out_csv, "a", encoding="utf-8", newline="") as f:
        if header:
            f.write("video,start_s,end_s,label,move_id\n")
        f.write(f"{row['video']},{row['start_s']:.3f},{row['end_s']:.3f},{row['label']},{row.get('move_id','')}\n")

def iter_videos(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    vids: List[Path] = []
    for ext in VIDEO_EXTS:
        vids += list(path.rglob(f"*{ext}"))
    return sorted(vids)

# ---------- etiqueta interactiva ----------

def label_session(video: Path, out_csv: Path, sugg_rows: List[Tuple[float,float,str,str]],
                  step_small: float, step_large: float, autoplay: bool,
                  moves: List[str], backend: str):

    vr = VideoReader(str(video), backend=backend)
    fps = vr.fps
    dur = (vr.nframes - 1) / fps if vr.nframes > 0 else 0.0

    win = "Labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    idx_seg = 0
    t = 0.0
    IN: Optional[float] = None
    OUT: Optional[float] = None
    LABEL = ""
    move_idx = 0 if moves else -1
    show_help = True  # ayuda visible al inicio

    def jump_to(tn: float):
        nonlocal t
        t = max(0.0, min(dur, tn))

    def apply_suggestion(i: int):
        nonlocal IN, OUT, LABEL, move_idx, t
        if 0 <= i < len(sugg_rows):
            s, e, lab, mov = sugg_rows[i]
            IN, OUT = float(s), float(e)
            LABEL = lab or ""
            if moves and mov in moves:
                move_idx = moves.index(mov)
            t = IN

    if sugg_rows:
        apply_suggestion(0)

    playing = autoplay
    last_time = time.time()

    while True:
        frame, fidx = vr.frame_at_time(t)
        disp = frame.copy()

        # overlay superior
        y0 = 28
        def put(txt, y):
            cv2.putText(disp, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(disp, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        put(f"Video: {video.name}  [{time_str(t)}/{time_str(dur)}]  FPS:{fps:.2f}  Backend:{vr.backend}", y0)
        put(f"IN: {time_str(IN) if IN is not None else '--:--.--'}   OUT: {time_str(OUT) if OUT is not None else '--:--.--'}", y0+28)
        cur_move = moves[move_idx] if (moves and move_idx>=0) else ""
        put(f"LABEL: {LABEL or '(sin)'}   MOVE: {cur_move or '(opcional)'}   Seg: {idx_seg+1}/{max(len(sugg_rows),1)}", y0+56)

        # barra de IN/OUT y posición
        if IN is not None and OUT is not None and IN < OUT:
            H = disp.shape[0]; W = disp.shape[1]
            x_in  = int((IN / max(dur,1e-6)) * (W-20)) + 10
            x_out = int((OUT / max(dur,1e-6)) * (W-20)) + 10
            cv2.rectangle(disp, (x_in, H-16), (x_out, H-8), (0,255,255), -1)
        x_cur = int((t / max(dur,1e-6)) * (disp.shape[1]-20)) + 10
        cv2.line(disp, (x_cur, 0), (x_cur, disp.shape[0]), (0,255,0), 1)

        # panel de ayuda
        draw_help_panel(disp, show_help)

        cv2.imshow(win, disp)

        # temporización de reproducción
        delay = 1
        if playing:
            now = time.time()
            dt = now - last_time
            last_time = now
            t += dt
            if t >= dur:
                playing = False
                t = dur

        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q') or key == 27:  # Q o Esc
            break
        elif key == ord('h'):
            show_help = not show_help
        elif key == 32:  # espacio
            playing = not playing
            last_time = time.time()
        elif key == ord('a'):
            jump_to(t - step_small)
        elif key == ord('d'):
            jump_to(t + step_small)
        elif key == ord('A'):
            jump_to(t - step_large)
        elif key == ord('D'):
            jump_to(t + step_large)
        elif key == ord('i'):
            IN = t
        elif key == ord('o'):
            OUT = t
        elif key == ord('1'):
            LABEL = LABELS[0]
        elif key == ord('2'):
            LABEL = LABELS[1]
        elif key == ord('3'):
            LABEL = LABELS[2]
        elif key == ord('m'):
            if moves:
                move_idx = 0 if move_idx < 0 else (move_idx + 1) % len(moves)
        elif key in (ord('\r'), ord('\n'), ord('n')):  # Enter o N
            if IN is not None and OUT is not None and OUT > IN and LABEL:
                row = {
                    "video": str(video),
                    "start_s": float(IN),
                    "end_s": float(OUT),
                    "label": LABEL,
                    "move_id": moves[move_idx] if (moves and move_idx>=0) else ""
                }
                write_row(out_csv, row)
                idx_seg += 1
                if idx_seg < len(sugg_rows):
                    apply_suggestion(idx_seg)
                else:
                    IN = OUT = None
                    LABEL = ""
                    if t < dur: jump_to(t + 0.01)
        elif key == ord('p'):
            if idx_seg > 0 and len(sugg_rows) > 0:
                idx_seg -= 1
                apply_suggestion(idx_seg)
        elif key == 8:  # Backspace
            IN = OUT = None
            LABEL = ""

    vr.close()
    cv2.destroyAllWindows()

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Etiquetador rápido de segmentos con atajos de teclado.")
    ap.add_argument("--suggest-csv", type=str, help="CSV de sugerencias (video,start_s,end_s[,label,move_id])")
    ap.add_argument("--video", type=str, help="Etiquetar un solo video")
    ap.add_argument("--videos-dir", type=str, help="Carpeta con videos para recorrer")
    ap.add_argument("--out-csv", type=str, required=True, help="Salida: video,start_s,end_s,label,move_id")
    ap.add_argument("--moves-file", type=str, help="TXT con un move_id por línea (opcional)")
    ap.add_argument("--step-small", type=float, default=0.10, help="Paso pequeño en segundos")
    ap.add_argument("--step-large", type=float, default=1.00, help="Paso grande en segundos")
    ap.add_argument("--autoplay", action="store_true", help="Arrancar reproduciendo")
    ap.add_argument("--backend", type=str, choices=["auto","decord","opencv"], default="auto",
                    help="Backend de decodificación de video")
    args = ap.parse_args()

    sugg_df = load_suggestions(Path(args.suggest_csv)) if args.suggest_csv else pd.DataFrame(columns=["video","start_s","end_s","label","move_id"])

    moves: List[str] = []
    if args.moves_file and Path(args.moves_file).exists():
        moves = [ln.strip() for ln in Path(args.moves_file).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # lista de videos a recorrer
    videos: List[Path] = []
    if args.video:
        videos = [Path(args.video)]
    elif args.videos_dir:
        videos = iter_videos(Path(args.videos_dir))
    else:
        if not sugg_df.empty:
            vids = sorted(set(sugg_df["video"].tolist()))
            videos = [Path(v) for v in vids if Path(v).exists()]
        else:
            sys.exit("Indica --video, --videos-dir o --suggest-csv con rutas válidas.")

    for vp in videos:
        if not vp.exists():
            print(f"[WARN] No existe: {vp}")
            continue
        # filtra sugerencias para este video
        if not sugg_df.empty:
            rows = sugg_df[sugg_df["video"].astype(str) == str(vp)]
            sugg_rows = [(float(r.start_s), float(r.end_s), str(r.label), str(r.move_id)) for _, r in rows.iterrows()]
        else:
            sugg_rows = []
        print(f"[INFO] Video: {vp.name}  sugerencias: {len(sugg_rows)}")
        label_session(
            vp, out_csv, sugg_rows,
            step_small=args.step_small, step_large=args.step_large,
            autoplay=args.autoplay, moves=moves, backend=args.backend
        )

if __name__ == "__main__":
    main()
