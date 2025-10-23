# src/segmentation/move_capture.py
from __future__ import annotations
import math, json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import cv2

# Mapeo indices MediaPipe Pose
LMK = dict(
    NOSE=0,
    L_EYE_IN=1, L_EYE=2, L_EYE_OUT=3, R_EYE_IN=4, R_EYE=5, R_EYE_OUT=6,
    L_EAR=7, R_EAR=8,
    L_MOUTH=9, R_MOUTH=10,
    L_SH=11, R_SH=12, L_ELB=13, R_ELB=14, L_WRIST=15, R_WRIST=16,
    L_PINKY=17, R_PINKY=18, L_INDEX=19, R_INDEX=20, L_THUMB=21, R_THUMB=22,
    L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30, L_FOOT=31, R_FOOT=32
)

# Conexiones para dibujar esqueleto (subset habitual)
CONNECTIONS = [
    (LMK["L_SH"],LMK["R_SH"]), (LMK["L_SH"],LMK["L_HIP"]), (LMK["R_SH"],LMK["R_HIP"]), (LMK["L_HIP"],LMK["R_HIP"]),
    (LMK["L_SH"],LMK["L_ELB"]), (LMK["L_ELB"],LMK["L_WRIST"]),
    (LMK["R_SH"],LMK["R_ELB"]), (LMK["R_ELB"],LMK["R_WRIST"]),
    (LMK["L_HIP"],LMK["L_KNEE"]), (LMK["L_KNEE"],LMK["L_ANK"]),
    (LMK["R_HIP"],LMK["R_KNEE"]), (LMK["R_KNEE"],LMK["R_ANK"]),
    (LMK["L_ANK"],LMK["L_HEEL"]), (LMK["R_ANK"],LMK["R_HEEL"]),
    (LMK["L_HEEL"],LMK["L_FOOT"]), (LMK["R_HEEL"],LMK["R_FOOT"])
]

END_EFFECTORS = ("L_WRIST","R_WRIST","L_ANK","R_ANK")

# ---------------- util numérica ----------------

def movavg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    win = min(win, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    y = (c[win:] - c[:-win]) / float(win)
    pad_l = win // 2
    pad_r = len(x) - len(y) - pad_l
    return np.pad(y, ((pad_l, pad_r), (0, 0)), mode="edge")

def central_diff(x: np.ndarray, fps: float) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x)
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    dx[0] = (x[1] - x[0]) * fps
    dx[-1] = (x[-1] - x[-2]) * fps
    return dx

def unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i-1]
        if d > 180:  out[i:] -= 360
        elif d < -180: out[i:] += 360
    return out

def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    BA = a - b; BC = c - b
    num = float(np.dot(BA, BC))
    den = float(np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
    cosv = max(-1.0, min(1.0, num/den))
    return math.degrees(math.acos(cosv))

def bucket_turn(delta_deg: float) -> str:
    d = ((delta_deg + 180) % 360) - 180
    opts = [(-180,"TURN_180"), (-135,"LEFT_135"), (-90,"LEFT_90"), (-45,"LEFT_45"),
            (0,"STRAIGHT"), (45,"RIGHT_45"), (90,"RIGHT_90"), (135,"RIGHT_135"), (180,"TURN_180")]
    return min(opts, key=lambda t: abs(d - t[0]))[1]

def dir_octant(vx: float, vy: float) -> str:
    ang = math.degrees(math.atan2(vy, vx))
    dirs = [("E",-22.5,22.5), ("SE",22.5,67.5), ("S",67.5,112.5), ("SW",112.5,157.5),
            ("W",157.5,180.0), ("W",-180.0,-157.5), ("NW",-157.5,-112.5),
            ("N",-112.5,-67.5), ("NE",-67.5,-22.5)]
    for lab,a,b in dirs:
        if a <= ang <= b: return lab
    return "E"

# ---------------- cargar series ----------------

def load_landmarks_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"video_id","frame","lmk_id","x","y"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"CSV inválido, faltan columnas: {req - set(df.columns)}")
    return df

def series_xy(df: pd.DataFrame, lmk_id: int, nframes: int) -> np.ndarray:
    sub = df[df["lmk_id"] == lmk_id][["frame","x","y"]]
    arr = np.full((nframes, 2), np.nan, dtype=np.float32)
    idx = sub["frame"].to_numpy(dtype=int)
    xy = sub[["x","y"]].to_numpy(np.float32)
    idx = np.clip(idx, 0, nframes-1)
    arr[idx] = xy
    for k in range(2):
        v = arr[:,k]
        if np.isnan(v).any():
            for i in range(1, len(v)):
                if np.isnan(v[i]) and not np.isnan(v[i-1]):
                    v[i] = v[i-1]
            for i in range(len(v)-2, -1, -1):
                if np.isnan(v[i]) and not np.isnan(v[i+1]):
                    v[i] = v[i+1]
            arr[:,k] = v
    return arr

# ---------------- helpers de heading/energía ----------------

def _heading_series_deg(lsh: np.ndarray, rsh: np.ndarray) -> np.ndarray:
    sh_vec = rsh - lsh
    heading = np.degrees(np.arctan2(sh_vec[:,1], sh_vec[:,0]))
    return unwrap_deg(heading)

def _energy_series(eff_xy: Dict[str,np.ndarray], fps: float, smooth_win: int) -> Tuple[np.ndarray, np.ndarray]:
    speeds = []
    for name in END_EFFECTORS:
        xy = movavg(eff_xy[name], smooth_win)
        v = np.linalg.norm(central_diff(xy, fps), axis=1)
        speeds.append(v)
    speeds = np.stack(speeds, axis=1)          # (T,4)
    sp = np.nanmax(speeds, axis=1)             # energía global (T,)
    return speeds, sp

# ---------------- segmentación ----------------

def detect_segments_from_speed(speeds: np.ndarray, fps: float,
                               vstart: float, vstop: float,
                               min_dur: float, min_gap: float) -> List[Tuple[int,int]]:
    smax = np.nanmax(speeds, axis=1)
    active = smax >= vstart
    segs: List[Tuple[int,int]] = []
    s = None
    for i, on in enumerate(active):
        if on and s is None:
            s = i
        elif not on and s is not None:
            segs.append((s, i-1)); s = None
    if s is not None:
        segs.append((s, len(active)-1))

    frames_min = int(round(min_dur * fps))
    frames_gap = int(round(min_gap * fps))
    out: List[Tuple[int,int]] = []
    last = None
    for a,b in segs:
        if (b - a + 1) < frames_min: continue
        if last and (a - last[1] - 1) <= frames_gap:
            last = (last[0], b)
        else:
            if last: out.append(last)
            last = (a,b)
    if last: out.append(last)
    return out

def _quantile_cut_segments(sp: np.ndarray, expected_n: int, min_frames: int) -> List[Tuple[int,int]]:
    T = len(sp)
    if expected_n <= 0 or T <= 1:
        return [(0, max(0, T-1))]
    e = np.maximum(sp, 0.0).astype(np.float64)
    E = np.cumsum(e + 1e-6)  # evita masa cero
    targets = np.linspace(0.0, float(E[-1]), expected_n+1)
    cuts = np.searchsorted(E, targets, side="left")
    cuts[0] = 0; cuts[-1] = T-1
    # de-duplicar y garantizar longitud mínima
    # avance de 1 frame cuando haya empates
    for k in range(1, len(cuts)):
        if cuts[k] <= cuts[k-1]:
            cuts[k] = min(T-1, cuts[k-1] + 1)
    # construir pares
    segs: List[Tuple[int,int]] = []
    for k in range(expected_n):
        a = int(cuts[k]); b = int(cuts[k+1])
        if b - a + 1 < min_frames:
            b = min(T-1, a + min_frames - 1)
        if k < expected_n-1 and b >= cuts[k+1]:
            b = max(a, cuts[k+1]-1)
        segs.append((a, b))
    # ajustar solapes residuales
    for k in range(1, len(segs)):
        prev_a, prev_b = segs[k-1]
        a, b = segs[k]
        if a <= prev_b:
            a = prev_b + 1
            if a > b: a = b
            segs[k] = (a, b)
    # clamp a límites
    segs = [(max(0,a), min(T-1,b)) for a,b in segs if a <= b]
    # si por ajustes se perdió alguno, rellenar uniformemente
    while len(segs) < expected_n:
        segs.append((segs[-1][1], min(T-1, segs[-1][1])))
    return segs[:expected_n]

def _disp(xy: np.ndarray) -> float:
    if len(xy) < 2: return 0.0
    d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
    return float(np.nansum(d))

def _choose_active_limb(eff_xy: Dict[str,np.ndarray], a: int, b: int) -> str:
    cand = {k: eff_xy[k][a:b+1] for k in END_EFFECTORS}
    return max(cand.items(), key=lambda kv: _disp(kv[1]))[0]

def resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    if len(xy) == 0: return np.zeros((0,2), np.float32)
    if len(xy) == 1: return np.tile(xy[:1], (n,1))
    d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-8:
        return np.tile(xy[:1], (n,1))
    t = np.linspace(0, s[-1], n)
    xs = np.interp(t, s, xy[:,0]); ys = np.interp(t, s, xy[:,1])
    return np.stack([xs, ys], axis=1)

# ---------------- clasificación de postura/patada ----------------

def classify_stance_frame(lmk: Dict[str,np.ndarray]) -> str:
    LANK, RANK = lmk["L_ANK"], lmk["R_ANK"]
    LKNE, RKNE = lmk["L_KNEE"], lmk["R_KNEE"]
    LHIP, RHIP = lmk["L_HIP"], lmk["R_HIP"]
    foot_dist = np.linalg.norm(LANK - RANK)
    hip_c = 0.5*(LHIP + RHIP)
    feet_c = 0.5*(LANK + RANK)
    hip_behind = (hip_c[1] > feet_c[1] + 0.02)
    L_knee = angle3(LHIP, LKNE, LANK)
    R_knee = angle3(RHIP, RKNE, RANK)
    knee_min = min(L_knee, R_knee)
    knee_max = max(L_knee, R_knee)
    if foot_dist > 0.10 and knee_min < 155 and knee_max > 165:
        return "ap_kubi"
    if hip_behind and foot_dist > 0.07 and knee_min > 160:
        return "dwit_kubi"
    if foot_dist <= 0.07 and hip_behind:
        return "beom_seogi"
    return "unknown"

def detect_kick_type(seg_xy: np.ndarray, hip_xy: np.ndarray) -> str:
    if len(seg_xy) < 5: return "none"
    rel_y = hip_xy[:,1] - seg_xy[:,1]
    amp = float(np.percentile(rel_y, 95) - np.percentile(rel_y, 5))
    if amp > 0.10:
        return "ap_chagi"
    return "none"

def arm_motion_direction(wrist_xy: np.ndarray) -> str:
    if len(wrist_xy) < 2: return "NONE"
    v = wrist_xy[-1] - wrist_xy[0]
    return dir_octant(float(v[0]), float(v[1]))

# ---------------- dataclasses ----------------

@dataclass
class MoveSegment:
    idx: int
    t_start: float
    t_end: float
    duration: float
    active_limb: str
    speed_peak: float
    rotation_deg: float
    rotation_bucket: str
    height_end: float
    path: List[Tuple[float, float]]  # [0..1]
    stance_pred: str
    kick_pred: str
    arm_dir: str

@dataclass
class CaptureResult:
    video_id: str
    fps: float
    nframes: int
    moves: List[MoveSegment]
    def to_json(self) -> str:
        return json.dumps({
            "video_id": self.video_id,
            "fps": self.fps,
            "nframes": self.nframes,
            "moves": [asdict(m) for m in self.moves]
        }, ensure_ascii=False, indent=2)

# ---------------- núcleo ----------------

def capture_moves_from_csv(csv_path: Path, *, video_path: Optional[Path],
                           vstart: float = 0.60, vstop: float = 0.20,
                           min_dur: float = 0.25, min_gap: float = 0.18,
                           smooth_win: int = 5, poly_n: int = 20,
                           min_path_norm: float = 0.02,
                           expected_n: Optional[int] = None) -> CaptureResult:
    df = load_landmarks_csv(csv_path)
    vid = Path(csv_path).stem
    nframes = int(df["frame"].max()) + 1
    fps = 30.0
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or nframes)
        cap.release()

    lsh = series_xy(df, LMK["L_SH"], nframes)
    rsh = series_xy(df, LMK["R_SH"], nframes)
    lhip = series_xy(df, LMK["L_HIP"], nframes)
    rhip = series_xy(df, LMK["R_HIP"], nframes)
    lank = series_xy(df, LMK["L_ANK"], nframes)
    rank = series_xy(df, LMK["R_ANK"], nframes)
    lwri = series_xy(df, LMK["L_WRIST"], nframes)
    rwri = series_xy(df, LMK["R_WRIST"], nframes)
    lkne = series_xy(df, LMK["L_KNEE"], nframes)
    rkne = series_xy(df, LMK["R_KNEE"], nframes)

    heading = _heading_series_deg(lsh, rsh)

    eff_xy: Dict[str,np.ndarray] = {
        "L_WRIST": lwri, "R_WRIST": rwri, "L_ANK": lank, "R_ANK": rank
    }

    speeds, sp = _energy_series(eff_xy, fps, smooth_win)

    # umbrales adaptativos
    med = np.nanmedian(sp)
    mad = np.nanmedian(np.abs(sp - med)) + 1e-6
    vstart_adapt = max(vstart, med + 2.0*mad)
    vstop_adapt  = max(vstop,  med + 1.0*mad)

    segs = detect_segments_from_speed(speeds, fps, vstart_adapt, vstop_adapt, min_dur, min_gap)

    # forzar N esperado (p.ej., 24 en Pal Jang) usando cortes por cuantiles de energía
    if isinstance(expected_n, int) and expected_n > 0:
        frames_min = int(round(min_dur * fps))
        if len(segs) != expected_n:
            segs = _quantile_cut_segments(sp, expected_n, frames_min)

    moves: List[MoveSegment] = []
    for i,(a,b) in enumerate(segs, start=1):
        # miembro activo por desplazamiento dentro del segmento
        limb = _choose_active_limb(eff_xy, a, b)
        seg_sp_col = {
            k: np.linalg.norm(central_diff(movavg(eff_xy[k][a:b+1], smooth_win), fps), axis=1) for k in END_EFFECTORS
        }
        v_peak = float(max(np.nanmax(v) if len(v) else 0.0 for v in seg_sp_col.values()))

        path_xy = eff_xy[limb][a:b+1, :]
        # cuando se fuerza expected_n no descartamos por trayectoria corta
        if not (isinstance(expected_n, int) and expected_n > 0):
            if len(path_xy) >= 2:
                path_len = float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))
            else:
                path_len = 0.0
            if path_len < min_path_norm:
                continue

        h0 = float(heading[a]); h1 = float(heading[b])
        dhead = ((h1 - h0 + 180) % 360) - 180
        bucket = bucket_turn(dhead)

        poly = resample_polyline(path_xy, poly_n)
        y_end = float(path_xy[-1, 1]) if len(path_xy) else float('nan')

        stance_votes = []
        for f in range(a, b+1):
            lmk = {
                "L_ANK": lank[f], "R_ANK": rank[f],
                "L_KNEE": lkne[f], "R_KNEE": rkne[f],
                "L_HIP": lhip[f], "R_HIP": rhip[f]
            }
            stance_votes.append(classify_stance_frame(lmk))
        if stance_votes:
            st = max(set(stance_votes), key=stance_votes.count)
            if st == "unknown":
                fd = float(np.linalg.norm(lank[a]-rank[a]))
                st = "ap_kubi" if fd>0.10 else ("beom_seogi" if fd<0.07 else "dwit_kubi")
        else:
            st = "unknown"

        hip_seg = 0.5*(lhip[a:b+1,:] + rhip[a:b+1,:])
        kick = "none"
        if limb in ("L_ANK","R_ANK") and len(path_xy) >= 2:
            kick = detect_kick_type(path_xy, hip_seg)
        a_dir = "NONE"
        if limb in ("L_WRIST","R_WRIST") and len(path_xy) >= 2:
            a_dir = arm_motion_direction(path_xy)

        moves.append(MoveSegment(
            idx=i, t_start=a/fps, t_end=b/fps, duration=max(1,(b-a+1))/fps,
            active_limb=limb, speed_peak=v_peak,
            rotation_deg=float(dhead), rotation_bucket=bucket,
            height_end=y_end, path=[(float(x), float(y)) for x,y in poly],
            stance_pred=st, kick_pred=kick, arm_dir=a_dir
        ))

    return CaptureResult(video_id=Path(csv_path).stem, fps=fps, nframes=nframes, moves=moves)

# ---------------- helpers para preview ----------------

def _draw_pose_from_csv_frame(frame: np.ndarray, csv_df: pd.DataFrame, fidx: int):
    """Dibuja esqueleto en 'frame' usando el CSV del frame 'fidx' (soporta x,y normalizados o en pixeles)."""
    h, w = frame.shape[:2]
    rows = csv_df[csv_df["frame"] == fidx]
    if rows.empty:
        return
    rows = rows[np.isfinite(rows["x"]) & np.isfinite(rows["y"])]
    if rows.empty:
        return
    in01 = (rows["x"].between(-0.1, 1.1)) & (rows["y"].between(-0.1, 1.1))
    normed = (in01.mean() >= 0.8)
    if not normed:
        mx = float(max(rows["x"].max(), rows["y"].max()))
        if np.isfinite(mx) and mx <= 2.0:
            normed = True

    pts = {}
    for _, r in rows.iterrows():
        x = float(r["x"]); y = float(r["y"])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if normed:
            px = int(round(max(0.0, min(1.0, x)) * w))
            py = int(round(max(0.0, min(1.0, y)) * h))
        else:
            px = int(round(max(0.0, min(float(w - 1), x))))
            py = int(round(max(0.0, min(float(h - 1), y))))
        pts[int(r["lmk_id"])] = (px, py)

    if not pts:
        return

    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
    for pid, p in pts.items():
        cv2.circle(frame, p, 3, (0, 255, 0), -1)

def render_preview(video_path: Path, result: CaptureResult, out_path: Path,
                   csv_path: Optional[Path] = None, draw_pose: bool = True) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = result.fps if result.fps > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W,H))

    per_frame: Dict[int,List[MoveSegment]] = {}
    for m in result.moves:
        a = int(round(m.t_start * fps))
        b = int(round(m.t_end * fps))
        for f in range(a, b+1):
            per_frame.setdefault(f, []).append(m)

    csv_df = None
    if draw_pose and csv_path and Path(csv_path).exists():
        csv_df = load_landmarks_csv(csv_path)

    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        if csv_df is not None:
            _draw_pose_from_csv_frame(frame, csv_df, fidx)

        segs = per_frame.get(fidx, [])
        y0 = 26
        cv2.putText(frame, f"t={fidx/fps:6.2f}s", (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"t={fidx/fps:6.2f}s", (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        y = y0 + 24
        for m in segs[:3]:
            txt = f"#{m.idx} {m.active_limb} {m.rotation_bucket} {m.stance_pred} {m.kick_pred}"
            cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 2, cv2.LINE_AA)
            y += 20
        for m in segs:
            pts = np.array([(int(px*W), int(py*H)) for (px,py) in m.path], dtype=np.int32)
            for k in range(1, len(pts)):
                cv2.line(frame, pts[k-1], pts[k], (0,255,255), 2)
            if len(pts):
                cv2.circle(frame, pts[-1], 4, (0,140,255), -1)
        out.write(frame); fidx += 1
    out.release(); cap.release()
    return out_path
