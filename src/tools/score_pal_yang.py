# src/tools/score_pal_yang.py
from __future__ import annotations
import argparse, sys, json, math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import load_landmarks_csv

# ----------------- carga spec poomsae / pose -----------------

def load_spec(path: Path) -> Dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if "moves" not in data:
        raise SystemExit("Spec inválido: falta 'moves'")
    flat: List[Dict] = []
    for m in data["moves"]:
        key = f"{m.get('idx','?')}{m.get('sub','')}"
        m2 = dict(m)
        m2["_key"] = key
        flat.append(m2)
    data["_flat_moves"] = flat
    return data

def load_pose_spec(path: Optional[Path]) -> Dict:
    cfg = {
        "schema": "v1.1",
        "tolerances": {"turn_deg_tol": 30.0, "arm_octant_tol": 1, "dtw_good": 0.25},
        "levels": {"olgul_max_rel": -0.02, "arae_min_rel": 1.03, "momtong_band":[0.05,0.95]},
        "stances": {
            "ap_kubi":    {"ankle_dist_min_sw": 0.90, "front_knee_max_deg": 155.0, "rear_knee_min_deg": 165.0},
            "dwit_kubi":  {"ankle_dist_min_sw": 0.55, "ankle_dist_max_sw": 1.20, "hip_minus_feet_center_min": 0.02,
                           "front_knee_min_deg": 165.0, "rear_knee_max_deg": 155.0},
            "beom_seogi": {"ankle_dist_max_sw": 0.60, "hip_minus_feet_center_min": 0.015}
        },
        "kicks": {
            "ap_chagi": {"amp_min": 0.20, "peak_above_hip_min": 0.25},
            "ttwieo_ap_chagi": {"amp_min": 0.30, "peak_above_hip_min": 0.30, "airborne_min_frac": 0.10}
        }
    }
    if not path:
        return cfg
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        for k,v in d.items():
            if isinstance(v, dict):
                cfg.setdefault(k, {})
                cfg[k].update(v)
            else:
                cfg[k] = v
    except Exception:
        pass
    return cfg

# ----------------- helpers geom/dirección -----------------

_OCTANTS = ["E","NE","N","NW","W","SW","S","SE"]
_OCT_TO_IDX = {o:i for i,o in enumerate(_OCTANTS)}
def _oct_idx(name: str) -> Optional[int]: return _OCT_TO_IDX.get(str(name).upper(), None)
def _wrap_oct_delta(a: int, b: int) -> int:
    d = abs(a - b) % 8
    return min(d, 8 - d)

def _oct_vec(name: str) -> np.ndarray:
    name = str(name).upper()
    ang_deg = {"E":0,"NE":-45,"N":-90,"NW":-135,"W":180,"SW":135,"S":90,"SE":45}.get(name,0)
    ang = math.radians(ang_deg)
    return np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)

def _unit_seq(poly: List[Tuple[float,float]]) -> np.ndarray:
    p = np.asarray(poly, np.float32)
    if len(p) < 2: return np.zeros((0,2), np.float32)
    d = np.diff(p, axis=0)
    n = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    u = d / n
    keep = (n[:,0] > 1e-6)
    return u[keep]

def _dtw(A: np.ndarray, B: np.ndarray) -> float:
    if len(A)==0 or len(B)==0: return 1.0
    n,m = len(A), len(B)
    C = np.zeros((n,m), np.float32)
    for i in range(n):
        dots = (A[i:i+1] @ B.T).ravel()
        C[i,:] = 0.5*(1.0 - np.clip(dots, -1.0, 1.0))
    D = np.full((n+1,m+1), np.inf, np.float32); D[0,0]=0.0
    for i in range(1,n+1):
        for j in range(1,m+1):
            D[i,j] = C[i-1,j-1] + min(D[i-1,j], D[i,j-1], D[i-1,j-1])
    return float(D[n,m] / max(n,m))

def _turn_code_deg(code: str) -> float:
    c = str(code).upper()
    if c in ("NONE","STRAIGHT",""): return 0.0
    if c == "LEFT_90": return -90.0
    if c == "RIGHT_90": return 90.0
    if c in ("LEFT_180","RIGHT_180","TURN_180"): return 180.0
    if c == "LEFT_270": return -270.0
    if c == "RIGHT_270": return 270.0
    try: return float(c)
    except: return 0.0

def _deg_diff(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

# ----------------- series / util landmarks -----------------

def _series_xy_range(df: pd.DataFrame, lmk_id: int, a: int, b: int) -> np.ndarray:
    L = max(1, b - a + 1)
    arr = np.full((L, 2), np.nan, np.float32)
    sub = df[(df["lmk_id"]==lmk_id) & (df["frame"]>=a) & (df["frame"]<=b)][["frame","x","y"]]
    if not sub.empty:
        idx = (sub["frame"].to_numpy(int) - a).clip(0, L-1)
        arr[idx] = sub[["x","y"]].to_numpy(np.float32)
    for k in range(2):
        v = arr[:,k]
        for i in range(1, L):
            if np.isnan(v[i]) and not np.isnan(v[i-1]):
                v[i] = v[i-1]
        for i in range(L-2, -1, -1):
            if np.isnan(v[i]) and not np.isnan(v[i+1]):
                v[i] = v[i+1]
        arr[:,k] = v
    return arr

def _wrist_rel_series_robust(df: pd.DataFrame, a: int, b: int, wrist_id: int) -> np.ndarray:
    wr = _series_xy_range(df, wrist_id, a, b)
    lsh = _series_xy_range(df, 11, a, b)
    rsh = _series_xy_range(df, 12, a, b)
    lhp = _series_xy_range(df, 23, a, b)
    rhp = _series_xy_range(df, 24, a, b)
    y_w  = wr[:,1]
    y_sh = 0.5*(lsh[:,1] + rsh[:,1])
    y_hp = 0.5*(lhp[:,1] + rhp[:,1])
    den = (y_hp - y_sh)
    den[den == 0] = np.nan
    rel = (y_w - y_sh) / den
    return rel

def _level_from_spec_thresholds(rel_end: float, levels_cfg: Dict[str,object]) -> str:
    if not isinstance(rel_end, (int,float)) or math.isnan(rel_end):
        return "MOMTONG"
    if rel_end <= float(levels_cfg.get("olgul_max_rel", -0.02)):
        return "OLGUL"
    if rel_end >= float(levels_cfg.get("arae_min_rel", 1.03)):
        return "ARAE"
    return "MOMTONG"

# --------- util para segmentos sintéticos ---------

def _heading_series_deg(df: pd.DataFrame) -> np.ndarray:
    maxf = int(df["frame"].max())
    lsh = _series_xy_range(df, 11, 0, maxf)
    rsh = _series_xy_range(df, 12, 0, maxf)
    v = rsh - lsh
    ang = np.degrees(np.arctan2(v[:,1], v[:,0]))
    out = ang.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i-1]
        if d > 180:  out[i:] -= 360
        elif d < -180: out[i:] += 360
    return out

def _estimate_rotation_deg(df: pd.DataFrame, a: int, b: int) -> float:
    hd = _heading_series_deg(df)
    a = int(np.clip(a, 0, len(hd)-1)); b = int(np.clip(b, 0, len(hd)-1))
    h0 = float(hd[a]); h1 = float(hd[b])
    d = ((h1 - h0 + 180.0) % 360.0) - 180.0
    return float(d)

def _disp(xy: np.ndarray) -> float:
    if len(xy) < 2: return 0.0
    d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
    return float(np.nansum(d))

def _choose_active_limb(df: pd.DataFrame, a: int, b: int) -> str:
    cand = {
        "L_WRIST": _series_xy_range(df, 15, a, b),
        "R_WRIST": _series_xy_range(df, 16, a, b),
        "L_ANK":   _series_xy_range(df, 27, a, b),
        "R_ANK":   _series_xy_range(df, 28, a, b),
    }
    return max(cand.items(), key=lambda kv: _disp(kv[1]))[0]

def _poly_from_limb(df: pd.DataFrame, limb: str, a: int, b: int, n: int = 20) -> List[Tuple[float,float]]:
    lid = {"L_WRIST":15,"R_WRIST":16,"L_ANK":27,"R_ANK":28}[limb]
    xy = _series_xy_range(df, lid, a, b)
    if len(xy) == 0:
        return []
    d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1)) if len(xy) >= 2 else np.array([0.0])
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        xs = np.full((n,), xy[0,0], np.float32); ys = np.full((n,), xy[0,1], np.float32)
    else:
        t = np.linspace(0.0, float(s[-1]), n)
        xs = np.interp(t, s, xy[:,0]); ys = np.interp(t, s, xy[:,1])
    return [(float(xs[i]), float(ys[i])) for i in range(n)]

# ----------------- postura / patada / codo -----------------

def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b; bc = c - b
    num = float(np.dot(ba, bc)); den = float(np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    return float(math.degrees(math.acos(np.clip(num/den, -1, 1))))

def _stance(df: pd.DataFrame, a: int, b: int) -> str:
    sub = df[(df["frame"]>=a) & (df["frame"]<=b)]
    ankL = sub[sub["lmk_id"]==27][["x","y"]].to_numpy()
    ankR = sub[sub["lmk_id"]==28][["x","y"]].to_numpy()
    shL  = sub[sub["lmk_id"]==11][["x","y"]].to_numpy()
    shR  = sub[sub["lmk_id"]==12][["x","y"]].to_numpy()
    if len(ankL)==0 or len(ankR)==0 or len(shL)==0 or len(shR)==0: return "unknown"
    aL = ankL.mean(axis=0); aR = ankR.mean(axis=0)
    sL = shL.mean(axis=0);  sR = shR.mean(axis=0)
    shoulder_w = float(np.linalg.norm(sR - sL) + 1e-6)
    base = float(np.linalg.norm(aR - aL))
    ratio = base / shoulder_w
    if ratio >= 0.30: return "ap_kubi"
    if ratio <= 0.18: return "beom_seogi"
    return "dwit_kubi"

def _kick_required(tech_kor: str, tech_es: str) -> bool:
    s = f"{tech_kor} {tech_es}".lower()
    return ("chagi" in s) or ("patada" in s)

def _kick_metrics(df: pd.DataFrame, a: int, b: int) -> Tuple[float,float]:
    sub = df[(df["frame"]>=a) & (df["frame"]<=b)]
    ank = sub[sub["lmk_id"].isin([27,28])][["frame","y"]]
    hips= sub[sub["lmk_id"].isin([23,24])][["frame","y"]]
    if ank.empty or hips.empty: return 0.0, 0.0
    hip = hips.groupby("frame")["y"].mean()
    ank_min = ank.groupby("frame")["y"].min()
    join = hip.to_frame("hip").join(ank_min.to_frame("ank"), how="inner")
    rel = (join["hip"].to_numpy(np.float32) - join["ank"].to_numpy(np.float32))
    amp = float(np.nanpercentile(rel,95) - np.nanpercentile(rel,5))
    peak = float(np.nanmax(rel)) if rel.size else 0.0
    return max(0.0, amp), max(0.0, peak)

def _elbow_angle(a, b, c) -> float:
    ba = a - b; bc = c - b
    num = float(np.dot(ba, bc)); den = float(np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    return float(math.degrees(math.acos(np.clip(num/den, -1, 1))))

def _median_elbow_extension(df: pd.DataFrame, a: int, b: int) -> float:
    sub = df[(df["frame"]>=a) & (df["frame"]<=b)]
    need = {11,12,13,14,15,16}
    if not set(sub["lmk_id"].unique()).intersection(need):
        return 150.0
    def get_xy(k):
        return sub[sub["lmk_id"]==k][["frame","x","y"]]
    LSH,RSH = get_xy(11), get_xy(12)
    LELB,RELB= get_xy(13), get_xy(14)
    LWR,RWR  = get_xy(15), get_xy(16)
    mL = LSH.merge(LELB,on="frame",suffixes=("_sh","_elb")).merge(LWR,on="frame")
    mR = RSH.merge(RELB,on="frame",suffixes=("_sh","_elb")).merge(RWR,on="frame")
    def seq_elbow(m):
        if m.empty: return np.array([],np.float32)
        sh = m[["x_sh","y_sh"]].to_numpy(np.float32)
        el = m[["x_elb","y_elb"]].to_numpy(np.float32)
        wr = m[["x","y"]].to_numpy(np.float32)
        return np.array([_elbow_angle(sh[i], el[i], wr[i]) for i in range(len(m))], np.float32)
    arrL = seq_elbow(mL); arrR = seq_elbow(mR)
    allv = np.concatenate([arrL, arrR]) if (arrL.size or arrR.size) else np.array([150.0],np.float32)
    return float(np.nanmedian(allv))

# ----------------- clasificación por sub-criterio -----------------

def _classify_rotation(rot_meas_deg: float, rot_exp_code: str, tol: float) -> str:
    exp_deg = _turn_code_deg(rot_exp_code)
    if abs(exp_deg) == 180.0:
        d = min(_deg_diff(rot_meas_deg,180.0), _deg_diff(rot_meas_deg,-180.0))
    else:
        d = _deg_diff(rot_meas_deg, exp_deg)
    if d <= tol: return "OK"
    if d <= (tol + 15.0): return "LEVE"
    return "GRAVE"

def _classify_level(level_meas: str, level_exp: str) -> str:
    order = {"ARAE":0, "MOMTONG":1, "OLGUL":2}
    if level_meas not in order or level_exp not in order: return "LEVE"
    dm = abs(order[level_meas] - order[level_exp])
    if dm == 0: return "OK"
    if dm == 1: return "LEVE"
    return "GRAVE"

def _classify_dir(arm_poly: List[Tuple[float,float]], dir_exp: Optional[str],
                  octant_slack: int, dtw_thr: float) -> str:
    if not dir_exp:
        return "SKIP"
    u = _unit_seq(arm_poly)
    if len(u)==0: return "LEVE"
    vmean = u.mean(axis=0)
    if np.linalg.norm(vmean) < 1e-6: return "LEVE"
    ang = math.degrees(math.atan2(vmean[1], vmean[0]))
    breaks = [(-22.5,"E"),(22.5,"SE"),(67.5,"S"),(112.5,"SW"),(157.5,"W"),
              (180.0,"W"),(-180.0,"W"),(-157.5,"NW"),(-112.5,"N"),(-67.5,"NE")]
    meas = "E"
    for th,lab in breaks:
        if ang <= th: meas = lab; break
    ie = _oct_idx(dir_exp); im = _oct_idx(meas)
    if ie is None or im is None: return "LEVE"
    delta = _wrap_oct_delta(ie, im)
    tmpl = np.tile(_oct_vec(dir_exp), (max(3,len(u)), 1))
    dtw = _dtw(u, tmpl)
    if delta == 0: return "OK"
    if delta <= octant_slack and dtw <= dtw_thr: return "OK"
    if delta <= 2: return "LEVE"
    return "GRAVE"

def _classify_extension(median_elbow_deg: float, min_ok: float = 150.0, min_leve: float = 120.0) -> str:
    if median_elbow_deg >= min_ok: return "OK"
    if median_elbow_deg >= min_leve: return "LEVE"
    return "GRAVE"

def _classify_stance(meas: str, exp: str) -> str:
    if exp == "": return "OK"
    if meas == exp: return "OK"
    rear = {"dwit_kubi","beom_seogi"}
    if meas in rear and exp in rear: return "LEVE"
    if meas == "unknown": return "LEVE"
    return "GRAVE"

def _classify_kick(amp: float, peak: float, thr_amp: float, thr_peak: Optional[float]) -> str:
    ok_amp  = amp  >= thr_amp
    ok_peak = (peak >= float(thr_peak)) if (thr_peak is not None) else True
    if ok_amp and ok_peak: return "OK"
    if (amp >= 0.5*thr_amp) or (thr_peak is not None and peak >= 0.5*float(thr_peak)):
        return "LEVE"
    return "GRAVE"

def _ded(cls: str, pen_leve: float, pen_grave: float) -> float:
    if cls == "GRAVE": return pen_grave
    if cls == "LEVE":  return pen_leve
    return 0.0

def _tech_category(tech_kor: str, tech_es: str) -> str:
    s = f"{tech_kor} {tech_es}".lower()
    if "chagi" in s or "patada" in s: return "KICK"
    if "makki" in s or "sonnal" in s or "batangson" in s or "bakkat" in s or "an makki" in s: return "BLOCK"
    if "jireugi" in s or "deungjumeok" in s or "palkup" in s or "uraken" in s or "chigi" in s or "teok" in s:
        return "STRIKE"
    return "ARMS"

def _severity_to_int(cls: str) -> int:
    return {"OK":0,"LEVE":1,"GRAVE":2}.get(cls, 0)

# ----------------- core scoring -----------------

def score_one_video(
    moves_json: Path, spec_json: Path, *,
    landmarks_root: Path, alias: str, subset: Optional[str],
    pose_spec: Optional[Path],
    penalty_leve: float, penalty_grave: float,
    clamp_min: float = 1.5
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    spec = load_spec(spec_json)
    cfg  = load_pose_spec(pose_spec)

    mv = json.loads(Path(moves_json).read_text(encoding="utf-8"))
    video_id = mv["video_id"]
    fps = float(mv.get("fps", 30.0))
    segs = mv.get("moves", [])

    csv_path = (landmarks_root / alias / subset / f"{video_id}.csv") if subset else (landmarks_root / alias / f"{video_id}.csv")
    if not csv_path.exists():
        raise SystemExit(f"CSV no encontrado para {video_id}: {csv_path}")
    df = load_landmarks_csv(csv_path)

    expected = spec["_flat_moves"]
    rows = []
    exactitud = 4.0

    tol  = cfg.get("tolerances", {})
    rot_tol   = float(tol.get("turn_deg_tol", 30.0))
    oct_slack = int(tol.get("arm_octant_tol", 1))
    dtw_thr   = float(tol.get("dtw_good", 0.25))
    lvl_cfg   = cfg.get("levels", {})
    kcfg      = cfg.get("kicks", {})
    ap_thr    = float(kcfg.get("ap_chagi",{}).get("amp_min", 0.20))
    peak_thr  = kcfg.get("ap_chagi",{}).get("peak_above_hip_min", None)
    if peak_thr is not None:
        peak_thr = float(peak_thr)

    nframes = int(df["frame"].max()) + 1
    def _uniform_window(i: int, N: int) -> Tuple[int,int]:
        a = int(round(i     * nframes / N))
        b = int(round((i+1) * nframes / N)) - 1
        a = max(0, min(a, nframes-1))
        b = max(a, min(b, nframes-1))
        return a, b

    for i in range(len(expected)):
        e = expected[i]
        s = segs[i] if i < len(segs) else None

        tech_kor = e.get("tech_kor",""); tech_es = e.get("tech_es","")
        level_exp = str(e.get("level","MOMTONG")).upper()
        stance_exp  = str(e.get("stance_code",""))
        dir_exp = e.get("dir_octant", None)

        if s is not None:
            t0 = float(s.get("t_start", 0.0)); t1 = float(s.get("t_end", t0))
            a  = int(round(t0*fps));           b  = int(round(t1*fps))
            a = max(0, min(a, nframes-1)); b = max(a, min(b, nframes-1))
            rot_meas_deg = float(s.get("rotation_deg", 0.0))
            limb = str(s.get("active_limb","")) or _choose_active_limb(df, a, b)
            arm_poly = s.get("path", []) or _poly_from_limb(df, limb, a, b, n=20)
        else:
            a, b = _uniform_window(i, len(expected))
            t0, t1 = a/fps, b/fps
            rot_meas_deg = _estimate_rotation_deg(df, a, b)
            limb = _choose_active_limb(df, a, b)
            arm_poly = _poly_from_limb(df, limb, a, b, n=20)

        cat = _tech_category(tech_kor, tech_es)

        rot_exp_code = str(e.get("turn","NONE")).upper()
        cls_rot = _classify_rotation(rot_meas_deg, rot_exp_code, rot_tol)

        wrist_id = 15 if limb == "L_WRIST" else (16 if limb == "R_WRIST" else (15 if _disp(_series_xy_range(df,15,a,b)) >= _disp(_series_xy_range(df,16,a,b)) else 16))
        rel = _wrist_rel_series_robust(df, a, b, wrist_id)
        y_rel_end = float(rel[-1]) if rel.size else float("nan")
        level_meas = _level_from_spec_thresholds(y_rel_end, lvl_cfg)
        cls_lvl = _classify_level(level_meas, level_exp)

        if limb in ("L_WRIST","R_WRIST") and dir_exp:
            cls_dir = _classify_dir(arm_poly, dir_exp, oct_slack, dtw_thr)
        else:
            cls_dir = "SKIP"

        med_elbow = _median_elbow_extension(df, a, b)
        cls_ext = _classify_extension(med_elbow)

        if cat == "BLOCK":
            subs = [cls_lvl, cls_rot]
            if cls_dir != "SKIP": subs.append(cls_dir)
            subs = [c for c in subs if c in ("OK","LEVE","GRAVE")]
            comp_arms = max(subs, key=_severity_to_int) if subs else "OK"
        elif cat == "STRIKE":
            subs = [cls_lvl, cls_ext, cls_rot]
            if cls_dir != "SKIP": subs.append(cls_dir)
            subs = [c for c in subs if c in ("OK","LEVE","GRAVE")]
            comp_arms = max(subs, key=_severity_to_int) if subs else "OK"
        elif cat == "KICK":
            comp_arms = "OK"
        else:
            comp_arms = max([cls_lvl, cls_rot], key=_severity_to_int)

        stance_meas = _stance(df, a, b)
        comp_legs   = _classify_stance(stance_meas, stance_exp)

        req_kick = _kick_required(tech_kor, tech_es)
        if req_kick:
            amp, peak = _kick_metrics(df, a, b)
            comp_kick = _classify_kick(amp, peak, ap_thr, peak_thr)
        else:
            amp, peak = (np.nan, np.nan)
            comp_kick = "OK"

        pen_arms = _ded(comp_arms, penalty_leve, penalty_grave)
        pen_legs = _ded(comp_legs, penalty_leve, penalty_grave)
        pen_kick = _ded(comp_kick, penalty_leve, penalty_grave)

        ded_total = pen_arms + pen_legs + pen_kick
        exactitud = max(clamp_min, exactitud - ded_total)

        move_acc = 1.0 - ded_total
        is_correct = (move_acc >= 0.90)

        fails = []
        if comp_arms in ("LEVE","GRAVE"): fails.append(f"brazos:{comp_arms.lower()}")
        if comp_legs in ("LEVE","GRAVE"): fails.append(f"piernas:{comp_legs.lower()}")
        if req_kick and comp_kick in ("LEVE","GRAVE"): fails.append(f"patada:{comp_kick.lower()}")
        fail_parts = "; ".join(fails) if fails else ("OK" if is_correct else "—")

        rows.append({
            "video_id": video_id,
            "M": e["_key"], "tech_kor": tech_kor, "tech_es": tech_es,
            "comp_arms": comp_arms, "comp_legs": comp_legs, "comp_kick": comp_kick,
            "pen_arms": pen_arms, "pen_legs": pen_legs, "pen_kick": pen_kick,
            "ded_total_move": round(ded_total,3), "exactitud_acum": round(exactitud,3),
            "move_acc": round(move_acc,3), "is_correct": "yes" if is_correct else "no",
            "fail_parts": fail_parts,
            "level(exp)": level_exp, "level(meas)": level_meas,
            "y_rel_end": round(y_rel_end,4) if (isinstance(y_rel_end,(int,float)) and not math.isnan(y_rel_end)) else np.nan,
            "rot(exp)": rot_exp_code, "rot(meas_deg)": round(rot_meas_deg,2),
            "dir(exp_oct)": dir_exp or "—",
            "dir_used": "yes" if (dir_exp and limb in ("L_WRIST","R_WRIST")) else ("spec_only" if dir_exp else "no"),
            "elbow_median_deg": round(med_elbow,1),
            "kick_req": "sí" if req_kick else "no",
            "kick_amp": round(float(amp),4) if isinstance(amp,(int,float)) else np.nan,
            "kick_peak": round(float(peak),4) if isinstance(peak,(int,float)) else np.nan,
            "t0": round(a/fps,3), "t1": round(b/fps,3), "dur_s": round((b-a)/fps,3)
        })

    df_det = pd.DataFrame(rows)

    if df_det.empty:
        df_sum = pd.DataFrame([{
            "video_id": video_id, "moves_expected": len(expected),
            "moves_detected": len(segs), "moves_scored": 0,
            "moves_correct_90p": 0,
            "ded_total": 0.0, "exactitud_final": 4.0,
            "pct_arms_ok": 0.0, "pct_legs_ok": 0.0, "pct_kick_ok": 0.0
        }])
    else:
        def pct_ok(col):
            m = df_det[col].values
            return float((m == "OK").sum()) / max(1, len(m))
        df_sum = pd.DataFrame([{
            "video_id": video_id,
            "moves_expected": len(expected),
            "moves_detected": len(segs),
            "moves_scored": int(len(df_det)),
            "moves_correct_90p": int((df_det["is_correct"] == "yes").sum()),
            "ded_total": round(float(df_det["ded_total_move"].sum()),3),
            "exactitud_final": round(float(df_det["exactitud_acum"].iloc[-1]),3),
            "pct_arms_ok": round(100.0*pct_ok("comp_arms"),2),
            "pct_legs_ok": round(100.0*pct_ok("comp_legs"),2),
            "pct_kick_ok": round(100.0*pct_ok("comp_kick"),2)
        }])
    return df_det, df_sum

def score_many_to_excel(
    *, moves_jsons: List[Path], spec_json: Path,
    out_xlsx: Path, landmarks_root: Path, alias: str, subset: Optional[str],
    pose_spec: Optional[Path],
    penalty_leve: float, penalty_grave: float, clamp_min: float = 1.5
):
    det_list, sum_list = [], []
    for mj in moves_jsons:
        try:
            det, summ = score_one_video(
                mj, spec_json,
                landmarks_root=landmarks_root, alias=alias, subset=subset, pose_spec=pose_spec,
                penalty_leve=penalty_leve, penalty_grave=penalty_grave, clamp_min=clamp_min
            )
            det_list.append(det); sum_list.append(summ)
        except Exception as e:
            print(f"[WARN] Scoring falló en {mj.name}: {e}")

    df_det = pd.concat(det_list, ignore_index=True) if det_list else pd.DataFrame()
    df_sum = pd.concat(sum_list, ignore_index=True) if sum_list else pd.DataFrame()

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        (df_det if not df_det.empty else pd.DataFrame(columns=[
            "video_id","M","tech_kor","tech_es",
            "comp_arms","comp_legs","comp_kick","pen_arms","pen_legs","pen_kick",
            "ded_total_move","exactitud_acum","move_acc","is_correct","fail_parts",
            "level(exp)","level(meas)","y_rel_end",
            "rot(exp)","rot(meas_deg)","dir(exp_oct)","dir_used",
            "elbow_median_deg","kick_req","kick_amp","kick_peak","t0","t1","dur_s"
        ])).to_excel(xw, index=False, sheet_name="detalle")
        (df_sum if not df_sum.empty else pd.DataFrame(columns=[
            "video_id","moves_expected","moves_detected","moves_scored","moves_correct_90p",
            "ded_total","exactitud_final",
            "pct_arms_ok","pct_legs_ok","pct_kick_ok"
        ])).to_excel(xw, index=False, sheet_name="resumen")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="Scoring Pal Jang (8yang) con fallback de segmentos para cubrir todo el spec.")
    ap.add_argument("--moves-json", help="JSON de movimientos (uno)")
    ap.add_argument("--moves-dir",  help="Carpeta con *_moves.json")
    ap.add_argument("--spec", required=True, help="Spec JSON 8yang")
    ap.add_argument("--out-xlsx", required=True, help="Salida Excel")
    ap.add_argument("--landmarks-root", default="data/landmarks", help="Raíz de landmarks")
    ap.add_argument("--alias", required=True, help="Alias (8yang)")
    ap.add_argument("--subset", default="", help="Subconjunto (train/val/test)")
    ap.add_argument("--pose-spec", default="", help="JSON con tolerancias y umbrales de pose")
    ap.add_argument("--penalty-leve", type=float, default=0.1)
    ap.add_argument("--penalty-grave", type=float, default=0.3)
    ap.add_argument("--clamp-min", type=float, default=1.5)

    args = ap.parse_args()

    spec = Path(args.spec)
    out_xlsx = Path(args.out_xlsx)
    landmarks_root = Path(args.landmarks_root)
    alias = args.alias.strip()
    subset = args.subset.strip() or None
    pose_spec = Path(args.pose_spec) if args.pose_spec else None

    jsons: List[Path] = []
    if args.moves_json:
        p = Path(args.moves_json)
        if not p.exists(): sys.exit(f"No existe moves-json: {p}")
        jsons = [p]
    elif args.moves_dir:
        d = Path(args.moves_dir)
        if not d.exists(): sys.exit(f"No existe moves-dir: {d}")
        jsons = sorted(d.glob(f"{alias}_*_moves.json"))
        if not jsons:
            sys.exit(f"No se hallaron JSONs en {d} con patrón {alias}_*_moves.json")
    else:
        sys.exit("Indica --moves-json o --moves-dir")

    score_many_to_excel(
        moves_jsons=jsons, spec_json=spec, out_xlsx=out_xlsx,
        landmarks_root=landmarks_root, alias=alias, subset=subset,
        pose_spec=pose_spec,
        penalty_leve=args.penalty_leve, penalty_grave=args.penalty_grave, clamp_min=args.clamp_min
    )

if __name__ == "__main__":
    main()
