# src/eval/pose_metrics.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json, math
import numpy as np
import pandas as pd

# ------------ Landmarks (MediaPipe Pose) ------------
LMK = dict(
    NOSE=0,
    L_EYE=2, R_EYE=5,
    L_EAR=7, R_EAR=8,
    L_SH=11, R_SH=12, L_ELB=13, R_ELB=14, L_WRIST=15, R_WRIST=16,
    L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30, L_FOOT_INDEX=31, R_FOOT_INDEX=32
)

# ------------- util de E/S y numérico -------------
def _load(df_or_csv) -> pd.DataFrame:
    return pd.read_csv(df_or_csv) if isinstance(df_or_csv,(str,Path)) else df_or_csv

def _xy(df: pd.DataFrame, lmk: int, nframes: int) -> np.ndarray:
    sub = df[df.lmk_id==lmk][["frame","x","y"]]
    arr = np.full((nframes,2), np.nan, np.float32)
    idx = np.clip(sub.frame.values.astype(int), 0, nframes-1)
    arr[idx] = sub[["x","y"]].values.astype(np.float32)
    # ffill/bfill
    for k in range(2):
        v = arr[:,k]
        for i in range(1,len(v)):
            if np.isnan(v[i]) and not np.isnan(v[i-1]): v[i]=v[i-1]
        for i in range(len(v)-2,-1,-1):
            if np.isnan(v[i]) and not np.isnan(v[i+1]): v[i]=v[i+1]
        arr[:,k]=v
    return arr

def _angle3(a, b, c) -> float:
    # ángulo ABC (en B) en grados
    ba = a - b; bc = c - b
    den = (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-9)
    cosv = float(np.dot(ba, bc) / den)
    return float(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))

def _dist(a, b) -> float:
    return float(np.linalg.norm(a-b))

def _unit(x):
    n = np.linalg.norm(x)
    return x / (n + 1e-9)

# ------------- carga de especificación JSON -------------
def load_pose_spec(path: Path | str) -> Dict:
    """
    JSON esperado (ejemplo más abajo):
    - "levels": umbrales relativos a hombro↔cadera
    - "stances": reglas con distancias y ángulos de rodilla
    - "arms": reglas geométricas para bloqueos/ataques
    - "kicks": reglas de amplitud/tiempo para ap chagi, ttwieo, etc.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))

# ------------- referencias por frame -------------
def _refs(xy: Dict[str,np.ndarray], f: int) -> Dict[str, np.ndarray | float]:
    LSH, RSH = xy["L_SH"][f], xy["R_SH"][f]
    LHP, RHP = xy["L_HIP"][f], xy["R_HIP"][f]
    SH_W = _dist(LSH, RSH) + 1e-6
    TORSO = _dist(0.5*(LSH+RSH), 0.5*(LHP+RHP)) + 1e-6
    center_sh = 0.5*(LSH + RSH)
    center_hp = 0.5*(LHP + RHP)
    return dict(
        sh_w=SH_W, torso=TORSO,
        sh_y=center_sh[1], hip_y=center_hp[1],
        center_sh=center_sh, center_hp=center_hp
    )

def _build_xy(df_or_csv) -> Tuple[Dict[str,np.ndarray], int]:
    df = _load(df_or_csv)
    nframes = int(df["frame"].max())+1
    xy = {k: _xy(df, v, nframes) for k,v in LMK.items()}
    return xy, nframes

# ---------------- niveles (ARAE/MOMTONG/OLGUL) ----------------
def classify_level_from_spec(y_wrist: float, ref: Dict, levels: Dict) -> str:
    """
    Normaliza por (hip_y - sh_y) y usa cuartiles (márgenes) desde JSON.
    """
    dy = (y_wrist - ref["sh_y"]) / (ref["hip_y"] - ref["sh_y"] + 1e-6)  # y crece hacia abajo
    # umbrales del JSON (ejemplo: olgul_max=0.0, arae_min=1.0, banda momtong=[0.05,0.85])
    olg_ulim = float(levels.get("olgul_max_rel", -0.02))
    arae_llim = float(levels.get("arae_min_rel", 1.02))
    mom_a, mom_b = levels.get("momtong_band", [0.05, 0.95])
    if dy <= olg_ulim:     return "OLGUL"
    if dy >= arae_llim:    return "ARAE"
    if mom_a <= dy <= mom_b: return "MOMTONG"
    # fallback al más cercano
    if dy < mom_a: return "OLGUL"
    return "ARAE"

# ---------------- stance (piernas) ----------------
def classify_stance_from_spec(xy: Dict[str,np.ndarray], f: int, spec: Dict) -> Tuple[str, Dict[str,float]]:
    ref = _refs(xy, f)
    LANK, RANK = xy["L_ANK"][f], xy["R_ANK"][f]
    LKNE, RKNE = xy["L_KNEE"][f], xy["R_KNEE"][f]
    LHIP, RHIP = xy["L_HIP"][f], xy["R_HIP"][f]
    ankle_dist_sw = _dist(LANK, RANK) / ref["sh_w"]

    # ángulos de rodilla
    L_knee = _angle3(LHIP, LKNE, LANK)
    R_knee = _angle3(RHIP, RKNE, RANK)
    knee_min, knee_max = min(L_knee, R_knee), max(L_knee, R_knee)

    # “cadera detrás” (peso atrás) en imagen: centro cadera por debajo del centro pies
    feet_c = 0.5*(LANK + RANK)
    hip_behind = float((ref["center_hp"][1] > feet_c[1]) * 1.0)

    feats = dict(ankle_dist_sw=ankle_dist_sw, knee_min=L_knee if L_knee<R_knee else R_knee,
                 knee_max=knee_max, hip_behind=hip_behind)

    st_spec = spec.get("stances", {})
    scores = {}

    def score_ap():
        t = st_spec.get("ap_kubi", {})
        ok_dist = ankle_dist_sw >= float(t.get("ankle_dist_min_sw", 0.90))
        ok_knees = (knee_min <= float(t.get("front_knee_max_deg", 155.0))) and (knee_max >= float(t.get("rear_knee_min_deg", 165.0)))
        return 1.0 if (ok_dist and ok_knees) else 0.0

    def score_dwit():
        t = st_spec.get("dwit_kubi", {})
        ok_dist = ankle_dist_sw >= float(t.get("ankle_dist_min_sw", 0.55))
        ok_hip  = hip_behind >= float(t.get("hip_behind_min", 0.5))
        ok_knee = knee_min >= float(t.get("front_knee_min_deg", 165.0))
        return 1.0 if (ok_dist and ok_hip and ok_knee) else 0.0

    def score_beom():
        t = st_spec.get("beom_seogi", {})
        ok_dist = ankle_dist_sw <= float(t.get("ankle_dist_max_sw", 0.60))
        ok_hip  = hip_behind >= float(t.get("hip_behind_min", 0.5))
        return 1.0 if (ok_dist and ok_hip) else 0.0

    scores["ap_kubi"] = score_ap()
    scores["dwit_kubi"] = score_dwit()
    scores["beom_seogi"] = score_beom()

    lab = max(scores.items(), key=lambda kv: kv[1])[0] if scores else "unknown"
    return lab, {**feats, **{f"score_{k}":v for k,v in scores.items()}}

# ---------------- brazos (bloqueo/ataque) ----------------
def arms_category_from_spec(xy: Dict[str,np.ndarray], f: int, spec: Dict) -> Tuple[str, str, Dict[str,float]]:
    """
    Devuelve (categoria, nivel, feats).
    Niveles por JSON; categoría por reglas geométricas (codo y posicionamiento).
    """
    ref = _refs(xy, f)
    WL, WR = xy["L_WRIST"][f], xy["R_WRIST"][f]
    EL, ER = xy["L_ELB"][f], xy["R_ELB"][f]
    LSH, RSH = xy["L_SH"][f], xy["R_SH"][f]
    center_x = 0.5*(LSH[0]+RSH[0])

    # mano activa = la más separada lateralmente
    dL = abs(WL[0] - LSH[0]); dR = abs(WR[0] - RSH[0])
    W_act, E_act = (WL, EL) if dL >= dR else (WR, ER)

    # nivel por JSON (cuartiles relativos)
    level = classify_level_from_spec(W_act[1], ref, spec.get("levels", {}))

    # ángulo de codo del brazo activo y altura
    elbow_ang = _angle3(LSH if dL>=dR else RSH, E_act, W_act)
    rel_y = (W_act[1] - ref["sh_y"]) / (ref["hip_y"] - ref["sh_y"] + 1e-6)
    rel_x_to_center = (W_act[0] - center_x) / (ref["sh_w"])

    feats = dict(elbow_deg=elbow_ang, wrist_rel_y=rel_y, wrist_rel_xc=rel_x_to_center)

    a = spec.get("arms", {})
    # Reglas (se pueden ajustar en JSON si quieres granularidad por técnica)
    # Momtong Makki (medio): puño a altura hombro, codo 90–120°, antebrazo mirando al centro
    if (a.get("momtong_makki", {}).get("use", True) and
        (0.45 <= rel_y <= 0.65) and (90.0 <= elbow_ang <= 120.0)):
        return "MOMTONG_MAKKI", level, feats

    # Olgul Makki (alto): muñeca ~1 puño por encima de frente; aproximamos como por sobre hombros
    if (a.get("olgul_makki", {}).get("use", True) and
        (rel_y <= a.get("olgul_makki",{}).get("rel_y_max", 0.05))):
        return "OLGUL_MAKKI", level, feats

    # Arae Makki (bajo): muñeca claramente bajo cadera
    if (a.get("arae_makki", {}).get("use", True) and
        (rel_y >= a.get("arae_makki",{}).get("rel_y_min", 1.05))):
        return "ARAE_MAKKI", level, feats

    # Bakkat / An (externo vs interno): discriminamos por x respecto al centro
    # (si la muñeca acaba cruzando la línea media → AN; si queda al “exterior” → BAKKAT)
    if a.get("bakkat_an", {}).get("use", True) and (90.0 <= elbow_ang <= 130.0) and (0.30 <= rel_y <= 0.80):
        cat = "MOMTONG_AN" if abs(rel_x_to_center) < a["bakkat_an"].get("xcross_thr", 0.15) else "MOMTONG_BAKKAT"
        return cat, level, feats

    # Jireugi: codo >160° (brazo casi extendido) a nivel momtong/olgul
    if a.get("jireugi", {}).get("use", True) and (elbow_ang >= a["jireugi"].get("elbow_min", 160.0)):
        return "JIREUGI", level, feats

    # Palkup yeop chigi (codo horizontal): codo muy flexionado y muñeca avanzando lateralmente
    if a.get("palkup_yeop", {}).get("use", True) and (elbow_ang <= a["palkup_yeop"].get("elbow_max", 100.0)) and (abs(rel_x_to_center) >= 0.25):
        return "PALKUP_YEOP_CHIGI", level, feats

    # Sonnal/Hansonnal/Batangson geometría ≈ Momtong Makki (no podemos ver mano abierta/cerrada)
    if a.get("sonnal_equiv", {}).get("use", True) and (90.0 <= elbow_ang <= 125.0) and (0.35 <= rel_y <= 0.75):
        return "SONNAL_GEOM", level, feats

    return "ARMS_OTHER", level, feats

# ---------------- patadas (segmento) ----------------
def detect_kick_segment(xy: Dict[str,np.ndarray], a: int, b: int, spec: Dict) -> Tuple[str, Dict[str,float]]:
    """
    Apoya en amplitud vertical del tobillo vs cadera y detección de doble pico.
    """
    LANK = xy["L_ANK"][a:b+1]; RANK = xy["R_ANK"][a:b+1]
    HIP  = 0.5*(xy["L_HIP"][a:b+1] + xy["R_HIP"][a:b+1])
    ref0 = _refs(xy, a)

    rel_y_L = (HIP[:,1] - LANK[:,1]) / (ref0["torso"])
    rel_y_R = (HIP[:,1] - RANK[:,1]) / (ref0["torso"])
    amp_L = float(np.percentile(rel_y_L, 95) - np.percentile(rel_y_L, 5))
    amp_R = float(np.percentile(rel_y_R, 95) - np.percentile(rel_y_R, 5))
    amp = max(amp_L, amp_R)

    kspec = spec.get("kicks", {})
    ap_thr = float(kspec.get("ap_chagi", {}).get("amp_min", 0.30))
    ttw_thr = float(kspec.get("ttwieo_ap_chagi", {}).get("amp_min", 0.35))
    dbl_gap = int(kspec.get("double_peak_max_gap_frames", 10))

    # picos crudos
    def peaks(x: np.ndarray, min_amp: float) -> List[int]:
        idx = []
        for i in range(1, len(x)-1):
            if x[i] > x[i-1] and x[i] > x[i+1] and x[i] > min_amp:
                idx.append(i)
        return idx

    pkL = peaks(rel_y_L, ap_thr)
    pkR = peaks(rel_y_R, ap_thr)
    any_pk = sorted(pkL + pkR)

    # ¿doble ap chagi? (Dubal dang-sang): dos picos cercanos en el mismo segmento
    double_ok = False
    for i in range(1, len(any_pk)):
        if (any_pk[i] - any_pk[i-1]) <= dbl_gap:
            double_ok = True; break

    # ¿fase aérea? ambos tobillos por encima del baseline (umbral pequeño) simultáneamente
    base_L = float(np.percentile(rel_y_L, 5))
    base_R = float(np.percentile(rel_y_R, 5))
    air = (rel_y_L > base_L + 0.05) & (rel_y_R > base_R + 0.05)
    airborne_frac = float(np.mean(air))

    feats = dict(amp=amp, amp_L=amp_L, amp_R=amp_R, npk=len(any_pk), airborne_frac=airborne_frac)

    if double_ok and amp >= ap_thr:
        return "DUBAL_DANGSANG_AP_CHAGI", feats
    if (amp >= ttw_thr) and (airborne_frac >= float(kspec.get("ttwieo_ap_chagi",{}).get("airborne_min_frac", 0.10))):
        return "TTWIEO_AP_CHAGI", feats
    if amp >= ap_thr:
        return "AP_CHAGI", feats
    return "NO_KICK", feats

# ---------------- API de alto nivel ----------------
def snapshot_metrics(df_or_csv, frame_idx: int, spec_json: Optional[Path | str] = None) -> Dict[str,object]:
    xy, nframes = _build_xy(df_or_csv)
    spec = load_pose_spec(spec_json) if spec_json else dict(levels={}, stances={}, arms={}, kicks={})
    # brazos
    arms_cat, arms_level, arms_feats = arms_category_from_spec(xy, frame_idx, spec)
    # stance
    stance_lab, stance_feats = classify_stance_from_spec(xy, frame_idx, spec)
    return dict(
        arms_cat=arms_cat, arms_level=arms_level, stance=stance_lab,
        **{f"arms_{k}":v for k,v in arms_feats.items()},
        **{f"stance_{k}":v for k,v in stance_feats.items()}
    )

def segment_metrics(df_or_csv, a: int, b: int, spec_json: Optional[Path | str] = None) -> Dict[str,object]:
    xy, nframes = _build_xy(df_or_csv)
    spec = load_pose_spec(spec_json) if spec_json else dict(levels={}, stances={}, arms={}, kicks={})

    # mayorías por frame para stance/arms level
    stance_votes: List[str] = []
    level_votes:  List[str] = []
    elbow_vals:   List[float] = []

    for f in range(a, b+1):
        arms_cat, arms_level, feats = arms_category_from_spec(xy, f, spec)
        stance_lab, sft = classify_stance_from_spec(xy, f, spec)
        stance_votes.append(stance_lab)
        level_votes.append(arms_level)
        elbow_vals.append(feats["elbow_deg"])

    def maj(v):
        return max(set(v), key=v.count) if v else "unknown"

    kick_cat, kick_feats = detect_kick_segment(xy, a, b, spec)

    return dict(
        stance=maj(stance_votes),
        arms_level=maj(level_votes),
        elbow_deg_med=float(np.nanmedian(elbow_vals)) if elbow_vals else np.nan,
        kick=kick_cat,
        **{f"kick_{k}":v for k,v in kick_feats.items()}
    )
