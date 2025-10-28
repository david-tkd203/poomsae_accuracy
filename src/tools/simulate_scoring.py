# src/tools/simulate_scoring.py
from __future__ import annotations
import argparse, sys, json, math, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import cv2

# --- path para usar módulos internos ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Reusamos helpers del scorer y el lector de CSV
from src.tools import score_pal_yang as SP
from src.segmentation.move_capture import load_landmarks_csv

# -------------------- dibujo esqueleto (copiado de move_capture) --------------------
LMK = dict(
    NOSE=0, L_EYE_IN=1, L_EYE=2, L_EYE_OUT=3, R_EYE_IN=4, R_EYE=5, R_EYE_OUT=6,
    L_EAR=7, R_EAR=8, L_MOUTH=9, R_MOUTH=10, L_SH=11, R_SH=12, L_ELB=13, R_ELB=14,
    L_WRIST=15, R_WRIST=16, L_PINKY=17, R_PINKY=18, L_INDEX=19, R_INDEX=20,
    L_THUMB=21, R_THUMB=22, L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30, L_FOOT=31, R_FOOT=32
)
CONNECTIONS = [
    (LMK["L_SH"],LMK["R_SH"]), (LMK["L_SH"],LMK["L_HIP"]), (LMK["R_SH"],LMK["R_HIP"]), (LMK["L_HIP"],LMK["R_HIP"]),
    (LMK["L_SH"],LMK["L_ELB"]), (LMK["L_ELB"],LMK["L_WRIST"]),
    (LMK["R_SH"],LMK["R_ELB"]), (LMK["R_ELB"],LMK["R_WRIST"]),
    (LMK["L_HIP"],LMK["L_KNEE"]), (LMK["L_KNEE"],LMK["L_ANK"]),
    (LMK["R_HIP"],LMK["R_KNEE"]), (LMK["R_KNEE"],LMK["R_ANK"]),
    (LMK["L_ANK"],LMK["L_HEEL"]), (LMK["R_ANK"],LMK["R_HEEL"]),
    (LMK["L_HEEL"],LMK["L_FOOT"]), (LMK["R_HEEL"],LMK["R_FOOT"])
]

def _draw_pose_from_csv_frame(frame: np.ndarray, csv_df: pd.DataFrame, fidx: int):
    h, w = frame.shape[:2]
    rows = csv_df[csv_df["frame"] == fidx]
    if rows.empty: return
    rows = rows[np.isfinite(rows["x"]) & np.isfinite(rows["y"])]
    if rows.empty: return
    in01 = (rows["x"].between(-0.1, 1.1)) & (rows["y"].between(-0.1, 1.1))
    normed = (in01.mean() >= 0.8)
    if not normed:
        mx = float(max(rows["x"].max(), rows["y"].max()))
        if np.isfinite(mx) and mx <= 2.0:
            normed = True
    pts = {}
    for _, r in rows.iterrows():
        x = float(r["x"]); y = float(r["y"])
        if not (np.isfinite(x) and np.isfinite(y)): continue
        if normed:
            px = int(round(max(0.0, min(1.0, x)) * w))
            py = int(round(max(0.0, min(1.0, y)) * h))
        else:
            px = int(round(max(0.0, min(float(w - 1), x))))
            py = int(round(max(0.0, min(float(h - 1), y))))
        pts[int(r["lmk_id"])] = (px, py)
    if not pts: return
    for a, b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 255), 2)
    for _, p in pts.items():
        cv2.circle(frame, p, 3, (0, 255, 0), -1)

# -------------------- utilidades --------------------
def _sec_to_video_frame(t: float, fps: float, nframes: int) -> int:
    f = int(round(t * fps))
    return max(0, min(nframes - 1, f))

def _ensure_ab_from_jsonseg(seg: dict, fps_json: float, nframes_json: int, fps_vid: float, nframes_vid: int):
    """
    Devuelve (a,b,pose_f) en índices de frame del VIDEO.
    - Si el JSON ya trae a/b/pose_f (son de CSV), convertimos por escala nframes.
    - Si no, usamos t_start/t_end con fps_json.
    """
    # Preferimos a/b/pose_f si están
    if "a" in seg and "b" in seg:
        a_csv = int(seg["a"]); b_csv = int(seg["b"])
        # mapeo a frames de video por proporción de longitudes
        if nframes_json > 1:
            s = (nframes_vid - 1) / float(max(1, nframes_json - 1))
        else:
            s = fps_vid / max(1e-6, fps_json)
        a = int(round(a_csv * s)); b = int(round(b_csv * s))
        pose_f = int(seg.get("pose_f", b_csv))
        pose_f = int(round(pose_f * s))
        a = max(0, min(nframes_vid-1, a))
        b = max(0, min(nframes_vid-1, b))
        pose_f = max(0, min(nframes_vid-1, pose_f))
        if b < a: a, b = b, a
        return a, b, pose_f
    # Si no hay a/b: usar tiempos del JSON
    t0 = float(seg.get("t_start", 0.0)); t1 = float(seg.get("t_end", 0.0))
    a = _sec_to_video_frame(t0, fps_vid, nframes_vid)
    b = _sec_to_video_frame(t1, fps_vid, nframes_vid)
    if b < a: a, b = b, a
    pose_f = b
    return a, b, pose_f

def _score_move_one(e: dict, seg: dict, df_csv: pd.DataFrame, a_csv: int, b_csv: int,
                    cfg: dict, penalty_leve: float, penalty_grave: float) -> dict:
    """Replica la lógica de score_pal_yang para un movimiento (usa helpers del módulo)."""
    tech_kor = e.get("tech_kor",""); tech_es = e.get("tech_es","")
    level_exp = str(e.get("level","MOMTONG")).upper()
    stance_exp  = str(e.get("stance_code",""))
    dir_exp = e.get("dir_octant", None)
    lead = str(e.get("lead","")).upper() or "L"

    tol  = cfg.get("tolerances", {})
    rot_tol   = float(tol.get("turn_deg_tol", 30.0))
    oct_slack = int(tol.get("arm_octant_tol", 1))
    dtw_thr   = float(tol.get("dtw_good", 0.25))
    lvl_cfg   = cfg.get("levels", {})
    kcfg      = cfg.get("kicks", {})
    ap_thr    = float(kcfg.get("ap_chagi",{}).get("amp_min", 0.20))
    peak_thr  = kcfg.get("ap_chagi",{}).get("peak_above_hip_min", None)
    plantar_thr = float(kcfg.get("ap_chagi",{}).get("plantar_min_deg", 150.0))
    gaze_thr    = float(kcfg.get("ap_chagi",{}).get("gaze_max_deg", 35.0))
    if peak_thr is not None: peak_thr = float(peak_thr)

    # Rotación
    rot_meas_deg = float(seg.get("rotation_deg", 0.0))
    rot_exp_code = str(e.get("turn","NONE")).upper()
    cls_rot = SP._classify_rotation(rot_meas_deg, rot_exp_code, rot_tol)

    # Nivel/brazo
    limb = str(seg.get("active_limb",""))
    wrist_id = 15 if limb == "L_WRIST" else 16
    rel = SP._wrist_rel_series_robust(df_csv, a_csv, b_csv, wrist_id)
    y_rel_end = float(rel[-1]) if rel.size else float("nan")
    level_meas = SP._level_from_spec_thresholds(y_rel_end, lvl_cfg)
    cls_lvl = SP._classify_level(level_meas, level_exp)

    # Dirección (si aplica)
    if limb in ("L_WRIST","R_WRIST") and dir_exp:
        cls_dir = SP._classify_dir(seg.get("path", []), dir_exp, oct_slack, dtw_thr)
    else:
        cls_dir = "SKIP"

    med_elbow = SP._median_elbow_extension(df_csv, a_csv, b_csv)
    cls_ext = SP._classify_extension(med_elbow)

    cat = SP._tech_category(tech_kor, tech_es)
    if cat == "BLOCK":
        subs = [cls_lvl, cls_rot]; 
        if cls_dir != "SKIP": subs.append(cls_dir)
        subs = [c for c in subs if c in ("OK","LEVE","GRAVE")]
        comp_arms = max(subs, key=SP._severity_to_int) if subs else "OK"
    elif cat == "STRIKE":
        subs = [cls_lvl, cls_ext, cls_rot]; 
        if cls_dir != "SKIP": subs.append(cls_dir)
        subs = [c for c in subs if c in ("OK","LEVE","GRAVE")]
        comp_arms = max(subs, key=SP._severity_to_int) if subs else "OK"
    elif cat == "KICK":
        comp_arms = "OK"
    else:
        comp_arms = max([cls_lvl, cls_rot], key=SP._severity_to_int)

    meas_lab, feats = SP._stance_plus(df_csv, a_csv, b_csv)
    rear_turn = SP._rear_foot_turn_deg(feats, front_side=lead)
    ankle_sw  = feats.get("ankle_dist_sw", SP._ankle_dist_sw(df_csv, a_csv, b_csv))
    comp_legs, legs_reason = SP._classify_stance_2024(meas_lab, stance_exp, ankle_sw, rear_turn, cfg)

    # Patada (si corresponde)
    req_kick = SP._kick_required(tech_kor, tech_es)
    kick_type_pred = str(seg.get("kick_pred",""))
    if req_kick:
        amp, peak = SP._kick_metrics(df_csv, a_csv, b_csv)
        kside = SP._kicking_side(df_csv, a_csv, b_csv) or lead
        plantar_deg = SP._plantar_angle_deg(df_csv, a_csv, b_csv, kside)
        gaze_deg    = SP._gaze_to_toe_deg(df_csv, a_csv, b_csv, kside)
        comp_kick, kick_reason = SP._classify_kick_2024(
            amp, peak, ap_thr, peak_thr, kick_type_pred, "ap_chagi",
            plantar_deg, plantar_thr, gaze_deg, gaze_thr
        )
    else:
        amp, peak = (np.nan, np.nan)
        plantar_deg = np.nan; gaze_deg = np.nan
        comp_kick = "OK"; kick_reason = "no_kick"

    pen_arms = SP._ded(comp_arms, penalty_leve, penalty_grave)
    pen_legs = SP._ded(comp_legs, penalty_leve, penalty_grave)
    pen_kick = SP._ded(comp_kick, penalty_leve, penalty_grave)
    ded_total = pen_arms + pen_legs + pen_kick

    return dict(
        comp_arms=comp_arms, comp_legs=comp_legs, comp_kick=comp_kick,
        pen_arms=pen_arms, pen_legs=pen_legs, pen_kick=pen_kick,
        ded_total=round(ded_total,3),
        level_meas=level_meas, y_rel_end=y_rel_end, rot_meas_deg=rot_meas_deg,
        dir_used=("yes" if (dir_exp and limb in ("L_WRIST","R_WRIST")) else ("spec_only" if dir_exp else "no")),
        elbow_median_deg=med_elbow,
        stance_meas=meas_lab, ankle_dist_sw=ankle_sw, rear_foot_turn_deg=rear_turn,
        kick_amp=amp, kick_peak=peak, kick_type_pred=kick_type_pred,
        kick_plantar_deg=plantar_deg, kick_gaze_deg=gaze_deg
    )

# -------------------- simulación con overlay --------------------
def main():
    ap = argparse.ArgumentParser(description="Simulador visual: reproduce video, muestra movimientos esperados vs detectados y aplica descuentos desde 4.0.")
    ap.add_argument("--video", required=True, help="Ruta al video .mp4")
    ap.add_argument("--csv",   required=True, help="CSV de landmarks normalizados del mismo video")
    ap.add_argument("--moves-json", required=True, help="JSON de movimientos (output de capture_moves/capture_moves_batch)")
    ap.add_argument("--spec", required=True, help="Spec del poomsae")
    ap.add_argument("--pose-spec", default="", help="Spec de pose (tolerancias)")
    ap.add_argument("--out-xlsx", required=True, help="Excel de salida con el detalle/sumario")
    ap.add_argument("--out-preview", default="", help="(Opcional) MP4 anotado a guardar")
    ap.add_argument("--show", action="store_true", help="Mostrar ventana en vivo")
    ap.add_argument("--speed", type=float, default=1.0, help="Factor de reproducción (1.0 = nativo)")
    ap.add_argument("--no-draw-pose", action="store_true", help="No dibujar esqueleto")
    ap.add_argument("--penalty-leve", type=float, default=0.1)
    ap.add_argument("--penalty-grave", type=float, default=0.3)
    ap.add_argument("--clamp-min", type=float, default=1.5)
    args = ap.parse_args()

    video_path = Path(args.video)
    csv_path   = Path(args.csv)
    moves_path = Path(args.moves_json)
    spec_path  = Path(args.spec)
    pose_spec  = Path(args.pose_spec) if args.pose_spec else None
    out_xlsx   = Path(args.out_xlsx)
    out_preview= Path(args.out_preview) if args.out_preview else None

    # Carga
    mv = json.loads(moves_path.read_text(encoding="utf-8"))
    spec = SP.load_spec(spec_path)
    cfg  = SP.load_pose_spec(pose_spec)

    expected = spec["_flat_moves"]
    segs = mv.get("moves", [])
    video_id = mv.get("video_id", video_path.stem)
    fps_json = float(mv.get("fps", 30.0))
    nframes_json = int(mv.get("nframes", 0) or 0)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"No se pudo abrir video: {video_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    NF  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    csv_df = load_landmarks_csv(csv_path) if csv_path.exists() else None

    # Precompute a,b,pose_f en FRAMES DE VIDEO, y el mismo rango en FRAMES DE CSV (para métricas).
    # Si el CSV tiene nframes distinto al video, mapeamos por proporción.
    nframes_csv = int(csv_df["frame"].max()) + 1 if csv_df is not None else NF
    if nframes_json > 1 and nframes_csv > 1:
        fps_csv_est = fps_json * ((nframes_csv - 1) / float(nframes_json - 1))
    else:
        fps_csv_est = fps_json

    def json_to_csv_f(frame_json: int) -> int:
        if nframes_json > 1 and nframes_csv > 1:
            s = (nframes_csv - 1) / float(nframes_json - 1)
            return int(round(frame_json * s))
        return frame_json

    mapping = []  # lista por movimiento: info + frames
    for i, e in enumerate(expected):
        seg = segs[i] if i < len(segs) else None
        if seg is None:
            mapping.append(dict(idx=i+1, expected=e, seg=None, aV=None, bV=None, poseV=None, aC=None, bC=None, poseC=None))
            continue
        aV, bV, poseV = _ensure_ab_from_jsonseg(seg, fps_json, nframes_json, FPS, NF)
        if "a" in seg and "b" in seg:
            aC, bC, poseC = int(seg["a"]), int(seg["b"]), int(seg.get("pose_f", seg["b"]))
        else:
            # Convertir desde tiempos al espacio del CSV con fps_csv_est
            aC = int(round(float(seg.get("t_start",0.0)) * fps_csv_est))
            bC = int(round(float(seg.get("t_end",0.0))   * fps_csv_est))
            poseC = bC
        # clamp y orden
        aC = max(0, min(nframes_csv-1, aC)); bC = max(0, min(nframes_csv-1, bC)); poseC = max(0, min(nframes_csv-1, poseC))
        if bC < aC: aC, bC = bC, aC
        mapping.append(dict(idx=i+1, expected=e, seg=seg, aV=aV, bV=bV, poseV=poseV, aC=aC, bC=bC, poseC=poseC))

    # VideoWriter (opcional)
    writer = None
    if out_preview:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_preview), fourcc, FPS, (W, H))

    # Estado de score
    score = 4.0
    applied = set()  # índices i donde ya apliqué descuento
    details_rows = []

    # Dibujar timeline simple
    def draw_timeline(img, fidx):
        y = H - 30
        cv2.rectangle(img, (10, y-8), (W-10, y+8), (40,40,40), -1)
        # marcas por movimiento
        for info in mapping:
            if info["aV"] is None or info["bV"] is None: continue
            x1 = 10 + int((W-20) * info["aV"] / max(1,NF-1))
            x2 = 10 + int((W-20) * info["bV"] / max(1,NF-1))
            cv2.rectangle(img, (x1, y-6), (x2, y+6), (80,160,255), -1)
        # cursor
        x = 10 + int((W-20) * fidx / max(1,NF-1))
        cv2.line(img, (x, y-10), (x, y+10), (0,255,255), 2)

    # Main loop
    delay_ms = int(max(1, round(1000.0 / max(1e-6, FPS * args.speed))))
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        # Pose overlay
        if csv_df is not None and not args.no_draw_pose:
            # aproximar: usamos frame de video ~ frame de csv vía escala
            if NF > 1 and nframes_csv > 1:
                s = (nframes_csv - 1) / float(NF - 1)
                f_csv = int(round(fidx * s))
            else:
                f_csv = fidx
            _draw_pose_from_csv_frame(frame, csv_df, f_csv)

        # ¿qué movimiento está activo ahora?
        active_idx = None
        for info in mapping:
            if info["aV"] is None or info["bV"] is None: continue
            if info["aV"] <= fidx <= info["bV"]:
                active_idx = info["idx"]
                break

        # Título y score
        y0 = 26
        tsec = fidx / max(1e-6, FPS)
        cv2.putText(frame, f"{video_id}  t={tsec:5.2f}s  fps={FPS:.1f}", (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"{video_id}  t={tsec:5.2f}s  fps={FPS:.1f}", (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        y0 += 28
        cv2.putText(frame, f"Score: {score:0.2f} / 4.00", (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {score:0.2f} / 4.00", (12,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
        y = y0 + 26

        # Info del movimiento activo
        if active_idx is not None:
            info = mapping[active_idx-1]
            e = info["expected"]; seg = info["seg"]
            label = f"#{info['idx']}  {e.get('tech_es','')}"
            cv2.putText(frame, label, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, label, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2, cv2.LINE_AA); y += 22
            if seg:
                det = f"det: {seg.get('active_limb','?')}  rot:{seg.get('rotation_bucket','?')}  stance:{seg.get('stance_pred','?')}  kick:{seg.get('kick_pred','none')}"
            else:
                det = "det: — (no segment)"
            cv2.putText(frame, det, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(frame, det, (12,y), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200,255,255), 2, cv2.LINE_AA); y += 20

        # Aplicar deducción cuando pasamos el final bV de un movimiento y aún no aplicado
        for info in mapping:
            if info["idx"] in applied: continue
            if info["bV"] is None:  # sin segmento => no aplica descuento pero marcamos detalle
                # Registrar "NS"
                rows_ns = dict(
                    video_id=video_id, M=f"{info['expected'].get('idx','?')}{info['expected'].get('sub','')}",
                    tech_kor=info['expected'].get('tech_kor',''), tech_es=info['expected'].get('tech_es',''),
                    comp_arms="NS", comp_legs="NS", comp_kick="NS",
                    pen_arms=0.0, pen_legs=0.0, pen_kick=0.0, ded_total_move=0.0,
                    exactitud_after=round(score,3),
                    t_applied=np.nan
                )
                details_rows.append(rows_ns)
                applied.add(info["idx"])
                continue
            if fidx >= info["bV"]:
                # Calcular métricas/penalización con frames del CSV
                if csv_df is not None and info["seg"] is not None:
                    res = _score_move_one(info["expected"], info["seg"], csv_df, info["aC"], info["bC"], cfg, args.penalty_leve, args.penalty_grave)
                    score = max(args.clamp_min, score - float(res["ded_total"]))
                    rows = dict(
                        video_id=video_id,
                        M=f"{info['expected'].get('idx','?')}{info['expected'].get('sub','')}",
                        tech_kor=info['expected'].get('tech_kor',''),
                        tech_es=info['expected'].get('tech_es',''),
                        comp_arms=res["comp_arms"], comp_legs=res["comp_legs"], comp_kick=res["comp_kick"],
                        pen_arms=res["pen_arms"], pen_legs=res["pen_legs"], pen_kick=res["pen_kick"],
                        ded_total_move=res["ded_total"], exactitud_after=round(score,3),
                        level_meas=res["level_meas"], y_rel_end=res["y_rel_end"], rot_meas_deg=res["rot_meas_deg"],
                        dir_used=res["dir_used"], elbow_median_deg=res["elbow_median_deg"],
                        stance_meas=res["stance_meas"], ankle_dist_sw=res["ankle_dist_sw"], rear_foot_turn_deg=res["rear_foot_turn_deg"],
                        kick_amp=res["kick_amp"], kick_peak=res["kick_peak"], kick_type_pred=res["kick_type_pred"],
                        kick_plantar_deg=res["kick_plantar_deg"], kick_gaze_deg=res["kick_gaze_deg"],
                        t0=info["aV"]/FPS, t1=info["bV"]/FPS, t_pose=info["poseV"]/FPS,
                        a_csv=info["aC"], b_csv=info["bC"], pose_csv=info["poseC"],
                        a_vid=info["aV"], b_vid=info["bV"], pose_vid=info["poseV"],
                        idx=info["idx"], active_limb=info["seg"].get("active_limb",""), rotation_bucket=info["seg"].get("rotation_bucket",""),
                        stance_pred=info["seg"].get("stance_pred",""), kick_pred=info["seg"].get("kick_pred",""),
                        t_applied=tsec
                    )
                else:
                    rows = dict(
                        video_id=video_id, M=f"{info['expected'].get('idx','?')}{info['expected'].get('sub','')}",
                        tech_kor=info['expected'].get('tech_kor',''), tech_es=info['expected'].get('tech_es',''),
                        comp_arms="NS", comp_legs="NS", comp_kick="NS",
                        pen_arms=0.0, pen_legs=0.0, pen_kick=0.0, ded_total_move=0.0,
                        exactitud_after=round(score,3),
                        t0=info["aV"]/FPS if info["aV"] is not None else np.nan,
                        t1=info["bV"]/FPS if info["bV"] is not None else np.nan,
                        t_pose=info["poseV"]/FPS if info["poseV"] is not None else np.nan,
                        idx=info["idx"], t_applied=tsec
                    )
                details_rows.append(rows)
                # Overlay del descuento aplicado
                y2 = 110
                txt = f"#{info['idx']} -> -{rows.get('ded_total_move',0.0):.2f}  (score {score:.2f})"
                cv2.putText(frame, txt, (12,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, txt, (12,y2), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0,100,255), 2, cv2.LINE_AA)
                applied.add(info["idx"])

        # Timeline
        draw_timeline(frame, fidx)

        # Salida
        if writer is not None:
            writer.write(frame)
        if args.show:
            cv2.imshow("Simulador de scoring", frame)
            k = cv2.waitKey(delay_ms) & 0xFF
            if k in (27, ord('q')):
                break
        else:
            # si no mostramos, al menos respetar tiempo al exportar
            if delay_ms > 0:
                # duerme más breve para no bloquear demasiado export
                time.sleep(delay_ms/1000.0 * 0.02)

        fidx += 1

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # ---------- Reporte a Excel ----------
    df_det = pd.DataFrame(details_rows)
    # resumen simple compatible
    if not df_det.empty:
        moves_expected = len(expected)
        moves_detected = int((df_det["comp_arms"] != "NS").sum())
        moves_scored   = moves_detected
        moves_correct_90p = int(((1.0 - df_det.get("ded_total_move",0.0)) >= 0.90).sum())
        ded_total = float(df_det.get("ded_total_move",0.0).sum())
        exactitud_final = round(float(df_det["exactitud_after"].iloc[-1]), 3)
        def pct_ok(col):
            m = df_det[col].values if col in df_det.columns else None
            if m is None: return np.nan
            mask = (m != "NS")
            n = int(mask.sum())
            return round(100.0*float((m[mask] == "OK").sum())/n,2) if n>0 else np.nan
        df_sum = pd.DataFrame([{
            "video_id": video_id,
            "moves_expected": moves_expected,
            "moves_detected": moves_detected,
            "moves_scored": moves_scored,
            "moves_correct_90p": moves_correct_90p,
            "ded_total": round(ded_total,3),
            "exactitud_final": exactitud_final,
            "pct_arms_ok": pct_ok("comp_arms"),
            "pct_legs_ok": pct_ok("comp_legs"),
            "pct_kick_ok": pct_ok("comp_kick")
        }])
    else:
        df_sum = pd.DataFrame([{
            "video_id": video_id,
            "moves_expected": len(expected),
            "moves_detected": 0, "moves_scored": 0, "moves_correct_90p": 0,
            "ded_total": 0.0, "exactitud_final": 4.0,
            "pct_arms_ok": np.nan, "pct_legs_ok": np.nan, "pct_kick_ok": np.nan
        }])

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
        df_det.to_excel(xw, index=False, sheet_name="detalle_sim")
        df_sum.to_excel(xw, index=False, sheet_name="resumen_sim")

    print(f"[OK] Simulación terminada. Excel -> {out_xlsx}")
    if out_preview:
        print(f"[OK] Preview anotado -> {out_preview}")

if __name__ == "__main__":
    main()
