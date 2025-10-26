# src/tools/export_snapshots.py
from __future__ import annotations
import argparse, math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd

VIDEO_EXTS = (".mp4", ".mkv", ".webm", ".mov", ".m4v", ".avi", ".mpg", ".mpeg")

# ---- dibujo de esqueleto desde CSV (x,y) ----
CONNECTIONS = [
    (11,12), (11,23), (12,24), (23,24),
    (11,13), (13,15),
    (12,14), (14,16),
    (23,25), (25,27),
    (24,26), (26,28),
    (27,29), (28,30),
    (29,31), (30,32)
]

# Umbrales por defecto (coherentes con score_pal_yang)
LEVELS = {"olgul_max_rel": -0.02, "arae_min_rel": 1.03, "momtong_band": [0.05, 0.95]}

# ------------------------------------------------------------
# Utilidades de CSV / normalización
# ------------------------------------------------------------
def _frame_rows(csv_df: pd.DataFrame, fidx: int) -> pd.DataFrame:
    r = csv_df[csv_df["frame"] == fidx]
    if r.empty: return r
    r = r[np.isfinite(r["x"]) & np.isfinite(r["y"])]
    return r

def _is_norm01(rows: pd.DataFrame) -> bool:
    if rows.empty: return True
    in01 = (rows["x"].between(-0.1,1.1)) & (rows["y"].between(-0.1,1.1))
    if in01.mean() >= 0.8: return True
    mx = float(max(rows["x"].max(), rows["y"].max()))
    return (np.isfinite(mx) and mx <= 2.0)

def _xy_px(rows: pd.DataFrame, W: int, H: int) -> dict[int, Tuple[int,int]]:
    pts = {}
    normed = _is_norm01(rows)
    for _, r in rows.iterrows():
        x = float(r["x"]); y = float(r["y"])
        if not (np.isfinite(x) and np.isfinite(y)): continue
        if normed:
            px = int(round(max(0.0, min(1.0, x)) * W))
            py = int(round(max(0.0, min(1.0, y)) * H))
        else:
            px = int(round(max(0.0, min(float(W-1), x))))
            py = int(round(max(0.0, min(float(H-1), y))))
        pts[int(r["lmk_id"])] = (px, py)
    return pts

# ------------------------------------------------------------
# Seguir a la persona: recorte por bbox de landmarks
# ------------------------------------------------------------
def _person_bbox_pixels(csv_df: pd.DataFrame, fidx: int, W: int, H: int,
                        margin: float = 0.25, square: bool = True, min_side: int = 480) -> Tuple[int,int,int,int]:
    rows = _frame_rows(csv_df, fidx)
    if rows.empty:
        return 0, 0, W, H
    pts = _xy_px(rows, W, H)
    if not pts:
        return 0, 0, W, H
    xs = [p[0] for p in pts.values()]
    ys = [p[1] for p in pts.values()]
    x0, x1 = int(max(0, min(xs))), int(min(W-1, max(xs)))
    y0, y1 = int(max(0, min(ys))), int(min(H-1, max(ys)))
    bw, bh = x1 - x0 + 1, y1 - y0 + 1

    # expandir por margen
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    if square:
        side = max(bw, bh)
        side = int(max(min_side, side * (1.0 + margin)))
        half = side // 2
        x0 = int(round(cx - half)); x1 = int(round(cx + half))
        y0 = int(round(cy - half)); y1 = int(round(cy + half))
    else:
        bw = int(max(min_side, bw * (1.0 + margin)))
        bh = int(max(min_side, bh * (1.0 + margin)))
        x0 = int(round(cx - bw/2)); x1 = int(round(cx + bw/2))
        y0 = int(round(cy - bh/2)); y1 = int(round(cy + bh/2))

    # clamp a bordes
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if (x1 - x0) < 2 or (y1 - y0) < 2:
        return 0, 0, W, H
    return x0, y0, x1, y1

# ------------------------------------------------------------
# Dibujo de pose (acepta ROI para seguir a la persona)
# ------------------------------------------------------------
def _draw_pose_from_csv_frame(frame: np.ndarray, csv_df: pd.DataFrame, fidx: int,
                              *, src_W: Optional[int] = None, src_H: Optional[int] = None,
                              roi: Optional[Tuple[int,int,int,int]] = None):
    h, w = frame.shape[:2]
    rows = _frame_rows(csv_df, fidx)
    if rows.empty: return

    if roi is None:
        pts = _xy_px(rows, w, h)
    else:
        # proyectar puntos desde frame original (src_W,src_H) al ROI y luego al frame recortado
        assert src_W is not None and src_H is not None
        x0, y0, x1, y1 = roi
        pts_glob = _xy_px(rows, src_W, src_H)
        pts = {}
        for k,(gx,gy) in pts_glob.items():
            px = gx - x0
            py = gy - y0
            if 0 <= px < (x1 - x0) and 0 <= py < (y1 - y0):
                pts[k] = (int(px), int(py))

    for a,b in CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0,255,255), 2)
    for _, p in pts.items():
        cv2.circle(frame, p, 3, (0,255,0), -1)

# ------------------------------------------------------------
# Texto / banner / logos
# ------------------------------------------------------------
def _ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def _put_text(frame, txt, org, scale=0.8, fg=(255,255,255), bg=(0,0,0), thick=2):
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thick+2, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)

def _banner(frame, title):
    h,w = frame.shape[:2]
    pad = 6
    cv2.rectangle(frame, (0,0), (w, 48+pad*2), (20,20,20), -1)
    _put_text(frame, title, (12, 36), scale=0.9, fg=(255,255,0))

def _logos(frame, logos_text: str, logos_paths: List[str]):
    h,w = frame.shape[:2]
    x = w-12
    placed = False
    for p in logos_paths:
        pth = Path(p)
        if not pth.exists(): continue
        lg = cv2.imread(str(pth), cv2.IMREAD_UNCHANGED)
        if lg is None: continue
        ratio = 28.0 / lg.shape[0]
        lg = cv2.resize(lg, (int(lg.shape[1]*ratio), 28))
        lh, lw = lg.shape[:2]
        x -= lw
        y2 = 10
        if lg.shape[2]==4:
            alpha = lg[:,:,3:4]/255.0
            roi = frame[y2:y2+lh, x:x+lw, :3]
            if roi.shape[0]==lh and roi.shape[1]==lw:
                frame[y2:y2+lh, x:x+lw, :3] = (alpha*lg[:,:,:3] + (1-alpha)*roi).astype(np.uint8)
        else:
            frame[y2:y2+lh, x:x+lw] = lg[:,:,:3]
        x -= 8
        placed = True
    if not placed:
        _put_text(frame, logos_text, (w-320, 36), scale=0.7, fg=(180,255,180))

# ------------------------------------------------------------
# Selección de frame “posición de poomsae”
# ------------------------------------------------------------
def _series_xy_one(csv_df: pd.DataFrame, lmk_id: int, f: int) -> Tuple[float,float]:
    r = csv_df[(csv_df["lmk_id"]==lmk_id) & (csv_df["frame"]==f)]
    if r.empty: return (np.nan, np.nan)
    return float(r.iloc[0]["x"]), float(r.iloc[0]["y"])

def _wrist_rel_y(csv_df: pd.DataFrame, f: int, wrist_id: int) -> float:
    _, yW = _series_xy_one(csv_df, wrist_id, f)
    _, yLS = _series_xy_one(csv_df, 11, f)
    _, yRS = _series_xy_one(csv_df, 12, f)
    _, yLH = _series_xy_one(csv_df, 23, f)
    _, yRH = _series_xy_one(csv_df, 24, f)
    if any(np.isnan(v) for v in [yW,yLS,yRS,yLH,yRH]): return np.nan
    y_sh = 0.5*(yLS + yRS); y_hp = 0.5*(yLH + yRH)
    den = (y_hp - y_sh)
    if den == 0: return np.nan
    return (yW - y_sh) / den

def _wrist_speed(csv_df: pd.DataFrame, f: int, wrist_id: int) -> float:
    x0,y0 = _series_xy_one(csv_df, wrist_id, max(0,f-1))
    x1,y1 = _series_xy_one(csv_df, wrist_id, f)
    if any(np.isnan(v) for v in [x0,y0,x1,y1]): return np.inf
    return float(np.hypot(x1-x0, y1-y0))

def _target_for_level(level_exp: str) -> float:
    level_exp = str(level_exp or "MOMTONG").upper()
    if level_exp == "OLGUL":   return LEVELS["olgul_max_rel"]
    if level_exp == "ARAE":    return LEVELS["arae_min_rel"]
    a,b = LEVELS["momtong_band"]; return 0.5*(a+b)

def _pose_frame_for_segment(csv_df: pd.DataFrame, a: int, b: int, level_exp: str) -> int:
    if b <= a: return a
    f0 = a + int(0.6*(b-a))
    T  = _target_for_level(level_exp)
    best_f, best_cost = b, float("inf")
    for f in range(f0, b+1):
        yL = _wrist_rel_y(csv_df, f, 15)
        yR = _wrist_rel_y(csv_df, f, 16)
        dL = abs(yL - T) if not np.isnan(yL) else np.inf
        dR = abs(yR - T) if not np.isnan(yR) else np.inf
        d  = min(dL, dR)
        vL = _wrist_speed(csv_df, f, 15)
        vR = _wrist_speed(csv_df, f, 16)
        v  = min(vL, vR)
        cost = d + 0.3*v
        if cost < best_cost:
            best_cost, best_f = cost, f
    return int(best_f)

# ------------------------------------------------------------
# Rotulación severidad
# ------------------------------------------------------------
def _label_for_row(r: pd.Series) -> Tuple[str, str]:
    ded = float(r["ded_total_move"])
    is_ok = str(r.get("is_correct","")).lower() == "yes"
    if is_ok or abs(ded) < 1e-6: return "CORRECTO", "0.0"
    if 0.08 <= ded <= 0.12:      return "ERROR LEVE", "-0.1"
    if 0.28 <= ded <= 0.32:      return "ERROR GRAVE", "-0.3"
    if ded < 0.2:                return "ERROR LEVE", f"-{ded:.1f}"
    return "ERROR GRAVE", f"-{ded:.1f}"

# ------------------------------------------------------------
# Exportar una fila (con seguimiento de persona opcional)
# ------------------------------------------------------------
def _export_one_row(cap, fps: float, csv_df: pd.DataFrame, row: pd.Series,
                    out_dir: Path, poomsae_title: str, logos: List[str],
                    video_stem: str, follow: bool, follow_margin: float,
                    follow_square: bool, follow_min: int,
                    tag_suffix: str = "") -> Optional[Path]:
    if any(pd.isna(row.get(k)) for k in ["t0","t1"]): return None
    t0, t1 = float(row["t0"]), float(row["t1"])
    a, b = int(round(t0*fps)), int(round(t1*fps))
    fidx = _pose_frame_for_segment(csv_df, a, b, level_exp=row.get("level(exp)", "MOMTONG"))

    # leer frame base
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        ok, frame = cap.read()
        if not ok: return None
    H, W = frame.shape[:2]

    # ROI para seguir persona
    roi = None
    if follow:
        x0,y0,x1,y1 = _person_bbox_pixels(csv_df, fidx, W, H, margin=follow_margin,
                                          square=follow_square, min_side=follow_min)
        roi = (x0,y0,x1,y1)
        frame = frame[y0:y1, x0:x1].copy()

    # pose sobre el frame (si hay ROI, proyecta a coordenadas recortadas)
    _draw_pose_from_csv_frame(frame, csv_df, fidx, src_W=W, src_H=H, roi=roi)

    # rótulos
    _banner(frame, f"POOMSAE: {poomsae_title}")
    _logos(frame, "MediaPipe  |  TensorFlow", logos)

    label, dedtxt = _label_for_row(row)
    y0txt = 70
    _put_text(frame, f"{label}   (deducción {dedtxt})", (12, y0txt), scale=0.9, fg=(50,220,255)); y0txt += 26
    tech = str(row.get("tech_es") or row.get("tech_kor") or "")
    _put_text(frame, f"M={row.get('M','?')}   {tech}", (12, y0txt), scale=0.8, fg=(255,255,255)); y0txt += 22
    _put_text(frame, f"t={fidx/fps:0.2f}s   ventana=[{t0:0.2f},{t1:0.2f}]s", (12, y0txt), scale=0.75, fg=(200,200,200)); y0txt += 22
    comp = f"brazos={row.get('comp_arms','')}  piernas={row.get('comp_legs','')}  patada={row.get('comp_kick','')}"
    _put_text(frame, comp, (12, y0txt), scale=0.75, fg=(200,255,200))

    sev = "ok" if label=="CORRECTO" else ("leve" if "LEVE" in label else "grave")
    msafe = str(row.get("M","?")).replace("/","-")
    out_name = f"{video_stem}_{msafe}_{sev}{tag_suffix}.png"
    out_path = out_dir / out_name
    cv2.imwrite(str(out_path), frame)
    return out_path

# ------------------------------------------------------------
# Selección de filas ejemplo
# ------------------------------------------------------------
def _pick_example_rows(df: pd.DataFrame) -> List[pd.Series]:
    d = df.dropna(subset=["t0","t1"]).copy()
    if d.empty: return []
    ok = d[d["is_correct"].astype(str).str.lower().eq("yes")]
    leve = d[d["ded_total_move"].between(0.08,0.12)]
    grave = d[d["ded_total_move"].between(0.28,0.32)]
    if ok.empty:
        ok = d.loc[[d["ded_total_move"].abs().idxmin()]]
    if leve.empty:
        i = (d["ded_total_move"]-0.10).abs().idxmin(); leve = d.loc[[i]]
    if grave.empty:
        g = d[d["ded_total_move"]>=0.20]
        if g.empty: g = d
        i = (g["ded_total_move"]-0.30).abs().idxmin(); grave = g.loc[[i]]
    return [ok.iloc[0], leve.iloc[0], grave.iloc[0]]

# ------------------------------------------------------------
# Export por video
# ------------------------------------------------------------
def _export_for_video(video_path: Path, csv_path: Path, det_rows: pd.DataFrame,
                      out_dir: Path, poomsae_title: str, logos: List[str],
                      export_all: bool, follow: bool, follow_margin: float,
                      follow_square: bool, follow_min: int):
    _ensure_dir(out_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] No se pudo abrir video: {video_path}"); return
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    csv_df = pd.read_csv(csv_path)
    if not {"video_id","frame","lmk_id","x","y"}.issubset(csv_df.columns):
        print(f"[WARN] CSV inválido: {csv_path}"); cap.release(); return

    rows = det_rows.copy()
    rows = rows[~rows["comp_arms"].eq("NS")]  # descarta sin segmento
    if rows.empty:
        print(f"[WARN] Sin filas anotadas para {video_path.stem}"); cap.release(); return

    if export_all:
        for _, r in rows.iterrows():
            p = _export_one_row(cap, fps, csv_df, r, out_dir, poomsae_title, logos,
                                video_path.stem, follow, follow_margin, follow_square, follow_min)
            if p: print(f"[OK] {p}")
    else:
        ex = _pick_example_rows(rows)
        for r in ex:
            p = _export_one_row(cap, fps, csv_df, r, out_dir, poomsae_title, logos,
                                video_path.stem, follow, follow_margin, follow_square, follow_min)
            if p: print(f"[OK] {p}")

    cap.release()

# ------------------------------------------------------------
# Helpers de localización
# ------------------------------------------------------------
def _video_for_id(video_root: Path, video_id: str) -> Optional[Path]:
    for ext in VIDEO_EXTS:
        p = video_root / f"{video_id}{ext}"
        if p.exists(): return p
    for p in video_root.rglob(f"{video_id}*"):
        if p.suffix.lower() in VIDEO_EXTS: return p
    return None

def _csv_for_id(csv_root: Path, video_id: str) -> Optional[Path]:
    p = csv_root / f"{video_id}.csv"
    return p if p.exists() else None

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Exporta fotogramas en la POSICIÓN de poomsae por movimiento, siguiendo a la persona (crop por landmarks), con severidad y rótulos."
    )
    # Modo 1: un solo video + csv explícitos
    ap.add_argument("--video", help="Ruta al video")
    ap.add_argument("--csv", help="CSV de landmarks (MediaPipe) para ese video")
    # Modo 2: batch a partir del Excel
    ap.add_argument("--xlsx", required=True, help="Reporte Excel de score_pal_yang.py (sheet 'detalle')")
    ap.add_argument("--video-root", default="", help="Raíz de videos, p.ej. data/raw_videos/8yang/train")
    ap.add_argument("--csv-root", default="", help="Raíz de CSV, p.ej. data/landmarks/8yang/train")
    ap.add_argument("--video-id", default="", help="Solo este video_id (opcional)")
    ap.add_argument("--out-dir", required=True, help="Carpeta de salida PNG")
    ap.add_argument("--poomsae", default="Taegeuk 8 Jang (Pal Yang)", help="Título en banner")
    ap.add_argument("--logos", default="", help="Rutas a logos (png) separadas por coma")

    # Seguimiento de persona (ON por defecto)
    ap.add_argument("--no-follow", action="store_true", help="Desactivar seguimiento (no recorta)")
    ap.add_argument("--follow-margin", type=float, default=0.25, help="Margen relativo del bbox")
    ap.add_argument("--follow-min", type=int, default=480, help="Lado mínimo del recorte")
    ap.add_argument("--follow-rect", action="store_true", help="Usar rectángulo (por defecto: cuadrado)")
    # Exportación
    ap.add_argument("--all", action="store_true", help="Exportar TODOS los movimientos (uno por fila)")

    args = ap.parse_args()
    logos = [p.strip() for p in args.logos.split(",") if p.strip()]
    out_dir = Path(args.out_dir)

    det = pd.read_excel(args.xlsx, sheet_name="detalle")
    need = {"video_id","t0","t1","M","tech_es","tech_kor","comp_arms","comp_legs","comp_kick","ded_total_move","is_correct","level(exp)"}
    miss = need - set(det.columns)
    if miss:
        raise SystemExit(f"El Excel 'detalle' no contiene columnas: {miss}")

    follow = (not args.no_follow)
    follow_square = (not args.follow_rect)

    # Modo 1
    if args.video and args.csv:
        vpath = Path(args.video); cpath = Path(args.csv)
        if not (vpath.exists() and cpath.exists()):
            raise SystemExit("Rutas --video o --csv inválidas")
        stem = vpath.stem
        vrows = det[det["video_id"].astype(str) == stem]
        if vrows.empty:
            raise SystemExit(f"No hay filas en Excel para video_id={stem}")
        _export_for_video(vpath, cpath, vrows, out_dir, args.poomsae, logos,
                          export_all=args.all, follow=follow, follow_margin=args.follow_margin,
                          follow_square=follow_square, follow_min=args.follow_min)
        return

    # Modo 2
    if not args.video_root or not args.csv_root:
        raise SystemExit("Batch: especifique --video-root y --csv-root")
    video_root = Path(args.video_root)
    csv_root = Path(args.csv_root)

    vids = sorted(det["video_id"].astype(str).unique().tolist())
    if args.video_id:
        vids = [v for v in vids if v == args.video_id]
        if not vids:
            raise SystemExit(f"video_id solicitado no presente en Excel: {args.video_id}")

    for vid in vids:
        vpath = _video_for_id(video_root, vid)
        cpath = _csv_for_id(csv_root, vid)
        if not vpath or not cpath:
            print(f"[WARN] Saltando {vid}: no se encontró video o csv"); continue
        vrows = det[det["video_id"].astype(str) == vid]
        _export_for_video(vpath, cpath, vrows, out_dir, args.poomsae, logos,
                          export_all=args.all, follow=follow, follow_margin=args.follow_margin,
                          follow_square=follow_square, follow_min=args.follow_min)

if __name__ == "__main__":
    main()
