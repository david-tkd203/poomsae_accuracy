# src/video_get.py
from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys
import os
from typing import Iterable, List, Tuple, Optional, Dict

import cv2
import numpy as np

# Permite ejecutar "python -m src.video_get"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.dataio.video_downloader import smart_download  # noqa: E402

SUBSET_CHOICES = ("train", "val", "test", "")
VIDEO_GLOB = ("*.mp4", "*.mkv", "*.webm", "*.mov", "*.m4v", "*.avi", "*.mpg", "*.mpeg")

# ---------------------- util ----------------------

def find_videos_in_dir(d: Path) -> List[Path]:
    vids: List[Path] = []
    if not d.exists():
        return vids
    for pat in VIDEO_GLOB:
        vids.extend(sorted(d.glob(pat)))
    return vids

def clean_stem(s: str, maxlen: int = 80) -> str:
    s = s.strip().lower().replace(" ", "_")
    s = re.sub(r"[^a-z0-9._-]+", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:maxlen] if s else "video"

def parse_playlist_index_from_name(name: str) -> int | None:
    m = re.match(r"^(\d{3})-", name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None

def sort_key_for_seq(p: Path) -> Tuple[int, float, str]:
    idx = parse_playlist_index_from_name(p.stem)
    return (idx if idx is not None else 10**9, p.stat().st_mtime, p.name)

def next_noncolliding(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem, ext = dst.stem, dst.suffix
    k = 1
    while True:
        cand = dst.with_name(f"{stem}-{k}{ext}")
        if not cand.exists():
            return cand
        k += 1

def rename_batch(files: Iterable[Path], *, alias: str, seq: bool,
                 seq_start: int = 1, seq_digits: int = 3,
                 dry_run: bool = False, keep_ext: bool = True,
                 scheme: str = "{alias}_{n:0Nd}") -> List[Tuple[Path, Path]]:
    """
    Renombra en su MISMO directorio. No mueve carpetas.
    scheme tokens:
      - {alias}  alias normalizado (p.ej. 8yang)
      - {n}      número consecutivo (usa {n:0Nd})
      - {stem}   nombre base limpiado del archivo original
    """
    scheme_fmt = scheme.replace("{n:0Nd}", "{n:0" + str(seq_digits) + "d}")

    renames: List[Tuple[Path, Path]] = []
    files_sorted = sorted(files, key=sort_key_for_seq)

    n = seq_start
    for src in files_sorted:
        ext = src.suffix.lower() if keep_ext else ".mp4"
        if seq:
            dst_name = scheme_fmt.format(alias=alias, n=n, stem=clean_stem(src.stem))
            n += 1
        else:
            dst_name = f"{alias}_{clean_stem(src.stem)}"
        dst = src.with_name(dst_name + ext)
        if dst == src:
            continue
        dst = next_noncolliding(dst)
        renames.append((src, dst))

    applied: List[Tuple[Path, Path]] = []
    for src, dst in renames:
        print(f"[RENAME] {src.name}  ->  {dst.name}" + ("  (dry-run)" if dry_run else ""))
        if not dry_run:
            src.rename(dst)
            applied.append((src, dst))
    return applied

def read_urls_from_table(file_path: Path,
                         url_col: str = "url",
                         alias_col: Optional[str] = None,
                         subset_col: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    rows: List[Dict[str, Optional[str]]] = []
    suffix = file_path.suffix.lower()
    if suffix in (".xlsx", ".xls", ".csv"):
        try:
            import pandas as pd  # type: ignore
            if suffix == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            cols = {c.lower(): c for c in df.columns}
            if url_col.lower() not in cols:
                raise SystemExit(f"[ERROR] La columna '{url_col}' no existe en {file_path.name}. Columnas: {list(df.columns)}")

            def getval(row, colname: Optional[str]) -> Optional[str]:
                if not colname:
                    return None
                if colname.lower() not in cols:
                    return None
                v = row[cols[colname.lower()]]
                if pd.isna(v):
                    return None
                v = str(v).strip()
                return v or None

            for _, r in df.iterrows():
                u = getval(r, url_col)
                if not u:
                    continue
                a = getval(r, alias_col)
                s = getval(r, subset_col)
                rows.append({"url": u, "alias": a, "subset": s})
            return rows
        except ImportError:
            if suffix != ".csv":
                raise SystemExit("[ERROR] Instala pandas y openpyxl para leer Excel: pip install pandas openpyxl")
            urls = []
            with file_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        urls.append(line)
            for u in urls:
                rows.append({"url": u, "alias": None, "subset": None})
            return rows
    else:
        raise SystemExit(f"[ERROR] Formato no soportado: {file_path}")

def coerce_subset(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    s2 = s.strip().lower()
    return s2 if s2 in SUBSET_CHOICES and s2 else None

# ---------------------- detección Jeonbi (inicio/fin) ----------------------

def _norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x)) if np.all(np.isfinite(x)) else np.nan

def _detect_jeonbi_window(
    vpath: Path,
    resize_w: int = 960,
    vis_min: float = 0.15,
    hold_sec: float = 0.40,
) -> Tuple[int, int, float]:
    """
    Heurística robusta:
      - Jeonbi estable si:
         * separación de tobillos/anchura de hombros ∈ [0.8, 1.3]
         * ambas muñecas a nivel bajo-pecho: y_rel ∈ [0.8, 1.2]
         * manos cerca del eje central: |x - x_mid_sh|/SW ≤ 0.35
         * manos casi quietas: Δpos ≤ 0.02 (norma)
      - t_start = flanco de subida de “separación de pies” previo al primer bloque estable,
                  o 0.3 s antes del inicio del primer bloque estable.
      - t_end   = final del último bloque estable.
    Devuelve (f_ini, f_fin, fps)
    """
    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        raise SystemExit("Falta mediapipe. Instala: pip install mediapipe opencv-python numpy")

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {vpath}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if nframes <= 0:
        cap.release()
        raise SystemExit(f"Video sin frames: {vpath}")

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                        enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # series
    sw = np.full(nframes, np.nan, np.float32)       # shoulder width
    ad = np.full(nframes, np.nan, np.float32)       # ankle distance
    yrelL = np.full(nframes, np.nan, np.float32)    # wristL rel
    yrelR = np.full(nframes, np.nan, np.float32)    # wristR rel
    dxL = np.full(nframes, np.nan, np.float32)      # |xL - mid|/SW
    dxR = np.full(nframes, np.nan, np.float32)      # |xR - mid|/SW
    vL = np.full(nframes, np.nan, np.float32)       # speed wrist L
    vR = np.full(nframes, np.nan, np.float32)       # speed wrist R

    prevL = None
    prevR = None

    fidx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        h, w = img.shape[:2]
        if resize_w and w != resize_w:
            img = cv2.resize(img, (resize_w, int(round(h * (resize_w / float(w))))), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lms = res.pose_landmarks.landmark
            def get_xy(i):
                lm = lms[i]
                v = float(getattr(lm, "visibility", 0.0))
                if v < vis_min:
                    return np.array([np.nan, np.nan], np.float32)
                x = float(getattr(lm, "x", np.nan))
                y = float(getattr(lm, "y", np.nan))
                if not np.isfinite(x) or not np.isfinite(y):
                    return np.array([np.nan, np.nan], np.float32)
                x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
                y = 0.0 if y < 0.0 else (1.0 if y > 1.0 else y)
                return np.array([x, y], np.float32)

            LSH = get_xy(11); RSH = get_xy(12)
            LHP = get_xy(23); RHP = get_xy(24)
            LWK = get_xy(15); RWK = get_xy(16)
            LAN = get_xy(27); RAN = get_xy(28)

            if np.all(np.isfinite(LSH)) and np.all(np.isfinite(RSH)):
                sw[fidx] = _norm(RSH - LSH)
                mid_sh = 0.5 * (LSH + RSH)
            else:
                mid_sh = np.array([np.nan, np.nan], np.float32)

            if np.all(np.isfinite(LAN)) and np.all(np.isfinite(RAN)):
                ad[fidx] = _norm(RAN - LAN)

            if np.all(np.isfinite(LHP)) and np.all(np.isfinite(RHP)) and np.all(np.isfinite(mid_sh)):
                y_sh = 0.5 * (LSH[1] + RSH[1]) if np.all(np.isfinite(LSH)) and np.all(np.isfinite(RSH)) else np.nan
                y_hp = 0.5 * (LHP[1] + RHP[1])
                den = (y_hp - y_sh) if (np.isfinite(y_hp) and np.isfinite(y_sh)) else np.nan
                if np.isfinite(den) and den != 0 and np.all(np.isfinite(LWK)) and np.all(np.isfinite(RWK)) and np.isfinite(sw[fidx]) and sw[fidx] > 0:
                    yrelL[fidx] = (LWK[1] - y_sh) / den
                    yrelR[fidx] = (RWK[1] - y_sh) / den
                    dxL[fidx] = abs(LWK[0] - mid_sh[0]) / sw[fidx]
                    dxR[fidx] = abs(RWK[0] - mid_sh[0]) / sw[fidx]

            if prevL is not None and np.all(np.isfinite(LWK)) and np.all(np.isfinite(RWK)):
                vL[fidx] = float(np.linalg.norm(LWK - prevL))
                vR[fidx] = float(np.linalg.norm(RWK - prevR))
            prevL = LWK if np.all(np.isfinite(LWK)) else prevL
            prevR = RWK if np.all(np.isfinite(RWK)) else prevR

        fidx += 1

    cap.release()
    pose.close()

    ad_sw = ad / (sw + 1e-9)
    cond_feet   = (ad_sw >= 0.8) & (ad_sw <= 1.3)
    cond_levelL = (yrelL >= 0.8) & (yrelL <= 1.2)
    cond_levelR = (yrelR >= 0.8) & (yrelR <= 1.2)
    cond_center = (dxL <= 0.35) & (dxR <= 0.35)
    cond_quiet  = (vL <= 0.02) & (vR <= 0.02)

    stable = cond_feet & cond_levelL & cond_levelR & cond_center & cond_quiet
    hold = max(1, int(round(hold_sec * fps)))

    # agrupar bloques estables con duración >= hold
    runs: List[Tuple[int,int]] = []
    s = None
    for i, ok in enumerate(stable):
        if ok and s is None:
            s = i
        elif (not ok) and s is not None:
            if i - s >= hold:
                runs.append((s, i-1))
            s = None
    if s is not None and (len(stable) - s) >= hold:
        runs.append((s, len(stable)-1))

    if not runs:
        # fallback: todo el video
        return 0, max(0, nframes-1), fps

    first_s, first_e = runs[0]
    last_s,  last_e  = runs[-1]

    # detectar flanco ascendente de “separación de pies” previo al primer bloque
    search_a = max(0, first_s - int(3.0 * fps))
    onset = first_s
    for f in range(search_a+1, first_s+1):
        if np.isfinite(ad_sw[f-1]) and np.isfinite(ad_sw[f]):
            if (ad_sw[f-1] < 0.5) and (ad_sw[f] >= 0.6):
                onset = f
                break
    return onset, last_e, fps

def _write_trim(
    vpath: Path,
    out_path: Path,
    f0: int,
    f1: int,
    fps: float
) -> None:
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir: {vpath}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f0))
    cur = max(0, f0)
    while cur <= f1:
        ok, fr = cap.read()
        if not ok:
            break
        out.write(fr)
        cur += 1
    out.release()
    cap.release()

def autotrim_one(
    vpath: Path,
    suffix: str = "_cut",
    pad_pre: float = 0.15,
    pad_post: float = 0.15,
    resize_w: int = 960,
    vis_min: float = 0.15,
    hold_sec: float = 0.40,
    replace: bool = False
) -> Optional[Path]:
    f0, f1, fps = _detect_jeonbi_window(vpath, resize_w=resize_w, vis_min=vis_min, hold_sec=hold_sec)
    n = int(cv2.VideoCapture(str(vpath)).get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n <= 0:
        return None
    pad0 = int(round(pad_pre * fps))
    pad1 = int(round(pad_post * fps))
    a = max(0, f0 - pad0)
    b = min(max(0, n-1), f1 + pad1)
    if a >= b:
        return None
    dst = vpath.with_name(vpath.stem + suffix + vpath.suffix)
    _write_trim(vpath, dst, a, b, fps)
    if replace:
        backup = vpath.with_name(vpath.stem + "_orig" + vpath.suffix)
        vpath.rename(backup)
        dst.rename(vpath)
        return vpath
    return dst

# ---------------------- CLI principal ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Descarga/normaliza videos y (opcional) recorta automáticamente al tramo Jeonbi→Jeonbi."
    )
    # Descarga individual o desde archivo de texto
    ap.add_argument("--url", type=str, help="URL única a descargar")
    ap.add_argument("--url-file", type=str, help="Archivo de texto con una URL por línea")

    # Descarga por Excel/CSV
    ap.add_argument("--xlsx", type=str, help="Ruta a Excel/CSV con enlaces (columna por defecto: 'url')")
    ap.add_argument("--xlsx-url-col", type=str, default="url", help="Nombre de la columna URL en el Excel/CSV")
    ap.add_argument("--xlsx-alias-col", type=str, help="Columna para alias por fila (opcional)")
    ap.add_argument("--xlsx-subset-col", type=str, help="Columna para subset por fila (opcional: train/val/test)")

    ap.add_argument("--out", type=str, required=True, help="Carpeta base de salida, p.ej. data/raw_videos")
    ap.add_argument("--cookies", type=str, help="cookies.txt (opcional)")

    # Alias/subset globales (se usan si no hay por fila)
    ap.add_argument("--alias", type=str, help="Alias canónico para carpeta/prefijo, p.ej. '8yang'.")
    ap.add_argument("--subset", type=str, default="train", choices=SUBSET_CHOICES, help="Subcarpeta: train/val/test o vacío")

    # Estrategia de descarga
    ap.add_argument("--no-ytdlp-first", action="store_true", help="No usar yt-dlp primero")
    ap.add_argument("--no-normalize-from-title", action="store_true", help="No derivar alias desde el título automáticamente")
    ap.add_argument("--no-skip-existing", action="store_true", help="No omitir archivos ya existentes")
    ap.add_argument("--format", type=str, help="Cadena de formato yt-dlp (p.ej. bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4)")

    # Renombrado
    ap.add_argument("--rename-downloaded", action="store_true", help="Tras descargar, renombrar SOLO los archivos descargados en esta ejecución")
    ap.add_argument("--rename-existing", action="store_true", help="Renombrar TODOS los videos existentes en la carpeta destino (no mueve carpetas)")
    ap.add_argument("--seq", action="store_true", help="Usar numeración ascendente en el renombrado")
    ap.add_argument("--seq-start", type=int, default=1, help="Número inicial para --seq")
    ap.add_argument("--seq-digits", type=int, default=3, help="Relleno de ceros para --seq (p.ej. 3 -> 001)")
    ap.add_argument("--dry-run", action="store_true", help="Simular renombrado sin aplicar cambios")
    ap.add_argument("--scheme", type=str, default="{alias}_{n:0Nd}", help="Esquema de nombre cuando --seq; tokens: {alias}, {n:0Nd}, {stem}")

    # Recorte automático Jeonbi→Jeonbi
    ap.add_argument("--autotrim-existing", action="store_true", help="Recortar TODOS los videos en alias/subset a [inicioJeonbi..finJeonbi]")
    ap.add_argument("--trim-pad-pre", type=float, default=0.15, help="Padding antes del inicio (s)")
    ap.add_argument("--trim-pad-post", type=float, default=0.15, help="Padding después del fin (s)")
    ap.add_argument("--trim-hold-sec", type=float, default=0.40, help="Duración mínima de estabilidad Jeonbi (s)")
    ap.add_argument("--trim-resize-w", type=int, default=960, help="Ancho para inferencia de pose (solo trimming)")
    ap.add_argument("--trim-vis-min", type=float, default=0.15, help="Visibility mínima para usar landmarks (solo trimming)")
    ap.add_argument("--trim-suffix", type=str, default="_cut", help="Sufijo del archivo recortado")
    ap.add_argument("--trim-replace", action="store_true", help="Reemplazar el archivo original por el recortado (guarda backup *_orig.ext)")

    args = ap.parse_args()

    out_dir = Path(args.out)
    cookies = Path(args.cookies) if args.cookies else None
    prefer_ytdlp = not args.no_ytdlp_first
    skip_existing = not args.no_skip_existing
    normalize_from_title = not args.no_normalize_from_title
    alias_global = args.alias.strip() if args.alias else None
    subset_global = args.subset if args.subset in SUBSET_CHOICES and args.subset else None

    # Construir lista de URLs
    work_items: List[Dict[str, Optional[str]]] = []

    if args.url:
        work_items.append({"url": args.url.strip(), "alias": alias_global, "subset": subset_global})

    if args.url_file:
        urls_file = Path(args.url_file)
        if not urls_file.exists():
            sys.exit(f"No existe --url-file: {urls_file}")
        for line in urls_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                work_items.append({"url": line, "alias": alias_global, "subset": subset_global})

    if args.xlsx:
        table_path = Path(args.xlsx)
        if not table_path.exists():
            sys.exit(f"No existe --xlsx: {table_path}")
        rows = read_urls_from_table(
            table_path,
            url_col=args.xlsx_url_col,
            alias_col=args.xlsx_alias_col,
            subset_col=args.xlsx_subset_col,
        )
        for r in rows:
            a = r.get("alias") or alias_global
            s = coerce_subset(r.get("subset")) or subset_global
            work_items.append({"url": r["url"], "alias": a, "subset": s})

    # Descarga
    all_downloaded: List[Path] = []
    for item in work_items:
        u = item["url"]
        if not u:
            continue
        try:
            paths = smart_download(
                u, out_dir,
                cookies=cookies,
                prefer_ytdlp=prefer_ytdlp,
                skip_existing=skip_existing,
                alias=item.get("alias"),
                subset=item.get("subset"),
                normalize_from_title=normalize_from_title,
                format_str=args.format
            )
            all_downloaded.extend(paths)
        except Exception as e:
            print(f"[ERROR] {u} -> {e}", file=sys.stderr)

    if all_downloaded:
        print("\nArchivos descargados:")
        for p in all_downloaded:
            print(f" - {p}")

    # Renombrado: determinar alias efectivo
    alias_for_rename = alias_global
    if (args.rename_downloaded or args.rename_existing or args.autotrim_existing) and not alias_for_rename:
        # intentar deducirlo desde la carpeta donde se descargó o existe material
        base = out_dir
        # caso: descarga previa
        if all_downloaded:
            parent = all_downloaded[0].parent
            if subset_global and parent.name == subset_global and parent.parent:
                alias_for_rename = parent.parent.name
            else:
                alias_for_rename = parent.name
        # caso: renombrado/recorte existentes sin descarga
        if not alias_for_rename and subset_global:
            # buscar una subcarpeta out/ALIAS/subset con vídeos
            for cand in base.iterdir():
                if cand.is_dir() and (cand / subset_global).exists():
                    vids = find_videos_in_dir(cand / subset_global)
                    if vids:
                        alias_for_rename = cand.name
                        break
        if not alias_for_rename:
            sys.exit("No se pudo inferir --alias. Proporcione --alias para renombrar/recortar existentes.")

    # Renombrar SOLO lo descargado
    if args.rename_downloaded and all_downloaded:
        print("\n[RENAMER] Normalizando NOMBRES de archivos DESCARGADOS…")
        rename_batch(
            all_downloaded,
            alias=alias_for_rename,
            seq=args.seq,
            seq_start=args.seq_start,
            seq_digits=args.seq_digits,
            dry_run=args.dry_run,
            keep_ext=True,
            scheme=args.scheme,
        )

    # Renombrar TODOS los existentes
    if args.rename_existing:
        target_dir = out_dir / alias_for_rename
        if subset_global:
            target_dir = target_dir / subset_global
        if not target_dir.exists():
            sys.exit(f"No existe carpeta destino para --rename-existing: {target_dir}")
        files = find_videos_in_dir(target_dir)
        if not files:
            print(f"[WARN] No hay videos en {target_dir}")
        else:
            print(f"\n[RENAMER] Normalizando TODOS los videos en: {target_dir}")
            rename_batch(
                files,
                alias=alias_for_rename,
                seq=args.seq,
                seq_start=args.seq_start,
                seq_digits=args.seq_digits,
                dry_run=args.dry_run,
                keep_ext=True,
                scheme=args.scheme,
            )

    # Recorte automático Jeonbi→Jeonbi para TODOS los existentes
    if args.autotrim_existing:
        target_dir = out_dir / alias_for_rename
        if subset_global:
            target_dir = target_dir / subset_global
        if not target_dir.exists():
            sys.exit(f"No existe carpeta destino para --autotrim-existing: {target_dir}")
        files = find_videos_in_dir(target_dir)
        if not files:
            print(f"[WARN] No hay videos en {target_dir}")
        else:
            print(f"\n[TRIM] Recortando Jeonbi→Jeonbi en: {target_dir}")
            for vp in files:
                try:
                    outp = autotrim_one(
                        vp,
                        suffix=args.trim_suffix,
                        pad_pre=args.trim_pad_pre,
                        pad_post=args.trim_pad_post,
                        resize_w=args.trim_resize_w,
                        vis_min=args.trim_vis_min,
                        hold_sec=args.trim_hold_sec,
                        replace=args.trim_replace
                    )
                    if outp:
                        print(f"[OK] {vp.name} -> {outp.name if outp != vp else vp.name}  (auto-trim)")
                    else:
                        print(f"[SKIP] {vp.name}  (no se pudo recortar)")
                except Exception as e:
                    print(f"[WARN] {vp.name}  {e}")

    # Si no se pidió descarga/renombrado/recorte y no hay flags, terminar
    if not (work_items or args.rename_downloaded or args.rename_existing or args.autotrim_existing):
        sys.exit("Nada que hacer.")

if __name__ == "__main__":
    main()
