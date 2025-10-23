# src/video_get.py
from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys
import os
from typing import Iterable, List, Tuple, Optional, Dict

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

# ---------------------- lectura Excel/CSV ----------------------

def read_urls_from_table(file_path: Path,
                         url_col: str = "url",
                         alias_col: Optional[str] = None,
                         subset_col: Optional[str] = None) -> List[Dict[str, Optional[str]]]:
    """
    Lee .xlsx o .csv y devuelve filas: {"url": str, "alias": Optional[str], "subset": Optional[str]}
    - Excel requiere pandas+openpyxl instalados.
    - CSV se lee con pandas si está disponible; si no, se hace fallback simple.
    """
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
            # Fallback muy básico para CSV sin pandas
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

# ---------------------- CLI principal ----------------------

def main():
    ap = argparse.ArgumentParser(
        description="Descarga videos (YouTube/playlist/enlace directo o Excel/CSV) y normaliza nombres."
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
    ap.add_argument("--alias", type=str,
                    help="Alias canónico para carpeta/prefijo, p.ej. '8yang'. Si falta y se usa playlist, se intenta derivar del título.")
    ap.add_argument("--subset", type=str, default="train", choices=SUBSET_CHOICES,
                    help="Subcarpeta: train/val/test o vacío")

    # Estrategia de descarga
    ap.add_argument("--no-ytdlp-first", action="store_true", help="No usar yt-dlp primero")
    ap.add_argument("--no-normalize-from-title", action="store_true", help="No derivar alias desde el título automáticamente")
    ap.add_argument("--no-skip-existing", action="store_true", help="No omitir archivos ya existentes")
    ap.add_argument("--format", type=str, help="Cadena de formato yt-dlp (p.ej. bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4)")
    # Renombrado
    ap.add_argument("--rename-downloaded", action="store_true",
                    help="Tras descargar, renombrar SOLO los archivos descargados en esta ejecución")
    ap.add_argument("--rename-existing", action="store_true",
                    help="Renombrar TODOS los videos existentes en la carpeta destino (no mueve carpetas)")
    ap.add_argument("--seq", action="store_true", help="Usar numeración ascendente en el renombrado")
    ap.add_argument("--seq-start", type=int, default=1, help="Número inicial para --seq")
    ap.add_argument("--seq-digits", type=int, default=3, help="Relleno de ceros para --seq (p.ej. 3 -> 001)")
    ap.add_argument("--dry-run", action="store_true", help="Simular renombrado sin aplicar cambios")
    ap.add_argument("--scheme", type=str, default="{alias}_{n:0Nd}",
                    help="Esquema de nombre cuando --seq; tokens: {alias}, {n:0Nd}, {stem}")

    args = ap.parse_args()

    out_dir = Path(args.out)
    cookies = Path(args.cookies) if args.cookies else None
    prefer_ytdlp = not args.no_ytdlp_first
    skip_existing = not args.no_skip_existing
    normalize_from_title = not args.no_normalize_from_title
    alias_global = args.alias.strip() if args.alias else None
    subset_global = args.subset if args.subset in SUBSET_CHOICES and args.subset else None

    # Construir lista de URLs desde varias fuentes
    work_items: List[Dict[str, Optional[str]]] = []  # cada item: {"url":..., "alias":..., "subset":...}

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

    if not work_items and not (args.rename_downloaded or args.rename_existing):
        sys.exit("Nada que hacer: indique --url/--url-file/--xlsx o alguna acción de renombrado (--rename-downloaded/--rename-existing)")

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

    # Si no hay descargas y tampoco flags de renombrado, terminar
    if not work_items and not (args.rename_downloaded or args.rename_existing):
        return

    # Determinar alias efectivo para renombrado si no se proporcionó
    alias_for_rename = alias_global
    if (args.rename_downloaded or args.rename_existing) and not alias_for_rename:
        if all_downloaded:
            sample = all_downloaded[0]
            parent = sample.parent
            if subset_global and parent.name == subset_global and parent.parent:
                alias_for_rename = parent.parent.name
            else:
                alias_for_rename = parent.name
        if args.rename_existing and not alias_for_rename:
            sys.exit("Para --rename-existing debe proporcionar --alias o haber descargado a una carpeta alias previamente.")

    # Renombrar SOLO lo descargado
    if args.rename_downloaded and all_downloaded:
        print("\n[RENAMER] Normalizando NOMBRES de archivos DESCARGADOS en esta ejecución…")
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

    # Renombrar todos los existentes de la carpeta destino alias/subset
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
            print(f"\n[RENAMER] Normalizando TODOS los videos existentes en: {target_dir}")
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

if __name__ == "__main__":
    main()
