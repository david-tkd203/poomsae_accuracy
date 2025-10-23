# src/tools/collect_links.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

import pandas as pd

# yt-dlp es robusto para extracción sin descarga
from yt_dlp import YoutubeDL

# ----------------------- extracción -----------------------

def ydl_base_opts() -> Dict:
    return {
        "quiet": True,
        "extract_flat": True,     # no descargues, devuelve metadatos
        "skip_download": True,
        "ignoreerrors": True,
        "noplaylist": False,
        "nocheckcertificate": True,
        "retries": 3,
    }

def extract_search(query: str, limit: int) -> List[Dict]:
    # yt-dlp permite "ytsearch{N}:query"
    q = f"ytsearch{limit}:{query}"
    with YoutubeDL(ydl_base_opts()) as ydl:
        info = ydl.extract_info(q, download=False)
    # info puede ser dict con entries
    entries = []
    if info and "entries" in info and isinstance(info["entries"], list):
        for e in info["entries"]:
            if e:
                e["_source_query"] = query
                entries.append(e)
    return entries

def extract_playlist(url: str) -> List[Dict]:
    with YoutubeDL(ydl_base_opts()) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = []
    if info and "entries" in info and isinstance(info["entries"], list):
        for e in info["entries"]:
            if e:
                e["_source_playlist"] = info.get("title") or "playlist"
                entries.append(e)
    return entries

# ----------------------- util -----------------------

_YANG8_PAT = re.compile(r"\b(8|viii)\b|\bpal\b|\bpal\s*jang\b", re.IGNORECASE)

def looks_like_yang8(title: str) -> bool:
    s = title.lower()
    # señales comunes: taeguk 8, pal jang, poomsae 8, taegeuk 8, 태극8장, 팔장
    keys = [
        "taeguk 8", "taegeuk 8", "poomsae 8", "8 jang", "8jang",
        "pal jang", "paljang", "태극8장", "팔장", "팔 장"
    ]
    if any(k in s for k in keys):
        return True
    if _YANG8_PAT.search(s):
        # combinada con palabra taeguk/poomsae aumenta confianza
        if "taeguk" in s or "taegeuk" in s or "poomsae" in s or "태극" in s:
            return True
    return False

def to_row(e: Dict) -> Dict:
    return {
        "video_id": e.get("id"),
        "title": e.get("title"),
        "channel": e.get("channel") or e.get("uploader"),
        "duration": e.get("duration"),
        "url": e.get("url") or e.get("webpage_url"),
        "source_query": e.get("_source_query"),
        "source_playlist": e.get("_source_playlist"),
    }

# ----------------------- pipeline -----------------------

def collect_links(queries: List[str],
                  playlists: List[str],
                  per_query: int,
                  min_total: int,
                  dedup_on_id: bool = True,
                  filter_yang8: bool = True) -> List[Dict]:

    bag: Dict[str, Dict] = {}  # id -> entry
    rows: List[Dict] = []

    # 1) queries
    for q in queries:
        entries = extract_search(q, limit=per_query)
        for e in entries:
            if not e:
                continue
            url = e.get("url") or e.get("webpage_url")
            vid = e.get("id") or url
            title = e.get("title") or ""
            if filter_yang8 and title and not looks_like_yang8(title):
                continue
            if dedup_on_id and vid in bag:
                continue
            bag[vid] = e

    # 2) playlists
    for p in playlists:
        entries = extract_playlist(p)
        for e in entries:
            if not e:
                continue
            url = e.get("url") or e.get("webpage_url")
            vid = e.get("id") or url
            title = e.get("title") or ""
            if filter_yang8 and title and not looks_like_yang8(title):
                continue
            if dedup_on_id and vid in bag:
                continue
            bag[vid] = e

    # 3) a filas
    for e in bag.values():
        rows.append(to_row(e))

    # 4) si no alcanza min_total, relajar filtro_yang8 en última pasada de refuerzo
    if len(rows) < min_total and filter_yang8:
        extra: Dict[str, Dict] = {}
        # re-ejecutar queries con más resultados y sin filtro
        for q in queries:
            entries = extract_search(q, limit=max(per_query, min_total))
            for e in entries:
                if not e:
                    continue
                vid = e.get("id") or (e.get("url") or e.get("webpage_url"))
                if vid in bag or vid in extra:
                    continue
                extra[vid] = e
                if len(bag) + len(extra) >= min_total:
                    break
        for e in extra.values():
            rows.append(to_row(e))

    # ordenar por canal+title para estabilidad
    rows.sort(key=lambda r: (str(r.get("channel") or ""), str(r.get("title") or "")))
    return rows

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="Genera un Excel con enlaces de Taeguk 8 (Pal Jang) sin descargar videos.")
    ap.add_argument("--out", required=True, help="Ruta al .xlsx de salida")
    ap.add_argument("--min", type=int, default=120, help="Mínimo de enlaces requeridos")
    ap.add_argument("--per-query", type=int, default=150, help="Resultados por consulta")
    ap.add_argument("--no-filter", action="store_true", help="No filtrar títulos por 8 Jang; exporta todo lo devuelto")
    ap.add_argument("--query", action="append", default=[], help="Consulta de búsqueda (repetible)")
    ap.add_argument("--playlist", action="append", default=[], help="URL de playlist (repetible)")
    ap.add_argument("--csv", help="Opcional: además del Excel, guardar CSV en esta ruta")
    args = ap.parse_args()

    # consultas por defecto si no se indican
    queries = args.query or [
        "taeguk 8 jang poomsae",
        "taegeuk 8 jang poomsae",
        "poomsae taeguk 8",
        "pal jang poomsae",
        "태극8장 품새",
        "팔장 품새",
    ]
    playlists = args.playlist or []

    rows = collect_links(
        queries=queries,
        playlists=playlists,
        per_query=args.per_query,
        min_total=args.min,
        dedup_on_id=True,
        filter_yang8=not args.no_filter
    )

    # a DataFrame y Excel
    df = pd.DataFrame(rows, columns=["video_id", "title", "channel", "duration", "url", "source_query", "source_playlist"])
    # eliminar filas sin URL
    df = df[df["url"].astype(bool)].copy()

    # si aún no se llega al mínimo, truncar el mínimo alcanzable sin repetir
    if len(df) < args.min:
        print(f"[WARN] Se obtuvieron {len(df)} enlaces (< {args.min}). Exportando todos los únicos disponibles.")

    out_xlsx = Path(args.out)
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_xlsx, index=False)
    print(f"[OK] Excel guardado: {out_xlsx}  ({len(df)} filas)")

    if args.csv:
        out_csv = Path(args.csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"[OK] CSV guardado: {out_csv}")

if __name__ == "__main__":
    main()
