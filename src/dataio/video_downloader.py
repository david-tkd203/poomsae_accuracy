# src/dataio/video_downloader.py
from __future__ import annotations

import os
import re
import sys
import time
import logging
import unicodedata
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from urllib.parse import urlparse, parse_qs, urlunparse

import requests
from requests.exceptions import RequestException

# yt-dlp es opcional; solo se importa cuando se usa
# pip install yt-dlp
# Requiere FFmpeg en PATH para mezclar streams
try:
    import yt_dlp  # noqa: F401
    _HAS_YTDLP = True
except Exception:
    _HAS_YTDLP = False

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
LOG = logging.getLogger("video_downloader")
if not LOG.handlers:
    LOG.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(handler)

VIDEO_EXT_RE = re.compile(r"\.(mp4|mov|mkv|webm|avi|flv|m4v|mpg|mpeg)$", re.IGNORECASE)

# --------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    # Limpiar y acortar nombres para Windows/Unix
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.category(c).startswith("C"))
    name = re.sub(r'[\\/*?:"<>|\r\n\t]+', "_", name)
    name = name.strip(" ._")
    # recorta longitud extrema:
    return name[:180] or "video"

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")

def env_proxies_from_env() -> Dict[str, str]:
    http = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    proxies: Dict[str, str] = {}
    if http:
        proxies["http"] = http
    if https:
        proxies["https"] = https
    return proxies

def is_probably_direct_file(url: str) -> bool:
    return bool(VIDEO_EXT_RE.search(url))

def head_content_type(url: str, timeout: int = 10, proxies: Optional[Dict[str,str]] = None) -> Optional[str]:
    try:
        r = requests.head(url, allow_redirects=True, timeout=timeout, proxies=proxies)
        ct = r.headers.get("Content-Type", "")
        return ct.lower() if ct else None
    except RequestException:
        return None

# --------------------------------------------------------------------
# Normalización de alias (8yang, 7yang, etc.)
# --------------------------------------------------------------------
_YANG_PATTS = [
    r'(?:^|\b)(yang)\s*[-_ ]*([0-9]{1,2})(?:\b|$)',
    r'(?:^|\b)([0-9]{1,2})\s*[-_ ]*(yang)(?:\b|$)',
    r'(?:^|\b)(tae?ge?uk)\s*[-_ ]*([0-9]{1,2})(?:\b|$)',
    r'(?:^|\b)([0-9]{1,2})\s*[-_ ]*(tae?ge?uk)(?:\b|$)',
    r'(?:^|\b)(poomsae)\s*[-_ ]*([0-9]{1,2})(?:\b|$)',
    r'(?:^|\b)([0-9]{1,2})\s*[-_ ]*(poomsae)(?:\b|$)',
]

def normalize_alias_from_title(title: str) -> Optional[str]:
    s = strip_accents((title or "").lower())
    s = re.sub(r'[^a-z0-9\s_-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    for patt in _YANG_PATTS:
        m = re.search(patt, s)
        if m:
            nums = [g for g in m.groups() if g and g.isdigit()]
            if nums:
                return f"{int(nums[0])}yang"
    return None

# --------------------------------------------------------------------
# Descarga HTTP directa (para URLs a archivos de video)
# --------------------------------------------------------------------
def http_download(
    url: str,
    out_dir: Path,
    filename: Optional[str] = None,
    skip_existing: bool = True,
    chunk_size: int = 1 << 20,  # 1 MB
) -> Path:
    ensure_dir(out_dir)
    proxies = env_proxies_from_env()

    if filename is None:
        parsed = urlparse(url)
        base = Path(parsed.path).name or "download"
        base = sanitize_filename(base)
        if not VIDEO_EXT_RE.search(base):
            base += ".mp4"
        filename = base

    out_path = out_dir / filename
    if skip_existing and out_path.exists():
        LOG.info(f"Ya existe: {out_path}")
        return out_path

    try:
        with requests.get(url, stream=True, allow_redirects=True, timeout=30, proxies=proxies) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            tmp_path = out_path.with_suffix(out_path.suffix + ".part")

            downloaded = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # si no hubo Content-Length, igual completamos
            tmp_path.replace(out_path)
            LOG.info(f"Descargado: {out_path} ({downloaded} bytes)")
            return out_path
    except RequestException as e:
        raise RuntimeError(f"Fallo HTTP al descargar: {url} -> {e}") from e

# --------------------------------------------------------------------
# yt-dlp helpers
# --------------------------------------------------------------------
def _extract_title_and_kind(url: str) -> Tuple[Optional[str], str]:
    """
    Usa yt-dlp para extraer metadata sin descargar.
    Retorna (title, kind) donde kind ∈ {'playlist','video','unknown'}.
    """
    if not _HAS_YTDLP:
        return None, "unknown"
    from yt_dlp import YoutubeDL

    opts = {
        "quiet": True,
        "skip_download": True,
        "extract_flat": True,
        "noplaylist": False,
        "ignoreerrors": True,
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            return None, "unknown"
        if "entries" in info:
            title = info.get("title") or info.get("playlist_title")
            return title, "playlist"
        else:
            title = info.get("title")
            return title, "video"

def ytdlp_download(
    url: str,
    out_dir: Path,
    *,
    prefix: Optional[str] = None,
    subset: Optional[str] = None,
    format_str: Optional[str] = None,
    cookies: Optional[Path] = None,
    skip_existing: bool = True
) -> List[Path]:
    if not _HAS_YTDLP:
        raise RuntimeError("yt-dlp no está instalado. Instala con: pip install yt-dlp")
    from yt_dlp import YoutubeDL

    # Normaliza Shorts a watch?v=...
    url = _normalize_youtube_url(url)

    target = Path(out_dir)
    if prefix:
        target = target / prefix
    if subset:
        target = target / subset
    ensure_dir(target)

    # Formato robusto (prioriza MP4/H.264 + M4A). Puedes sobreescribir con --format
    ytdlp_format = (
        format_str
        or os.getenv("YTDLP_FORMAT")
        or "bestvideo[ext=mp4][vcodec^=avc1]+bestaudio[ext=m4a]/"
           "best[ext=mp4]/best"
    )
    name_prefix = f"{prefix}-" if prefix else ""
    outtmpl = str(target / f"{name_prefix}%(playlist_index)03d-%(title).200s-%(id)s.%(ext)s")

    started_at = time.time()
    finished_files: List[Path] = []

    def _progress_hook(d):
        if d.get("status") == "finished":
            fn = d.get("filename")
            if fn:
                finished_files.append(Path(fn))

    base_opts = dict(
        outtmpl=outtmpl,
        outtmpl_na_placeholder="000",
        restrictfilenames=True,
        ignoreerrors=True,
        merge_output_format="mp4",
        format=ytdlp_format,
        nocheckcertificate=True,
        retries=5,
        fragment_retries=5,
        continuedl=True,
        windowsfilenames=True,
        quiet=False,
        no_warnings=True,
        postprocessors=[{"key": "FFmpegVideoConvertor", "preferedformat": "mp4"}],
        progress_hooks=[_progress_hook],
        noplaylist=False,  # soporta playlists también
        # endurecer contra 403:
        concurrent_fragment_downloads=1,   # evita demasiados hilos a la vez
        sleep_requests=2,                  # duerme entre peticiones
        max_sleep_interval=4,
        geo_bypass=True,
        force_ipv4=True,
        http_headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/124.0 Safari/537.36"),
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            "Referer": "https://www.youtube.com/",
        },
        extractor_args={
            # probar varios clientes para Shorts
            "youtube": {"player_client": ["android", "web"]}
        },
    )
    if cookies and Path(cookies).exists():
        base_opts["cookiefile"] = str(cookies)

    # Intento 1
    try:
        with YoutubeDL(base_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError(f"No se pudo extraer/descargar: {url}")
    except Exception as e1:
        # Intento 2 (fallback): cambia orden de clientes y relaja filtros
        fallback_opts = base_opts.copy()
        fallback_opts["extractor_args"] = {"youtube": {"player_client": ["web", "ios"]}}
        # formato más laxo
        if not format_str and not os.getenv("YTDLP_FORMAT"):
            fallback_opts["format"] = "bestvideo+bestaudio/best"
        try:
            with YoutubeDL(fallback_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise RuntimeError(f"No se pudo extraer/descargar: {url}")
        except Exception as e2:
            raise RuntimeError(f"yt-dlp falló en ambos intentos para {url} -> {e1} / {e2}") from e2

    # Confirmar salidas
    results: List[Path] = []
    seen: set[Path] = set()
    for p in finished_files:
        if p.exists():
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp); results.append(p)
            continue
        for ext in (".mp4", ".mkv", ".webm", ".m4v"):
            cand = p.with_suffix(ext)
            if cand.exists():
                rp = cand.resolve()
                if rp not in seen:
                    seen.add(rp); results.append(cand)
                break

    if not results:
        for ext in ("*.mp4", "*.mkv", "*.webm", "*.m4v"):
            for fp in target.glob(ext):
                if fp.stat().st_mtime >= started_at - 2:
                    rp = fp.resolve()
                    if rp not in seen:
                        seen.add(rp); results.append(fp)

    uniq: List[Path] = []
    seen2: set[Path] = set()
    for p in results:
        if p and p.exists():
            rp = p.resolve()
            if rp not in seen2:
                seen2.add(rp); uniq.append(p)

    LOG.info(f"Descargas: {len(uniq)} archivo(s) en {target}")
    return uniq
# --------------------------------------------------------------------
# Orquestador con normalización de alias y fallback HTTP
# --------------------------------------------------------------------
def smart_download(
    url: str,
    out_dir: Path | str,
    *,
    cookies: Optional[Path | str] = None,
    prefer_ytdlp: bool = True,
    skip_existing: bool = True,
    alias: Optional[str] = None,
    subset: Optional[str] = None,
    normalize_from_title: bool = True,
    format_str: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Path]:
    """
    - Si alias está definido, se usa como nombre canónico (ej. '8yang').
    - Si no, y normalize_from_title=True, se intenta extraer título con yt-dlp y derivar alias ('8yang').
    - Para yt-dlp, guarda en: out_dir / [alias] / [subset]
    - Para HTTP directo, guarda en: out_dir / [alias] / [subset] y prefija filename con alias.
    """
    out_base = Path(out_dir)
    ensure_dir(out_base)

    # Resolver alias si no fue provisto
    eff_alias = alias
    if eff_alias is None and normalize_from_title and prefer_ytdlp and _HAS_YTDLP:
        title, kind = _extract_title_and_kind(url)
        if title:
            cand = normalize_alias_from_title(title)
            if cand:
                eff_alias = cand

    # 1) Preferir yt-dlp
    if prefer_ytdlp:
        try:
            opts = {}
            if limit is not None:
                opts["playlistend"] = int(limit)

            return ytdlp_download(
                url,
                out_base,
                prefix=eff_alias,
                subset=subset,
                format_str=format_str,
                cookies=Path(cookies) if cookies else None,
                skip_existing=skip_existing,
            )
        except Exception as e:
            LOG.warning(f"yt-dlp falló, probando HTTP directo si aplica. Motivo: {e}")

    # 2) HTTP directo (solo si la URL parece archivo de video)
    target = out_base
    if eff_alias:
        target = target / eff_alias
    if subset:
        target = target / subset
    ensure_dir(target)

    if is_probably_direct_file(url):
        parsed = urlparse(url)
        base = Path(parsed.path).name or "download.mp4"
        base = sanitize_filename(base)
        if not VIDEO_EXT_RE.search(base):
            base += ".mp4"
        filename = f"{eff_alias + '-' if eff_alias else ''}"
        p = http_download(url, target, filename=filename, skip_existing=skip_existing)
        return [p]

    ct = head_content_type(url)
    if ct and ct.startswith("video"):
        parsed = urlparse(url)
        base = Path(parsed.path).name or "download.mp4"
        base = sanitize_filename(base)
        if not VIDEO_EXT_RE.search(base):
            base += ".mp4"
        filename = f"{eff_alias + '-' if eff_alias else ''}{base}"
        p = http_download(url, target, filename=filename, skip_existing=skip_existing)
        return [p]

    # 3) Último intento: yt-dlp si inicialmente estaba deshabilitado
    if not prefer_ytdlp:
        if not _HAS_YTDLP:
            raise RuntimeError("yt-dlp no está instalado y la URL no es un archivo directo de video.")
        return ytdlp_download(
            url,
            out_base,
            prefix=eff_alias,
            subset=subset,
            format_str=format_str,
            cookies=Path(cookies) if cookies else None,
            skip_existing=skip_existing,
        )

    raise RuntimeError(f"No se pudo determinar método de descarga para: {url}")

def _normalize_youtube_url(u: str) -> str:
    """
    Convierte URLs 'shorts' y 'youtu.be' a forma watch?v=ID para mayor compatibilidad.
    Mantiene playlists intactas.
    """
    try:
        pr = urlparse(u)
        host = pr.netloc.lower()
        path = pr.path
        # playlist: no tocar
        if "youtube.com" in host and path.startswith("/playlist"):
            return u
        # youtu.be/<id>
        if host in ("youtu.be",):
            vid = path.strip("/").split("/")[0]
            if vid:
                return f"https://www.youtube.com/watch?v={vid}"
        # youtube.com/shorts/<id>
        if "youtube.com" in host and path.startswith("/shorts/"):
            parts = [p for p in path.split("/") if p]
            # ['/shorts','<id>']
            if len(parts) >= 2 and parts[0] == "shorts":
                vid = parts[1]
                # conserva la query existente (si trae si=... no hace falta)
                q = parse_qs(pr.query)
                # reconstruir como watch?v=<id>
                return f"https://www.youtube.com/watch?v={vid}"
        return u
    except Exception:
        return u
