# src/obs/mosaic_view.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import math
import cv2
import numpy as np

PAD_COLOR = (28, 28, 32)  # gris oscuro agradable


# ---------------- utilidades internas ----------------
def _safe_bgr(img: np.ndarray) -> np.ndarray:
    """Devuelve BGR válido; maneja None, GRAY y BGRA."""
    if img is None:
        return np.zeros((240, 320, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    # evita imágenes vacías
    if img.size == 0 or img.shape[0] <= 1 or img.shape[1] <= 1:
        return np.zeros((240, 320, 3), np.uint8)
    return img


def _letterbox(img: np.ndarray, dst_size: Tuple[int, int],
               pad_color=PAD_COLOR) -> np.ndarray:
    """Ajusta con barras (no deforma): dst_size=(W,H)."""
    img = _safe_bgr(img)
    dst_w, dst_h = int(dst_size[0]), int(dst_size[1])
    if dst_w <= 0 or dst_h <= 0:
        return img

    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return np.full((dst_h, dst_w, 3), pad_color, np.uint8)

    src_ratio = w / float(h)
    dst_ratio = dst_w / float(dst_h)

    if src_ratio > dst_ratio:
        new_w = dst_w
        new_h = max(1, int(round(new_w / src_ratio)))
    else:
        new_h = dst_h
        new_w = max(1, int(round(new_h * src_ratio)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((dst_h, dst_w, 3), pad_color, dtype=np.uint8)
    y0 = (dst_h - new_h) // 2
    x0 = (dst_w - new_w) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """Etiqueta en esquina sup-izq con auto-escala según alto del tile."""
    if not text:
        return img
    im = img.copy()
    H = im.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    # escala y grosor proporcionales al alto
    fs = max(0.45, min(1.0, H / 400.0))
    th = 1 if H < 400 else 2
    (tw, th_text), _ = cv2.getTextSize(text, font, fs, th)
    bx, by = 12, 16
    pad = 6
    cv2.rectangle(im,
                  (bx - pad, by - th_text - pad),
                  (bx + tw + pad, by + pad),
                  (0, 0, 0), -1)
    cv2.putText(im, text, (bx, by), font, fs, (240, 240, 240), th, cv2.LINE_AA)
    return im


def _best_grid(n: int, target_ratio: float, max_cols: int = 6) -> Tuple[int, int]:
    """
    Elige (rows, cols) que:
      1) se acerquen al aspect target_ratio = W/H del contenedor
      2) minimicen celdas vacías
      3) respeten tope de columnas
    """
    if n <= 0:
        return (0, 0)
    best = (1, n)
    best_cost = 1e9
    for cols in range(1, min(max_cols, n) + 1):
        rows = int(math.ceil(n / cols))
        # diferencia de aspect del mosaico y penalización por celdas vacías
        mosaic_ratio = cols / float(rows)
        aspect_cost = abs(math.log((mosaic_ratio + 1e-9) / (target_ratio + 1e-9)))
        empty = rows * cols - n
        empty_cost = empty * 0.15  # penaliza filas/cols sobrantes
        cost = aspect_cost + empty_cost
        if cost < best_cost:
            best_cost = cost
            best = (rows, cols)
    return best


def _stack_grid(tiles: List[np.ndarray], rows: int, cols: int,
                tile_size: Tuple[int, int]) -> np.ndarray:
    """Apila tiles en (rows x cols). Si faltan, rellena con PAD_COLOR."""
    W, H = tile_size
    blank = np.full((H, W, 3), PAD_COLOR, dtype=np.uint8)
    grid_rows = []
    idx = 0
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            if idx < len(tiles):
                row_imgs.append(tiles[idx])
            else:
                row_imgs.append(blank)
            idx += 1
        grid_rows.append(np.hstack(row_imgs))
    return np.vstack(grid_rows)


# ---------------- API principal ----------------
def build_mosaic(
    frames: Dict[str, np.ndarray],
    target_height: int = 720,
    layout: str = "grid",
    kinect_mode: str = "streams",
    *,
    target_width: Optional[int] = None,
    grid: Optional[Tuple[int, int]] = None,   # (rows, cols) o None=auto
    max_tiles: Optional[int] = None,          # límite de fuentes a mostrar
    prefer_aspect: float = 16/9,
    max_cols: int = 6,
    show_labels: bool = True,
    pad_color: Tuple[int, int, int] = PAD_COLOR,
    kinect_first: bool = True                 # ordena Kinect primero
) -> Optional[np.ndarray]:
    """
    Construye un mosaico responsivo:
      - Auto elige matriz (rows x cols) según aspect del contenedor.
      - Si target_width y target_height están definidos, el mosaico NO excede ese rectángulo.
      - Cada tile se letterboxea (no deforma).
      - 'grid=(r,c)' fuerza tamaño de grilla.
      - 'max_tiles' limita cuántas fuentes se usan.
    """
    if not frames:
        return None

    # ---- orden de nombres: Kinect primero si se quiere ----
    names = list(frames.keys())
    if kinect_first:
        def _key(n):
            # orden lógico de streams de Kinect
            order = {
                "Kinect RGB": 0, "Kinect Depth": 1,
                "Kinect IR": 2, "Kinect BodyIndex": 3,
                "Kinect (Microsoft Kinect v2)": 0
            }
            base = 0 if any(n.startswith(k) for k in order.keys()) else 1
            sub = order.get(n, 99)
            return (base, sub, n.lower())
        names.sort(key=_key)
    else:
        names.sort()

    if max_tiles is not None and max_tiles > 0:
        names = names[:max_tiles]
    N = len(names)

    # ---- aspect del contenedor destino ----
    if target_width and target_height and target_width > 0 and target_height > 0:
        dst_ratio = float(target_width) / float(target_height)
    else:
        dst_ratio = float(prefer_aspect)

    # ---- tamaño de grilla ----
    if grid and grid[0] > 0 and grid[1] > 0:
        rows, cols = int(grid[0]), int(grid[1])
    else:
        rows, cols = _best_grid(N, dst_ratio, max_cols=max_cols)

    rows = max(1, rows)
    cols = max(1, cols)

    # ---- tamaño de cada tile (respeta contenedor si ambos lados dados) ----
    if target_width and target_height and target_width > 0 and target_height > 0:
        # Ajuste exacto al rectángulo final
        tile_w = max(1, target_width // cols)
        tile_h = max(1, target_height // rows)
        final_w = tile_w * cols
        final_h = tile_h * rows
    else:
        # Fallback por altura; ancho del tile según prefer_aspect
        tile_h = max(1, target_height // rows if target_height else 360)
        tile_w = max(1, int(round(tile_h * prefer_aspect)))
        final_w = tile_w * cols
        final_h = tile_h * rows

    # ---- construir tiles ----
    tiles: List[np.ndarray] = []
    for name in names:
        img = _safe_bgr(frames.get(name))
        thumb = _letterbox(img, (tile_w, tile_h), pad_color=pad_color)
        if show_labels:
            thumb = _label(thumb, name)
        tiles.append(thumb)

    mosaic = _stack_grid(tiles, rows, cols, (tile_w, tile_h))

    # ---- si nos dieron target_width/height, aseguremos encaje exacto ----
    if target_width and target_height and target_width > 0 and target_height > 0:
        # Si por división entera sobran píxeles, centramos con letterbox global
        if mosaic.shape[1] != target_width or mosaic.shape[0] != target_height:
            mosaic = _letterbox(mosaic, (target_width, target_height), pad_color=pad_color)
    else:
        # Garantiza al menos target_height (si se pasó) manteniendo aspect
        if target_height and mosaic.shape[0] != target_height:
            # Ajuste vertical “agradable”
            tgt_w = int(round(final_w * (target_height / float(final_h))))
            mosaic = _letterbox(mosaic, (tgt_w, target_height), pad_color=pad_color)

    return mosaic
