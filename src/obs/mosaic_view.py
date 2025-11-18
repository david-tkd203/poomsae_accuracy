#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np

PRIMARY_NAME = "Kinect RGB"      # en modo 'streams'
PRIMARY_FALLBACK = "Kinect (Microsoft Kinect v2)"  # en modo 'composite'

BG = (28, 28, 33)      # fondo
GUTTER = 10            # margen entre “tarjetas”
PAD = 8                # padding interno para etiqueta


def _label(img, text):
    if img is None:
        return None
    h, w = img.shape[:2]
    overlay = img.copy()
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    bx, by = PAD, PAD + th + 2
    bw, bh = tw + 2*PAD, th + 2*PAD
    cv2.rectangle(overlay, (PAD, PAD), (PAD + bw, PAD + bh), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    cv2.putText(img, text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA)
    return img


def _stack_grid(named_imgs, cols, tile_size):
    """Devuelve una grilla (H, W, 3) con márgenes y fondo."""
    if not named_imgs:
        return None
    tile_w, tile_h = tile_size
    rows = int(math.ceil(len(named_imgs) / float(cols)))
    W = cols * tile_w + (cols + 1) * GUTTER
    H = rows * tile_h + (rows + 1) * GUTTER
    canvas = np.full((H, W, 3), BG, dtype=np.uint8)

    i = 0
    for r in range(rows):
        for c in range(cols):
            if i >= len(named_imgs):
                break
            name, im = named_imgs[i]
            i += 1
            if im is None:
                continue
            # resize manteniendo aspecto
            h, w = im.shape[:2]
            scale = min(tile_w / float(w), tile_h / float(h))
            nw, nh = int(w*scale), int(h*scale)
            thumb = cv2.resize(im, (nw, nh))
            thumb = _label(thumb, name)

            y0 = GUTTER + r*(tile_h + GUTTER) + (tile_h - nh)//2
            x0 = GUTTER + c*(tile_w + GUTTER) + (tile_w - nw)//2
            canvas[y0:y0+nh, x0:x0+nw] = thumb
    return canvas


def build_studio(frames, target_height=540, kinect_mode="streams"):
    """
    Vista Studio:
      - Panel grande a la izquierda: Kinect RGB (streams) o composite (fallback).
      - Columna derecha: mini-grid 2x2 con Depth/IR/BodyIndex + webcams.
    """
    if not frames:
        return None

    # elegir primario
    primary_key = PRIMARY_NAME if kinect_mode == "streams" else PRIMARY_FALLBACK
    if primary_key not in frames:
        # fallback al primero
        primary_key = list(frames.keys())[0]

    primary = frames.get(primary_key)
    if primary is None:
        return None

    ph, pw = primary.shape[:2]
    # altura objetivo para el panel grande
    left_h = target_height
    left_w = int(pw * (left_h / float(ph)))
    left = cv2.resize(primary, (left_w, left_h))
    left = _label(left, primary_key)

    # recopilar “thumbnails” (todo lo que no sea el primario)
    thumbs = []
    for name, img in frames.items():
        if name == primary_key or img is None:
            continue
        thumbs.append((name, img))

    # grid 2x2 (o lo que entre)
    grid = None
    if thumbs:
        # cada tile ~ 40% del alto del panel izquierdo
        tile_h = max(160, int(left_h * 0.45))
        tile_w = max(280, int(left_w * 0.55))
        grid = _stack_grid(thumbs, cols=1, tile_size=(tile_w, tile_h))  # columna

        # igualar altura a la izquierda
        gh = grid.shape[0]
        grid = cv2.resize(grid, (grid.shape[1], left_h if gh != left_h else gh))

    # armar canvas con fondo y gutter central
    if grid is not None:
        H = max(left_h, grid.shape[0]) + 2*GUTTER
        W = left_w + grid.shape[1] + 3*GUTTER
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        # posiciones
        yL = (H - left_h)//2
        xL = GUTTER
        yR = (H - grid.shape[0])//2
        xR = left_w + 2*GUTTER
        canvas[yL:yL+left_h, xL:xL+left_w] = left
        canvas[yR:yR+grid.shape[0], xR:xR+grid.shape[1]] = grid
    else:
        H = left_h + 2*GUTTER
        W = left_w + 2*GUTTER
        canvas = np.full((H, W, 3), BG, dtype=np.uint8)
        canvas[GUTTER:GUTTER+left_h, GUTTER:GUTTER+left_w] = left

    return canvas


def build_grid(frames, target_height=540):
    """Vista Grid general con fondo y márgenes."""
    if not frames:
        return None
    imgs = [(k, v) for k, v in frames.items() if v is not None]
    if not imgs:
        return None

    # calcular grid casi cuadrada
    n = len(imgs)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))

    # alto de cada tile ~ target_height/rows
    tile_h = max(180, int(target_height / rows))
    # ancho: usar relación 16:9 aproximada si no conocemos w/h
    tile_w = int(tile_h * 16 / 9)

    return _stack_grid(imgs, cols=cols, tile_size=(tile_w, tile_h))


def build_mosaic(frames, target_height=540, layout="studio", kinect_mode="streams"):
    if layout == "grid":
        return build_grid(frames, target_height)
    return build_studio(frames, target_height, kinect_mode)
