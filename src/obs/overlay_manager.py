#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2


def draw_hud(canvas, lines, pos="bottom-left", margin=12):
    if canvas is None:
        return canvas
    h, w = canvas.shape[:2]

    # base de texto
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thick = 2
    line_h = 24

    # calcular tamaño del panel
    text_w = 0
    for s in lines:
        (tw, th), _ = cv2.getTextSize(s, font, scale, thick)
        text_w = max(text_w, tw)
    panel_w = text_w + 2 * margin
    panel_h = line_h * len(lines) + 2 * margin

    # esquina
    if pos == "bottom-left":
        x0, y0 = margin, h - panel_h - margin
    elif pos == "top-left":
        x0, y0 = margin, margin
    elif pos == "top-right":
        x0, y0 = w - panel_w - margin, margin
    else:  # bottom-right
        x0, y0 = w - panel_w - margin, h - panel_h - margin

    # panel semitransparente
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
    alpha = 0.35
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # texto
    y = y0 + margin + 18
    for s in lines:
        cv2.putText(canvas, s, (x0 + margin, y), font, scale, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_h
    return canvas
