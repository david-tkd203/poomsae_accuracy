#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# Intrínsecos aproximados del sensor de profundidad del Kinect v2 (512x424)
# Se usan SOLO como fallback si no está disponible el CoordinateMapper nativo.
KINECTV2_DEPTH_INTRINSICS = {
    "fx": 366.1,     # focal x (px)
    "fy": 366.1,     # focal y (px)
    "cx": 258.2,     # centro x (px)
    "cy": 204.2,     # centro y (px)
    "depth_scale": 0.001,  # mm -> m
}

def depth_to_points_approx(
    depth_u16: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    depth_scale: float = 0.001,
    decimate: int = 1,
    valid_min: int = 250,     # mm
    valid_max: int = 8000     # mm
) -> np.ndarray:
    """
    Convierte el mapa de profundidad (uint16, mm) a nube XYZ (m) usando modelo pinhole.
    Devuelve array (N,3). Usa sólo cada 'decimate' píxeles para aligerar.
    """
    if depth_u16.ndim != 2:
        raise ValueError("depth_u16 debe ser (H, W)")

    H, W = depth_u16.shape
    step = max(1, int(decimate))

    # Cuadrícula (u, v) con salto 'step'
    v, u = np.mgrid[0:H:step, 0:W:step]
    d = depth_u16[::step, ::step].astype(np.float32)  # mm

    # Filtro de válidos
    mask = (d >= valid_min) & (d <= valid_max)
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32)

    d = d[mask] * depth_scale  # m
    u = u[mask].astype(np.float32)
    v = v[mask].astype(np.float32)

    # Pinhole
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    z = d

    pts = np.stack((x, y, z), axis=-1).astype(np.float32)
    return pts

def write_ply_ascii(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """
    Escribe un .ply ASCII con vértices (x,y,z) y opcionalmente (r,g,b).
    'points': (N,3) float32
    'colors': (N,3) uint8 (opcional)
    """
    if points is None or points.size == 0:
        raise ValueError("No hay puntos para escribir.")

    N = points.shape[0]
    has_color = colors is not None and colors.shape[0] == N

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if has_color:
            for (x, y, z), (r, g, b) in zip(points, colors):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in points:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
