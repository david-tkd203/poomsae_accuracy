# src/obs/pointcloud_qt.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Tuple, Literal
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# -----------------------------------------------------------------------------
# OpenGL backend (pyqtgraph)
# -----------------------------------------------------------------------------
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:
    gl = None
    pg = None
    _HAS_GL = False


# -----------------------------------------------------------------------------
# Helpers numéricos
# -----------------------------------------------------------------------------
def _finite_mask(a: np.ndarray) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros((0,), dtype=bool)
    if a.ndim != 2 or a.shape[1] != 3:
        return np.zeros((a.shape[0],), dtype=bool)
    return np.isfinite(a).all(axis=1)


def _bbox_center_radius(pts: np.ndarray) -> Tuple[np.ndarray, float]:
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = (mn + mx) * 0.5
    radius = float(np.max(mx - mn)) * 0.5
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    return center.astype(np.float32), radius


def _fit_distance_for_fov(radius: float, fov_deg: float, safety: float = 1.4) -> float:
    fov = max(10.0, min(120.0, float(fov_deg)))
    theta = np.deg2rad(fov * 0.5)
    base = radius / max(1e-6, np.tan(theta))
    return float(base * safety)


def _voxel_downsample(pts: np.ndarray,
                      colors: Optional[np.ndarray],
                      voxel: float) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if voxel is None or voxel <= 0:
        return pts, colors
    if pts is None or pts.size == 0:
        return pts, colors

    grid = np.floor(pts / float(voxel)).astype(np.int32, copy=False)
    key = (grid[:, 0].astype(np.int64) * 73856093 ^
           grid[:, 1].astype(np.int64) * 19349663 ^
           grid[:, 2].astype(np.int64) * 83492791)
    _, idx = np.unique(key, return_index=True)
    idx = np.sort(idx)

    pts_ds = pts[idx]
    cols_ds = colors[idx] if (colors is not None and colors.shape[0] == pts.shape[0]) else None
    return pts_ds, cols_ds


# -----------------------------------------------------------------------------
# Widget principal
# -----------------------------------------------------------------------------
class PointCloudWidget(QtWidgets.QWidget):
    """
    Visor 3D para nubes:
      - FOV y planos de recorte configurables.
      - Auto-fit del encuadre (una vez o continuo).
      - Acumulación de frames con presupuesto de puntos.
      - Downsample por vóxel para aliviar carga.
      - Tamaño de punto ajustable.
      - Ejes y grilla opcionales.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_GL:
            lab = QtWidgets.QLabel(
                "Visor 3D deshabilitado.\n\n"
                "Instala OpenGL para pyqtgraph:\n"
                "  pip install pyqtgraph PyOpenGL PyOpenGL-accelerate"
            )
            lab.setStyleSheet("color:#ccc; padding:16px;")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(lab)
            # Estados mínimos para no romper setters
            self.view = None
            self.scatter = None
            self.axes = None
            self.grid = None
            self._point_size = 3.0
            self._auto_fit_mode: Literal['off','once','continuous'] = 'off'
            self._did_fit_once = False
            self._accumulate = False
            self._accum_pts = None
            self._accum_cols = None
            self._accum_budget = 2_000_000
            self._voxel_m: Optional[float] = None
            return

        # --- Vista GL ---
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view, 1)

        # Fondo: en algunas versiones no existe; proteger
        try:
            if pg is not None:
                self.view.setBackgroundColor(pg.mkColor(20, 20, 24))
        except Exception:
            pass

        # Cámara inicial (FOV amplio para ver más campo)
        self.view.opts['fov'] = 90.0
        self.view.opts['elevation'] = 20.0
        self.view.opts['azimuth'] = 45.0
        self.view.opts['distance'] = 3.0
        self.view.opts['center'] = QtGui.QVector3D(0.0, 0.8, 1.5)
        self.view.opts['clip'] = (0.01, 12000.0)

        # Grilla XY
        self.grid = gl.GLGridItem()
        try:
            self.grid.setSize(x=6, y=6, z=1)
        except Exception:
            try:
                self.grid.scale(1, 1, 1)
            except Exception:
                pass
        # En algunas versiones no existe setDepthValue
        try:
            self.grid.setDepthValue(1)
        except Exception:
            pass
        self.view.addItem(self.grid)

        # Ejes (1 m) — puede faltar en builds antiguas
        try:
            self.axes = gl.GLAxisItem()
            self.axes.setSize(1.0, 1.0, 1.0)
            self.axes.translate(0, 0, 0)
            self.view.addItem(self.axes)
        except Exception:
            self.axes = None

        # Nube de puntos
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque')
        self.view.addItem(self.scatter)

        # Estado
        self._point_size = 2.5
        self._auto_fit_mode: Literal['off','once','continuous'] = 'once'
        self._did_fit_once = False
        self._accumulate = False
        self._accum_pts: Optional[np.ndarray] = None
        self._accum_cols: Optional[np.ndarray] = None
        self._accum_budget = 2_000_000
        self._voxel_m: Optional[float] = None

        self.scatter.setData(size=self._point_size)

        # Atajos útiles
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, activated=self.fit_now)
        QtWidgets.QShortcut(QtGui.QKeySequence("C"), self, activated=self.clear_cloud)
        QtWidgets.QShortcut(
            QtGui.QKeySequence("G"), self,
            activated=lambda: self.set_grid_visible(False if self.grid is None else not self.grid.isVisible())
        )
        if self.axes is not None:
            QtWidgets.QShortcut(
                QtGui.QKeySequence("A"), self,
                activated=lambda: self.set_axes_visible(not self.axes.isVisible())
            )

    # ----------------- API de control -----------------
    def set_point_size(self, px: float):
        self._point_size = float(px)
        if self.scatter:
            self.scatter.setData(size=self._point_size)

    def set_fov(self, deg: float):
        if self.view:
            self.view.opts['fov'] = max(10.0, min(120.0, float(deg)))

    def set_clip(self, near: float, far: float):
        if self.view:
            n = max(1e-4, float(near))
            f = max(n + 1e-3, float(far))
            self.view.opts['clip'] = (n, f)

    def set_auto_fit(self, enabled: bool):
        self.set_auto_fit_mode('once' if enabled else 'off')

    def set_auto_fit_mode(self, mode: Literal['off','once','continuous']):
        mode = str(mode).lower()
        if mode not in ('off', 'once', 'continuous'):
            mode = 'once'
        self._auto_fit_mode = mode
        if mode != 'once':
            self._did_fit_once = False

    def set_accumulate(self, enabled: bool, max_points: int = 2_000_000):
        self._accumulate = bool(enabled)
        self._accum_budget = int(max_points)
        if not self._accumulate:
            self._accum_pts = None
            self._accum_cols = None

    def set_voxel_size(self, voxel_m: Optional[float]):
        if voxel_m is None or voxel_m <= 0:
            self._voxel_m = None
        else:
            self._voxel_m = float(voxel_m)

    def set_orbit(self, elev: float, azim: float, dist: Optional[float] = None):
        if not self.view:
            return
        self.view.opts['elevation'] = float(elev)
        self.view.opts['azimuth'] = float(azim)
        if dist is not None:
            self.view.opts['distance'] = float(dist)

    def set_center(self, x: float, y: float, z: float):
        if self.view:
            self.view.opts['center'] = QtGui.QVector3D(float(x), float(y), float(z))

    def set_axes_visible(self, flag: bool):
        if self.axes is not None:
            self.axes.setVisible(bool(flag))

    def set_grid_visible(self, flag: bool):
        if self.grid is not None:
            self.grid.setVisible(bool(flag))

    def clear_cloud(self):
        self._accum_pts = None
        self._accum_cols = None
        self._did_fit_once = False
        if self.scatter:
            self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def fit_now(self):
        if not (self.view and self.scatter):
            return
        data = self.scatter.opts.get('pos', None)
        if data is None or data.size == 0:
            return
        try:
            self._apply_fit(np.asarray(data))
            self._did_fit_once = True
        except Exception:
            pass

    # ----------------- Render / actualización -----------------
    def _apply_fit(self, pts_valid: np.ndarray):
        if pts_valid is None or pts_valid.size == 0 or self.view is None:
            return
        center, radius = _bbox_center_radius(pts_valid)
        self.view.opts['center'] = QtGui.QVector3D(float(center[0]),
                                                   float(center[1]),
                                                   float(center[2]))
        dist = _fit_distance_for_fov(radius, self.view.opts.get('fov', 90.0))
        self.view.opts['distance'] = dist
        try:
            s = max(2.0, float(radius) * 2.5)
            self.grid.setSize(x=s, y=s, z=1)
        except Exception:
            pass

    def _update_accum(self, pts: np.ndarray, cols: Optional[np.ndarray]):
        if pts is None or pts.size == 0:
            return
        if self._accum_pts is None:
            self._accum_pts = pts.copy()
            self._accum_cols = cols.copy() if cols is not None else None
        else:
            self._accum_pts = np.vstack((self._accum_pts, pts))
            if cols is not None:
                if self._accum_cols is None:
                    self._accum_cols = cols.copy()
                else:
                    self._accum_cols = np.vstack((self._accum_cols, cols))

        if self._accum_pts.shape[0] > self._accum_budget:
            k = self._accum_pts.shape[0] - self._accum_budget
            self._accum_pts = self._accum_pts[k:, :]
            if self._accum_cols is not None:
                self._accum_cols = self._accum_cols[k:, :]

    def update_cloud(self,
                     points: np.ndarray,
                     colors: Optional[np.ndarray] = None,
                     auto_fit: Optional[bool] = None):
        if self.view is None or self.scatter is None:
            return
        if points is None or points.size == 0:
            return

        m = _finite_mask(points)
        if m.sum() == 0:
            return
        pts = points[m].astype(np.float32, copy=False)
        cols = colors[m] if (colors is not None and colors.shape[0] == points.shape[0]) else None

        if self._accumulate:
            self._update_accum(pts, cols)
            pts_draw = self._accum_pts
            cols_draw = self._accum_cols
        else:
            pts_draw = pts
            cols_draw = cols

        if self._voxel_m and self._voxel_m > 0:
            pts_draw, cols_draw = _voxel_downsample(pts_draw, cols_draw, self._voxel_m)

        if pts_draw is None or pts_draw.size == 0:
            return

        if cols_draw is not None and cols_draw.shape[0] == pts_draw.shape[0]:
            c = (cols_draw.astype(np.float32) / 255.0)
        else:
            c = np.ones((pts_draw.shape[0], 3), dtype=np.float32) * 0.85

        self.scatter.setData(pos=pts_draw, color=c, size=self._point_size)

        if auto_fit is None:
            if self._auto_fit_mode == 'continuous':
                try:
                    self._apply_fit(pts_draw)
                except Exception:
                    pass
            elif self._auto_fit_mode == 'once' and not self._did_fit_once:
                try:
                    self._apply_fit(pts_draw)
                    self._did_fit_once = True
                except Exception:
                    pass
        elif bool(auto_fit):
            try:
                self._apply_fit(pts_draw)
                self._did_fit_once = True
            except Exception:
                pass
