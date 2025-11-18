# src/obs/pointcloud_qt.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

try:
    import pyqtgraph.opengl as gl
    _HAS_GL = True
except Exception:
    gl = None
    _HAS_GL = False


def _finite_mask(a: np.ndarray) -> np.ndarray:
    if a is None or a.size == 0:
        return np.zeros((0,), dtype=bool)
    if a.ndim != 2 or a.shape[1] != 3:
        return np.zeros((a.shape[0],), dtype=bool)
    return np.isfinite(a).all(axis=1)


def _bbox_center_radius(pts: np.ndarray) -> tuple[np.ndarray, float]:
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
                      colors: np.ndarray | None,
                      voxel: float) -> tuple[np.ndarray, np.ndarray | None]:
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


class PointCloudWidget(QtWidgets.QWidget):
    """
    Visor 3D para nubes con:
      - Auto-fit (una sola vez) al primer update.
      - Ajuste FOV y clip planes.
      - Acumulación opcional de frames.
      - Downsample por vóxel.
      - Tamaño de punto configurable.

    API:
      set_point_size(px: float)
      set_fov(deg: float)
      set_clip(near: float, far: float)
      set_auto_fit(enabled: bool)
      set_accumulate(enabled: bool, max_points: int = 2_000_000)
      set_voxel_size(voxel_m: float | None)
      clear_cloud()
      update_cloud(points, colors=None, auto_fit: bool | None = None)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _HAS_GL:
            lab = QtWidgets.QLabel(
                "Instala pyqtgraph[opengl] para ver 3D:\n\npip install pyqtgraph PyOpenGL"
            )
            lab.setStyleSheet("color:#ccc; padding:16px;")
            lab.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(lab)
            self.view = None
            self.scatter = None
            self._point_size = 3.0
            self._auto_fit_on_first = True
            self._did_fit_once = False
            self._accumulate = False
            self._accum_pts = None
            self._accum_cols = None
            self._accum_budget = 2_000_000
            self._voxel_m = None
            return

        # --- Vista ---
        self.view = gl.GLViewWidget()
        layout.addWidget(self.view, 1)

        # Cámara inicial (FOV amplio para ver más campo)
        self.view.opts['fov'] = 80.0
        self.view.opts['elevation'] = 20.0
        self.view.opts['azimuth'] = 45.0
        self.view.opts['distance'] = 2.5
        self.view.opts['center'] = QtGui.QVector3D(0.0, 0.8, 1.5)
        self.view.opts['clip'] = (0.01, 5000.0)

        # Grid de referencia
        self.grid = gl.GLGridItem()
        try:
            self.grid.setSize(x=4, y=4, z=1)
        except Exception:
            self.grid.scale(1, 1, 1)
        self.view.addItem(self.grid)

        # Nube
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque')
        self.view.addItem(self.scatter)

        # Estado
        self._point_size = 3.0
        self._auto_fit_on_first = True
        self._did_fit_once = False
        self._accumulate = False
        self._accum_pts = None
        self._accum_cols = None
        self._accum_budget = 2_000_000
        self._voxel_m = None

        self.scatter.setData(size=self._point_size)

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
        self._auto_fit_on_first = bool(enabled)

    def set_accumulate(self, enabled: bool, max_points: int = 2_000_000):
        self._accumulate = bool(enabled)
        self._accum_budget = int(max_points)
        if not self._accumulate:
            self._accum_pts = None
            self._accum_cols = None

    def set_voxel_size(self, voxel_m: float | None):
        if voxel_m is None or voxel_m <= 0:
            self._voxel_m = None
        else:
            self._voxel_m = float(voxel_m)

    def clear_cloud(self):
        self._accum_pts = None
        self._accum_cols = None
        if self.scatter:
            self.scatter.setData(pos=np.zeros((0, 3), dtype=np.float32))

    # ----------------- Render -----------------
    def _apply_fit(self, pts_valid: np.ndarray):
        if pts_valid is None or pts_valid.size == 0 or self.view is None:
            return
        center, radius = _bbox_center_radius(pts_valid)
        self.view.opts['center'] = QtGui.QVector3D(float(center[0]),
                                                   float(center[1]),
                                                   float(center[2]))
        dist = _fit_distance_for_fov(radius, self.view.opts.get('fov', 80.0))
        self.view.opts['distance'] = dist
        try:
            s = max(2.0, float(radius) * 2.5)
            self.grid.setSize(x=s, y=s, z=1)
        except Exception:
            pass

    def _update_accum(self, pts: np.ndarray, cols: np.ndarray | None):
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

    def update_cloud(self, points: np.ndarray, colors: np.ndarray | None = None, auto_fit: bool | None = None):
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

        if cols_draw is not None and cols_draw.shape[0] == pts_draw.shape[0]:
            c = (cols_draw.astype(np.float32) / 255.0)
        else:
            c = np.ones((pts_draw.shape[0], 3), dtype=np.float32) * 0.85

        self.scatter.setData(pos=pts_draw, color=c, size=self._point_size)

        do_fit = self._auto_fit_on_first if (auto_fit is None) else bool(auto_fit)
        if do_fit and not self._did_fit_once:
            try:
                self._apply_fit(pts_draw)
                self._did_fit_once = True
            except Exception:
                pass
