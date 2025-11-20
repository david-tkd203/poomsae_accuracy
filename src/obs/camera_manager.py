# src/obs/camera_manager.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import platform
import threading
import time
from collections import OrderedDict
from typing import Optional, Dict, Tuple, List

import cv2
import numpy as np

# Intento de importar PyKinect2 solo para saber si está disponible
try:
    from pykinect2 import PyKinectV2  # noqa: F401
    KINECT_LIB = True
except Exception:
    KINECT_LIB = False


# ---------------- ThreadedCam: lectura no bloqueante ----------------
class ThreadedCam:
    def __init__(self, index: int, backend: int, width=1280, height=720, fps=30):
        self.index = index
        self.backend = backend
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir webcam {index} ({backend}).")

        # Intento de configuración “razonable”
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS,         fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self._lock = threading.Lock()
        self._last = None
        self._stopping = threading.Event()
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stopping.is_set():
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._last = frame
            else:
                time.sleep(0.01)

    def read(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last is None:
                return None
            return self._last.copy()

    def stop(self):
        self._stopping.set()
        self._t.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass


# ---------------- CameraManager ----------------
class CameraManager:
    """
    Administra Kinect + webcams y expone frames como OrderedDict[str, np.ndarray (BGR)].

    kinect_mode:
      - "streams": fuentes separadas -> "Kinect RGB", "Kinect Depth", "Kinect IR", "Kinect BodyIndex"
      - "composite": un único "Kinect (Microsoft Kinect v2)" armado por el backend

    Parámetros extra:
      - max_webcams_visible: límite de webcams a retornar (None = sin límite)
      - rescan_interval_sec: si >0, re-sondea cada N segundos para detectar cámaras nuevas
      - exclude_indices: set de índices a ignorar
      - exclude_name_contains: substrings (lower) para ignorar por nombre
    """

    def __init__(self,
                 kinect_backend=None,
                 kinect_mode: str = "streams",
                 max_probe: int = 6,
                 consecutive_fail_stop: int = 3,
                 exclude_indices: Optional[List[int]] = None,
                 exclude_name_contains: Optional[List[str]] = None,
                 max_webcams_visible: Optional[int] = None,
                 rescan_interval_sec: float = 0.0):
        self._kinect_backend = kinect_backend
        self._kinect = None

        self.kinect_mode = str(kinect_mode)
        self._exclude_indices = set(exclude_indices or [])
        self._exclude_name_contains = [s.lower() for s in (exclude_name_contains or [])]

        self._webcams: "OrderedDict[str, ThreadedCam]" = OrderedDict()
        self._webcam_names: List[str] = []

        self._max_probe = int(max_probe)
        self._consecutive_fail_stop = int(consecutive_fail_stop)
        self._max_webcams_visible = max_webcams_visible
        self._rescan_interval = float(rescan_interval_sec)
        self._last_scan_t = 0.0

        self._discover_webcams(initial=True)

    # ---------- API pública ----------
    def set_kinect_mode(self, mode: str):
        self.kinect_mode = "streams" if str(mode).lower() == "streams" else "composite"

    def set_max_webcams_visible(self, n: Optional[int]):
        self._max_webcams_visible = None if n is None else int(max(0, n))

    def list_cameras(self) -> List[str]:
        cams = []
        if KINECT_LIB:
            if self.kinect_mode == "streams":
                cams.extend(["Kinect RGB", "Kinect Depth", "Kinect IR", "Kinect BodyIndex"])
            else:
                cams.append("Kinect (Microsoft Kinect v2)")
        cams.extend(self._webcam_names)
        return cams

    def get_all_frames(self) -> "OrderedDict[str, np.ndarray]":
        out: "OrderedDict[str, np.ndarray]" = OrderedDict()

        # ---- Kinect ----
        if self._ensure_kinect_open():
            fd = self._kinect.get_frame()
            if fd:
                if self.kinect_mode == "composite":
                    comp = self._kinect.visualize_frame(fd)
                    if comp is not None:
                        out["Kinect (Microsoft Kinect v2)"] = _ensure_bgr(comp)
                else:
                    if 'rgb' in fd:
                        out["Kinect RGB"] = _ensure_bgr(fd['rgb'])
                    if 'depth' in fd:
                        out["Kinect Depth"] = _ensure_bgr(self._kinect.depth_to_bgr(fd['depth']))
                    if 'ir' in fd:
                        out["Kinect IR"] = _ensure_bgr(self._kinect.ir_to_bgr(fd['ir']))
                    if 'body_index' in fd:
                        out["Kinect BodyIndex"] = _ensure_bgr(_bodyindex_to_bgr(fd['body_index']))

        # ---- Re-sondeo opcional de webcams ----
        if self._rescan_interval > 0.0 and (time.time() - self._last_scan_t) >= self._rescan_interval:
            self._discover_webcams(initial=False)

        # ---- Webcams ----
        visible = 0
        for name, tc in list(self._webcams.items()):
            if self._is_name_excluded(name):
                continue
            if (self._max_webcams_visible is not None) and (visible >= self._max_webcams_visible):
                break
            frame = tc.read()
            if frame is not None:
                out[name] = _ensure_bgr(frame)
                visible += 1
        return out

    def toggle_kinect_record(self):
        if self._kinect and hasattr(self._kinect, "recording"):
            if self._kinect.recording:
                self._kinect.stop_recording()
                return False
            else:
                self._kinect.start_recording()
                return True
        print("[INFO] Grabación de Kinect no disponible.")
        return None

    def release_all(self):
        for _, tc in list(self._webcams.items()):
            try:
                tc.stop()
            except Exception:
                pass
        self._webcams.clear()
        self._webcam_names.clear()

        if self._kinect and hasattr(self._kinect, "cleanup"):
            try:
                self._kinect.cleanup()
            except Exception:
                pass
        self._kinect = None

    # ---------- internos ----------
    def _is_index_excluded(self, idx: int) -> bool:
        return idx in self._exclude_indices

    def _is_name_excluded(self, name: str) -> bool:
        low = name.lower()
        return any(substr in low for substr in self._exclude_name_contains)

    def _discover_webcams(self, initial: bool):
        self._last_scan_t = time.time()
        # Evitar reprobar si ya hay cámaras y es un escaneo inicial
        fail_streak = 0
        for i in range(self._max_probe):
            if self._is_index_excluded(i):
                continue
            # Si ya hay una webcam con ese índice, salta (en rescan)
            if any(name.startswith(f"Webcam {i} ") for name in self._webcam_names):
                continue

            tc = self._open_threaded_cam(i)
            if tc is not None:
                name = f"Webcam {i} (DSHOW)"
                try:
                    if tc.backend == cv2.CAP_MSMF:
                        name = f"Webcam {i} (MSMF)"
                except Exception:
                    pass
                if self._is_name_excluded(name):
                    tc.stop()
                    continue
                # Registrar
                self._webcams[name] = tc
                self._webcam_names.append(name)
                fail_streak = 0
            else:
                fail_streak += 1
                if initial and fail_streak >= self._consecutive_fail_stop:
                    break

    def _open_threaded_cam(self, index: int) -> Optional[ThreadedCam]:
        if platform.system() == "Windows":
            backend_order = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
        else:
            backend_order = [0]
        for be in backend_order:
            try:
                return ThreadedCam(index, be)
            except Exception:
                continue
        return None

    def _ensure_kinect_open(self) -> bool:
        if not KINECT_LIB:
            return False
        if self._kinect is not None:
            return True
        if self._kinect_backend is None:
            return False
        self._kinect = self._kinect_backend
        try:
            return bool(self._kinect.initialize())
        except Exception as e:
            print(f"[KINECT] Error inicializando backend: {e}")
            return False


# ---------------- helpers de formato ----------------
def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    """Convierte GRAY/BGRA a BGR para unificación del mosaic."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[-1] == 4:
        return img[..., :3]
    return img

def _bodyindex_to_bgr(bidx: np.ndarray) -> np.ndarray:
    if bidx is None:
        return None
    if bidx.dtype != np.uint8:
        bidx = bidx.astype(np.uint8, copy=False)
    palette = np.zeros((256, 3), dtype=np.uint8)
    palette[0] = (255, 0, 0); palette[1] = (0, 255, 0); palette[2] = (0, 0, 255)
    palette[3] = (255, 255, 0); palette[4] = (255, 0, 255); palette[5] = (0, 255, 255)
    palette[255] = (0, 0, 0)
    return palette[bidx]
