#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import platform
import threading
import time
from collections import OrderedDict

import cv2
import numpy as np

try:
    from pykinect2 import PyKinectV2  # noqa: F401
    KINECT_LIB = True
except Exception:
    KINECT_LIB = False


# ---- Webcam en hilo (no bloquea el loop) ----
class ThreadedCam:
    def __init__(self, index, backend, width=1280, height=720, fps=30):
        self.index = index
        self.backend = backend
        self.cap = cv2.VideoCapture(index, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir webcam {index} ({backend}).")

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

    def read(self):
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


class CameraManager:
    """
    - kinect_mode: "composite" (render del backend) | "streams" (RGB/Depth/IR/BodyIndex separados)
    - Excluye webcams por índice o por substring del nombre.
    """
    def __init__(self,
                 kinect_backend=None,
                 kinect_mode="streams",
                 max_probe=6,
                 consecutive_fail_stop=3,
                 exclude_indices=None,
                 exclude_name_contains=None):
        self._kinect_backend = kinect_backend
        self._kinect = None

        self.kinect_mode = kinect_mode  # 'composite' | 'streams'

        self._exclude_indices = set(exclude_indices or [])
        self._exclude_name_contains = [s.lower() for s in (exclude_name_contains or [])]

        self._webcams = OrderedDict()   # name -> ThreadedCam
        self._webcam_names = []

        self._max_probe = max_probe
        self._consecutive_fail_stop = consecutive_fail_stop

        self._discover_webcams()

    # ---------- API ----------
    def list_cameras(self):
        cams = []
        if KINECT_LIB:
            cams.append("Kinect (Microsoft Kinect v2)")
        cams.extend(self._webcam_names)
        return cams

    def get_all_frames(self):
        """
        OrderedDict nombre->frame BGR.
        - En kinect_mode='streams': añade Kinect RGB/Depth/IR/BodyIndex por separado.
        - En kinect_mode='composite': un único "Kinect (Microsoft Kinect v2)" ya compuesto.
        """
        out = OrderedDict()

        if self._ensure_kinect_open():
            if self.kinect_mode == "composite":
                fd = self._kinect.get_frame()
                if fd:
                    out["Kinect (Microsoft Kinect v2)"] = self._kinect.visualize_frame(fd)
            else:
                fd = self._kinect.get_frame()
                if fd:
                    if 'rgb' in fd:  out["Kinect RGB"] = fd['rgb']
                    if 'depth' in fd: out["Kinect Depth"] = self._kinect.depth_to_bgr(fd['depth'])
                    if 'ir' in fd:    out["Kinect IR"] = self._kinect.ir_to_bgr(fd['ir'])
                    if 'body_index' in fd:
                        bidx = fd['body_index']
                        if bidx.dtype != np.uint8:
                            bidx = bidx.astype(np.uint8, copy=False)
                        palette = np.zeros((256, 3), dtype=np.uint8)
                        palette[0] = (255, 0, 0); palette[1] = (0, 255, 0); palette[2] = (0, 0, 255)
                        palette[3] = (255, 255, 0); palette[4] = (255, 0, 255); palette[5] = (0, 255, 255)
                        palette[255] = (0, 0, 0)
                        out["Kinect BodyIndex"] = palette[bidx]

        # Webcams
        for name, tc in list(self._webcams.items()):
            if self._is_name_excluded(name):
                continue
            frame = tc.read()
            if frame is not None:
                out[name] = frame

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
        for name, tc in list(self._webcams.items()):
            try:
                tc.stop()
            except Exception:
                pass
        self._webcams.clear()

        if self._kinect and hasattr(self._kinect, "cleanup"):
            try:
                self._kinect.cleanup()
            except Exception:
                pass
        self._kinect = None

    # ---------- Internos ----------
    def _is_index_excluded(self, idx: int) -> bool:
        return idx in self._exclude_indices

    def _is_name_excluded(self, name: str) -> bool:
        return any(substr in name.lower() for substr in self._exclude_name_contains)

    def _discover_webcams(self):
        fail_streak = 0
        for i in range(self._max_probe):
            if self._is_index_excluded(i):
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
                self._webcams[name] = tc
                self._webcam_names.append(name)
                fail_streak = 0
            else:
                fail_streak += 1
                if fail_streak >= self._consecutive_fail_stop:
                    break

    def _open_threaded_cam(self, index):
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

    def _ensure_kinect_open(self):
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
