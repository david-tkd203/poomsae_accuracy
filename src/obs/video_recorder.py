# src/obs/video_recorder.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import time, re
import cv2
import numpy as np

def _safe_filename(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s*\([^)]*\)", "", name)          # quita " (DSHOW)" etc.
    name = re.sub(r"[^\w]+", "_", name)               # separadores -> _
    bad = '<>:"/\\|?*'
    for ch in bad: name = name.replace(ch, "_")
    return name.strip("_")

def _to_bgr_u8(img: np.ndarray) -> np.ndarray:
    if img is None: return None
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)

def _even_wh(w: int, h: int) -> tuple[int, int]:
    return (w - (w % 2), h - (h % 2))

def _letterbox(img: np.ndarray, target_wh: tuple[int, int]) -> np.ndarray:
    tw, th = target_wh
    ih, iw = img.shape[:2]
    if iw == tw and ih == th: return img
    scale = min(tw / float(iw), th / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((th, tw, 3), np.uint8)
    x0, y0 = (tw - nw) // 2, (th - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

class MultiRecorder:
    """
    Graba 1 archivo por FUENTE + (opcional) 'scene_mosaic'.
    - FPS por-fuente (evita aceleraciones); si no hay dato, usa fps_default.
    - Fallback de códecs: preferido → mp4v → avc1 → XVID → MJPG.
    - Fija tamaño por writer y hace letterbox si cambia en tiempo real.
    - session_log.txt con tamaños, códec y frames escritos.
    """
    EXT_FOR = {"mp4v": ".mp4", "avc1": ".mp4", "XVID": ".avi", "MJPG": ".avi"}

    def __init__(self, out_root="recordings", fps_default=30, codec="mp4v",
                 scale=1.0, record_mosaic=False, fps_map: dict[str, float] | None = None):
        self.out_root = Path(out_root)
        self.fps_default = int(max(1, fps_default))
        self.codec_pref = codec
        self.scale = float(scale)
        self.record_mosaic = bool(record_mosaic)
        self.fps_map = {(_safe_filename(k)): float(v) for k, v in (fps_map or {}).items()}

        self.session_dir: Path | None = None
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._writer_codecs: dict[str, str] = {}
        self._writer_sizes: dict[str, tuple[int, int]] = {}
        self._writer_fps: dict[str, float] = {}
        self._frame_counts: dict[str, int] = {}
        self._started = False
        self._log_lines: list[str] = []

    # ---- ciclo ----
    def start(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.out_root / f"session_{ts}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._writers.clear(); self._writer_codecs.clear()
        self._writer_sizes.clear(); self._frame_counts.clear(); self._writer_fps.clear()
        self._log_lines = [f"Session start: {ts}", f"Root: {self.session_dir}"]
        self._started = True
        return self.session_dir

    def stop(self):
        for wr in self._writers.values():
            try: wr.release()
            except Exception: pass
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_lines.append("---- Frame counts ----")
        for k, n in sorted(self._frame_counts.items()):
            self._log_lines.append(f"{k}: {n} frames @ {self._writer_fps.get(k, self.fps_default)} fps")
        self._log_lines.append(f"Session end: {ts}")
        (self.session_dir / "session_log.txt").write_text("\n".join(self._log_lines), encoding="utf-8")
        self._writers.clear(); self._started = False
        return self.session_dir

    # ---- internos ----
    def _fps_for(self, key_norm: str) -> float:
        v = self.fps_map.get(key_norm, self.fps_default)
        try:
            v = float(v)
            if v < 1: v = self.fps_default
            if v > 120: v = 120.0
        except Exception:
            v = self.fps_default
        return v

    def _try_open(self, path: Path, size_wh, codec: str, fps: float):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        wr = cv2.VideoWriter(str(path), fourcc, float(fps), size_wh)
        return wr if wr.isOpened() else None

    def _open_with_fallbacks(self, key_norm: str, size_wh0: tuple[int,int], fps: float):
        w0, h0 = size_wh0
        if self.scale != 1.0:
            w0, h0 = int(w0 * self.scale), int(h0 * self.scale)
        w0, h0 = _even_wh(w0, h0)
        tried = []
        for c in [self.codec_pref, "mp4v", "avc1", "XVID", "MJPG"]:
            ext = self.EXT_FOR.get(c, ".mp4")
            path = self.session_dir / f"{key_norm}{ext}"
            wr = self._try_open(path, (w0, h0), c, fps)
            tried.append(f"{c}->{path.name}")
            if wr is not None:
                self._writers[key_norm] = wr
                self._writer_codecs[key_norm] = c
                self._writer_sizes[key_norm] = (w0, h0)
                self._writer_fps[key_norm] = fps
                self._frame_counts[key_norm] = 0
                self._log_lines.append(f"[OK] {key_norm}: {w0}x{h0}@{fps:.2f} {c} -> {path.name}")
                return wr
        self._log_lines.append(f"[ERR] {key_norm}: cannot open writer. Tried: {', '.join(tried)}")
        return None

    def _get_writer(self, key: str, frame_shape) -> tuple[cv2.VideoWriter | None, str]:
        key_norm = _safe_filename(key)
        if key_norm in self._writers:
            return self._writers[key_norm], key_norm
        h, w = frame_shape[:2]
        fps = self._fps_for(key_norm)
        wr = self._open_with_fallbacks(key_norm, (w, h), fps)
        return wr, key_norm

    # ---- pública ----
    def write_frames(self, frames: dict[str, np.ndarray], mosaic: np.ndarray | None = None):
        if not self._started or self.session_dir is None:
            return

        for key, img in frames.items():
            if img is None: continue
            frame = _to_bgr_u8(img)
            wr, key_norm = self._get_writer(key, frame.shape)
            if wr is None: continue
            tw, th = self._writer_sizes[key_norm]
            if (frame.shape[1], frame.shape[0]) != (tw, th):
                frame = _letterbox(frame, (tw, th))
            wr.write(frame)
            self._frame_counts[key_norm] += 1

        if self.record_mosaic and mosaic is not None:
            frame = _to_bgr_u8(mosaic)
            wr, key_norm = self._get_writer("scene_mosaic", frame.shape)
            if wr is not None:
                tw, th = self._writer_sizes[key_norm]
                if (frame.shape[1], frame.shape[0]) != (tw, th):
                    frame = _letterbox(frame, (tw, th))
                wr.write(frame)
                self._frame_counts[key_norm] += 1
