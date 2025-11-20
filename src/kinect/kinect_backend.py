#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KinectPoomsaeCapture (Kinect v2, PyKinect2)
- Activa: Color, Depth, Infrared, Body, BodyIndex.
- Grabación continua del RGB (sin acumular en memoria).
- Visualización con esqueletos (color space), y conversión depth/IR estable.
- 3D: Nube de puntos desde Depth (CoordinateMapper nativo o intrínsecos aprox.),
      color de nube desde RGB (Depth->ColorSpace) o por profundidad,
      y grabación periódica a .ply.

Mejoras:
- Cache de BodyIndex para usarlo en 3D.
- get_point_cloud(..., mask_to_bodies=False, color_by_bidx=False)
  * mask_to_bodies: filtra puntos donde BodyIndex==255
  * color_by_bidx: colorea por persona (0..5) en vez de RGB/z
"""

from pathlib import Path
import math
import time
import json
import ctypes

import cv2
import numpy as np

# --- PyKinect2 + utilidades 3D ---
try:
    from pykinect2 import PyKinectRuntime, PyKinectV2
    from src.kinect.pointcloud_utils import (
        depth_to_points_approx,
        write_ply_ascii,
        KINECTV2_DEPTH_INTRINSICS
    )
    KINECT_AVAILABLE = True
except Exception:
    KINECT_AVAILABLE = False


class KinectPoomsaeCapture:
    def __init__(self,
                 enable_color=True,
                 enable_depth=True,
                 enable_ir=True,
                 enable_audio=True,
                 enable_body=True,
                 enable_body_index=True):
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.enable_ir = enable_ir
        self.enable_audio = enable_audio
        self.enable_body = enable_body
        self.enable_body_index = enable_body_index

        # --- 3D / Point Cloud ---
        self._pc_record_dir = None
        self._pc_every_n = 10
        self._pc_decimate = 2
        self._pc_colorize = True
        self._pc_use_mapper = True  # intenta usar CoordinateMapper si está disponible
        self._frame_3d_counter = 0

        self.kinect = None
        self.sensors = {}
        self.frame_count = 0

        # tamaños nativos
        self.color_size = (1920, 1080)   # (w, h)
        self.depth_ir_size = (512, 424)  # (w, h)

        # caches para 3D/streams
        self._last_depth_cache = None
        self._last_color_cache = None
        self._last_bidx_cache  = None  # ⬅️ NUEVO: BodyIndex

        # grabación RGB continua
        self.recording = False
        self._writer = None
        self.record_dir = None
        self.record_meta = []
        self.record_fps = 30
        self.record_scale = 0.6667  # 1920x1080 -> ~1280x720
        self.record_codec = "mp4v"  # 'mp4v' funciona en Windows sin dependencias

    # ---------------- Inicialización ----------------
    def initialize(self) -> bool:
        if not KINECT_AVAILABLE:
            print("[KINECT] PyKinect2 no disponible o Kinect no conectada.")
            return False
        try:
            sources = 0
            if self.enable_color:
                sources |= PyKinectV2.FrameSourceTypes_Color
            if self.enable_depth:
                sources |= PyKinectV2.FrameSourceTypes_Depth
            if self.enable_ir:
                sources |= PyKinectV2.FrameSourceTypes_Infrared
            if self.enable_audio:
                sources |= PyKinectV2.FrameSourceTypes_Audio
            if self.enable_body:
                sources |= PyKinectV2.FrameSourceTypes_Body
            if self.enable_body_index:
                sources |= PyKinectV2.FrameSourceTypes_BodyIndex

            self.kinect = PyKinectRuntime.PyKinectRuntime(sources)
            self.sensors = {
                'color': bool(self.enable_color and getattr(self.kinect, 'color_frame_desc', None)),
                'depth': bool(self.enable_depth and getattr(self.kinect, 'depth_frame_desc', None)),
                'ir':    bool(self.enable_ir and getattr(self.kinect, 'infrared_frame_desc', None)),
                'audio': bool(self.enable_audio),
                'body':  bool(self.enable_body),
                'body_index': bool(self.enable_body_index),
            }
            activos = ", ".join([k for k, v in self.sensors.items() if v])
            print(f"Sensores activos: {activos if activos else 'ninguno'}")
            return True
        except Exception as e:
            print(f"[KINECT] Error inicializando Kinect: {e}")
            return False

    # ---------------- Grabación RGB continua ----------------
    def start_recording(self, output_dir="kinect_recordings",
                        codec=None, fps=None, scale=None):
        """Inicia writer de video y limpia metadatos (grabación continua)."""
        if codec: self.record_codec = codec
        if fps:   self.record_fps = fps
        if scale is not None: self.record_scale = float(scale)

        self.record_dir = Path(output_dir)
        self.record_dir.mkdir(exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._rgb_path = self.record_dir / f"kinect_{ts}_rgb.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.record_codec)

        out_w = int(self.color_size[0] * self.record_scale)
        out_h = int(self.color_size[1] * self.record_scale)
        self._writer = cv2.VideoWriter(str(self._rgb_path), fourcc, self.record_fps, (out_w, out_h))

        self.record_meta = []
        self.recording = True
        print(f"🔴 Grabación ON → {self._rgb_path.name} ({out_w}x{out_h} @ {self.record_fps}fps)")

    def stop_recording(self):
        """Cierra writer y guarda JSON con metadatos (ángulos, etc.)."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        if not self.record_dir:
            print("[KINECT] No hay sesión de grabación.")
            self.recording = False
            return None

        meta_p = self.record_dir / f"{self._rgb_path.stem}_bodies_metrics.json"
        with open(meta_p, "w", encoding="utf-8") as f:
            json.dump(self.record_meta, f, indent=2, ensure_ascii=False, default=str)

        print(f"⏹️ Grabación OFF\n✅ RGB: {self._rgb_path}\n📊 Métricas: {meta_p}")
        self.recording = False
        return self.record_dir

    # ---------------- Conversores auxiliares ----------------
    @staticmethod
    def depth_to_bgr(depth_u16, clip=(500, 4500)):
        """Normaliza depth (mm) a 8-bit y aplica colormap estable."""
        d = np.clip(depth_u16.astype(np.float32), clip[0], clip[1])
        d = ((d - clip[0]) / (clip[1] - clip[0]) * 255.0).astype(np.uint8)
        return cv2.applyColorMap(d, cv2.COLORMAP_JET)

    @staticmethod
    def ir_to_bgr(ir_u16):
        """Normaliza IR con percentil robusto para evitar saturación."""
        ir = ir_u16.astype(np.float32)
        lo, hi = np.percentile(ir, [1, 99])
        if hi <= lo:
            lo, hi = ir.min(), ir.max() if ir.max() > ir.min() else (0, 1)
        ir = np.clip((ir - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(ir, cv2.COLORMAP_BONE)

    # ---------------- 3D helpers (CoordinateMapper) ----------------
    def _map_depth_to_camera_space(self, depth_u16: np.ndarray):
        """Usa CoordinateMapper para convertir depth -> CameraSpace (XYZ en metros)."""
        try:
            H, W = depth_u16.shape
            N = W * H
            DepthArray = ctypes.c_ushort * N
            depth_buf = DepthArray(*depth_u16.flatten().tolist())

            CSP_t = PyKinectV2._CameraSpacePoint * N
            csp = CSP_t()

            # La firma en PyKinect2 acepta (N, depth_buf, W, H, csp)
            self.kinect._mapper.MapDepthFrameToCameraSpace(
                ctypes.c_uint(N), depth_buf,
                ctypes.c_uint(W), ctypes.c_uint(H),
                csp
            )

            pts = np.empty((N, 3), dtype=np.float32)
            for i in range(N):
                pts[i, 0] = csp[i].x
                pts[i, 1] = csp[i].y
                pts[i, 2] = csp[i].z
            return pts
        except Exception as e:
            print(f"[3D] Mapper MapDepthFrameToCameraSpace falló: {e}")
            return None

    def _map_depth_to_color_space(self, depth_u16: np.ndarray):
        """Depth -> ColorSpace (x,y en imagen RGB). Devuelve lista de _ColorSpacePoint o None."""
        try:
            H, W = depth_u16.shape
            N = W * H
            DepthArray = ctypes.c_ushort * N
            depth_buf = DepthArray(*depth_u16.flatten().tolist())

            COL_t = PyKinectV2._ColorSpacePoint * N
            colsp = COL_t()

            self.kinect._mapper.MapDepthFrameToColorSpace(
                ctypes.c_uint(N), depth_buf,
                ctypes.c_uint(W), ctypes.c_uint(H),
                colsp
            )
            return colsp
        except Exception as e:
            print(f"[3D] Mapper MapDepthFrameToColorSpace falló: {e}")
            return None

    # ---------------- API de Nube de Puntos ----------------
    def get_point_cloud(self,
                        decimate: int = 2,
                        colorize: bool = True,
                        prefer_mapper: bool = True,
                        mask_to_bodies: bool = False,
                        color_by_bidx: bool = False):
        """
        Devuelve dict con:
          'points': (N,3) XYZ (m)
          'colors': (N,3) RGB 0..255 o None

        - prefer_mapper: usa CoordinateMapper nativo si es posible
        - decimate: muestreo en la grilla depth (1=sin decimar)
        - mask_to_bodies: si hay BodyIndex, filtra puntos con bidx!=255
        - color_by_bidx: si hay BodyIndex, colorea por persona (0..5)
        - colorize: si True y no color_by_bidx -> intenta RGB; si falla -> z-colormap
        """
        if self.kinect is None or not self.sensors.get('depth', False):
            return None

        step = max(1, int(decimate))

        # Depth (con caché)
        if self.kinect.has_new_depth_frame():
            depth_frame = self.kinect.get_last_depth_frame().reshape(
                (self.depth_ir_size[1], self.depth_ir_size[0])
            ).astype(np.uint16)
            self._last_depth_cache = depth_frame.copy()
        elif self._last_depth_cache is not None:
            depth_frame = self._last_depth_cache
        else:
            return None

        H, W = depth_frame.shape

        # BodyIndex (si se usará)
        bidx_frame = None
        if (mask_to_bodies or color_by_bidx) and self.sensors.get('body_index', False):
            if hasattr(self.kinect, 'has_new_body_index_frame') and self.kinect.has_new_body_index_frame():
                b = self.kinect.get_last_body_index_frame()
                bidx_frame = b.reshape((H, W)).astype(np.uint8, copy=False)
                self._last_bidx_cache = bidx_frame.copy()
            elif self._last_bidx_cache is not None:
                bidx_frame = self._last_bidx_cache

        # Color frame (para colorear con RGB si procede)
        color_frame = None
        if colorize and self.sensors.get('color', False) and not color_by_bidx:
            if self.kinect.has_new_color_frame():
                cf = self.kinect.get_last_color_frame().reshape(
                    (self.color_size[1], self.color_size[0], 4)
                ).astype(np.uint8)[..., :3]
                self._last_color_cache = cf.copy()
                color_frame = cf
            elif self._last_color_cache is not None:
                color_frame = self._last_color_cache

        # 1) XYZ (preferente: mapper nativo)
        points = None
        if prefer_mapper:
            pts_all = self._map_depth_to_camera_space(depth_frame)
            if pts_all is not None:
                pts_dec = pts_all.reshape(H, W, 3)[::step, ::step, :].reshape(-1, 3)
                points = pts_dec

        # 2) Fallback aproximado
        if points is None:
            intr = KINECTV2_DEPTH_INTRINSICS
            points = depth_to_points_approx(
                depth_frame, intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                depth_scale=intr["depth_scale"], decimate=step
            )
            if points is None or points.size == 0:
                return None

        # --- máscara de válidos (finito y z>0) ---
        valid = np.isfinite(points).all(axis=1)
        if valid.any():
            z = points[:, 2]
            valid &= (z > 0)
        if not valid.any():
            return None

        # --- máscara por BodyIndex (opcional) ---
        mask_bodies = None
        bidx_dec = None
        if bidx_frame is not None:
            bidx_dec = bidx_frame[::step, ::step].reshape(-1)
            mask_bodies = (bidx_dec != 255)
            if mask_to_bodies:
                valid &= mask_bodies
                if not valid.any():
                    return {"points": np.zeros((0, 3), np.float32), "colors": None}

        # --- Colores ---
        colors = None
        if color_by_bidx and bidx_dec is not None:
            # Paleta 6 personas + fondo
            palette = np.zeros((256, 3), dtype=np.uint8)
            palette[0] = (255, 0, 0)      # R
            palette[1] = (0, 255, 0)      # G
            palette[2] = (0, 0, 255)      # B
            palette[3] = (255, 255, 0)    # Y
            palette[4] = (255, 0, 255)    # M
            palette[5] = (0, 255, 255)    # C
            palette[255] = (40, 40, 40)   # Fondo oscuro
            colors = palette[bidx_dec]
            colors = colors.astype(np.uint8, copy=False)
            colors = colors[valid]
        elif colorize:
            if color_frame is not None and prefer_mapper:
                # RGB real usando Depth->ColorSpace para los índices decimados
                colsp_all = self._map_depth_to_color_space(depth_frame)
                if colsp_all is not None:
                    vs, us = np.mgrid[0:H:step, 0:W:step]
                    idxs = (vs * W + us).ravel()
                    col_list = np.zeros((idxs.size, 3), dtype=np.uint8)
                    for i, idx in enumerate(idxs):
                        cpt = colsp_all[idx]
                        cx, cy = int(cpt.x), int(cpt.y)
                        if 0 <= cx < self.color_size[0] and 0 <= cy < self.color_size[1]:
                            col_list[i] = color_frame[cy, cx]
                        else:
                            col_list[i] = (0, 0, 0)
                    colors = col_list[valid]
                else:
                    # fallback z-colormap
                    z = points[:, 2]
                    z = z[valid]
                    z_norm = (z - z.min()) / max(1e-6, (z.max() - z.min()))
                    colors = np.stack([
                        255 * z_norm,
                        255 * (1.0 - np.abs(z_norm - 0.5) * 2.0),
                        255 * (1.0 - z_norm)
                    ], axis=-1).astype(np.uint8)
            else:
                # z-colormap
                z = points[:, 2]
                z = z[valid]
                z_norm = (z - z.min()) / max(1e-6, (z.max() - z.min()))
                colors = np.stack([
                    255 * z_norm,
                    255 * (1.0 - np.abs(z_norm - 0.5) * 2.0),
                    255 * (1.0 - z_norm)
                ], axis=-1).astype(np.uint8)

        # Aplicar máscara final de válidos (y, si corresponde, de cuerpos)
        points = points[valid]
        # si no generamos colors arriba, puede quedar None
        if colors is not None and colors.shape[0] != points.shape[0]:
            # Asegurar alineación (rara vez necesario si usamos 'valid' correctamente)
            colors = colors[:points.shape[0]]

        return {"points": points.astype(np.float32, copy=False), "colors": colors}

    def save_point_cloud_ply(self, out_path: str, decimate=2, colorize=True, prefer_mapper=True):
        pc = self.get_point_cloud(decimate=decimate, colorize=colorize, prefer_mapper=prefer_mapper)
        if pc is None or pc["points"] is None or pc["points"].size == 0:
            return False
        write_ply_ascii(out_path, pc["points"], pc["colors"])
        return True

    # ---------------- Grabación PLY (3D) ----------------
    def start_pointcloud_recording(self, out_dir: str, every_n: int = 10, decimate: int = 2,
                                   colorize: bool = True, prefer_mapper: bool = True):
        self._pc_record_dir = Path(out_dir)
        self._pc_record_dir.mkdir(parents=True, exist_ok=True)
        self._pc_every_n = max(1, int(every_n))
        self._pc_decimate = int(max(1, decimate))
        self._pc_colorize = bool(colorize)
        self._pc_use_mapper = bool(prefer_mapper)
        self._frame_3d_counter = 0
        print(f"[3D] Grabación PLY: dir={self._pc_record_dir}, cada {self._pc_every_n} frames, decimate={self._pc_decimate}")

    def tick_pointcloud_recording(self):
        """Llama en cada loop/tick; si corresponde, guarda un frame .ply."""
        if self._pc_record_dir is None:
            return
        self._frame_3d_counter += 1
        if self._frame_3d_counter % self._pc_every_n != 0:
            return
        idx = self._frame_3d_counter
        out = self._pc_record_dir / f"pc_{idx:06d}.ply"
        ok = self.save_point_cloud_ply(
            str(out),
            decimate=self._pc_decimate,
            colorize=self._pc_colorize,
            prefer_mapper=self._pc_use_mapper
        )
        if ok:
            print(f"[3D] guardado {out}")

    def stop_pointcloud_recording(self):
        self._pc_record_dir = None
        print("[3D] Grabación PLY detenida.")

    # ---------------- Lectura de frame ----------------
    def get_frame(self):
        if not self.kinect:
            return None

        fd = {}
        self.frame_count += 1

        # Color
        if self.sensors['color'] and self.kinect.has_new_color_frame():
            cf = self.kinect.get_last_color_frame()
            rgb = cf.reshape((self.color_size[1], self.color_size[0], 4)).astype(np.uint8)[..., :3]
            fd['rgb'] = rgb
            self._last_color_cache = rgb.copy()

        # Depth
        if self.sensors['depth'] and self.kinect.has_new_depth_frame():
            df = self.kinect.get_last_depth_frame()
            depth_u16 = df.reshape((self.depth_ir_size[1], self.depth_ir_size[0])).astype(np.uint16, copy=False)
            fd['depth'] = depth_u16
            self._last_depth_cache = depth_u16.copy()

        # IR
        if self.sensors['ir'] and self.kinect.has_new_infrared_frame():
            irf = self.kinect.get_last_infrared_frame()
            fd['ir'] = irf.reshape((self.depth_ir_size[1], self.depth_ir_size[0])).astype(np.uint16, copy=False)

        # BodyIndex
        if self.sensors['body_index'] and hasattr(self.kinect, 'has_new_body_index_frame') \
           and self.kinect.has_new_body_index_frame():
            bidx = self.kinect.get_last_body_index_frame()
            bidx = bidx.reshape((self.depth_ir_size[1], self.depth_ir_size[0])).astype(np.uint8, copy=False)
            fd['body_index'] = bidx
            self._last_bidx_cache = bidx.copy()   # ⬅️ NUEVO: cache

        # Bodies + métricas
        fd['bodies'] = []
        fd['biomechanics'] = []
        if self.sensors['body'] and self.kinect.has_new_body_frame():
            bodies = self.kinect.get_last_body_frame()
            if bodies is not None:
                for i in range(self.kinect.max_body_count):
                    body = bodies.bodies[i]
                    if not body.is_tracked:
                        continue
                    joints = body.joints

                    landmarks = {}
                    for j in range(PyKinectV2.JointType_Count):
                        p = joints[j].Position
                        landmarks[j] = (p.x, p.y, p.z, 1.0)

                    color_pts = self.kinect.body_joints_to_color_space(joints)
                    px = {j: (int(color_pts[j].x), int(color_pts[j].y)) for j in range(PyKinectV2.JointType_Count)}

                    fd['bodies'].append({'landmarks': landmarks, 'color_pixels': px})

                    angles  = self._calculate_angles(landmarks)
                    metrics = self._biomechanics_metrics(landmarks, angles)
                    fd['biomechanics'].append({'angles': angles, 'metrics': metrics})

        # Timestamps
        ts = None
        if self.sensors['color'] and hasattr(self.kinect, '_last_color_frame_time'):
            ts = self.kinect._last_color_frame_time
        elif self.sensors['depth'] and hasattr(self.kinect, '_last_depth_frame_time'):
            ts = self.kinect._last_depth_frame_time
        elif self.sensors['body'] and hasattr(self.kinect, '_last_body_frame_time'):
            ts = self.kinect._last_body_frame_time
        fd['timestamp'] = ts if ts is not None else self.frame_count
        fd['frame_number'] = self.frame_count

        # ---- grabación continua del RGB ----
        if self.recording and 'rgb' in fd and self._writer is not None:
            frame = fd['rgb']
            if self.record_scale != 1.0:
                ow = int(self.color_size[0] * self.record_scale)
                oh = int(self.color_size[1] * self.record_scale)
                frame = cv2.resize(frame, (ow, oh))
            self._writer.write(frame)

            # guarda solo metadatos (cuerpos/ángulos)
            self.record_meta.append({
                'frame': int(fd['frame_number']),
                'timestamp': fd.get('timestamp'),
                'bodies': fd.get('bodies'),
                'biomechanics': fd.get('biomechanics'),
            })

        # ---- tick de grabación PLY (3D) si está activa ----
        try:
            self.tick_pointcloud_recording()
        except Exception:
            pass

        return fd if fd else None

    # ---------------- Ángulos y métricas ----------------
    def _angle3d(self, a, b, c):
        try:
            ax, ay, az = a[0], a[1], a[2]
            bx, by, bz = b[0], b[1], b[2]
            cx, cy, cz = c[0], c[1], c[2]
            ba = (ax - bx, ay - by, az - bz)
            bc = (cx - bx, cy - by, cz - bz)
            dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
            nba = math.sqrt(ba[0]**2 + ba[1]**2 + ba[2]**2)
            nbc = math.sqrt(bc[0]**2 + bc[1]**2 + bc[2]**2)
            if nba == 0 or nbc == 0:
                return None
            cosang = max(-1.0, min(1.0, dot/(nba*nbc)))
            return math.degrees(math.acos(cosang))
        except Exception:
            return None

    def _calculate_angles(self, landmarks):
        J = PyKinectV2.JointType
        angles = {}
        for side in ('Left', 'Right'):
            try:
                angles[f'{side.lower()}_elbow'] = self._angle3d(
                    landmarks[getattr(J, f'{side}Shoulder')],
                    landmarks[getattr(J, f'{side}Elbow')],
                    landmarks[getattr(J, f'{side}Wrist')])
            except Exception:
                pass
            try:
                angles[f'{side.lower()}_knee'] = self._angle3d(
                    landmarks[getattr(J, f'{side}Hip')],
                    landmarks[getattr(J, f'{side}Knee')],
                    landmarks[getattr(J, f'{side}Ankle')])
            except Exception:
                pass
            try:
                angles[f'{side.lower()}_shoulder'] = self._angle3d(
                    landmarks[getattr(J, f'{side}Hip')],
                    landmarks[getattr(J, f'{side}Shoulder')],
                    landmarks[getattr(J, f'{side}Elbow')])
            except Exception:
                pass
            try:
                angles[f'{side.lower()}_hip'] = self._angle3d(
                    landmarks[getattr(J, f'{side}Shoulder')],
                    landmarks[getattr(J, f'{side}Hip')],
                    landmarks[getattr(J, f'{side}Knee')])
            except Exception:
                pass
        return angles

    def _biomechanics_metrics(self, landmarks, angles):
        metrics = {}
        try:
            l_sh_y = landmarks[PyKinectV2.JointType_ShoulderLeft][1]
            r_sh_y = landmarks[PyKinectV2.JointType_ShoulderRight][1]
            l_hip_y = landmarks[PyKinectV2.JointType_HipLeft][1]
            r_hip_y = landmarks[PyKinectV2.JointType_HipRight][1]
            metrics['shoulder_balance'] = abs(l_sh_y - r_sh_y)
            metrics['hip_balance']      = abs(l_hip_y - r_hip_y)
        except Exception:
            metrics['shoulder_balance'] = None
            metrics['hip_balance']      = None

        try:
            base = landmarks[PyKinectV2.JointType_SpineBase]
            top  = landmarks[PyKinectV2.JointType_SpineShoulder]
            v = (top[0]-base[0], top[1]-base[1], top[2]-base[2])
            vlen = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
            if vlen > 0:
                cosang = max(-1.0, min(1.0, abs(v[1]/vlen)))
                metrics['torso_inclination_deg'] = round(math.degrees(math.acos(cosang)), 2)
            else:
                metrics['torso_inclination_deg'] = None
        except Exception:
            metrics['torso_inclination_deg'] = None

        for k in ('left_elbow', 'right_elbow', 'left_knee', 'right_knee'):
            metrics[f'extension_{k}'] = angles.get(k, None)
        for k in ('left_hip', 'right_hip', 'left_knee', 'right_knee'):
            ang = angles.get(k, None)
            metrics[f'grave_{k}'] = (ang is not None) and (ang < 120 or ang > 180)

        return metrics

    # ---------------- Visualización (composite rápido) ----------------
    def _skeleton_pairs(self):
        J = PyKinectV2.JointType
        return [
            (J.SpineBase, J.SpineMid), (J.SpineMid, J.SpineShoulder), (J.SpineShoulder, J.Neck), (J.Neck, J.Head),
            (J.SpineShoulder, J.ShoulderLeft), (J.ShoulderLeft, J.ElbowLeft), (J.ElbowLeft, J.WristLeft), (J.WristLeft, J.HandLeft),
            (J.SpineShoulder, J.ShoulderRight), (J.ShoulderRight, J.ElbowRight), (J.ElbowRight, J.WristRight), (J.WristRight, J.HandRight),
            (J.SpineBase, J.HipLeft), (J.HipLeft, J.KneeLeft), (J.KneeLeft, J.AnkleLeft), (J.AnkleLeft, J.FootLeft),
            (J.SpineBase, J.HipRight), (J.HipRight, J.KneeRight), (J.KneeRight, J.AnkleRight), (J.AnkleRight, J.FootRight)
        ]

    def visualize_frame(self, fd):
        """Composición rápida: RGB a la izquierda, panel derecho con Depth/IR/BodyIndex (los que existan)."""
        rgb  = fd.get('rgb')
        depth = fd.get('depth')
        ir    = fd.get('ir')
        bidx  = fd.get('body_index')

        W, H = self.color_size
        base = rgb.copy() if rgb is not None else np.zeros((H, W, 3), np.uint8)
        right_tiles = []

        if depth is not None:
            right_tiles.append(cv2.resize(self.depth_to_bgr(depth), (W, H)))
        if ir is not None:
            right_tiles.append(cv2.resize(self.ir_to_bgr(ir), (W, H)))
        if bidx is not None:
            if bidx.dtype != np.uint8:
                bidx = bidx.astype(np.uint8, copy=False)
            palette = np.zeros((256, 3), dtype=np.uint8)
            palette[0] = (255, 0, 0); palette[1] = (0, 255, 0); palette[2] = (0, 0, 255)
            palette[3] = (255, 255, 0); palette[4] = (255, 0, 255); palette[5] = (0, 255, 255)
            palette[255] = (0, 0, 0)
            color_bidx = palette[bidx]
            right_tiles.append(cv2.resize(color_bidx, (W, H), interpolation=cv2.INTER_NEAREST))

        if right_tiles:
            right = np.vstack(right_tiles)
            right = cv2.resize(right, (W, H))
            vis = np.hstack((base, right))
        else:
            vis = base

        # esqueletos
        if rgb is not None and 'bodies' in fd:
            for b in fd['bodies']:
                pxmap = b.get('color_pixels', {})
                # puntos
                for jid, (px, py) in pxmap.items():
                    if 0 <= px < W and 0 <= py < H:
                        cv2.circle(vis, (px, py), 3, (0, 255, 0), -1)
                # líneas
                for a, c in self._skeleton_pairs():
                    if a in pxmap and c in pxmap:
                        pa, pc = pxmap[a], pxmap[c]
                        if 0 <= pa[0] < W and 0 <= pa[1] < H and 0 <= pc[0] < W and 0 <= pc[1] < H:
                            cv2.line(vis, pa, pc, (0, 255, 0), 2)

        # Info
        y = 28
        for t in [f"Kinect v2 - Frame {fd.get('frame_number',0)}",
                  f"Sensores: {', '.join([k for k,v in self.sensors.items() if v])}"]:
            cv2.putText(vis, t, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            y += 28
        return vis

    # ---------------- Limpieza ----------------
    def cleanup(self):
        if self.kinect:
            self.kinect.close()
            self.kinect = None
            print("Kinect liberada.")
