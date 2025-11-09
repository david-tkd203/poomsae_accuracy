# src/segmentation/move_capture.py
from __future__ import annotations
import math, json, yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import cv2
import scipy.signal as signal

# ---------------- Configuración Global ----------------
class PoomsaeConfig:
    """Carga y gestiona las especificaciones del Poomsae"""
    
    def __init__(self, spec_path: Path, pose_spec_path: Path, config_path: Path):
        self.spec_path = spec_path
        self.pose_spec_path = pose_spec_path
        self.config_path = config_path
        self.poomsae_spec = None
        self.pose_spec = None
        self.app_config = None
        
        self.load_configs()
    
    def load_configs(self):
        """Carga todas las configuraciones"""
        try:
            # Cargar configuración principal
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.app_config = yaml.safe_load(f)
            
            # Cargar especificación del Poomsae
            with open(self.spec_path, 'r', encoding='utf-8') as f:
                self.poomsae_spec = json.load(f)
            
            # Cargar especificación de poses
            with open(self.pose_spec_path, 'r', encoding='utf-8') as f:
                self.pose_spec = json.load(f)
                
            print(f"[CONFIG] Cargadas especificaciones para {self.poomsae_spec['poomsae']}")
            
        except Exception as e:
            print(f"[CONFIG] Error cargando configuraciones: {e}")
            raise
    
    def get_move_spec(self, move_idx: int, sub: str = None) -> Optional[Dict]:
        """Obtiene la especificación para un movimiento específico"""
        if not self.poomsae_spec:
            return None
            
        for move in self.poomsae_spec['moves']:
            if move.get('idx') == move_idx:
                if sub is None or move.get('sub') == sub:
                    return move
        return None
    
    def get_stance_thresholds(self, stance_code: str) -> Dict:
        """Obtiene umbrales para una postura específica"""
        if not self.pose_spec or 'stances' not in self.pose_spec:
            return {}
        return self.pose_spec['stances'].get(stance_code, {})
    
    def get_kick_thresholds(self, kick_type: str) -> Dict:
        """Obtiene umbrales para un tipo de patada"""
        if not self.pose_spec or 'kicks' not in self.pose_spec:
            return {}
        return self.pose_spec['kicks'].get(kick_type, {})

# ---------------- Índices MediaPipe Pose ----------------
LMK = dict(
    NOSE=0,
    L_EYE_IN=1, L_EYE=2, L_EYE_OUT=3, R_EYE_IN=4, R_EYE=5, R_EYE_OUT=6,
    L_EAR=7, R_EAR=8,
    L_MOUTH=9, R_MOUTH=10,
    L_SH=11, R_SH=12, L_ELB=13, R_ELB=14, L_WRIST=15, R_WRIST=16,
    L_PINKY=17, R_PINKY=18, L_INDEX=19, R_INDEX=20, L_THUMB=21, R_THUMB=22,
    L_HIP=23, R_HIP=24, L_KNEE=25, R_KNEE=26, L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30, L_FOOT=31, R_FOOT=32
)

CONNECTIONS = [
    (LMK["L_SH"],LMK["R_SH"]), (LMK["L_SH"],LMK["L_HIP"]), (LMK["R_SH"],LMK["R_HIP"]), (LMK["L_HIP"],LMK["R_HIP"]),
    (LMK["L_SH"],LMK["L_ELB"]), (LMK["L_ELB"],LMK["L_WRIST"]),
    (LMK["R_SH"],LMK["R_ELB"]), (LMK["R_ELB"],LMK["R_WRIST"]),
    (LMK["L_HIP"],LMK["L_KNEE"]), (LMK["L_KNEE"],LMK["L_ANK"]),
    (LMK["R_HIP"],LMK["R_KNEE"]), (LMK["R_KNEE"],LMK["R_ANK"]),
    (LMK["L_ANK"],LMK["L_HEEL"]), (LMK["R_ANK"],LMK["R_HEEL"]),
    (LMK["L_HEEL"],LMK["L_FOOT"]), (LMK["R_HEEL"],LMK["R_FOOT"])
]

END_EFFECTORS = ("L_WRIST","R_WRIST","L_ANK","R_ANK")

# ---------------- Segmentador con Especificaciones ----------------
class SpecAwareSegmenter:
    """
    Segmentador que utiliza las especificaciones del Poomsae 8yang
    """
    
    def __init__(self, config: PoomsaeConfig):
        self.config = config
        self.poomsae_spec = config.poomsae_spec
        self.pose_spec = config.pose_spec
        
        # Parámetros de segmentación basados en la especificación
        seg_config = config.app_config.get('segmentation', {})
        self.min_segment_frames = seg_config.get('min_segment_frames', 8)
        self.max_pause_frames = seg_config.get('max_pause_frames', 6)
        self.activity_thresh = 0.1
        self.peak_thresh = 0.2
        self.min_peak_dist = 6
        self.smooth_win = 3

    def _compute_poomsae_aware_energy(self, angles_dict, landmarks_dict, fps):
        """Calcula energía considerando los movimientos esperados del 8yang"""
        energies = []
        
        # 1. Energía de efectores con pesos según especificación
        effector_energy = self._weighted_effector_energy(landmarks_dict, fps)
        energies.append(effector_energy)
        
        # 2. Energía angular específica para Taekwondo
        angular_energy = self._taekwondo_angular_energy(angles_dict, fps)
        energies.append(angular_energy)
        
        # 3. Energía de rotación (crítica para giros en 8yang)
        rotation_energy = self._rotation_energy(landmarks_dict, fps)
        energies.append(rotation_energy)
        
        # Combinar con pesos optimizados para 8yang
        weights = [0.5, 0.3, 0.2]
        total_energy = np.zeros_like(effector_energy)
        
        for energy, weight in zip(energies, weights):
            if len(energy) == len(total_energy) and np.max(energy) > 0:
                energy_norm = energy / (np.max(energy) + 1e-6)
                total_energy += weight * energy_norm
        
        return total_energy

    def _weighted_effector_energy(self, landmarks_dict, fps):
        """Energía de efectores con pesos según movimientos del 8yang"""
        effectors = ['L_WRIST', 'R_WRIST', 'L_ANK', 'R_ANK']
        energies = []
        
        for effector in effectors:
            if effector in landmarks_dict and len(landmarks_dict[effector]) > 1:
                pos = landmarks_dict[effector]
                movement = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[0:1]), axis=1)
                energy = movement * fps
                
                # Aplicar peso según tipo de efector
                if 'WRIST' in effector:
                    energy *= 1.2  # Más peso a movimientos de brazos
                elif 'ANK' in effector:
                    energy *= 1.0  # Peso estándar para piernas
                    
                energies.append(energy)
        
        if not energies:
            return np.zeros(1000)
            
        return np.max(energies, axis=0)

    def _taekwondo_angular_energy(self, angles_dict, fps):
        """Energía angular específica para movimientos de Taekwondo"""
        if not angles_dict:
            return np.zeros(1000)
            
        grads = []
        # ✅ MEJORADO: Incluir todos los ángulos disponibles (codos, caderas, rodillas)
        priority_angles = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee', 'left_hip', 'right_hip']
        
        for angle_name in priority_angles:
            if angle_name in angles_dict:
                angle_series = angles_dict[angle_name]
                if len(angle_series) > 1:
                    grad = np.abs(np.gradient(angle_series)) * fps
                    
                    # Pesos según importancia para Taekwondo
                    if 'elbow' in angle_name:
                        grad *= 1.3  # Muy importante para brazos
                    elif 'knee' in angle_name:
                        grad *= 1.5  # Crítico para posturas
                    elif 'hip' in angle_name:
                        grad *= 1.0  # Estándar para rotaciones
                    
                    grads.append(grad)
        
        if not grads:
            return np.zeros(1000)
            
        return np.mean(grads, axis=0)

    def _rotation_energy(self, landmarks_dict, fps):
        """Energía de rotación para detectar giros"""
        if 'L_SH' not in landmarks_dict or 'R_SH' not in landmarks_dict:
            return np.zeros(1000)
            
        l_shoulder = landmarks_dict['L_SH']
        r_shoulder = landmarks_dict['R_SH']
        
        if len(l_shoulder) < 2:
            return np.zeros(1000)
        
        # Vector de hombros para orientación
        shoulder_vec = r_shoulder - l_shoulder
        orientations = np.arctan2(shoulder_vec[:, 1], shoulder_vec[:, 0])
        
        # Cambio de orientación (muy importante en 8yang)
        orientation_change = np.abs(np.gradient(orientations)) * fps * 180/np.pi
        return orientation_change

    def _smooth_signal(self, signal):
        """Suavizar señal"""
        if len(signal) < self.smooth_win:
            return signal
        window = np.ones(self.smooth_win) / self.smooth_win
        return np.convolve(signal, window, mode='same')

    def _find_poomsae_peaks(self, energy_signal):
        """Encontrar picos considerando la estructura del 8yang"""
        if len(energy_signal) < 3:
            return []
        
        smoothed = self._smooth_signal(energy_signal)
        
        # Umbral adaptativo basado en percentiles
        q75 = np.percentile(smoothed, 75)
        q25 = np.percentile(smoothed, 25)
        dynamic_threshold = q25 + 0.5 * (q75 - q25)
        
        peaks, properties = signal.find_peaks(
            smoothed, 
            height=max(self.peak_thresh, dynamic_threshold),
            distance=self.min_peak_dist,
            prominence=0.08  # Menor prominencia para capturar más movimientos
        )
        
        return peaks.tolist()

    def find_segments(self, angles_dict, landmarks_dict, fps):
        """Encuentra segmentos usando conocimiento del 8yang"""
        if not angles_dict and not landmarks_dict:
            return []
        
        # Calcular energía considerando el Poomsae
        energy = self._compute_poomsae_aware_energy(angles_dict, landmarks_dict, fps)
        
        if len(energy) == 0:
            return []
        
        # Encontrar picos
        peaks = self._find_poomsae_peaks(energy)
        
        # Crear segmentos
        segments = []
        used_frames = set()
        
        for peak in peaks:
            start, end = self._expand_segment_around_peak(energy, peak, len(energy))
            
            # Verificar superposición
            overlap = False
            for seg_start, seg_end in segments:
                if not (end < seg_start or start > seg_end):
                    overlap = True
                    break
            
            if not overlap and (end - start) >= self.min_segment_frames:
                segments.append((start, end))
                used_frames.update(range(start, end + 1))
        
        # Ordenar y fusionar
        segments.sort(key=lambda x: x[0])
        merged_segments = self._merge_segments(segments)
        
        print(f"[SEGMENTER] Detectados {len(merged_segments)} segmentos para 8yang")
        return merged_segments

    def _expand_segment_around_peak(self, energy_signal, peak_idx, total_frames):
        """Expandir segmento alrededor de un pico"""
        start = peak_idx
        while start > 0 and energy_signal[start] > self.activity_thresh:
            start -= 1
        start = max(0, start)
        
        end = peak_idx
        while end < total_frames - 1 and energy_signal[end] > self.activity_thresh:
            end += 1
        end = min(total_frames - 1, end)
        
        # Asegurar longitud mínima según especificación
        if (end - start) < self.min_segment_frames:
            expansion = self.min_segment_frames - (end - start)
            start = max(0, start - expansion // 2)
            end = min(total_frames - 1, end + expansion - (start - max(0, start - expansion // 2)))
        
        return start, end

    def _merge_segments(self, segments):
        """Fusionar segmentos cercanos"""
        if not segments:
            return []
            
        merged = [segments[0]]
        for (x, y) in segments[1:]:
            px, py = merged[-1]
            if (x - py - 1) <= self.max_pause_frames:
                merged[-1] = (px, y)
            else:
                merged.append((x, y))
        return merged

# ---------------- Utilidades Numéricas ----------------
def movavg(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x
    win = min(win, len(x))
    c = np.cumsum(np.insert(x, 0, 0.0, axis=0), axis=0)
    y = (c[win:] - c[:-win]) / float(win)
    pad_l = win // 2
    pad_r = len(x) - len(y) - pad_l
    return np.pad(y, ((pad_l, pad_r), (0, 0)), mode="edge")

def central_diff(x: np.ndarray, fps: float) -> np.ndarray:
    if len(x) < 3:
        return np.zeros_like(x)
    dx = np.zeros_like(x)
    dx[1:-1] = (x[2:] - x[:-2]) * (fps / 2.0)
    dx[0] = (x[1] - x[0]) * fps
    dx[-1] = (x[-1] - x[-2]) * fps
    return dx

def unwrap_deg(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i-1]
        if d > 180:  out[i:] -= 360
        elif d < -180: out[i:] += 360
    return out

def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    BA = a - b; BC = c - b
    num = float(np.dot(BA, BC))
    den = float(np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-9)
    cosv = max(-1.0, min(1.0, num/den))
    return math.degrees(math.acos(cosv))

def bucket_turn(delta_deg: float) -> str:
    """Convierte diferencia angular a categoría de giro"""
    d = ((delta_deg + 180) % 360) - 180
    opts = [(-180,"TURN_180"), (-135,"LEFT_135"), (-90,"LEFT_90"), (-45,"LEFT_45"),
            (0,"STRAIGHT"), (45,"RIGHT_45"), (90,"RIGHT_90"), (135,"RIGHT_135"), (180,"TURN_180")]
    return min(opts, key=lambda t: abs(d - t[0]))[1]

def dir_octant(vx: float, vy: float) -> str:
    """Determina dirección en octantes"""
    ang = math.degrees(math.atan2(vy, vx))
    dirs = [("E",-22.5,22.5), ("SE",22.5,67.5), ("S",67.5,112.5), ("SW",112.5,157.5),
            ("W",157.5,180.0), ("W",-180.0,-157.5), ("NW",-157.5,-112.5),
            ("N",-112.5,-67.5), ("NE",-67.5,-22.5)]
    for lab,a,b in dirs:
        if a <= ang <= b: return lab
    return "E"

# ---------------- Carga de Datos ----------------
def load_landmarks_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    req = {"video_id","frame","lmk_id","x","y"}
    if not req.issubset(df.columns):
        raise RuntimeError(f"CSV inválido, faltan columnas: {req - set(df.columns)}")
    return df

def series_xy(df: pd.DataFrame, lmk_id: int, nframes: int) -> np.ndarray:
    """
    Extrae serie temporal de coordenadas (x,y) para un landmark.
    Aplica forward-fill y backward-fill para interpolar gaps de NaN.
    Si todo el landmark está ausente, retorna array con 0.5 (centro de imagen).
    """
    sub = df[df["lmk_id"] == lmk_id][["frame","x","y"]]
    arr = np.full((nframes, 2), np.nan, dtype=np.float32)
    
    if len(sub) > 0:
        idx = sub["frame"].to_numpy(dtype=int)
        xy = sub[["x","y"]].to_numpy(np.float32)
        idx = np.clip(idx, 0, nframes-1)
        arr[idx] = xy
    
    # Interpolación para valores faltantes: forward-fill y backward-fill
    for k in range(2):
        v = arr[:,k]
        if np.isnan(v).any():
            # Forward fill
            last_valid = None
            for i in range(len(v)):
                if not np.isnan(v[i]):
                    last_valid = v[i]
                elif last_valid is not None:
                    v[i] = last_valid
            
            # Backward fill
            last_valid = None
            for i in range(len(v)-1, -1, -1):
                if not np.isnan(v[i]):
                    last_valid = v[i]
                elif last_valid is not None:
                    v[i] = last_valid
            
            arr[:,k] = v
    
    # Si todavía quedan NaN (landmark nunca apareció), usar centro de imagen
    arr = np.nan_to_num(arr, nan=0.5)
    
    return arr

# ---------------- Clasificación con Especificaciones ----------------
class PoseClassifier:
    """Clasificador que utiliza las especificaciones de pose"""
    
    def __init__(self, config: PoomsaeConfig, use_ml=False, ml_model_path=None):
        self.config = config
        self.pose_spec = config.pose_spec
        self.use_ml = use_ml
        self.ml_classifier = None
        
        if use_ml and ml_model_path:
            try:
                import sys
                from pathlib import Path
                # Agregar src a path para imports
                src_path = Path(__file__).parent.parent
                if str(src_path) not in sys.path:
                    sys.path.insert(0, str(src_path))
                
                from model.stance_classifier import StanceClassifier
                self.ml_classifier = StanceClassifier.load(ml_model_path)
                print(f"[CLASSIFIER] ✅ Modelo ML cargado: {ml_model_path}")
            except Exception as e:
                print(f"[CLASSIFIER] ⚠️ Error cargando ML, usando heurístico: {e}")
                self.use_ml = False

    def classify_stance(self, landmarks_dict, frame_idx: int, debug=False) -> str:
        """Clasifica postura usando especificaciones o ML"""
        # Si ML está habilitado, usarlo
        if self.use_ml and self.ml_classifier:
            return self._classify_with_ml(landmarks_dict, frame_idx, debug)
        
        # Fallback a clasificador heurístico
        return self._classify_heuristic(landmarks_dict, frame_idx, debug)
    
    def _classify_with_ml(self, landmarks_dict, frame_idx: int, debug=False) -> str:
        """Clasificador ML usando features extraídas"""
        try:
            from features.stance_features import extract_stance_features_from_dict
            
            # Verificar que landmarks necesarios existen
            required_lmks = ['L_ANK', 'R_ANK', 'L_KNEE', 'R_KNEE', 'L_HIP', 'R_HIP', 
                           'L_SH', 'R_SH', 'L_HEEL', 'R_HEEL', 'L_FOOT', 'R_FOOT']
            
            for lmk in required_lmks:
                if lmk not in landmarks_dict:
                    if debug:
                        print(f"[ML] Frame {frame_idx}: Missing landmark {lmk}, fallback heurístico")
                    return self._classify_heuristic(landmarks_dict, frame_idx, debug)
            
            # Extraer features del frame actual
            features = extract_stance_features_from_dict(landmarks_dict, frame_idx)
            
            # Verificar NaN en features
            if any(np.isnan(v) for v in features.values()):
                if debug:
                    print(f"[ML] Frame {frame_idx}: NaN en features, fallback heurístico")
                return self._classify_heuristic(landmarks_dict, frame_idx, debug)
            
            # Preparar array de features
            FEATURE_NAMES = ['ankle_dist_sw', 'hip_offset_x', 'hip_offset_y', 
                           'knee_angle_left', 'knee_angle_right',
                           'foot_angle_left', 'foot_angle_right', 'hip_behind_feet']
            
            X = np.array([[features[k] for k in FEATURE_NAMES]])
            
            # Predecir
            prediction = self.ml_classifier.predict(X)[0]
            
            if debug:
                proba = self.ml_classifier.predict_proba(X)[0]
                print(f"[ML] Frame {frame_idx}: {prediction} (proba: {max(proba):.2f})")
            
            return prediction
            
        except Exception as e:
            if debug:
                print(f"[ML] Frame {frame_idx}: Error en predicción: {e}, fallback heurístico")
            return self._classify_heuristic(landmarks_dict, frame_idx, debug)
    
    def _classify_heuristic(self, landmarks_dict, frame_idx: int, debug=False) -> str:
        """Clasificador heurístico original"""
        try:
            lank = landmarks_dict['L_ANK'][frame_idx]
            rank = landmarks_dict['R_ANK'][frame_idx]
            lkne = landmarks_dict['L_KNEE'][frame_idx]
            rkne = landmarks_dict['R_KNEE'][frame_idx]
            lhip = landmarks_dict['L_HIP'][frame_idx]
            rhip = landmarks_dict['R_HIP'][frame_idx]
            lsh = landmarks_dict['L_SH'][frame_idx]
            rsh = landmarks_dict['R_SH'][frame_idx]
            
            # Validar que no haya NaN
            all_points = [lank, rank, lkne, rkne, lhip, rhip, lsh, rsh]
            if any(np.any(np.isnan(pt)) for pt in all_points):
                if debug:
                    print(f"[CLASSIFIER] Frame {frame_idx}: Landmarks con NaN detectados")
                return "moa_seogi"
            
            # Distancia entre pies (normalizada por ancho de hombros)
            shoulder_width = np.linalg.norm(rsh - lsh)
            
            # Validar shoulder width mínimo
            if shoulder_width < 0.02:
                if debug:
                    print(f"[CLASSIFIER] Frame {frame_idx}: Shoulder width muy pequeño ({shoulder_width:.4f})")
                return "moa_seogi"
            
            foot_dist = np.linalg.norm(lank - rank) / shoulder_width
            
            # Ángulos de rodilla
            l_knee_angle = angle3(lhip, lkne, lank)
            r_knee_angle = angle3(rhip, rkne, rank)
            
            if debug:
                print(f"[CLASSIFIER] Frame {frame_idx}: foot_dist={foot_dist:.4f}, "
                      f"l_knee={l_knee_angle:.1f}°, r_knee={r_knee_angle:.1f}°, "
                      f"shoulder_w={shoulder_width:.4f}")
            
            # Verificar contra especificaciones en orden correcto
            if self._check_ap_kubi(foot_dist, l_knee_angle, r_knee_angle, debug):
                stance = "ap_kubi"
            elif self._check_dwit_kubi(foot_dist, l_knee_angle, r_knee_angle, debug):
                stance = "dwit_kubi"
            elif self._check_beom_seogi(foot_dist, debug):
                stance = "beom_seogi"
            else:
                stance = "moa_seogi"
            
            if debug:
                print(f"[CLASSIFIER] Frame {frame_idx}: Clasificado como {stance}")
            
            return stance
                
        except Exception as e:
            print(f"[CLASSIFIER] Error clasificando postura en frame {frame_idx}: {e}")
            return "moa_seogi"

    def _check_ap_kubi(self, foot_dist, l_knee_angle, r_knee_angle, debug=False):
        """Verifica si cumple especificación de ap_kubi - SIMPLIFICADO"""
        # SIMPLIFICACIÓN: Solo usar ankle_dist (sin knee_angles)
        # Umbral ajustado basado en P70 del análisis empírico
        result = foot_dist >= 2.75
        
        if debug:
            print(f"    ap_kubi: dist>={2.75:.2f} → {result}")
        
        return result

    def _check_dwit_kubi(self, foot_dist, l_knee_angle, r_knee_angle, debug=False):
        """Verifica si cumple especificación de dwit_kubi - SIMPLIFICADO"""
        # SIMPLIFICACIÓN: Solo usar ankle_dist (sin knee_angles)
        # Umbral ajustado: entre P10 y P70 del análisis
        result = 0.5 <= foot_dist < 2.75
        
        if debug:
            print(f"    dwit_kubi: {0.5:.1f}<=dist<{2.75:.2f} → {result}")
        
        return result

    def _check_beom_seogi(self, foot_dist, debug=False):
        """Verifica si cumple especificación de beom_seogi - SIMPLIFICADO"""
        # SIMPLIFICACIÓN: Solo usar ankle_dist
        # Umbral ajustado basado en P10 del análisis
        result = foot_dist < 0.5
        
        if debug:
            print(f"    beom_seogi: dist<{0.5:.1f} → {result}")
        
        return result

    def classify_kick(self, ankle_xy: np.ndarray, hip_xy: np.ndarray, kick_type: str = None) -> str:
        """Clasifica patada usando especificaciones"""
        if len(ankle_xy) < 8:
            return "none"
        
        # Altura relativa a cadera
        rel_height = hip_xy[:,1] - ankle_xy[:,1]
        max_height = np.max(rel_height)
        
        # Usar especificaciones si están disponibles
        if self.pose_spec and 'kicks' in self.pose_spec:
            if kick_type in self.pose_spec['kicks']:
                spec = self.pose_spec['kicks'][kick_type]
                amp_min = spec.get('amp_min', 0.2)
                peak_min = spec.get('peak_above_hip_min', 0.25)
                
                if max_height >= peak_min:
                    return kick_type
            else:
                # Clasificación genérica
                ap_spec = self.pose_spec['kicks'].get('ap_chagi', {})
                ap_min = ap_spec.get('peak_above_hip_min', 0.25)
                
                if max_height >= ap_min:
                    return "ap_chagi"
                elif max_height >= 0.15:
                    return "arae_chagi"
        
        # Fallback a lógica original
        if max_height > 0.15:
            return "ap_chagi"
        elif max_height > 0.08:
            return "arae_chagi"
        
        return "none"

# ---------------- DataClasses ----------------
@dataclass
class MoveSegment:
    idx: int
    a: int                 # frame inicio
    b: int                 # frame fin
    t_start: float
    t_end: float
    duration: float
    active_limb: str
    speed_peak: float
    rotation_deg: float
    rotation_bucket: str
    height_end: float
    path: List[Tuple[float, float]]
    stance_pred: str
    kick_pred: str
    arm_dir: str
    pose_f: int
    pose_t: float
    # NUEVO: Información de la especificación
    expected_tech: str = ""
    expected_stance: str = ""
    expected_turn: str = ""
    match_score: float = 0.0

@dataclass
class CaptureResult:
    video_id: str
    fps: float
    nframes: int
    moves: List[MoveSegment]
    poomsae_spec: Dict = None
    
    def to_json(self) -> str:
        import numpy as np
        
        def convert_numpy(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy(item) for item in obj]
            return obj
        
        data = {
            "video_id": self.video_id,
            "fps": float(self.fps),
            "nframes": int(self.nframes),
            "moves": [convert_numpy(asdict(m)) for m in self.moves]
        }
        if self.poomsae_spec:
            data["poomsae_spec"] = self.poomsae_spec
        return json.dumps(data, ensure_ascii=False, indent=2)

# ---------------- Función Principal Mejorada ----------------
def capture_moves_with_spec(csv_path: Path, 
                           video_path: Optional[Path] = None,
                           config_path: Path = Path("config/default.yaml"),
                           spec_path: Path = Path("config/patterns/8yang_spec.json"),
                           pose_spec_path: Path = Path("config/patterns/pose_spec.json"),
                           use_ml_classifier: bool = False,
                           ml_model_path: Optional[Path] = None
                          ) -> CaptureResult:
    """
    Captura movimientos usando las especificaciones del Poomsae 8yang
    """
    
    # Cargar configuraciones
    config = PoomsaeConfig(spec_path, pose_spec_path, config_path)
    
    # Cargar datos
    df = load_landmarks_csv(csv_path)
    nframes = int(df["frame"].max()) + 1
    
    # Determinar FPS
    fps = 30.0
    if video_path and video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()
    
    print(f"[CAPTURE] Procesando {Path(csv_path).stem} para 8yang")
    print(f"[CAPTURE] Frames: {nframes}, FPS: {fps:.1f}")
    
    # Preparar datos
    landmarks_dict = {}
    landmark_points = ['L_SH', 'R_SH', 'L_HIP', 'R_HIP', 
                      'L_WRIST', 'R_WRIST', 'L_ANK', 'R_ANK',
                      'L_KNEE', 'R_KNEE', 'L_ELB', 'R_ELB',
                      'L_HEEL', 'R_HEEL', 'L_FOOT', 'R_FOOT']  # ✅ AGREGAR talones y pies para ML
    
    for point in landmark_points:
        if point in LMK:
            landmarks_dict[point] = series_xy(df, LMK[point], nframes)
    
    # Calcular ángulos para Taekwondo
    angles_dict = {}
    try:
        l_hip = landmarks_dict['L_HIP']
        r_hip = landmarks_dict['R_HIP']
        l_knee = landmarks_dict['L_KNEE']
        r_knee = landmarks_dict['R_KNEE'] 
        l_ankle = landmarks_dict['L_ANK']
        r_ankle = landmarks_dict['R_ANK']
        
        left_knee_angles = []
        right_knee_angles = []
        
        min_len = min(len(l_hip), len(l_knee), len(l_ankle), 
                     len(r_hip), len(r_knee), len(r_ankle))
        
        for i in range(min_len):
            left_angle = angle3(l_hip[i], l_knee[i], l_ankle[i])
            right_angle = angle3(r_hip[i], r_knee[i], r_ankle[i])
            left_knee_angles.append(left_angle)
            right_knee_angles.append(right_angle)
        
        angles_dict['left_knee'] = np.array(left_knee_angles)
        angles_dict['right_knee'] = np.array(right_knee_angles)
        
        # ✅ AGREGAR: Ángulos de codo (crítico para STRIKES/BLOCKS)
        if 'L_SH' in landmarks_dict and 'L_ELB' in landmarks_dict and 'L_WRIST' in landmarks_dict:
            l_sh = landmarks_dict['L_SH']
            l_elb = landmarks_dict['L_ELB']
            l_wri = landmarks_dict['L_WRIST']
            r_sh = landmarks_dict['R_SH']
            r_elb = landmarks_dict['R_ELB']
            r_wri = landmarks_dict['R_WRIST']
            
            left_elbow_angles = []
            right_elbow_angles = []
            
            min_elb_len = min(len(l_sh), len(l_elb), len(l_wri),
                             len(r_sh), len(r_elb), len(r_wri))
            
            for i in range(min_elb_len):
                left_elb_angle = angle3(l_sh[i], l_elb[i], l_wri[i])
                right_elb_angle = angle3(r_sh[i], r_elb[i], r_wri[i])
                left_elbow_angles.append(left_elb_angle)
                right_elbow_angles.append(right_elb_angle)
            
            angles_dict['left_elbow'] = np.array(left_elbow_angles)
            angles_dict['right_elbow'] = np.array(right_elbow_angles)
        
        # ✅ AGREGAR: Ángulos de cadera (importante para detección de giros)
        if 'L_SH' in landmarks_dict and 'L_HIP' in landmarks_dict and 'L_KNEE' in landmarks_dict:
            left_hip_angles = []
            right_hip_angles = []
            
            for i in range(min_len):
                left_hip_angle = angle3(l_sh[i], l_hip[i], l_knee[i])
                right_hip_angle = angle3(r_sh[i], r_hip[i], r_knee[i])
                left_hip_angles.append(left_hip_angle)
                right_hip_angles.append(right_hip_angle)
            
            angles_dict['left_hip'] = np.array(left_hip_angles)
            angles_dict['right_hip'] = np.array(right_hip_angles)
        
    except Exception as e:
        print(f"[CAPTURE] Error calculando ángulos: {e}")
    
    # Configurar segmentador y clasificador
    segmenter = SpecAwareSegmenter(config)
    classifier = PoseClassifier(config, use_ml=use_ml_classifier, ml_model_path=ml_model_path)
    
    # Encontrar segmentos
    segments = segmenter.find_segments(angles_dict, landmarks_dict, fps)
    
    # Filtrar por duración máxima
    poomsae_config = config.app_config.get('poomsae_8yang', {})
    max_duration = poomsae_config.get('max_duration', 2.0)
    max_frames = int(fps * max_duration)
    
    filtered_segments = [(s, e) for s, e in segments if (e - s) <= max_frames]
    print(f"[CAPTURE] Segmentos filtrados: {len(filtered_segments)}")
    
    # Crear objetos MoveSegment con información de especificación
    moves = []
    lsh = landmarks_dict['L_SH']
    rsh = landmarks_dict['R_SH']
    lwri = landmarks_dict['L_WRIST']
    rwri = landmarks_dict['R_WRIST']
    lank = landmarks_dict['L_ANK']
    rank = landmarks_dict['R_ANK']
    lhip = landmarks_dict['L_HIP']
    rhip = landmarks_dict['R_HIP']
    
    eff_xy = {"L_WRIST": lwri, "R_WRIST": rwri, "L_ANK": lank, "R_ANK": rank}
    heading = _heading_series_deg(lsh, rsh)
    
    for i, (a, b) in enumerate(filtered_segments, start=1):
        if a >= b or a >= nframes or b >= nframes:
            continue
            
        # Determinar extremidad activa
        limb = _choose_active_limb(eff_xy, a, b)
        
        # Obtener especificación esperada para este movimiento
        move_spec = config.get_move_spec(i)
        expected_tech = move_spec.get('tech_es', '') if move_spec else ""
        expected_stance = move_spec.get('stance_expect', '') if move_spec else ""
        expected_turn = move_spec.get('turn_expect', '') if move_spec else ""
        
        # Calcular características
        path_xy = eff_xy[limb][a:b+1, :]
        path_resampled = resample_polyline(path_xy, 20)
        
        # Velocidad pico
        seg_sp_col = {
            k: np.linalg.norm(central_diff(movavg(eff_xy[k][a:b+1], 3), fps), axis=1) 
            for k in END_EFFECTORS
        }
        v_peak = float(max(np.nanmax(v) if len(v) else 0.0 for v in seg_sp_col.values()))
        
        # Rotación
        rot_deg = heading[b] - heading[a] if len(heading) > b else 0.0
        rot_bucket = bucket_turn(rot_deg)
        
        # Altura final
        height_end = float(path_xy[-1, 1]) if len(path_xy) > 0 else 0.0
        
        # Frame de pose
        pose_f = _choose_pose_frame(
            np.nanmax([np.linalg.norm(central_diff(eff_xy[k], fps), axis=1) for k in END_EFFECTORS], axis=0),
            a, b
        )
        
        # Clasificaciones usando especificaciones
        stance_pred = classifier.classify_stance(landmarks_dict, pose_f)
        
        # Detección de patada con especificación
        hip_xy = 0.5 * (lhip + rhip)
        kick_type = move_spec.get('kick_type', None) if move_spec else None
        kick_pred = classifier.classify_kick(eff_xy[limb][a:b+1], hip_xy[a:b+1], kick_type)
        
        # Dirección de brazo
        arm_dir = arm_motion_direction(eff_xy[limb][a:b+1]) if limb in ["L_WRIST", "R_WRIST"] else "NONE"
        
        # Calcular score de coincidencia con especificación
        match_score = _calculate_match_score(stance_pred, kick_pred, rot_bucket, 
                                           expected_stance, expected_turn, move_spec)
        
        move = MoveSegment(
            idx=i,
            a=a, b=b,
            t_start=a/fps, t_end=b/fps,
            duration=(b-a)/fps,
            active_limb=limb,
            speed_peak=v_peak,
            rotation_deg=rot_deg,
            rotation_bucket=rot_bucket,
            height_end=height_end,
            path=[(float(x), float(y)) for x, y in path_resampled],
            stance_pred=stance_pred,
            kick_pred=kick_pred,
            arm_dir=arm_dir,
            pose_f=pose_f,
            pose_t=pose_f/fps,
            expected_tech=expected_tech,
            expected_stance=expected_stance,
            expected_turn=expected_turn,
            match_score=match_score
        )
        moves.append(move)
    
    print(f"[CAPTURE] Movimientos finales con especificación: {len(moves)}")
    return CaptureResult(
        video_id=Path(csv_path).stem, 
        fps=fps, 
        nframes=nframes, 
        moves=moves,
        poomsae_spec=config.poomsae_spec
    )

# ---------------- Funciones Auxiliares ----------------
def _choose_active_limb(eff_xy: Dict[str,np.ndarray], a: int, b: int) -> str:
    """Determina la extremidad más activa en el segmento"""
    def _disp(xy: np.ndarray) -> float:
        if len(xy) < 2: return 0.0
        d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
        return float(np.nansum(d))
    
    cand = {k: eff_xy[k][a:b+1] for k in END_EFFECTORS}
    return max(cand.items(), key=lambda kv: _disp(kv[1]))[0]

def _choose_pose_frame(smax: np.ndarray, a: int, b: int) -> int:
    """Selecciona frame estable para análisis de pose"""
    if b <= a: return a
    lo = a + int(0.6 * (b - a))
    lo = min(max(lo, a), b)
    sub = smax[lo:b+1]
    if sub.size == 0 or not np.isfinite(sub).any():
        return b
    off = int(np.nanargmin(sub))
    return lo + off

def arm_motion_direction(wrist_xy: np.ndarray) -> str:
    """Determina dirección del movimiento de brazo"""
    if len(wrist_xy) < 2: return "NONE"
    v = wrist_xy[-1] - wrist_xy[0]
    return dir_octant(float(v[0]), float(v[1]))

def _calculate_match_score(stance_actual: str, kick_actual: str, turn_actual: str,
                          stance_expected: str, turn_expected: str, move_spec: Dict) -> float:
    """Calcula score de coincidencia con especificación"""
    score = 0.0
    factors = 0
    
    # Coincidencia de postura
    if stance_expected:
        factors += 1
        if stance_actual == stance_expected:
            score += 1.0
        elif stance_actual in ["ap_kubi", "dwit_kubi", "beom_seogi"]:  # Posturas similares
            score += 0.5
    
    # Coincidencia de giro
    if turn_expected:
        factors += 1
        if turn_actual == turn_expected:
            score += 1.0
        elif turn_expected != "NONE" and turn_actual != "STRAIGHT":  # Algún giro detectado
            score += 0.7
    
    # Coincidencia de patada
    if move_spec and move_spec.get('category') == 'KICK':
        factors += 1
        expected_kick = move_spec.get('kick_type')
        if kick_actual == expected_kick:
            score += 1.0
        elif kick_actual != "none":  # Alguna patada detectada
            score += 0.5
    
    return score / factors if factors > 0 else 0.0

def _heading_series_deg(lsh: np.ndarray, rsh: np.ndarray) -> np.ndarray:
    """Calcula orientación del cuerpo"""
    sh_vec = rsh - lsh
    heading = np.degrees(np.arctan2(sh_vec[:,1], sh_vec[:,0]))
    return unwrap_deg(heading)

def resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    """Remuestrea polilínea a n puntos"""
    if len(xy) == 0: return np.zeros((0,2), np.float32)
    if len(xy) == 1: return np.tile(xy[:1], (n,1))
    d = np.sqrt(np.sum(np.diff(xy, axis=0)**2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] <= 1e-8: return np.tile(xy[:1], (n,1))
    t = np.linspace(0, s[-1], n)
    xs = np.interp(t, s, xy[:,0]); ys = np.interp(t, s, xy[:,1])
    return np.stack([xs, ys], axis=1)

# ---------------- Función Legacy para Compatibilidad ----------------
def capture_moves_enhanced(csv_path: Path, 
                          video_path: Optional[Path] = None,
                          expected_movements: int = 24,
                          min_duration: float = 0.2,
                          max_duration: float = 2.0,
                          sensitivity: float = 0.9
                         ) -> CaptureResult:
    """
    Función legacy que ahora usa el sistema con especificaciones
    """
    print("[LEGACY] Usando sistema con especificaciones 8yang")
    return capture_moves_with_spec(csv_path, video_path)

def capture_moves_from_csv(csv_path: Path, *, video_path: Optional[Path],
                           vstart: float = 0.40, vstop: float = 0.15,
                           min_dur: float = 0.20, min_gap: float = 0.15,
                           smooth_win: int = 3, poly_n: int = 20,
                           min_path_norm: float = 0.015,
                           expected_n: Optional[int] = None) -> CaptureResult:
    """
    Función legacy - ahora usa la versión con especificaciones
    """
    print("[LEGACY] Redirigiendo a sistema con especificaciones")
    return capture_moves_with_spec(csv_path, video_path)