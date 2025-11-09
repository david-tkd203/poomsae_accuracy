"""
Extractor de features para clasificación de posturas.
Calcula features geométricas a partir de landmarks de MediaPipe Pose.
"""
from __future__ import annotations
from typing import Tuple, Dict, Union
import numpy as np
import pandas as pd

def _xy_mean(df: pd.DataFrame, lmk_id: int, a: int, b: int) -> Tuple[float, float]:
    """Media de coordenadas x,y para un landmark en rango de frames"""
    sub = df[(df["lmk_id"] == lmk_id) & (df["frame"] >= a) & (df["frame"] <= b)][["x", "y"]]
    if sub.empty:
        return (np.nan, np.nan)
    m = sub.mean().to_numpy(np.float32)
    return float(m[0]), float(m[1])

def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo en grados entre 3 puntos (b es el vértice)"""
    ba = a - b
    bc = c - b
    num = float(np.dot(ba, bc))
    den = float(np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    ang = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    return float(ang)

def extract_stance_features(df: pd.DataFrame, a: int, b: int) -> Dict[str, float]:
    """
    Extrae 9 features geométricas para clasificación de posturas.
    
    Args:
        df: DataFrame con landmarks (columnas: frame, lmk_id, x, y, z)
        a: Frame inicial del segmento
        b: Frame final del segmento
    
    Returns:
        Dict con 9 features:
            - ankle_dist_sw: Distancia entre tobillos / ancho de hombros
            - hip_offset_x: Offset horizontal de cadera respecto a centro de pies
            - hip_offset_y: Offset vertical de cadera (positivo = atrás)
            - knee_angle_left: Ángulo de rodilla izquierda (grados)
            - knee_angle_right: Ángulo de rodilla derecha (grados)
            - foot_angle_left: Orientación pie izquierdo (grados)
            - foot_angle_right: Orientación pie derecho (grados)
            - hip_behind_feet: Indicador binario si cadera está atrás de pies
            - knee_angle_diff: Diferencia absoluta entre ángulos de rodillas
    """
    # Landmarks clave (promedio en el segmento [a,b])
    LANK = np.array(_xy_mean(df, 27, a, b), np.float32)
    RANK = np.array(_xy_mean(df, 28, a, b), np.float32)
    LKNE = np.array(_xy_mean(df, 25, a, b), np.float32)
    RKNE = np.array(_xy_mean(df, 26, a, b), np.float32)
    LHIP = np.array(_xy_mean(df, 23, a, b), np.float32)
    RHIP = np.array(_xy_mean(df, 24, a, b), np.float32)
    LSH = np.array(_xy_mean(df, 11, a, b), np.float32)
    RSH = np.array(_xy_mean(df, 12, a, b), np.float32)
    LHEEL = np.array(_xy_mean(df, 29, a, b), np.float32)
    RHEEL = np.array(_xy_mean(df, 30, a, b), np.float32)
    LFOOT = np.array(_xy_mean(df, 31, a, b), np.float32)
    RFOOT = np.array(_xy_mean(df, 32, a, b), np.float32)
    
    # Validar datos esenciales
    if any(np.isnan(v).any() for v in [LANK, RANK, LSH, RSH, LHIP, RHIP]):
        return {k: np.nan for k in [
            "ankle_dist_sw", "hip_offset_x", "hip_offset_y",
            "knee_angle_left", "knee_angle_right",
            "foot_angle_left", "foot_angle_right",
            "hip_behind_feet", "knee_angle_diff"
        ]}
    
    # FEATURE 1: Distancia entre tobillos normalizada
    sh_w = float(np.linalg.norm(RSH - LSH) + 1e-6)
    ankle_dist = float(np.linalg.norm(RANK - LANK))
    ankle_dist_sw = ankle_dist / sh_w
    
    # FEATURE 2-3: Offset de cadera respecto a centro de pies
    feet_center = 0.5 * (LANK + RANK)
    hip_center = 0.5 * (LHIP + RHIP)
    hip_offset_x = float(hip_center[0] - feet_center[0])
    hip_offset_y = float(hip_center[1] - feet_center[1])  # Y+ = atrás en imagen
    
    # FEATURE 4-5: Ángulos de rodillas
    if not any(np.isnan(v).any() for v in [LHIP, LKNE, LANK]):
        knee_angle_left = _angle3(LHIP, LKNE, LANK)
    else:
        knee_angle_left = np.nan
    
    if not any(np.isnan(v).any() for v in [RHIP, RKNE, RANK]):
        knee_angle_right = _angle3(RHIP, RKNE, RANK)
    else:
        knee_angle_right = np.nan
    
    # FEATURE 6-7: Orientación de pies
    def foot_angle(heel, foot):
        if np.isnan(heel).any() or np.isnan(foot).any():
            return np.nan
        vec = foot - heel
        if np.linalg.norm(vec) < 1e-6:
            return np.nan
        return float(np.degrees(np.arctan2(vec[1], vec[0])))
    
    foot_angle_left = foot_angle(LHEEL, LFOOT)
    foot_angle_right = foot_angle(RHEEL, RFOOT)
    
    # FEATURE 8: Indicador de cadera atrás
    hip_behind_feet = 1.0 if hip_offset_y > 0 else 0.0
    
    # FEATURE 9: Diferencia de ángulos de rodillas (para detectar asimetría)
    if not np.isnan(knee_angle_left) and not np.isnan(knee_angle_right):
        knee_angle_diff = abs(knee_angle_left - knee_angle_right)
    else:
        knee_angle_diff = np.nan
    
    return {
        "ankle_dist_sw": ankle_dist_sw,
        "hip_offset_x": hip_offset_x,
        "hip_offset_y": hip_offset_y,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "foot_angle_left": foot_angle_left,
        "foot_angle_right": foot_angle_right,
        "hip_behind_feet": hip_behind_feet,
        "knee_angle_diff": knee_angle_diff
    }

# Feature names para fácil referencia
FEATURE_NAMES = [
    "ankle_dist_sw",
    "hip_offset_x",
    "hip_offset_y",
    "knee_angle_left",
    "knee_angle_right",
    "foot_angle_left",
    "foot_angle_right",
    "hip_behind_feet",
    "knee_angle_diff"
]

def extract_features_batch(df: pd.DataFrame, segments: list) -> np.ndarray:
    """
    Extrae features para múltiples segmentos.
    
    Args:
        df: DataFrame con landmarks
        segments: Lista de tuplas (a, b) con rangos de frames
    
    Returns:
        Array numpy (n_segments, n_features)
    """
    features = []
    for a, b in segments:
        feat_dict = extract_stance_features(df, a, b)
        features.append([feat_dict[name] for name in FEATURE_NAMES])
    return np.array(features, dtype=np.float32)

def extract_stance_features_from_dict(landmarks_dict: Dict[str, np.ndarray], 
                                      frame_idx: int) -> Dict[str, float]:
    """
    Extrae features para clasificación de posturas desde landmarks_dict.
    
    Args:
        landmarks_dict: Diccionario con arrays numpy (nframes, 2) para cada landmark
        frame_idx: Índice del frame a analizar
    
    Returns:
        Dict con 8 features (igual que extract_stance_features)
    """
    # Obtener coordenadas del frame específico
    try:
        LANK = landmarks_dict['L_ANK'][frame_idx].astype(np.float32)
        RANK = landmarks_dict['R_ANK'][frame_idx].astype(np.float32)
        LKNE = landmarks_dict['L_KNEE'][frame_idx].astype(np.float32)
        RKNE = landmarks_dict['R_KNEE'][frame_idx].astype(np.float32)
        LHIP = landmarks_dict['L_HIP'][frame_idx].astype(np.float32)
        RHIP = landmarks_dict['R_HIP'][frame_idx].astype(np.float32)
        LSH = landmarks_dict['L_SH'][frame_idx].astype(np.float32)
        RSH = landmarks_dict['R_SH'][frame_idx].astype(np.float32)
        LHEEL = landmarks_dict['L_HEEL'][frame_idx].astype(np.float32)
        RHEEL = landmarks_dict['R_HEEL'][frame_idx].astype(np.float32)
        LFOOT = landmarks_dict['L_FOOT'][frame_idx].astype(np.float32)
        RFOOT = landmarks_dict['R_FOOT'][frame_idx].astype(np.float32)
    except (KeyError, IndexError) as e:
        # Retornar NaN si faltan landmarks
        return {name: np.nan for name in FEATURE_NAMES}
    
    # 1. Distancia entre tobillos normalizada
    ankle_dist = float(np.linalg.norm(RANK - LANK))
    shoulder_width = float(np.linalg.norm(RSH - LSH))
    ankle_dist_sw = ankle_dist / (shoulder_width + 1e-6)
    
    # 2. Centro de cadera y centro de pies
    hip_center = (LHIP + RHIP) / 2.0
    feet_center = (LANK + RANK) / 2.0
    
    # 3. Offsets de cadera
    hip_offset_x = float(hip_center[0] - feet_center[0])
    hip_offset_y = float(hip_center[1] - feet_center[1])
    
    # 4. Ángulos de rodilla
    knee_angle_left = _angle3(LHIP, LKNE, LANK)
    knee_angle_right = _angle3(RHIP, RKNE, RANK)
    
    # 5. Orientación de pies
    left_foot_vec = LFOOT - LHEEL
    right_foot_vec = RFOOT - RHEEL
    foot_angle_left = float(np.degrees(np.arctan2(left_foot_vec[1], left_foot_vec[0])))
    foot_angle_right = float(np.degrees(np.arctan2(right_foot_vec[1], right_foot_vec[0])))
    
    # 6. Indicador binario de cadera atrás de pies
    hip_behind_feet = 1.0 if hip_offset_y > 0 else 0.0
    
    return {
        "ankle_dist_sw": ankle_dist_sw,
        "hip_offset_x": hip_offset_x,
        "hip_offset_y": hip_offset_y,
        "knee_angle_left": knee_angle_left,
        "knee_angle_right": knee_angle_right,
        "foot_angle_left": foot_angle_left,
        "foot_angle_right": foot_angle_right,
        "hip_behind_feet": hip_behind_feet
    }
