#!/usr/bin/env python3
"""
Debug: Verificar cálculo de rel_level
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.segmentation.move_capture import load_landmarks_csv

# Cargar CSV
csv_path = Path("data/landmarks/8yang/8yang_001.csv")
df = load_landmarks_csv(csv_path)

print("Verificación de cálculo de rel_level (y_wrist_rel)")
print("=" * 100)
print("\nEstructura de datos:")
print(f"  Columnas: {df.columns.tolist()}")
print(f"  Filas: {len(df)}")
print(f"  Frames: {df['frame'].min()}-{df['frame'].max()}")
print(f"  Landmarks: {df['lmk_id'].unique()}")

# IDs en MediaPipe:
# 11 = L_SH (left shoulder)
# 12 = R_SH (right shoulder)
# 15 = L_WRIST (left wrist)
# 16 = R_WRIST (right wrist)
# 23 = L_HIP (left hip)
# 24 = R_HIP (right hip)

# Analizar un rango de frames
print("\n\nAnálisis de coordenadas Y para un movimiento específico:")
print("-" * 100)

# Tomar frames 300-400 (un rango en medio del video)
a, b = 300, 350

print(f"\nFrames {a}-{b}:")

def get_y_values(df, lmk_id, a, b):
    sub = df[(df['lmk_id'] == lmk_id) & (df['frame'] >= a) & (df['frame'] <= b)]
    return sub[['frame', 'y']].set_index('frame')

y_lsh = get_y_values(df, 11, a, b)
y_rsh = get_y_values(df, 12, a, b)
y_lhp = get_y_values(df, 23, a, b)
y_rhp = get_y_values(df, 24, a, b)
y_lwrist = get_y_values(df, 15, a, b)
y_rwrist = get_y_values(df, 16, a, b)

# Combinar
combined = pd.DataFrame({
    'Y_L_SH': y_lsh['y'],
    'Y_R_SH': y_rsh['y'],
    'Y_L_HP': y_lhp['y'],
    'Y_R_HP': y_rhp['y'],
    'Y_L_WRIST': y_lwrist['y'],
    'Y_R_WRIST': y_rwrist['y']
})

combined['Y_SH_MID'] = (combined['Y_L_SH'] + combined['Y_R_SH']) / 2
combined['Y_HP_MID'] = (combined['Y_L_HP'] + combined['Y_R_HP']) / 2
combined['REL_L_WRIST'] = (combined['Y_L_WRIST'] - combined['Y_SH_MID']) / (combined['Y_HP_MID'] - combined['Y_SH_MID'])
combined['REL_R_WRIST'] = (combined['Y_R_WRIST'] - combined['Y_SH_MID']) / (combined['Y_HP_MID'] - combined['Y_SH_MID'])

print("\nPrimeros 5 frames:")
print(combined.head().to_string())

print("\n\nEstadísticas de rel_level:")
for col in ['REL_L_WRIST', 'REL_R_WRIST']:
    vals = combined[col].dropna()
    if len(vals) > 0:
        print(f"\n{col}:")
        print(f"  min: {vals.min():.4f}")
        print(f"  max: {vals.max():.4f}")
        print(f"  mean: {vals.mean():.4f}")
        print(f"  median: {vals.median():.4f}")
        print(f"  values < -0.02 (OLGUL): {(vals <= -0.02).sum()}")
        print(f"  values in [-0.02, 1.03) (MOMTONG): {((vals > -0.02) & (vals < 1.03)).sum()}")
        print(f"  values >= 1.03 (ARAE): {(vals >= 1.03).sum()}")

print("\n\nInterpretación de alturas:")
print(f"  Y_SH (hombro): típicamente ~{combined['Y_SH_MID'].mean():.0f} (ARRIBA)")
print(f"  Y_HP (cadera): típicamente ~{combined['Y_HP_MID'].mean():.0f} (ABAJO)")
print(f"  Y_L_WRIST (muñeca izq): ~{combined['Y_L_WRIST'].mean():.0f}")
print(f"  Y_R_WRIST (muñeca der): ~{combined['Y_R_WRIST'].mean():.0f}")

print("\n✓ En MediaPipe Y aumenta hacia ABAJO")
print("  OLGUL (arriba): Y_WRIST < Y_SH → rel < 0")
print("  MOMTONG (medio): Y_SH < Y_WRIST < Y_HP → rel ∈ [0, 1]")
print("  ARAE (abajo): Y_WRIST > Y_HP → rel > 1")
