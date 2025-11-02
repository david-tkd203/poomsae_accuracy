#!/usr/bin/env python3
"""
Analizar discrepancia de rotaciones
"""
import pandas as pd
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

df = pd.read_excel("reports/8yang_001_CORRECTED.xlsx")
spec = json.load(open('config/patterns/8yang_spec.json'))

# Map spec
spec_map = {}
for m in spec['moves']:
    key = f"{m['idx']}{m.get('sub', '')}"
    spec_map[key] = m

print("=" * 100)
print("ANÁLISIS DE ROTACIONES ESPERADAS VS MEDIDAS")
print("=" * 100)

# Comparar rotaciones esperadas vs medidas
turns_deg = {
    "NONE": 0,
    "LEFT_90": -90,
    "RIGHT_90": 90,
    "LEFT_180": 180,
    "RIGHT_180": 180,
    "LEFT_270": -270,
    "RIGHT_270": 270,
}

print("\n1. DISCREPANCIA DE ROTACIÓN POR MOVIMIENTO:")
print("-" * 100)
print("Move | exp_turn        | exp_deg | meas_deg | diff_deg | comp_arms")
print("-" * 100)

for idx, row in df.iterrows():
    m_key = row['M']
    if m_key in spec_map:
        spec_m = spec_map[m_key]
        turn_code = spec_m.get('turn', 'NONE')
        exp_deg = turns_deg.get(turn_code, 0)
        meas_deg = row['rot(meas_deg)']
        
        # Calcular diferencia de ángulos
        diff = abs(meas_deg - exp_deg)
        if diff > 180:
            diff = 360 - diff
        
        print(f"{m_key:4s} | {turn_code:15s} | {exp_deg:7.1f} | {meas_deg:8.2f} | {diff:8.1f} | {row['comp_arms']}")

print("\n\n2. ESTADÍSTICAS POR TIPO DE GIRO:")
print("-" * 100)
for turn_code, exp_deg in turns_deg.items():
    rows = [r for idx, r in df.iterrows() if r['M'] in spec_map and spec_map[r['M']].get('turn', 'NONE') == turn_code]
    if rows:
        meas_degs = [r['rot(meas_deg)'] for r in rows]
        print(f"\n{turn_code:15s} (exp={exp_deg:7.1f}°):")
        print(f"  count: {len(rows)}")
        print(f"  meas: min={min(meas_degs):7.1f}°, max={max(meas_degs):7.1f}°, mean={sum(meas_degs)/len(meas_degs):7.1f}°")
        grave_count = sum(1 for r in rows if r['comp_arms'] == 'GRAVE')
        ok_count = sum(1 for r in rows if r['comp_arms'] == 'OK')
        print(f"  arms: {grave_count} GRAVE, {ok_count} OK")

print("\n\n3. TOLERANCIA ACTUAL (rot_tol=30°) vs RESULTADOS:")
print("-" * 100)
print("Según _classify_rotation con tol=30°:")
print("  OK si diff <= 30°")
print("  LEVE si 30° < diff <= 45°")
print("  GRAVE si diff > 45°")
