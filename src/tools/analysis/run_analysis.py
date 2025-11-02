#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Suite de análisis para debugging y validación del sistema de scoring.
Uso: python src/tools/analysis/run_analysis.py [command]

Comandos disponibles:
  - level       : Analizar mediciones de nivel
  - rotation    : Analizar discrepancias de rotación
  - scoring     : Resumen de scoring
  - arms        : Analizar causas de GRAVE en brazos
  - all         : Ejecutar todos los análisis
"""

import sys
import io
import locale
import json
from pathlib import Path
from typing import Dict

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
ROOT = Path(__file__).resolve().parents[3]  # Subir 3 niveles: src/tools/analysis -> raíz
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np


def analyze_level(xlsx_path: Path = None):
    """Analizar mediciones de nivel"""
    if xlsx_path is None:
        xlsx_path = ROOT / "reports/8yang_001_CORRECTED.xlsx"
    
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    
    df = pd.read_excel(xlsx_path)
    
    print("\n" + "=" * 100)
    print("ANÁLISIS DE MEDICIONES DE NIVEL")
    print("=" * 100)
    
    # Matriz de confusión
    print("\nMatriz de confusión (level_exp vs level_meas):")
    matrix = pd.crosstab(df['level(exp)'], df['level(meas)'], margins=True)
    print(matrix)
    
    # Estadísticas
    print("\nEstadísticas de y_rel_end:")
    for level_exp in ['ARAE', 'MOMTONG', 'OLGUL']:
        subset = df[df['level(exp)'] == level_exp]
        if len(subset) > 0:
            y_vals = subset['y_rel_end'].dropna()
            print(f"\n{level_exp}: n={len(subset)}, n_medidos={len(y_vals)}")
            if len(y_vals) > 0:
                print(f"  min={y_vals.min():.4f}, max={y_vals.max():.4f}, "
                      f"mean={y_vals.mean():.4f}, median={y_vals.median():.4f}")


def analyze_rotation(xlsx_path: Path = None):
    """Analizar discrepancias de rotación"""
    if xlsx_path is None:
        xlsx_path = ROOT / "reports/8yang_001_CORRECTED.xlsx"
    
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    
    df = pd.read_excel(xlsx_path)
    spec_path = ROOT / "config/patterns/8yang_spec.json"
    spec = json.loads(spec_path.read_text())
    
    # Map spec
    spec_map = {}
    for m in spec['moves']:
        key = f"{m['idx']}{m.get('sub', '')}"
        spec_map[key] = m
    
    turns_deg = {
        "NONE": 0, "LEFT_90": -90, "RIGHT_90": 90,
        "LEFT_180": 180, "RIGHT_180": 180,
        "LEFT_270": -270, "RIGHT_270": 270,
    }
    
    print("\n" + "=" * 100)
    print("ANÁLISIS DE ROTACIONES ESPERADAS VS MEDIDAS")
    print("=" * 100)
    
    print("\nDiscrepancias principales:")
    print("Move | Expected Turn   | Exp(°) | Meas(°) | Diff(°) | Status")
    print("-" * 70)
    
    for idx, row in df.iterrows():
        m_key = row['M']
        if m_key in spec_map:
            turn_code = spec_map[m_key].get('turn', 'NONE')
            exp_deg = turns_deg.get(turn_code, 0)
            meas_deg = row['rot(meas_deg)']
            
            diff = abs(meas_deg - exp_deg)
            if diff > 180:
                diff = 360 - diff
            
            if diff > 30:  # Solo mostrar discrepancias significativas
                print(f"{m_key:4s} | {turn_code:15s} | {exp_deg:6.0f} | {meas_deg:7.2f} | {diff:7.1f} | "
                      f"{'⚠️  PROBLEMA' if diff > 45 else 'Aceptable'}")
    
    # Estadísticas
    print("\n\nEstadísticas por tipo de giro:")
    for turn_code, exp_deg in turns_deg.items():
        rows = [r for idx, r in df.iterrows() if r['M'] in spec_map and spec_map[r['M']].get('turn', 'NONE') == turn_code]
        if rows:
            meas_degs = [r['rot(meas_deg)'] for r in rows]
            grave_count = sum(1 for r in rows if r['comp_arms'] == 'GRAVE')
            ok_count = sum(1 for r in rows if r['comp_arms'] == 'OK')
            print(f"\n{turn_code:15s} (exp={exp_deg:7.0f}°): n={len(rows)}")
            print(f"  Medido: min={min(meas_degs):7.1f}°, max={max(meas_degs):7.1f}°, mean={sum(meas_degs)/len(meas_degs):7.1f}°")
            print(f"  Arms: {ok_count} OK, {grave_count} GRAVE")


def analyze_scoring(xlsx_path: Path = None):
    """Resumen de scoring"""
    if xlsx_path is None:
        xlsx_path = ROOT / "reports/8yang_001_CORRECTED.xlsx"
    
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    
    df = pd.read_excel(xlsx_path)
    
    print("\n" + "=" * 100)
    print("RESUMEN DE SCORING")
    print("=" * 100)
    
    print(f"\nTotal movimientos: {len(df)}")
    print(f"Correctos (is_correct=yes): {(df['is_correct'] == 'yes').sum()}/36 ({(df['is_correct'] == 'yes').sum()/36*100:.1f}%)")
    print(f"Exactitud promedio: {df['exactitud_acum'].mean():.3f}/10")
    
    print("\nComponentes:")
    for col in ['comp_arms', 'comp_legs', 'comp_kick']:
        print(f"\n{col}:")
        counts = df[col].value_counts()
        for val, cnt in counts.items():
            pct = cnt / len(df) * 100
            print(f"  {val:10s}: {cnt:3d} ({pct:5.1f}%)")


def analyze_arms(xlsx_path: Path = None):
    """Analizar causa de GRAVE en brazos"""
    if xlsx_path is None:
        xlsx_path = ROOT / "reports/8yang_001_CORRECTED.xlsx"
    
    if not xlsx_path.exists():
        print(f"❌ No encontrado: {xlsx_path}")
        return
    
    df = pd.read_excel(xlsx_path)
    df_grave = df[df['comp_arms'] == 'GRAVE']
    df_ok = df[df['comp_arms'] == 'OK']
    
    print("\n" + "=" * 100)
    print("ANÁLISIS DE CAUSA DE GRAVE EN BRAZOS")
    print("=" * 100)
    
    print(f"\nTotal GRAVE: {len(df_grave)}")
    print(f"Total OK: {len(df_ok)}")
    
    # Codos
    ok_elbow = df_ok['elbow_median_deg'].dropna()
    grave_elbow = df_grave['elbow_median_deg'].dropna()
    
    print(f"\nExtensión de codo:")
    print(f"  OK:    min={ok_elbow.min():.1f}°, max={ok_elbow.max():.1f}°, mean={ok_elbow.mean():.1f}°")
    print(f"  GRAVE: min={grave_elbow.min():.1f}°, max={grave_elbow.max():.1f}°, mean={grave_elbow.mean():.1f}°")
    
    # Fail parts
    print(f"\nRazones de GRAVE (primeros 10):")
    for idx, (i, row) in enumerate(df_grave.iterrows()):
        if idx >= 10:
            break
        fail = row['fail_parts'] if pd.notna(row['fail_parts']) else '—'
        print(f"  {row['M']:4s}: {fail}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == 'all':
        commands = ['scoring', 'level', 'rotation', 'arms']
    else:
        commands = [sys.argv[1]]
    
    for cmd in commands:
        if cmd == 'level':
            analyze_level()
        elif cmd == 'rotation':
            analyze_rotation()
        elif cmd == 'scoring':
            analyze_scoring()
        elif cmd == 'arms':
            analyze_arms()
        else:
            print(f"❌ Comando desconocido: {cmd}")
            print(__doc__)
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
