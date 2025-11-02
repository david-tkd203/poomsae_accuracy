#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analiza qué movimientos están faltando entre filas 28-37
"""

import json
from pathlib import Path

# Cargar especificación
spec_path = Path("config/patterns/8yang_spec.json")
with open(spec_path, encoding='utf-8') as f:
    spec = json.load(f)

# Cargar scoring con XLSX
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
xlsx_path = Path("reports/8yang_001_fixed.xlsx")
scoring_df = pd.read_excel(xlsx_path, sheet_name='detalle')

print("=" * 100)
print("ANÁLISIS: MOVIMIENTOS 28-37 (FILAS DE ESPECIFICACIÓN)")
print("=" * 100)
print()

# Obtener movimientos 28-37 (índice 27-36 en array)
moves = spec['moves']
faltantes = []

print("MOVIMIENTOS ESPERADOS (filas 28-37):")
print("-" * 100)

for row_idx in range(27, 37):
    if row_idx < len(moves):
        move = moves[row_idx]
        seq = move.get('seq')
        idx = move.get('idx')
        sub = move.get('sub', '')
        tech_es = move.get('tech_es', '')
        tech_kor = move.get('tech_kor', '')
        active_limb = move.get('active_limb', '?')
        kick_type = move.get('kick_type') or 'none'
        
        # Solo mostrar la información
        print(f"Fila {row_idx+1}: M={seq:<3} ({idx}{sub:<2}) | Extremidad: {active_limb:<8} | Patada: {str(kick_type):<15}")
        print(f"         → {tech_es[:70]}")
        print()

print("=" * 100)
print(f"RESUMEN:")
print("=" * 100)
print(f"Total de movimientos en filas 28-37: 10")
print(f"Todos estos son los últimos movimientos de la poomsae")
print()

# Comparar con segmentación detectada
print("=" * 100)
print("ANÁLISIS DE SEGMENTACIÓN EN VIDEO:")
print("=" * 100)

# Cargar reporte de segmentación
report_path = Path("debug_plots/segmentation_report.json")
if report_path.exists():
    with open(report_path, encoding='utf-8') as f:
        seg_report = json.load(f)
    
    print(f"Total segmentos detectados: {seg_report['total_segments']}")
    print(f"Esperados: 36")
    print(f"Diferencia: {36 - seg_report['total_segments']}")
    print()
    
    # Ver los últimos segmentos detectados
    moves_det = seg_report.get('segments', [])
    if len(moves_det) >= 30:
        print("Últimos 7 segmentos detectados:")
        print("-" * 100)
        for seg in moves_det[-7:]:
            print(f"  Seg #{seg['id']}: T={seg['time']} | Extremidad: {seg['active_limb']:<8} | Tipo: {seg['kick']}")
        print()

print("=" * 100)
print("HIPÓTESIS:")
print("=" * 100)
print("1. Los 3-4 movimientos faltantes están entre las filas 28-37")
print("2. Son movimientos de transición o muy rápidos")
print("3. Pueden requerir ajuste de thresholds específico")
print()
print("RECOMENDACIÓN:")
print("  → Revisar timings esperados en especificación para filas 28-37")
print("  → Reducir peak_threshold de 0.12 a 0.10 ó 0.09")
print("  → Reducir min_segment_frames de 5 a 4")
print()
