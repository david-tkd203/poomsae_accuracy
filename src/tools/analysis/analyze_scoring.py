#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analiza qué está mal en el scoring de score_pal_yang
"""

import pandas as pd
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cargar datos
xlsx_path = Path("reports/8yang_001_final.xlsx")
df = pd.read_excel(xlsx_path, sheet_name='detalle')

print("=" * 100)
print("ANÁLISIS DETALLADO DE SCORING - score_pal_yang")
print("=" * 100)
print()

# 1. Verificar que tenemos 36 movimientos
print(f"Total movimientos en XLSX: {len(df)}")
print()

# 2. Ver las primeras filas
print("PRIMERAS 10 FILAS DEL SCORING:")
print("-" * 100)
for idx, row in df.head(10).iterrows():
    M = row.get('M', 'N/A')
    tech_es = str(row.get('tech_es', 'N/A'))[:50]
    comp_arms = row.get('comp_arms', 'N/A')
    comp_legs = row.get('comp_legs', 'N/A')
    comp_kick = row.get('comp_kick', 'N/A')
    is_correct = row.get('is_correct', 'N/A')
    fail_parts = row.get('fail_parts', 'N/A')
    
    print(f"\nM={M}: {tech_es}")
    print(f"  Brazos: {comp_arms} | Piernas: {comp_legs} | Patada: {comp_kick}")
    print(f"  ¿Correcto?: {is_correct} | Errores: {fail_parts}")

print("\n" + "=" * 100)
print("ANÁLISIS DE COMPONENTES:")
print("=" * 100)

# 3. Estadísticas de componentes
print("\nComp_arms (brazos):")
print(f"  OK: {(df['comp_arms'] == 'OK').sum()}")
print(f"  GRAVE: {(df['comp_arms'] == 'GRAVE').sum()}")

print("\nComp_legs (piernas):")
print(f"  OK: {(df['comp_legs'] == 'OK').sum()}")
print(f"  GRAVE: {(df['comp_legs'] == 'GRAVE').sum()}")

print("\nComp_kick (patadas):")
print(f"  OK: {(df['comp_kick'] == 'OK').sum()}")
print(f"  GRAVE: {(df['comp_kick'] == 'GRAVE').sum()}")

# 4. Exactitud
print("\n" + "=" * 100)
print("EXACTITUD:")
print("=" * 100)
print(f"\nExactitud acumulada promedio: {df['exactitud_acum'].mean():.4f}")
print(f"Movimientos correctos (is_correct=yes): {(df['is_correct'] == 'yes').sum()}/36")
print(f"Movimientos incorrectos: {(df['is_correct'] == 'no').sum()}/36")

# 5. Deductions
print("\n" + "=" * 100)
print("DEDUCTIONS:")
print("=" * 100)
print(f"\nPen_arms (deduction brazos):")
print(f"  Total: {df['pen_arms'].sum():.2f}")
print(f"  Promedio: {df['pen_arms'].mean():.4f}")

print(f"\nPen_legs (deduction piernas):")
print(f"  Total: {df['pen_legs'].sum():.2f}")
print(f"  Promedio: {df['pen_legs'].mean():.4f}")

print(f"\nPen_kick (deduction patadas):")
print(f"  Total: {df['pen_kick'].sum():.2f}")
print(f"  Promedio: {df['pen_kick'].mean():.4f}")

print(f"\nDed_total_move (total deduction por movimiento):")
print(f"  Total: {df['ded_total_move'].sum():.2f}")
print(f"  Promedio: {df['ded_total_move'].mean():.4f}")

# 6. Problemas potenciales
print("\n" + "=" * 100)
print("PROBLEMAS DETECTADOS:")
print("=" * 100)

issues = []

# Verificar si todos los movimientos tienen M
if df['M'].isna().any():
    issues.append("❌ Algunos movimientos no tienen valor M")

# Verificar si hay filas con 'NO_SEG'
if 'NO_SEG' in str(df['tech_es'].values):
    count_noseg = (df['tech_es'].astype(str).str.contains('NO_SEG', na=False)).sum()
    issues.append(f"❌ Hay {count_noseg} filas con 'NO_SEG' (no segmentados)")

# Verificar si exactitud es muy baja
if df['exactitud_acum'].mean() < 2.0:
    issues.append(f"⚠️  Exactitud promedio muy baja: {df['exactitud_acum'].mean():.2f}/10")

# Verificar si hay demasiadas deductions
if df['ded_total_move'].sum() > 15:
    issues.append(f"⚠️  Demasiadas deductions totales: {df['ded_total_move'].sum():.2f}")

# Verificar si todos los movimientos son "GRAVE"
grave_count = ((df['comp_arms'] == 'GRAVE') | (df['comp_legs'] == 'GRAVE') | (df['comp_kick'] == 'GRAVE')).sum()
if grave_count > 20:
    issues.append(f"⚠️  Demasiados movimientos con errores GRAVE: {grave_count}/36")

if issues:
    for issue in issues:
        print(f"  {issue}")
else:
    print("  ✅ No hay problemas evidentes")

print("\n" + "=" * 100)
print("RECOMENDACIONES:")
print("=" * 100)

print("""
1. Si hay muchas deductions (-0.3):
   → El scoring puede estar siendo demasiado estricto
   → Revisar los parámetros de tolerancia en score_pal_yang.py
   
2. Si exactitud es baja:
   → Verificar si la clasificación de brazos/piernas es correcta
   → Revisar si los landmarks están bien posicionados
   
3. Si hay NO_SEG:
   → Significa que algunos movimientos no se segmentaron
   → Nuestro fix de 36 movimientos debe haber resuelto esto
   
4. Si detectas problemas específicos:
   → Consulta el archivo score_pal_yang.py
   → Revisa la clasificación de movimientos (brazos vs piernas)
""")

print("=" * 100)
