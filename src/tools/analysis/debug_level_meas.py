#!/usr/bin/env python3
"""
Debug script para ver qué niveles se están midiendo vs. esperados
"""
import json
import pandas as pd
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cargar el XLSX de scoring
xlsx_path = Path("reports/8yang_001_final.xlsx")
df = pd.read_excel(xlsx_path)

print("=" * 100)
print("ANÁLISIS DE MEDICIONES DE NIVEL (level)")
print("=" * 100)

# Filtrar columnas relevantes
cols_of_interest = ['M', 'tech_es', 'level(exp)', 'level(meas)', 'comp_arms', 'pen_arms', 'y_rel_end']
df_nivel = df[cols_of_interest].copy()

# Ver ejemplos de brazos marcados como GRAVE
print("\n1. MOVIMIENTOS DE BRAZOS MARCADOS COMO 'GRAVE':")
print("-" * 100)
df_grave = df[df['comp_arms'] == 'GRAVE']
print(f"Total con comp_arms=GRAVE: {len(df_grave)}")
print("\nPrimeros 5 ejemplos:")
print(df_grave[cols_of_interest].head(10).to_string())

print("\n\n2. CASOS DONDE level(meas) != level(exp):")
print("-" * 100)
df_level_mismatch = df[df['level(meas)'] != df['level(exp)']]
print(f"Total con mismatch: {len(df_level_mismatch)}")
print("\nDetalles:")
for idx, row in df_level_mismatch.iterrows():
    print(f"  M={row['M']}: {row['tech_es']:<30} "
          f"level_exp={row['level(exp)']:>8} "
          f"level_meas={row['level(meas)']:>8} "
          f"y_rel_end={row['y_rel_end']:>8} "
          f"comp_arms={row['comp_arms']:>5}")

print("\n\n3. ESTADÍSTICAS DE y_rel_end POR NIVEL ESPERADO:")
print("-" * 100)
for level_exp in ['ARAE', 'MOMTONG', 'OLGUL']:
    subset = df[df['level(exp)'] == level_exp]
    if len(subset) > 0:
        y_vals = subset['y_rel_end'].dropna()
        print(f"\nlevel(exp)={level_exp}: n={len(subset)}, n_con_medida={len(y_vals)}")
        if len(y_vals) > 0:
            print(f"  y_rel_end: min={y_vals.min():.4f}, max={y_vals.max():.4f}, "
                  f"mean={y_vals.mean():.4f}, median={y_vals.median():.4f}")
            print(f"  Rango ARAE: < {-0.02}")
            print(f"  Rango MOMTONG: [{-0.02}, {1.03})")
            print(f"  Rango ARAE: >= {1.03}")

print("\n\n4. MATRIZ DE CONFUSIÓN (level(exp) vs level(meas)):")
print("-" * 100)
matrix = pd.crosstab(df['level(exp)'], df['level(meas)'], margins=True)
print(matrix)

print("\n\n5. RESUMEN DE PROBLEMAS:")
print("-" * 100)
# Contar cuántos GRAVE por categoría
grave_by_cat = {}
for tech_kor, tech_es in zip(df['tech_kor'], df['tech_es']):
    if pd.isna(tech_kor):
        tech_kor = ""
    if "chagi" in f"{tech_kor} {tech_es}".lower() or "patada" in tech_es.lower():
        cat = "KICK"
    elif "makki" in f"{tech_kor} {tech_es}".lower() or "sonnal" in tech_es.lower():
        cat = "BLOCK"
    elif "jireugi" in f"{tech_kor} {tech_es}".lower():
        cat = "STRIKE"
    else:
        cat = "ARMS"
    if cat not in grave_by_cat:
        grave_by_cat[cat] = {"OK": 0, "LEVE": 0, "GRAVE": 0}
    grave_by_cat[cat][df.loc[df['tech_es'] == tech_es, 'comp_arms'].iloc[0]] += 1

for cat, counts in grave_by_cat.items():
    total = sum(counts.values())
    grave_pct = counts['GRAVE'] / total * 100 if total > 0 else 0
    print(f"{cat:8s}: OK={counts['OK']:2d}, LEVE={counts['LEVE']:2d}, GRAVE={counts['GRAVE']:2d} ({grave_pct:5.1f}%)")
