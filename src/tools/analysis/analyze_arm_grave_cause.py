#!/usr/bin/env python3
"""
Analizar por qué hay GRAVE en brazos
"""
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

df = pd.read_excel("reports/8yang_001_CORRECTED.xlsx")

print("=" * 100)
print("ANÁLISIS DE CAUSA RAÍZ DE GRAVE EN BRAZOS")
print("=" * 100)

# Ver los detalles completos de movimientos GRAVE
print("\n1. DETALLES DE MOVIMIENTOS GRAVE:")
print("-" * 100)

df_grave = df[df['comp_arms'] == 'GRAVE'].sort_values('M')

cols_to_show = ['M', 'tech_es', 'level(meas)', 'rot(meas_deg)', 'elbow_median_deg', 'fail_parts']
print(df_grave[cols_to_show].to_string())

print("\n\n2. ANÁLISIS DE COMPONENTES DE BRAZOS:")
print("-" * 100)
print("Mostrar 'fail_parts' para ver qué falló:")
for idx, row in df_grave.iterrows():
    if pd.notna(row['fail_parts']) and row['fail_parts'] != 'OK':
        print(f"  M={row['M']:4s}: {row['fail_parts']}")

print("\n\n3. ESTADÍSTICAS DE ROTACIÓN:")
print("-" * 100)
rot_vals = df_grave['rot(meas_deg)'].dropna()
print(f"Rotación medida en GRAVE: min={rot_vals.min():.1f}°, max={rot_vals.max():.1f}°, mean={rot_vals.mean():.1f}°")

print("\n\n4. ESTADÍSTICAS DE EXTENSIÓN DE CODO:")
print("-" * 100)
elbow_vals = df_grave['elbow_median_deg'].dropna()
print(f"Extensión de codo en GRAVE: min={elbow_vals.min():.1f}°, max={elbow_vals.max():.1f}°, mean={elbow_vals.mean():.1f}°")
print(f"Umbrales: OK >= 150°, LEVE >= 120°, GRAVE < 120°")

print("\n\n5. MOVIMIENTOS OK (para comparar):")
print("-" * 100)
df_ok = df[df['comp_arms'] == 'OK']
print("Primeros 5:")
print(df_ok[cols_to_show].head().to_string())

print("\n\n6. COMPARATIVA DE CODOS (OK vs GRAVE):")
print("-" * 100)
ok_elbow = df_ok['elbow_median_deg'].dropna()
grave_elbow = df_grave['elbow_median_deg'].dropna()

print(f"OK:    min={ok_elbow.min():.1f}°, max={ok_elbow.max():.1f}°, mean={ok_elbow.mean():.1f}°, median={ok_elbow.median():.1f}°")
print(f"GRAVE: min={grave_elbow.min():.1f}°, max={grave_elbow.max():.1f}°, mean={grave_elbow.mean():.1f}°, median={grave_elbow.median():.1f}°")
