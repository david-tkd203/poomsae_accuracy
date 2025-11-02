#!/usr/bin/env python3
"""
Analizar qué niveles se miden en los movimientos GRAVE
"""
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

df = pd.read_excel("reports/8yang_001_CORRECTED.xlsx")

print("=" * 100)
print("ANÁLISIS DE NIVELES EN MOVIMIENTOS GRAVE")
print("=" * 100)

print("\n1. MOVIMIENTOS CON comp_arms='GRAVE':")
print("-" * 100)
df_grave_arms = df[df['comp_arms'] == 'GRAVE']
print(f"Total: {len(df_grave_arms)}")
print(f"\nlevel(exp) distribution:")
for level, count in df_grave_arms['level(exp)'].value_counts().items():
    pct = count / len(df_grave_arms) * 100
    print(f"  {level}: {count:2d} ({pct:5.1f}%)")

print(f"\nlevel(meas) distribution:")
for level, count in df_grave_arms['level(meas)'].value_counts().items():
    pct = count / len(df_grave_arms) * 100
    print(f"  {level}: {count:2d} ({pct:5.1f}%)")

print(f"\ny_rel_end statistics en GRAVE:")
y_vals = df_grave_arms['y_rel_end'].dropna()
if len(y_vals) > 0:
    print(f"  count: {len(y_vals)}")
    print(f"  min: {y_vals.min():.4f}")
    print(f"  max: {y_vals.max():.4f}")
    print(f"  mean: {y_vals.mean():.4f}")
    print(f"  median: {y_vals.median():.4f}")
    print(f"  25th: {y_vals.quantile(0.25):.4f}")
    print(f"  75th: {y_vals.quantile(0.75):.4f}")

print("\n\n2. MOVIMIENTOS CON comp_arms='OK':")
print("-" * 100)
df_ok_arms = df[df['comp_arms'] == 'OK']
print(f"Total: {len(df_ok_arms)}")
print(f"\nlevel(exp) distribution:")
for level, count in df_ok_arms['level(exp)'].value_counts().items():
    pct = count / len(df_ok_arms) * 100
    print(f"  {level}: {count:2d} ({pct:5.1f}%)")

print(f"\nlevel(meas) distribution:")
for level, count in df_ok_arms['level(meas)'].value_counts().items():
    pct = count / len(df_ok_arms) * 100
    print(f"  {level}: {count:2d} ({pct:5.1f}%)")

print(f"\ny_rel_end statistics en OK:")
y_vals = df_ok_arms['y_rel_end'].dropna()
if len(y_vals) > 0:
    print(f"  count: {len(y_vals)}")
    print(f"  min: {y_vals.min():.4f}")
    print(f"  max: {y_vals.max():.4f}")
    print(f"  mean: {y_vals.mean():.4f}")
    print(f"  median: {y_vals.median():.4f}")
    print(f"  25th: {y_vals.quantile(0.25):.4f}")
    print(f"  75th: {y_vals.quantile(0.75):.4f}")

print("\n\n3. UMBRALES ACTUALES vs SUGERIDOS:")
print("-" * 100)
print("Actuales: OLGUL <= 0.25, MOMTONG (0.25-0.75), ARAE >= 0.75")
print("\nSugeridos basados en datos reales:")
ok_vals = df_ok_arms['y_rel_end'].dropna()
grave_vals = df_grave_arms['y_rel_end'].dropna()

if len(ok_vals) > 0 and len(grave_vals) > 0:
    print(f"  OK: min={ok_vals.min():.4f}, max={ok_vals.max():.4f}, median={ok_vals.median():.4f}")
    print(f"  GRAVE: min={grave_vals.min():.4f}, max={grave_vals.max():.4f}, median={grave_vals.median():.4f}")

print("\n\n4. DETALLES DE CASOS GRAVE:")
print("-" * 100)
print("Primeros 10 movimientos con comp_arms='GRAVE':")
for idx, row in df_grave_arms.head(10).iterrows():
    print(f"  M={row['M']:4s} {row['tech_es']:<35s} "
          f"exp={row['level(exp)']:>10s} meas={row['level(meas)']:>8s} "
          f"y_rel={row['y_rel_end']:>7.4f} elbow={row['elbow_median_deg']:>6.1f}°")
