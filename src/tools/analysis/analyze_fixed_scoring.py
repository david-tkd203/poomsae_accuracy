#!/usr/bin/env python3
"""
Analizar los resultados del scoring con umbrales corregidos
"""
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Cargar el XLSX de scoring con umbrales corregidos
xlsx_path = Path("reports/8yang_001_CORRECTED.xlsx")
df = pd.read_excel(xlsx_path)

print("=" * 100)
print("ANÁLISIS DE SCORING CON UMBRALES CORREGIDOS - 36 MOVIMIENTOS")
print("=" * 100)

# Resumen general
print("\n1. RESUMEN GENERAL:")
print("-" * 100)
print(f"Total de movimientos: {len(df)}")
print(f"Movimientos correctos (is_correct=yes): {(df['is_correct'] == 'yes').sum()}")
print(f"Movimientos incorrectos: {(df['is_correct'] == 'no').sum()}")
print(f"Exactitud acumulada promedio: {df['exactitud_acum'].mean():.3f}/10")

# Contar severidad por componente
print("\n2. CLASIFICACIÓN DE COMPONENTES:")
print("-" * 100)
for col in ['comp_arms', 'comp_legs', 'comp_kick']:
    print(f"\n{col}:")
    counts = df[col].value_counts()
    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"  {val:10s}: {cnt:3d} ({pct:5.1f}%)")

# Ver ejemplos de movimientos correctos
print("\n3. MOVIMIENTOS CORRECTOS (is_correct=yes):")
print("-" * 100)
df_correct = df[df['is_correct'] == 'yes']
print(f"Total: {len(df_correct)}")
if len(df_correct) > 0:
    print("\nDetalles:")
    for idx, row in df_correct.iterrows():
        print(f"  {row['M']:4s}: {row['tech_es']:<50s} "
              f"exactitud={row['exactitud_acum']:>5.3f} "
              f"arms={row['comp_arms']:>5s} legs={row['comp_legs']:>5s} kick={row['comp_kick']:>5s}")

# Ver qué GRAVE persisten
print("\n4. MOVIMIENTOS CON ERRORES (is_correct=no):")
print("-" * 100)
df_wrong = df[df['is_correct'] == 'no']
print(f"Total: {len(df_wrong)}")

# Contar GRAVE por componente
grave_counts = {
    'arms': (df['comp_arms'] == 'GRAVE').sum(),
    'legs': (df['comp_legs'] == 'GRAVE').sum(),
    'kick': (df['comp_kick'] == 'GRAVE').sum(),
}
print(f"\nErrores GRAVE por componente:")
for comp, cnt in grave_counts.items():
    pct = cnt / len(df) * 100
    print(f"  {comp}: {cnt:2d} ({pct:5.1f}%)")

# Ver cambios comparado con run anterior
print("\n5. MATRIZ DE CONFUSIÓN (level(exp) vs level(meas)):")
print("-" * 100)
matrix = pd.crosstab(df['level(exp)'], df['level(meas)'], margins=True)
print(matrix)

# Contar NS (no segmentados)
print("\n6. MOVIMIENTOS SIN SEGMENTACIÓN (NS):")
print("-" * 100)
ns_count = (df['comp_arms'] == 'NS').sum()
print(f"Total NS: {ns_count}")

print("\n7. EXACTITUD POR MOVIMIENTO:")
print("-" * 100)
print(f"Min: {df['exactitud_acum'].min():.3f}")
print(f"Max: {df['exactitud_acum'].max():.3f}")
print(f"Mean: {df['exactitud_acum'].mean():.3f}")
print(f"Median: {df['exactitud_acum'].median():.3f}")

print("\n" + "=" * 100)
print("COMPARACIÓN ANTES VS AHORA")
print("=" * 100)
print("ANTES (26 movimientos con umbrales originales):")
print("  Correctos: 2/26 (7.7%)")
print("  Grave brazos: 16/26 (61.5%)")
print("  Grave piernas: 9/26 (34.6%)")
print("  Exactitud promedio: 1.78/10")
print("\nAHORA (36 movimientos con umbrales corregidos):")
print(f"  Correctos: {(df['is_correct'] == 'yes').sum()}/36 ({(df['is_correct'] == 'yes').sum()/36*100:.1f}%)")
print(f"  Grave brazos: {grave_counts['arms']}/36 ({grave_counts['arms']/36*100:.1f}%)")
print(f"  Grave piernas: {grave_counts['legs']}/36 ({grave_counts['legs']/36*100:.1f}%)")
print(f"  Exactitud promedio: {df['exactitud_acum'].mean():.2f}/10")

print("\n" + "=" * 100)

# Resumen general
print("\n1. RESUMEN GENERAL:")
print("-" * 100)
print(f"Total de movimientos: {len(df)}")
print(f"Movimientos correctos (is_correct=yes): {(df['is_correct'] == 'yes').sum()}")
print(f"Movimientos incorrectos: {(df['is_correct'] == 'no').sum()}")
print(f"Exactitud acumulada promedio: {df['exactitud_acum'].mean():.3f}/10")

# Contar severidad por componente
print("\n2. CLASIFICACIÓN DE COMPONENTES:")
print("-" * 100)
for col in ['comp_arms', 'comp_legs', 'comp_kick']:
    print(f"\n{col}:")
    counts = df[col].value_counts()
    for val, cnt in counts.items():
        pct = cnt / len(df) * 100
        print(f"  {val:10s}: {cnt:3d} ({pct:5.1f}%)")

# Ver ejemplos de movimientos correctos
print("\n3. MOVIMIENTOS CORRECTOS (is_correct=yes):")
print("-" * 100)
df_correct = df[df['is_correct'] == 'yes']
print(f"Total: {len(df_correct)}")
if len(df_correct) > 0:
    print("\nDetalles:")
    for idx, row in df_correct.iterrows():
        print(f"  {row['M']:4s}: {row['tech_es']:<50s} "
              f"exactitud={row['exactitud_acum']:>5.3f} "
              f"arms={row['comp_arms']:>5s} legs={row['comp_legs']:>5s} kick={row['comp_kick']:>5s}")

# Ver qué GRAVE persisten
print("\n4. MOVIMIENTOS CON ERRORES (is_correct=no):")
print("-" * 100)
df_wrong = df[df['is_correct'] == 'no']
print(f"Total: {len(df_wrong)}")

# Contar GRAVE por componente
grave_counts = {
    'arms': (df['comp_arms'] == 'GRAVE').sum(),
    'legs': (df['comp_legs'] == 'GRAVE').sum(),
    'kick': (df['comp_kick'] == 'GRAVE').sum(),
}
print(f"\nErrores GRAVE por componente:")
for comp, cnt in grave_counts.items():
    pct = cnt / len(df) * 100
    print(f"  {comp}: {cnt:2d} ({pct:5.1f}%)")

# Ver cambios comparado con run anterior
print("\n5. MATRIZ DE CONFUSIÓN (level(exp) vs level(meas)):")
print("-" * 100)
matrix = pd.crosstab(df['level(exp)'], df['level(meas)'], margins=True)
print(matrix)

# Contar NS (no segmentados)
print("\n6. MOVIMIENTOS SIN SEGMENTACIÓN (NS):")
print("-" * 100)
ns_count = (df['comp_arms'] == 'NS').sum()
print(f"Total NS: {ns_count}")
if ns_count > 0:
    df_ns = df[df['comp_arms'] == 'NS']
    print("Detalles:")
    for idx, row in df_ns.iterrows():
        print(f"  {row['M']:4s}: {row['tech_es']:<50s}")

print("\n7. EXACTITUD POR MOVIMIENTO:")
print("-" * 100)
print(f"Min: {df['exactitud_acum'].min():.3f}")
print(f"Max: {df['exactitud_acum'].max():.3f}")
print(f"Mean: {df['exactitud_acum'].mean():.3f}")
print(f"Median: {df['exactitud_acum'].median():.3f}")

print("\n" + "=" * 100)
print("CONCLUSIÓN: Comparar con el análisis anterior")
print("=" * 100)
print("Anterior:")
print("  Correctos: 2/36 (5.5%)")
print("  Grave brazos: 16/36")
print("  Grave piernas: 9/36")
print("  Exactitud promedio: 1.78/10")
print("\nActual:")
print(f"  Correctos: {(df['is_correct'] == 'yes').sum()}/36 ({(df['is_correct'] == 'yes').sum()/36*100:.1f}%)")
print(f"  Grave brazos: {grave_counts['arms']}/36")
print(f"  Grave piernas: {grave_counts['legs']}/36")
print(f"  Exactitud promedio: {df['exactitud_acum'].mean():.2f}/10")
