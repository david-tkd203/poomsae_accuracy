#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script temporal para crear un reporte simulado del evaluador 2
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Leer reporte original
df_det = pd.read_excel('reports/8yang_010_score.xlsx', sheet_name='detalle')
df_res = pd.read_excel('reports/8yang_010_score.xlsx', sheet_name='resumen')

print(f"📄 Reporte original: {len(df_det)} movimientos")
print(f"   Exactitud original: {df_res['exactitud_final'].iloc[0]:.2f}")
print()

# Modificar aleatoriamente 5 evaluaciones (cambiar OK a LEVE)
indices = np.random.choice(len(df_det), size=5, replace=False)
componentes = ['comp_arms', 'comp_legs', 'comp_kick']

print("🔄 Modificando evaluaciones para simular evaluador 2:")
for idx in indices:
    comp = np.random.choice(componentes)
    original = df_det.loc[idx, comp]
    
    if original == 'OK':
        df_det.loc[idx, comp] = 'LEVE'
        comp_name = comp.replace('comp_', '').upper()
        print(f"   • Índice {idx}: {comp_name} cambiado de OK → LEVE")

# Recalcular exactitud
brazos_ok = (df_det['comp_arms'] == 'OK').sum()
piernas_ok = (df_det['comp_legs'] == 'OK').sum()
patadas_ok = (df_det['comp_kick'] == 'OK').sum()
total_ok = brazos_ok + piernas_ok + patadas_ok

exactitud = total_ok / 108 * 4.0
df_res.loc[0, 'exactitud_final'] = round(exactitud, 2)

print()
print(f"📊 Nueva exactitud del evaluador 2: {exactitud:.2f}")
print()

# Guardar reporte del evaluador 2
output_path = 'reports/8yang_010_evaluador2.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    df_det.to_excel(writer, sheet_name='detalle', index=False)
    df_res.to_excel(writer, sheet_name='resumen', index=False)

print(f"✅ Reporte del evaluador 2 guardado: {output_path}")
