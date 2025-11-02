#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extrae resumen de scoring del XLSX generado
"""

import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def summarize_xlsx(xlsx_path):
    """Resume el contenido del archivo XLSX de scoring"""
    
    if not Path(xlsx_path).exists():
        print(f"Error: Archivo no existe: {xlsx_path}")
        return
    
    try:
        # Leer Excel
        xl = pd.ExcelFile(xlsx_path)
        print(f"Hojas disponibles: {xl.sheet_names}\n")
        
        # Leer hoja principal
        df = pd.read_excel(xlsx_path, sheet_name=0)
        
        print("=" * 100)
        print("RESUMEN DE SCORING - 8YANG_001")
        print("=" * 100)
        print(f"\nTotal de movimientos en dataset: {len(df)}")
        print(f"\nColumnas: {list(df.columns)}\n")
        
        # Estadísticas básicas - Buscar columna de exactitud
        acc_col = None
        for col in df.columns:
            if 'exactitud' in col.lower() or 'acc' in col.lower():
                acc_col = col
                break
        
        if acc_col:
            try:
                acc_vals = pd.to_numeric(df[acc_col], errors='coerce')
                mean_acc = acc_vals.mean()
                print(f"Exactitud promedio ({acc_col}): {mean_acc:.4f}")
                print(f"Exactitud mínima: {acc_vals.min():.4f}")
                print(f"Exactitud máxima: {acc_vals.max():.4f}")
            except:
                print("No se pudo calcular exactitud")
        else:
            print("Columna de exactitud no encontrada")
        
        # Contar movimientos segmentados vs no segmentados
        if 'moves_detected' in df.columns:
            segmented = (df['moves_detected'] > 0).sum()
            not_segmented = (df['moves_detected'] == 0).sum()
            print(f"\nMovimientos segmentados: {segmented}")
            print(f"Movimientos NO segmentados: {not_segmented}")
            print(f"Porcentaje de éxito: {100*segmented/len(df):.1f}%")
        
        # Detalles de deductions
        ded_cols = [c for c in df.columns if 'deduction' in c.lower()]
        if ded_cols:
            print(f"\nColumnas de deductions: {ded_cols}")
            for col in ded_cols:
                avg_ded = df[col].sum()
                print(f"  {col}: Total deductions = {avg_ded:.2f}")
        
        # Mostrar primeras 10 filas
        print("\n" + "=" * 100)
        print("PRIMERAS 10 MOVIMIENTOS:")
        print("=" * 100)
        print(df.head(10).to_string())
        
        # Distribución de exactitudes
        print("\n" + "=" * 100)
        print("DISTRIBUCIÓN DE VALORES:")
        print("=" * 100)
        print(df.describe().to_string())
        
        print("\n" + "=" * 100)
        print("ANÁLISIS COMPLETO: OK")
        print("=" * 100)
        
    except Exception as e:
        print(f"Error al procesar Excel: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    xlsx_file = "reports/8yang_001_fixed.xlsx"
    summarize_xlsx(xlsx_file)
