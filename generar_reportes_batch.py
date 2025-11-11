#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generar reportes en batch para múltiples videos
"""
import subprocess
import sys
from pathlib import Path

# Lista de TODOS los videos disponibles (excepto 006 que ya fue procesado)
videos_to_process = [
    "8yang_001",
    "8yang_002", 
    "8yang_003",
    "8yang_004",
    "8yang_005",
    "8yang_007",
    "8yang_008",
    "8yang_009",
    "8yang_010",
    "8yang_011",
    "8yang_012",
    "8yang_013",
    "8yang_014",
    "8yang_015",
    "8yang_016",
    "8yang_017",
    "8yang_018",
    "8yang_019",
    "8yang_020",
    "8yang_021",
    "8yang_022",
    "8yang_023",
    "8yang_024",
    "8yang_025",
    "8yang_026",
    "8yang_027",
    "8yang_028",
    "8yang_029",
]

print("="*80)
print("🔄 GENERACIÓN BATCH DE REPORTES DE SCORING")
print("="*80)
print()
print(f"Videos a procesar: {len(videos_to_process)}")
print()

results = []

for i, video_id in enumerate(videos_to_process, 1):
    print(f"[{i}/{len(videos_to_process)}] Procesando {video_id}...")
    
    moves_json = f"data/moves_ml/8yang_{video_id}_moves.json"
    out_xlsx = f"reports/{video_id}_score.xlsx"
    
    # Verificar que existe el archivo de movimientos
    if not Path(moves_json).exists():
        print(f"   ⚠️  Archivo no encontrado: {moves_json}")
        results.append((video_id, "ERROR", "Archivo no encontrado"))
        continue
    
    try:
        # Ejecutar el scoring
        cmd = [
            sys.executable, "-m", "src.tools.score_pal_yang",
            "--moves-json", moves_json,
            "--spec", "config/patterns/8yang_spec.json",
            "--out-xlsx", out_xlsx,
            "--landmarks-root", "data/landmarks",
            "--alias", "8yang",
            "--subset", "train"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✅ Reporte generado: {out_xlsx}")
            results.append((video_id, "OK", out_xlsx))
        else:
            print(f"   ❌ Error al generar reporte")
            print(f"      {result.stderr[:200]}")
            results.append((video_id, "ERROR", result.stderr[:100]))
    except Exception as e:
        print(f"   ❌ Excepción: {str(e)[:100]}")
        results.append((video_id, "ERROR", str(e)[:100]))
    
    print()

# Resumen
print("="*80)
print("📊 RESUMEN DE GENERACIÓN")
print("="*80)
print()

exitosos = [r for r in results if r[1] == "OK"]
errores = [r for r in results if r[1] == "ERROR"]

print(f"✅ Exitosos: {len(exitosos)}/{len(videos_to_process)}")
print(f"❌ Errores:  {len(errores)}/{len(videos_to_process)}")
print()

if exitosos:
    print("Videos procesados correctamente:")
    for video_id, status, path in exitosos:
        print(f"   • {video_id} → {path}")
    print()

if errores:
    print("Videos con errores:")
    for video_id, status, error in errores:
        print(f"   • {video_id}: {error[:80]}")
    print()

print("="*80)
