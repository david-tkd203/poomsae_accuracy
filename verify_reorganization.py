#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de verificación post-reorganización.
Verifica que:
1. Estructura de carpetas es correcta
2. Imports funcionan desde nueva ubicación
3. Scripts principales son accesibles
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def check_structure():
    """Verificar estructura de carpetas"""
    print("=" * 80)
    print("VERIFICACIÓN DE ESTRUCTURA DE CARPETAS")
    print("=" * 80)
    
    checks = {
        "docs/": ROOT / "docs",
        "src/tools/": ROOT / "src" / "tools",
        "src/tools/analysis/": ROOT / "src" / "tools" / "analysis",
        "config/": ROOT / "config",
        "data/": ROOT / "data",
    }
    
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name:30} {'OK' if exists else 'FALTA'}")
        if not exists:
            all_ok = False
    
    return all_ok

def check_files():
    """Verificar archivos principales"""
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE ARCHIVOS PRINCIPALES")
    print("=" * 80)
    
    # Archivos en raíz
    root_files = {
        "README.md": ROOT / "README.md",
        "requirements.txt": ROOT / "requirements.txt",
    }
    
    print("\nEn carpeta raíz:")
    all_ok = True
    for name, path in root_files.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name:30} {'OK' if exists else 'FALTA'}")
        if not exists:
            all_ok = False
    
    # Archivos importantes en docs/
    print("\nEn docs/:")
    doc_files = list((ROOT / "docs").glob("*.md")) + list((ROOT / "docs").glob("*.txt"))
    print(f"✓ {len(doc_files)} archivos de documentación")
    
    # Archivos importantes en analysis/
    print("\nEn src/tools/analysis/:")
    analysis_files = list((ROOT / "src" / "tools" / "analysis").glob("*.py"))
    print(f"✓ {len(analysis_files)} scripts de análisis")
    
    for f in sorted(analysis_files):
        print(f"  - {f.name}")
    
    return all_ok

def check_imports():
    """Verificar que imports funcionan"""
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE IMPORTS")
    print("=" * 80)
    
    # Test 1: Importar src modules
    print("\nTest 1: Importar módulos src...")
    try:
        sys.path.insert(0, str(ROOT))
        from src.segmentation.move_capture import load_landmarks_csv
        from src.eval.spec_validator import SpecValidator
        print("✓ Imports desde src/ funcionan correctamente")
        return True
    except Exception as e:
        print(f"✗ Error en imports: {e}")
        return False

def check_scripts():
    """Verificar que scripts principales funcionan"""
    print("\n" + "=" * 80)
    print("VERIFICACIÓN DE SCRIPTS PRINCIPALES")
    print("=" * 80)
    
    scripts = [
        ("run_analysis.py", ROOT / "src" / "tools" / "analysis" / "run_analysis.py"),
        ("capture_36_moves.py", ROOT / "src" / "tools" / "capture_36_moves.py"),
        ("score_pal_yang.py", ROOT / "src" / "tools" / "score_pal_yang.py"),
    ]
    
    all_ok = True
    for name, path in scripts:
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name:30} {'OK' if exists else 'FALTA'}")
        if not exists:
            all_ok = False
    
    return all_ok

def main():
    """Ejecutar todas las verificaciones"""
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "VERIFICACIÓN POST-REORGANIZACIÓN".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print()
    
    results = {
        "Estructura": check_structure(),
        "Archivos": check_files(),
        "Imports": check_imports(),
        "Scripts": check_scripts(),
    }
    
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    all_passed = all(results.values())
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {check}")
    
    print()
    if all_passed:
        print("✓ ¡TODA LA REORGANIZACIÓN COMPLETADA CON ÉXITO!")
        print()
        print("Próximos pasos:")
        print("1. Reparar cálculo de rotación (move_capture.py línea 727)")
        print("2. Re-generar JSON con: python src/tools/capture_36_moves.py")
        print("3. Re-ejecutar scoring con: python src/tools/score_pal_yang.py")
        print("4. Analizar resultados con: python src/tools/analysis/run_analysis.py all")
    else:
        print("✗ Hay problemas por resolver")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
