"""
Script integral para el flujo completo de entrenamiento ML poomsae.
- Ejecuta todos los pasos: landmarks, features, dataset, modelo
- Configuración flexible por argumentos
- Validación, logs y manejo de errores
- Permite ejecución parcial y personalizada
"""
import argparse
import subprocess
from pathlib import Path
import sys

def check_exists(path, is_dir=False):
    p = Path(path)
    if is_dir:
        return p.exists() and p.is_dir()
    return p.exists()

def run_step(cmd, desc):
    print(f"\n[{desc}] Ejecutando: {' '.join(str(c) for c in cmd)}")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Falló el paso '{desc}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flujo integral de entrenamiento ML poomsae")
    parser.add_argument('--alias', default='8yang', help='Alias del poomsae')
    parser.add_argument('--video-dir', default='data/raw_videos/8yang/train', help='Directorio de videos de entrenamiento')
    parser.add_argument('--landmarks-dir', default='data/landmarks/8yang/train', help='Directorio de landmarks')
    parser.add_argument('--features-dir', default='data/features/8yang/train', help='Directorio de features')
    parser.add_argument('--annotations-dir', default='data/annotations/8yang/train', help='Directorio de anotaciones')
    parser.add_argument('--dataset-csv', default='data/labels/stance_labels_auto.csv', help='CSV final de entrenamiento')
    parser.add_argument('--model-out', default='data/models/stance_classifier.pkl', help='Ruta de salida del modelo ML')
    parser.add_argument('--steps', nargs='+', default=['landmarks','features','dataset','modelo'], choices=['landmarks','features','dataset','modelo'], help='Pasos a ejecutar')
    parser.add_argument('--log', action='store_true', help='Mostrar logs avanzados')
    args = parser.parse_args()

    def log(msg):
        if args.log:
            print(f"[LOG] {msg}")

    # 1. Extracción de landmarks
    if 'landmarks' in args.steps:
        if not check_exists(args.video_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de videos: {args.video_dir}")
            sys.exit(1)
        log("Extrayendo landmarks...")
        run_step([
            "python", "src/tools/extract_landmarks.py",
            "--in", args.video_dir,
            "--out-root", args.landmarks_dir,
            "--alias", args.alias,
            "--overwrite"
        ], "Extracción de landmarks")

    # 2. Generación de features biomecánicos
    if 'features' in args.steps:
        if not check_exists(args.landmarks_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de landmarks: {args.landmarks_dir}")
            sys.exit(1)
        log("Generando features biomecánicos...")
        run_step([
            "python", "src/features/feature_extractor.py",
            "--in-root", args.landmarks_dir,
            "--out-root", args.features_dir,
            "--alias", args.alias,
            "--overwrite"
        ], "Generación de features")

    # 3. Construcción del dataset
    if 'dataset' in args.steps:
        # Crear carpeta de features si no existe
        Path(args.features_dir).mkdir(parents=True, exist_ok=True)
        # Crear carpeta de anotaciones si no existe
        Path(args.annotations_dir).mkdir(parents=True, exist_ok=True)
        if not check_exists(args.features_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de features: {args.features_dir}")
            sys.exit(1)
        if not check_exists(args.annotations_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de anotaciones: {args.annotations_dir}")
            sys.exit(1)
        log("Construyendo dataset de entrenamiento...")
        run_step([
            "python", "src/model/dataset_builder.py",
            "--features-root", args.features_dir,
            "--annotations-root", args.annotations_dir,
            "--out", args.dataset_csv
        ], "Construcción del dataset")

    # 4. Entrenamiento del modelo ML
    if 'modelo' in args.steps:
        if not check_exists(args.dataset_csv):
            print(f"[ERROR] No existe el CSV de entrenamiento: {args.dataset_csv}")
            sys.exit(1)
        log("Entrenando modelo ML...")
        run_step([
            "python", "-m", "src.model.stance_classifier",
            "--csv", args.dataset_csv,
            "--output", args.model_out
        ], "Entrenamiento del modelo ML")

    print("\nFlujo integral finalizado. Modelo guardado en:", args.model_out)
