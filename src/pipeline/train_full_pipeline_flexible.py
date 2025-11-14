"""
Script flexible para el flujo completo de entrenamiento ML:
- Permite ejecutar pasos individuales: landmarks, features, dataset, modelo
- Configuración de rutas y alias por argumentos
- Validación de archivos y manejo de errores
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
    parser = argparse.ArgumentParser(description="Flujo flexible de entrenamiento ML poomsae")
    parser.add_argument('--alias', default='8yang', help='Alias del poomsae')
    parser.add_argument('--video-dir', default='data/raw_videos/8yang/train', help='Directorio de videos de entrenamiento')
    parser.add_argument('--landmarks-dir', default='data/landmarks/8yang/train', help='Directorio de landmarks')
    parser.add_argument('--features-dir', default='data/features/8yang/train', help='Directorio de features')
    parser.add_argument('--annotations-dir', default='data/annotations/8yang/train', help='Directorio de anotaciones')
    parser.add_argument('--dataset-csv', default='data/labels/stance_labels_auto.csv', help='CSV final de entrenamiento')
    parser.add_argument('--model-out', default='data/models/stance_classifier.pkl', help='Ruta de salida del modelo ML')
    parser.add_argument('--steps', nargs='+', default=['landmarks','features','dataset','modelo'], choices=['landmarks','features','dataset','modelo'], help='Pasos a ejecutar')
    args = parser.parse_args()

    # 1. Extracción de landmarks
    if 'landmarks' in args.steps:
        if not check_exists(args.video_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de videos: {args.video_dir}")
            sys.exit(1)
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
        run_step([
            "python", "src/features/feature_extractor.py",
            "--in-root", args.landmarks_dir,
            "--out-root", args.features_dir,
            "--alias", args.alias,
            "--overwrite"
        ], "Generación de features")

    # 3. Construcción del dataset
    if 'dataset' in args.steps:
        if not check_exists(args.features_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de features: {args.features_dir}")
            sys.exit(1)
        if not check_exists(args.annotations_dir, is_dir=True):
            print(f"[ERROR] No existe el directorio de anotaciones: {args.annotations_dir}")
            sys.exit(1)
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
        run_step([
            "python", "-m", "src.model.stance_classifier",
            "--csv", args.dataset_csv,
            "--output", args.model_out
        ], "Entrenamiento del modelo ML")

    print("\nFlujo flexible finalizado. Modelo guardado en:", args.model_out)
