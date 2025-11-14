"""
Script automatizado para el flujo completo:
1. Extracción de landmarks
2. Generación de features biomecánicos
3. Construcción del dataset (features + etiquetas)
4. Entrenamiento del modelo ML
"""
import subprocess
from pathlib import Path

# Configuración de rutas y alias
alias = "8yang"
video_dir = Path("data/raw_videos/8yang/train")
landmarks_dir = Path(f"data/landmarks/{alias}/train")
features_dir = Path(f"data/features/{alias}/train")
annotations_dir = Path(f"data/annotations/{alias}/train")
dataset_csv = Path(f"data/labels/stance_labels_auto.csv")
model_out = Path(f"data/models/stance_classifier.pkl")

# 1. Extracción de landmarks
print("[1] Extrayendo landmarks...")
subprocess.run([
    "python", "src/tools/extract_landmarks.py",
    "--in", str(video_dir),
    "--out-root", str(landmarks_dir),
    "--alias", alias,
    "--overwrite"
], check=True)

# 2. Generación de features biomecánicos
print("[2] Generando features biomecánicos...")
subprocess.run([
    "python", "src/features/feature_extractor.py",
    "--in-root", str(landmarks_dir),
    "--out-root", str(features_dir),
    "--alias", alias,
    "--overwrite"
], check=True)

# 3. Construcción del dataset (features + etiquetas)
print("[3] Construyendo dataset de entrenamiento...")
subprocess.run([
    "python", "src/model/dataset_builder.py",
    "--features-root", str(features_dir),
    "--annotations-root", str(annotations_dir),
    "--out", str(dataset_csv)
], check=True)

# 4. Entrenamiento del modelo ML
print("[4] Entrenando modelo ML...")
subprocess.run([
    "python", "-m", "src.model.stance_classifier",
    "--csv", str(dataset_csv),
    "--output", str(model_out)
], check=True)

print("\nFlujo completo finalizado. Modelo guardado en:", model_out)

# Sugerencias de mejora:
# - Validar existencia de archivos antes de cada paso.
# - Permitir configuración flexible de rutas y parámetros.
# - Integrar logs y manejo de errores más detallado.
# - Permitir ejecución parcial de pasos (flags opcionales).
# - Mejorar modularidad de los scripts para facilitar importación directa en vez de subprocess.
