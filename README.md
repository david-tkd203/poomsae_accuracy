# Poomsae Accuracy Assessment System# POOMSAE ACCURACYPOOMSAE ACCURACY — README DE USO Y COMANDOS



> **Automated technical accuracy analysis system for Taekwondo WT Poomsae using computer vision and machine learning.**



[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)

[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green.svg)](https://mediapipe.dev/)

[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange.svg)](https://scikit-learn.org/)

[![License](https://img.shields.io/badge/License-Academic-red.svg)](LICENSE)

## Descripción del Proyecto

---

Sistema para apoyar la evaluación de “exactitud” en Poomsae (Taekwondo WT) a partir de video. Flujo general: captura multivista (más adelante 4 cámaras + Kinect), extracción de puntos corporales (MediaPipe cuando corresponda), cálculo de medidas simples (ángulos, alineaciones, profundidad de posturas, estabilidad), reglas de deducción (−0,1 / −0,3) y un modelo clásico de ML (no deep learning) para clasificar movimiento como correcto / error leve / error grave. Incluye utilidades para: descarga masiva de videos, eliminación de fondo, pre-segmentación por actividad y etiquetado rápido, entrenamiento e inferencia offline/tiempo real.

## Overview

Este sistema permite evaluar la precisión técnica de ejecuciones de poomsae mediante el procesamiento de video y análisis de postura. El proyecto incluye:

This research project presents an objective, automated framework for evaluating technical precision in Taekwondo poomsae performances through advanced video analysis and machine learning. The system integrates computer vision, biomechanical analysis, and pattern recognition to deliver quantitative assessments of movement execution quality.

Requisitos

### Core Capabilities

- **Detección de postura corporal**: Extracción de landmarks usando MediaPipe Pose----------

- **Human Pose Estimation**: Extraction of 33 anatomical landmarks using MediaPipe Pose framework

- **Temporal Segmentation**: Automated detection and isolation of individual movement sequences- **Segmentación automática de movimientos**: Identificación y captura de movimientos individuales dentro de la ejecución- Windows 10/11 64-bit (probado en PowerShell).

- **Stance Classification**: Hybrid ML/heuristic system achieving 65.11% match rate with reference patterns

- **Pattern Validation**: Conformance verification against formal movement specifications- **Clasificación de posturas (stances)**: Sistema híbrido con clasificador ML (Random Forest) y heurísticas geométricas- Python 3.12.x recomendado.

- **Performance Analytics**: Comprehensive reporting with accuracy metrics and deduction analysis

- **Validación contra especificaciones**: Comparación de movimientos detectados contra patrones de referencia (JSON specs)  - Nota: si más adelante usas MediaPipe, evita 3.13 (no hay wheel oficial estable al momento).

### Research Context

- **Generación de reportes**: Análisis detallado con métricas de exactitud y deducciones- FFmpeg en PATH (requerido por yt-dlp para remux/conversión).

Developed as part of graduate thesis research in biomechanical analysis of martial arts techniques. The system addresses the challenge of objective technical assessment in poomsae evaluation, where traditional judging relies heavily on subjective human observation.

  - Verifica: `ffmpeg -version`

---

## Requisitos- Git (opcional).

## Table of Contents



- [System Requirements](#system-requirements)

- [Installation](#installation)- **Sistema Operativo**: Windows 10/11 64-bit (PowerShell)Creación del entorno e instalación

- [Project Architecture](#project-architecture)

- [Usage Guide](#usage-guide)- **Python**: 3.12.x recomendado (compatible con MediaPipe)----------------------------------

  - [Video Acquisition](#1-video-acquisition)

  - [Landmark Extraction](#2-landmark-extraction)- **FFmpeg**: Requerido para procesamiento de video1) Crear entorno virtual (PowerShell):

  - [Movement Segmentation](#3-movement-segmentation)

  - [Pattern Validation](#4-pattern-validation)  - Verificar: `ffmpeg -version`   python -m venv .venv

- [Classification Methods](#classification-methods)

- [Performance Analysis](#performance-analysis)- **Git**: Opcional para control de versiones   .\.venv\Scripts\Activate.ps1

- [Configuration](#configuration)

- [Troubleshooting](#troubleshooting)

- [References](#references)

## Instalación2) Actualizar herramientas base:

---

   python -m pip install --upgrade pip setuptools wheel

## System Requirements

### 1. Crear entorno virtual (PowerShell)

### Environment

- **Operating System**: Windows 10/11 (64-bit)3) Instalar dependencias principales:

- **Shell**: PowerShell 5.1 or later

- **Python**: 3.12.x (recommended for MediaPipe compatibility)```powershell   pip install -r requirements.txt

- **FFmpeg**: Latest stable release (required for video processing)

python -m venv .venv

### Hardware Recommendations

- **Processor**: Intel i5/AMD Ryzen 5 or better.\.venv\Scripts\Activate.ps14) (Opcional) Variables de entorno en `.env`:

- **RAM**: 8GB minimum, 16GB recommended

- **Storage**: 10GB free space for video data and models```   # proxies (si tu red lo requiere)

- **GPU**: Optional (CUDA-compatible for accelerated inference)

   HTTP_PROXY=http://user:pass@host:port

### Software Dependencies

### 2. Actualizar herramientas base   HTTPS_PROXY=http://user:pass@host:port

Core libraries installed via `requirements.txt`:

- `mediapipe>=0.10.0` - Pose estimation

- `opencv-python>=4.8.0` - Video processing

- `scikit-learn>=1.3.0` - Machine learning```powershell   # formato de descarga (yt-dlp)

- `numpy>=1.24.0` - Numerical computing

- `pandas>=2.0.0` - Data manipulationpython -m pip install --upgrade pip setuptools wheel   YTDLP_FORMAT=bv*+ba/b

- `yt-dlp>=2023.0.0` - Video downloading

```

---

Estructura del repo (carpetas clave)

## Installation

### 3. Instalar dependencias------------------------------------

### Quick Start

C:.

```powershell

# 1. Clone repository```powershell│  .env

git clone https://github.com/david-tkd203/poomsae_accuracy.git

cd poomsae_accuracypip install -r requirements.txt│  README.md



# 2. Create virtual environment```│  requirements.txt

python -m venv .venv

.\.venv\Scripts\Activate.ps1│



# 3. Upgrade package managers### 4. (Opcional) Configurar variables de entorno├─config

python -m pip install --upgrade pip setuptools wheel

│    default.yaml

# 4. Install dependencies

pip install -r requirements.txtCrear archivo `.env` en la raíz del proyecto:│



# 5. Verify FFmpeg installation├─data

ffmpeg -version

``````bash│  ├─annotations         # CSV de segmentos/labels



### Environment Configuration# Proxies (si tu red lo requiere)│  ├─landmarks           # puntos corporales exportados (cuando se usen)



Create `.env` file in project root (optional):HTTP_PROXY=http://user:pass@host:port│  ├─models              # modelos ML entrenados



```bashHTTPS_PROXY=http://user:pass@host:port│  ├─raw_videos          # videos originales

# Network proxy settings (if required)

HTTP_PROXY=http://user:pass@host:port│  └─raw_videos_bg       # videos con fondo eliminado

HTTPS_PROXY=http://user:pass@host:port

# Formato de descarga para yt-dlp│

# Video download format preference

YTDLP_FORMAT=bv*+ba/bYTDLP_FORMAT=bv*+ba/b└─src

```

```   │  config.py

---

   │  main_offline.py        # evaluación desde archivo

## Project Architecture

## Estructura del Proyecto   │  main_realtime.py       # evaluación desde cámara

```

poomsae-accuracy/   │  video_get.py           # CLI descargas/renombrado

│

├── config/                          # System configuration```   │  __init__.py

│   ├── default.yaml                # Main configuration file

│   └── patterns/                   # Movement specificationspoomsae-accuracy/   │

│       ├── 8yang_spec.json         # Palgwae 8 reference pattern

│       └── pose_spec.json          # Pose detection parameters├─ config/                    # Configuración del sistema   ├─dataio

│

├── data/                           # Data directory│  ├─ default.yaml           # Configuración principal   │    video_downloader.py  # motor de descargas

│   ├── annotations/                # Segmentation annotations

│   │   └── moves/                  # Movement JSON files│  └─ patterns/              # Especificaciones de poomsae   │    video_source.py      # lectura robusta de video

│   ├── labels/                     # Training datasets

│   │   └── stance_labels_auto.csv # Automated stance labels (664 samples)│     ├─ 8yang_spec.json     # Patrón de referencia para 8yang   │    report_generator.py  # generación de informes

│   ├── landmarks/                  # Extracted pose landmarks

│   │   └── 8yang/                  # Organized by poomsae form│     └─ pose_spec.json      # Configuración de pose   │    __init__.py

│   ├── models/                     # Trained ML models

│   │   └── stance_classifier_final.pkl  # Random Forest classifier (7.3 MB)│   │

│   ├── moves_ml/                   # ML-based predictions

│   └── raw_videos/                 # Source video files├─ data/                      # Datos del proyecto   ├─features

│       └── 8yang/                  # Organized by form and subset

│           ├── train/│  ├─ annotations/           # Anotaciones y segmentos   │    angles.py            # utilidades geométricas de ángulos

│           ├── val/

│           └── test/│  │  └─ moves/              # Movimientos segmentados (JSON)   │    feature_extractor.py # extracción de features por segmento

│

├── reports/                        # Analysis outputs│  ├─ labels/                # Datasets de entrenamiento   │

│   ├── comparison_baseline_vs_ml.txt    # Comparative analysis

│   └── feature_analysis.png             # Feature distribution plots│  │  └─ stance_labels_auto.csv   ├─model

│

├── src/                            # Source code│  ├─ landmarks/             # Landmarks exportados   │    dataset_builder.py   # construye dataset (features + labels)

│   ├── dataio/                     # I/O operations

│   │   ├── video_downloader.py    # YouTube/web video acquisition│  │  └─ 8yang/              # Por poomsae   │    train.py             # entrena modelo ML clásico (p.ej., RandomForest/SVM)

│   │   ├── video_source.py        # Video reading utilities

│   │   └── report_generator.py    # Report formatting│  ├─ models/                # Modelos entrenados   │    infer.py             # inferencia con modelo entrenado

│   │

│   ├── pose/                       # Pose estimation backends│  │  └─ stance_classifier_final.pkl   │

│   │   ├── mediapipe_backend.py   # MediaPipe integration

│   │   ├── csv_backend.py         # CSV landmark loader│  ├─ moves_ml/              # Predicciones con ML   ├─pose

│   │   └── base.py                # Abstract base classes

│   ││  └─ raw_videos/            # Videos originales   │    base.py

│   ├── segmentation/              # Temporal segmentation

│   │   ├── segmenter.py           # Movement boundary detection│     └─ 8yang/train/        # Organizados por alias/subset   │    csv_backend.py

│   │   └── move_capture.py        # Movement extraction pipeline

│   ││   │    mediapipe_backend.py # se integra más adelante (Python 3.12)

│   ├── features/                  # Feature engineering

│   │   ├── stance_features.py     # Geometric feature extraction├─ reports/                   # Reportes generados   │

│   │   └── angles.py              # Biomechanical angle calculations

│   ││  ├─ comparison_baseline_vs_ml.txt   ├─preprocess

│   ├── model/                     # Machine learning

│   │   └── stance_classifier.py   # Random Forest stance classifier│  └─ feature_analysis.png   │    bg_remove.py         # eliminación de fondo (Rembg/BackgroundMatting o similar)

│   │

│   ├── eval/                      # Evaluation modules│   │

│   │   ├── patterns.py            # Pattern matching algorithms

│   │   ├── spec_validator.py      # Specification validation└─ src/                       # Código fuente   ├─rules

│   │   └── stance_estimator.py    # Stance estimation logic

│   │   ├─ dataio/                # Entrada/salida de datos   │    deduction_rules.py   # mapeo a −0,1 / −0,3 por movimiento

│   └── tools/                     # Command-line utilities

│       ├── extract_landmarks.py          # Batch landmark extraction   │  ├─ video_downloader.py # Descarga de videos (yt-dlp)   │

│       ├── capture_moves.py              # Single-video segmentation

│       ├── capture_moves_batch.py        # Batch segmentation   │  └─ video_source.py     # Lectura de video   ├─segmentation

│       └── score_pal_yang.py             # Scoring engine

│   │   │    segmenter.py         # segmentación temporal (movimiento)

├── RESUMEN_ML_FINAL.md            # Technical documentation (Spanish)

├── requirements.txt               # Python dependencies   ├─ features/              # Extracción de características   │

└── README.md                      # This file

```   │  ├─ angles.py           # Cálculos geométricos   └─utils



---   │  └─ stance_features.py  # Features para clasificación        geometry.py



## Usage Guide   │        smoothing.py



### 1. Video Acquisition   ├─ model/                 # Aprendizaje automático        timing.py



#### Download from YouTube Playlist   │  └─ stance_classifier.py # Random Forest para posturas



```powershell   │Comandos útiles (PowerShell)

python -m src.video_get `

  --url "https://youtube.com/playlist?list=PLAYLIST_ID" `   ├─ pose/                  # Detección de postura----------------------------

  --out "data\raw_videos" `

  --alias 8yang `   │  ├─ mediapipe_backend.py # Backend MediaPipe

  --subset train

```   │  └─ csv_backend.py      # Backend desde CSVA) Descarga de videos y normalización de nombres



#### Download Individual Video   │-----------------------------------------------



```powershell   ├─ eval/                  # Evaluación y validación1) Descargar una playlist con alias y subset:

python -m src.video_get `

  --url "https://www.youtube.com/watch?v=VIDEO_ID" `   │  ├─ patterns.py         # Comparación con patrones   python -m src.video_get --url "https://youtube.com/playlist?list=PL4sEO8KDGlNXxRjmvUz_a3g8MGDLwv1BG" --out ".\data\raw_videos" --alias 8yang --subset train

  --out "data\raw_videos" `

  --alias 8yang `   │  ├─ spec_validator.py   # Validación de specs

  --subset val

```   │  └─ stance_estimator.py # Estimación de posturas2) Descargar un único video (YouTube u otro sitio soportado por yt-dlp):



#### Batch Download from URL List   │   python -m src.video_get --url "https://www.youtube.com/watch?v=VIDEO_ID" --out ".\data\raw_videos" --alias 8yang --subset val



```powershell   ├─ segmentation/          # Segmentación de movimientos

python -m src.video_get `

  --url-file "urls.txt" `   │  ├─ segmenter.py        # Detección de segmentos3) Descargar enlaces directos a archivo .mp4/.mkv:

  --out "data\raw_videos" `

  --alias 8yang `   │  └─ move_capture.py     # Captura de movimientos   python -m src.video_get --url "https://servidor/archivo.mp4" --out ".\data\raw_videos" --alias 8yang --subset test

  --subset train

```   │



#### Automated Renaming   └─ tools/                 # Herramientas CLI4) Descargar múltiples URLs desde un .txt (una por línea):



```powershell      ├─ extract_landmarks.py      # Extrae landmarks de videos   python -m src.video_get --url-file ".\urls.txt" --out ".\data\raw_videos" --alias 8yang --subset train

# Rename downloaded files with sequential numbering

python -m src.video_get `      ├─ capture_moves.py          # Segmenta movimientos (un video)

  --out "data\raw_videos" `

  --alias 8yang `      ├─ capture_moves_batch.py    # Procesamiento por lotes5) Renombrar SOLO lo descargado en la ejecución (con numeración):

  --rename-downloaded `

  --seq --seq-start 1 --seq-digits 3 `      └─ score_pal_yang.py         # Scoring de ejecuciones   python -m src.video_get --rename-downloaded --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}" --out ".\data\raw_videos" --alias 8yang

  --scheme "{alias}_{n:0Nd}"

```

# Rename all existing files in directory

python -m src.video_get `6) Renombrar TODOS los existentes en la carpeta destino alias/subset:

  --out "data\raw_videos" `

  --alias 8yang --subset train `## Funcionalidad Principal   python -m src.video_get --rename-existing --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}" --out ".\data\raw_videos" --alias 8yang --subset train

  --rename-existing `

  --seq --seq-start 1 --seq-digits 3 `

  --scheme "{alias}_{n:0Nd}"

```### 1. Extracción de Landmarks7) Si tus URLs están en Excel (.xlsx), conviértelas a .txt con Python:



---   python - << "PY"



### 2. Landmark ExtractionProcesa videos para extraer puntos corporales (33 landmarks de MediaPipe):   import pandas as pd



Extract 33 MediaPipe pose landmarks from video:   df = pd.read_excel(r".\links_8yang.xlsx")



```powershell```powershell   df.iloc[:,0].dropna().to_csv(r".\urls.txt", index=False, header=False)

python -m src.tools.extract_landmarks `

  --video "data\raw_videos\8yang\train\8yang_001.mp4" `python -m src.tools.extract_landmarks `   PY

  --out "data\landmarks\8yang\8yang_001.csv" `

  --fps 30  --video "data\raw_videos\8yang\train\8yang_001.mp4" `

```

  --out "data\landmarks\8yang\8yang_001.csv" `B) Eliminación de fondo (pre-procesamiento)

**Output Format**: CSV file with columns for frame number and x, y, z coordinates for each landmark.

  --fps 30-------------------------------------------

---

```1) Un solo video:

### 3. Movement Segmentation

   python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train\8yang_001.mp4" --out ".\data\raw_videos_bg\8yang\train\8yang_001_bg.mp4"

#### Single Video Processing

### 2. Segmentación de Movimientos

```powershell

python -m src.tools.capture_moves `2) Carpeta completa (si el script soporta lotes, según implementación):

  --landmarks-file "data\landmarks\8yang\8yang_001.csv" `

  --video-file "data\raw_videos\8yang\train\8yang_001.mp4" `Detecta y segmenta movimientos individuales dentro de una ejecución:   # ejemplo típico: (si tu bg_remove.py trae modo carpeta)

  --spec "config\patterns\8yang_spec.json" `

  --out "data\annotations\moves\8yang_001_moves.json"   python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train" --out ".\data\raw_videos_bg\8yang\train"

```

```powershell

#### Batch Processing (Heuristic Method)

# Un solo videoC) Segmentación previa y etiquetado rápido

```powershell

python -m src.tools.capture_moves_batch `python -m src.tools.capture_moves `------------------------------------------

  --landmarks-root "data\landmarks" `

  --videos-root "data\raw_videos" `  --landmarks-file "data\landmarks\8yang\8yang_001.csv" `1) Pre-segmentación por actividad (auto-candidatos):

  --alias "8yang" `

  --subset "train" `  --video-file "data\raw_videos\8yang\train\8yang_001.mp4" `   python -m src.annot.auto_seg --in ".\data\raw_videos\8yang\train" --out ".\data\annotations\8yang_cands.csv" --alias 8yang

  --spec "config\patterns\8yang_spec.json" `

  --out-dir "data\annotations\moves"  --spec "config\patterns\8yang_spec.json" `

```

  --out "data\annotations\moves\8yang_001_moves.json"2) Etiquetado rápido de un video contra el CSV de candidatos:

#### Batch Processing (ML Classifier)

```   python -m src.annot.quick_label --video ".\data\raw_videos\8yang\train\8yang_001.mp4" --cands ".\data\annotations\8yang_cands.csv" --out ".\data\annotations\8yang_labels.csv"

```powershell

python -m src.tools.capture_moves_batch `

  --landmarks-root "data\landmarks" `

  --videos-root "data\raw_videos" `#### Procesamiento por lotes (batch)   Atajos típicos en la ventana (según implementación):

  --alias "8yang" `

  --subset "train" `   - Espacio: play/pausa

  --spec "config\patterns\8yang_spec.json" `

  --out-dir "data\moves_ml" ````powershell   - A/D: retroceder/avanzar segmento

  --use-ml-classifier `

  --ml-model "data\models\stance_classifier_final.pkl"# Baseline (heurístico)   - Flechas: ±0.10s / ±0.50s

```

python -m src.tools.capture_moves_batch `   - 1/2/3: correcto / error_leve / error_grave

---

  --landmarks-root "data\landmarks" `   - S: guardar

### 4. Pattern Validation

  --videos-root "data\raw_videos" `

Validate segmented movements against reference specification:

  --alias "8yang" --subset "train" `D) Extracción de características y entrenamiento (ML clásico)

```powershell

python -m src.tools.score_pal_yang `  --spec "config\patterns\8yang_spec.json" `-------------------------------------------------------------

  --moves "data\moves_ml\8yang_001_moves.json" `

  --spec "config\patterns\8yang_spec.json"  --out-dir "data\annotations\moves"1) Construcción de dataset (features + labels):

```

```   python -m src.model.dataset_builder --labels ".\data\annotations\8yang_labels.csv" --raw ".\data\raw_videos" --features ".\data\features\8yang" --landmarks ".\data\landmarks\8yang"

---



## Classification Methods

```powershell2) Entrenamiento del modelo (ej. RandomForest):

The system implements two complementary approaches to stance classification:

# Con clasificador ML   python -m src.model.train --features ".\data\features\8yang" --out ".\data\models\8yang_rf.pkl" --model rf --params "{'n_estimators':300,'max_depth':20}"

### Heuristic Classifier (Baseline)

python -m src.tools.capture_moves_batch `

**Methodology**: Rule-based geometric analysis

- Ankle distance measurements (normalized by shoulder width)  --landmarks-root "data\landmarks" `3) Inferencia sobre nuevos videos etiquetados o en vivo:

- Hip positioning and alignment

- Knee angle analysis  --videos-root "data\raw_videos" `   # offline (archivo):

- Foot orientation vectors

  --alias "8yang" --subset "train" `   python -m src.model.infer --model ".\data\models\8yang_rf.pkl" --video ".\data\raw_videos\8yang\val\8yang_021.mp4" --report ".\data\reports\8yang_021.xlsx"

**Performance**:

- Match rate: **40.96%** (290/708 movements)  --spec "config\patterns\8yang_spec.json" `

- Predicts 4 stance types: `ap_kubi`, `dwit_kubi`, `beom_seogi`, `moa_seogi`

- High prediction diversity (realistic distribution)  --out-dir "data\moves_ml" `   # tiempo real (cámara 0):



**Advantages**:  --use-ml-classifier `   python -m src.main_realtime --model ".\data\models\8yang_rf.pkl" --camera 0

- No training data required

- Interpretable decision logic  --ml-model "data\models\stance_classifier_final.pkl"

- Robust to class imbalance

```E) Ejecución de los “mains”

---

---------------------------

### Machine Learning Classifier

### 3. Descarga de Videos1) Evaluación desde archivo (pipeline completo si ya hay modelo):

**Methodology**: Random Forest ensemble classifier

- **Algorithm**: RandomForestClassifier (scikit-learn)   python -m src.main_offline --model ".\data\models\8yang_rf.pkl" --video ".\data\raw_videos\8yang\test\8yang_050.mp4" --report ".\data\reports\8yang_050.xlsx"

- **Hyperparameters**: n_estimators=300, max_depth=20, class_weight='balanced_subsample'

- **Training Data**: 664 labeled movements (531 train / 133 test, 80/20 split)Descarga videos desde YouTube (playlists o individuales):



**Feature Set** (8 geometric features):2) Evaluación en vivo (más adelante integrar Kinect y multivista):

1. `ankle_dist_sw` - Ankle distance normalized by shoulder width

2. `hip_offset_x` - Horizontal hip displacement```powershell   python -m src.main_realtime --model ".\data\models\8yang_rf.pkl" --camera 0

3. `hip_offset_y` - Vertical hip displacement

4. `knee_angle_left` - Left knee flexion angle# Playlist completa

5. `knee_angle_right` - Right knee flexion angle

6. `foot_angle_left` - Left foot orientationpython -m src.video_get `Qué hace cada módulo

7. `foot_angle_right` - Right foot orientation

8. `hip_behind_feet` - Hip posterior positioning indicator  --url "https://youtube.com/playlist?list=PLAYLIST_ID" `--------------------



**Performance Metrics**:  --out ".\data\raw_videos" `- src/video_get.py

- Test Accuracy: **56.4%**

- Cross-Validation: **52.7% ± 2.2%** (5-fold)  --alias 8yang --subset train  CLI de descarga y renombrado. Usa dataio/video_downloader.py. Permite alias (“8yang”), subset (train/val/test), `--url-file`, y normalización de nombres con secuencia.

- Match rate: **65.11%** (461/708 movements)

- Predicts 3 stance types (eliminates `moa_seogi`)



**Feature Analysis** (ANOVA F-test):# Video individual- src/dataio/video_downloader.py

- Best discriminative feature: `knee_angle_left` (F=1.94, p=0.14)

- **All features**: p > 0.14 (not statistically significant at α=0.05)python -m src.video_get `  Motor de descarga. yt-dlp (YouTube/playlist/sites) o HTTP directo. Normaliza alias desde títulos (ej. “Yang 8” → “8yang”) y estructura carpetas de salida.

- **Conclusion**: Simple geometric features insufficient for robust discrimination

  --url "https://www.youtube.com/watch?v=VIDEO_ID" `

**Limitations**:

- Extreme bias toward majority class (`ap_kubi`: 86.7% of predictions)  --out ".\data\raw_videos" `- src/preprocess/bg_remove.py

- Reduced prediction diversity compared to heuristic method

- Higher match rate reflects learned bias in reference specifications rather than superior accuracy  --alias 8yang --subset val  Eliminación de fondo (matting/segmentation). Produce un video con el practicante recortado sobre fondo transparente o plano.



---



## Performance Analysis# Múltiples URLs desde archivo .txt- src/segmentation/segmenter.py



### Comparative Evaluation (30 Videos)python -m src.video_get `  Segmentación temporal por movimiento para cortar ejecuciones en unidades más limpias o detectar inicios/finales.



| Metric | Heuristic | ML Classifier | Difference |  --url-file ".\urls.txt" `

|--------|-----------|---------------|------------|

| **Match Rate** | 40.96% | 65.11% | +59.0% |  --out ".\data\raw_videos" `- src/features/feature_extractor.py y src/features/angles.py

| **Stance Diversity** | 4 types | 3 types | -25.0% |

| **ap_kubi Ratio** | 30.8% | 86.7% | +181.5% |  --alias 8yang --subset train  A partir de landmarks, calcula medidas: ángulos articulares, alineaciones, profundidad, estabilidad, etc.

| **dwit_kubi Ratio** | 52.5% | 12.4% | -76.4% |

| **beom_seogi Ratio** | 7.7% | 0.9% | -88.8% |```

| **moa_seogi Ratio** | 8.9% | 0.0% | -100.0% |

- src/rules/deduction_rules.py

### Confusion Matrix (ML Classifier - Test Set)

#### Renombrado automático  Traduce medidas por movimiento a deducciones reglamentarias: −0,1 (leve) y −0,3 (grave).

```

                Predicted

              ap_kubi  beom_seogi  dwit_kubi

Actual```powershell- src/model/dataset_builder.py

ap_kubi         69        0          8       (90% recall)

beom_seogi      11        0          1       ( 0% recall)# Renombrar videos descargados con numeración  Construye el dataset combinando segmentos etiquetados y features, listo para entrenamiento.

dwit_kubi       38        0          6       (14% recall)

```python -m src.video_get `



**Interpretation**:  --out ".\data\raw_videos" `- src/model/train.py

The ML classifier achieves higher match rates with reference specifications but demonstrates severe class imbalance. The heuristic method provides more realistic stance diversity, suggesting that the "ground truth" from specifications may itself be biased. This finding highlights the challenge of developing reliable labeled datasets for martial arts movement analysis.

  --alias 8yang `  Entrena un modelo ML clásico (RandomForest, SVM, GradientBoosting) sin deep learning. Guarda `.pkl`.

---

  --rename-downloaded --seq --seq-start 1 --seq-digits 3 `

## Configuration

  --scheme "{alias}_{n:0Nd}"- src/model/infer.py

### System Configuration (`config/default.yaml`)

  Carga modelo y predice sobre nuevos videos; opcionalmente genera reporte (Excel/JSON).

Primary configuration file containing:

- Segmentation thresholds# Renombrar todos los existentes en carpeta

- Smoothing parameters

- Movement detection sensitivitypython -m src.video_get `- src/main_offline.py / src/main_realtime.py

- Output formatting options

  --out ".\data\raw_videos" `  Entradas de alto nivel para ejecutar el pipeline sobre archivos o cámaras en tiempo real.

### Pattern Specifications (`config/patterns/8yang_spec.json`)

  --alias 8yang --subset train `

JSON specification defining reference movement sequence:

  --rename-existing --seq --seq-start 1 --seq-digits 3 `- src/pose/*

```json

{  --scheme "{alias}_{n:0Nd}"  Backends de pose. `mediapipe_backend.py` se integra cuando el entorno esté listo (Python 3.12). `csv_backend.py` permite reproducir landmarks precomputados.

  "poomsae_name": "palgwae_8",

  "total_moves": 36,```

  "moves": [

    {- src/dataio/report_generator.py

      "move_id": 0,

      "name": "joonbi",## Componentes del Sistema  Ensambla un informe con nota de exactitud, errores y marcas de tiempo (p. ej. Excel).

      "expected_stance": "moa_seogi",

      "expected_duration_range": [1.0, 2.5]

    },

    {### Clasificación de Posturas- src/utils/*

      "move_id": 1,

      "name": "olgul_maki_l",  Utilidades transversales (geometría, suavizado temporal, temporización).

      "expected_stance": "ap_kubi",

      "expected_duration_range": [0.8, 1.5]El sistema incluye dos métodos de clasificación:

    }

    // ... 34 more movementsBuenas prácticas asumidas

  ]

}#### 1. Heurístico (Baseline)-------------------------

```

- Basado en reglas geométricas (distancias, ángulos)- POO y modularización: componentes separados (descarga, preproceso, features, reglas, modelo).

**Specification Structure**:

- `move_id`: Sequential movement index- Predice 4 posturas: `ap_kubi`, `dwit_kubi`, `beom_seogi`, `moa_seogi`- No hardcodear rutas: usar argumentos de CLI y config/default.yaml.

- `name`: Movement technique name (Korean terminology)

- `expected_stance`: Reference stance classification- **Match rate con spec**: 40.96%- `.env` para proxies y parámetros sensibles.

- `expected_duration_range`: Temporal bounds in seconds

- **Ventaja**: Mayor diversidad en predicciones- Reproducibilidad: scripts que generan salidas deterministas a partir de insumos claros.

---

- Sin CNN/Deep Learning: modelos clásicos de scikit-learn.

## Generated Data Formats

#### 2. Machine Learning (Random Forest)

### Movement JSON Output

- Modelo entrenado con 664 movimientos etiquetadosSolución de problemas comunes

```json

{- 8 características geométricas (ankle_dist, hip_offset, knee_angles, etc.)-----------------------------

  "metadata": {

    "video_file": "8yang_001.mp4",- **Accuracy**: 56.4% (test), 52.7% ± 2.2% (CV)- “ModuleNotFoundError” al ejecutar `python -m src.algo`:

    "fps": 30.0,

    "total_frames": 2847,- **Match rate con spec**: 65.11%  Activa el venv y ejecuta SIEMPRE con `python -m ...` desde la raíz del repo.

    "processing_date": "2025-11-09"

  },- **Limitación**: Sesgo hacia `ap_kubi` (86.7% de predicciones)

  "moves": [

    {- yt-dlp se queja de FFmpeg:

      "move_id": 0,

      "name": "olgul_maki_l",### Feature Extractor  Instala FFmpeg y agrégalo a PATH. Reabre PowerShell.

      "frame_start": 45,

      "frame_end": 78,

      "duration_sec": 1.10,

      "stance_pred": "ap_kubi",Extrae características geométricas de landmarks:- MediaPipe no instala en 3.13:

      "confidence": 0.85,

      "match_expected": true  Usa Python 3.12.x para esa parte del pipeline.

    }

    // ... additional movements```python

  ],

  "summary": {from src.features.stance_features import extract_stance_features_from_dict- Descarga sin archivos:

    "total_moves_detected": 36,

    "match_rate": 0.6389,  Playlist privada / región restringida → usa `--cookies cookies.txt`.

    "avg_confidence": 0.78

  }# 8 features: ankle_dist_sw, hip_offset_x, hip_offset_y,  Comprueba la conectividad y el formato `YTDLP_FORMAT` en `.env`.

}

```# knee_angle_left, knee_angle_right, foot_angle_left,



---# foot_angle_right, hip_behind_feet- Videos H.265/HEVC no reproducen:



## Command-Line Interface Referencefeatures = extract_stance_features_from_dict(landmarks_dict, frame_idx)  Forzar `--format "bv*+ba/b"` o convertir con FFmpeg a H.264/AAC.



### Available Tools```



| Command | Purpose | Usage Context |Roadmap breve

|---------|---------|---------------|

| `src.video_get` | Video acquisition and renaming | Data collection |### Validador de Especificaciones-------------

| `src.tools.extract_landmarks` | MediaPipe pose extraction | Preprocessing |

| `src.tools.capture_moves` | Single-video segmentation | Individual analysis |- Integrar Kinect (profundidad) y multivista.

| `src.tools.capture_moves_batch` | Batch segmentation | Large-scale processing |

| `src.tools.score_pal_yang` | Pattern validation and scoring | Performance evaluation |Compara movimientos detectados con patrones de referencia:- Completar backend de pose con MediaPipe en 3.12.



### Complete Pipeline Example- Añadir evaluaciones cruzadas con jueces y métricas: F1, MAE, ICC, latencia.



```powershell```json- Extender a más poomsae y a Pareja/Equipo.

# Step 1: Download training videos

python -m src.video_get `// config/patterns/8yang_spec.json

  --url-file "training_urls.txt" `

  --out "data\raw_videos" `{# Playlist YouTube → carpeta data/raw_videos/8yang/train

  --alias 8yang --subset train

  "moves": [python -m src.video_get --url "https://youtube.com/playlist?list=PL4sEO8KDGlNXxRjmvUz_a3g8MGDLwv1BG" --out ".\data\raw_videos" --alias 8yang --subset train

# Step 2: Extract pose landmarks

python -m src.tools.extract_landmarks `    {"name": "joonbi", "expected_stance": "moa_seogi"},

  --video "data\raw_videos\8yang\train\8yang_001.mp4" `

  --out "data\landmarks\8yang\8yang_001.csv"    {"name": "olgul_maki_l", "expected_stance": "ap_kubi"},# Video individual (YouTube)



# Step 3: Segment movements with ML classifier    ...python -m src.video_get --url "https://www.youtube.com/watch?v=VIDEO_ID" --out ".\data\raw_videos" --alias 8yang --subset val

python -m src.tools.capture_moves `

  --landmarks-file "data\landmarks\8yang\8yang_001.csv" `  ]

  --video-file "data\raw_videos\8yang\train\8yang_001.mp4" `

  --spec "config\patterns\8yang_spec.json" `}# Enlace directo a .mp4/.mkv

  --out "data\moves_ml\8yang_001_moves.json" `

  --use-ml-classifier ````python -m src.video_get --url "https://servidor/archivo.mp4" --out ".\data\raw_videos" --alias 8yang --subset test

  --ml-model "data\models\stance_classifier_final.pkl"



# Step 4: Validate and score

python -m src.tools.score_pal_yang `## Análisis de Rendimiento# Varias URLs desde .txt (una por línea)

  --moves "data\moves_ml\8yang_001_moves.json" `

  --spec "config\patterns\8yang_spec.json"python -m src.video_get --url-file ".\urls.txt" --out ".\data\raw_videos" --alias 8yang --subset train

```

### Comparación Baseline vs ML (30 videos)

---

# Evitar inferencia de alias por título

## Troubleshooting

| Métrica | Baseline | ML | Mejora |python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-normalize-from-title

### Common Issues

|---------|----------|----|----|

#### ModuleNotFoundError

| **Match rate con spec** | 40.96% | 65.11% | +59% |# No usar yt-dlp primero (preferir HTTP directo si aplica)

```powershell

# Ensure virtual environment is activated| **Diversidad de posturas** | 4 | 3 | -1 |python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-ytdlp-first

.\.venv\Scripts\Activate.ps1

| **Predicciones ap_kubi** | 30.8% | 86.7% | +182% |

# Always use module syntax from project root

python -m src.tools.capture_moves ...# No omitir existentes (re-descarga)

```

**Nota**: El ML mejora el match rate pero muestra sesgo hacia la clase mayoritaria. El clasificador heurístico ofrece mayor diversidad realista.python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-skip-existing

#### FFmpeg Not Found



```bash

# Verify FFmpeg installation### Resultados del Modelo ML# Cookies (si la playlist lo requiere)

ffmpeg -version

python -m src.video_get --url "URL" --out ".\data\raw_videos" --cookies ".\cookies.txt"

# Windows: Add FFmpeg to System PATH

# Download from: https://ffmpeg.org/download.html- **Test Accuracy**: 56.4%

```

- **Cross-Validation**: 52.7% ± 2.2% (5-fold)# Renombrar SOLO lo descargado en esta ejecución (con numeración)

#### MediaPipe Installation Failed (Python 3.13)

- **Dataset**: 664 movimientos (531 train / 133 test)python -m src.video_get --out ".\data\raw_videos" --alias 8yang --rename-downloaded --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}"

```bash

# MediaPipe requires Python 3.12.x or lower- **Algoritmo**: Random Forest (n_estimators=300, max_depth=20)

python --version

# Renombrar TODOS los existentes en alias/subset (sin descargar nada)

# If 3.13 detected, install Python 3.12:

# https://www.python.org/downloads/release/python-3120/**Análisis de características**:python -m src.video_get --out ".\data\raw_videos" --alias 8yang --subset train --rename-existing --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}"

```

- Todas las 8 features geométricas mostraron **p-value > 0.14** (ANOVA)

#### YouTube Download Errors

- Feature más discriminativa: `knee_angle_left` (F=1.94, p=0.14)# Simulación de renombrado (sin cambios)

```powershell

# Use cookies for private/restricted playlists- Conclusión: Características simples insuficientes para discriminación robustapython -m src.video_get --out ".\data\raw_videos" --alias 8yang --rename-existing --seq --dry-run

python -m src.video_get `

  --url "PLAYLIST_URL" `

  --cookies "cookies.txt"

## Archivos de Configuración# Un video → salida con fondo removido

# Verify network connectivity and proxy settings in .env

```python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train\8yang_001.mp4" --out ".\data\raw_videos_bg\8yang\train\8yang_001_bg.mp4"



#### Low Classification Accuracy### `config/default.yaml`



**Potential causes**:# Carpeta completa (si el script soporta directorio)

1. **Insufficient training data**: Expand labeled dataset beyond 664 samples

2. **Feature engineering**: Consider temporal features (velocity, acceleration)Configuración principal del sistema (thresholds, parámetros de segmentación)python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train" --out ".\data\raw_videos_bg\8yang\train"

3. **Camera angle variation**: Standardize recording conditions

4. **Lighting conditions**: Apply preprocessing normalization



**Recommendations**:### `config/patterns/8yang_spec.json`# Genera dataset tabular a partir de labels + landmarks/features

- Manual labeling of 200+ additional samples

- Implement hybrid classifier (ML + heuristics)python -m src.model.dataset_builder ^

- Add domain-specific feature engineering

- Explore temporal sequence models (LSTM/GRU)Especificación de referencia para el poomsae 8yang:  --labels ".\data\annotations\8yang_labels.csv" ^



---- 36 movimientos definidos  --landmarks ".\data\landmarks\8yang" ^



## Future Development- Postura esperada por movimiento  --features-out ".\data\features\8yang_dataset.parquet" ^



### Planned Enhancements- Usado para validación y scoring  --fps-target 30 --window 0.50 --hop 0.25 --norm zscore



- **Multi-camera fusion**: Integration of 4-camera setup with depth sensing (Kinect)

- **Temporal modeling**: LSTM/Transformer architectures for sequence classification

- **Real-time processing**: Optimized pipeline for live video analysis## Datos Generados# RandomForest con validación

- **Extended poomsae coverage**: Support for Taegeuk, Koryo, Keumgang forms

- **Transition analysis**: Inter-movement flow and rhythm assessmentpython -m src.model.train ^

- **Power metrics**: Integration of acceleration-based power estimation

### `data/annotations/moves/`  --dataset ".\data\features\8yang_dataset.parquet" ^

### Research Extensions

  --model-out ".\data\models\rf_8yang.joblib" ^

- **Inter-rater reliability**: Comparison with expert judge assessments

- **Cross-cultural validation**: Dataset expansion with international practitionersArchivos JSON con movimientos segmentados:  --clf random_forest ^

- **Explainable AI**: Saliency mapping for interpretable feedback

- **Mobile deployment**: Edge device optimization for on-site evaluation  --cv 5 --seed 42



---```json



## References{# SVM lineal



### Technical Documentation  "fps": 30.0,python -m src.model.train --dataset ".\data\features\8yang_dataset.parquet" --model-out ".\data\models\svm_8yang.joblib" --clf svm_linear --cv 5 --seed 42



- **ML Implementation Details**: See `RESUMEN_ML_FINAL.md` for comprehensive analysis  "moves": [

- **Comparative Analysis**: `reports/comparison_baseline_vs_ml.txt`

- **Feature Analysis**: `reports/feature_analysis.png`    {# Regresión logística



### Citations      "name": "olgul_maki_l",python -m src.model.train --dataset ".\data\features\8yang_dataset.parquet" --model-out ".\data\models\logreg_8yang.joblib" --clf logreg --cv 5 --seed 42



- MediaPipe Pose: [Lugaresi et al., 2019](https://arxiv.org/abs/1906.08172)      "frame_start": 45,

- Random Forest: [Breiman, 2001](https://link.springer.com/article/10.1023/A:1010933404324)

- Taekwondo Technical Standards: World Taekwondo (WT) Competition Rules      "frame_end": 78,# A partir de video con pipeline integrado (pose → features → predicción → reporte)



---      "stance_pred": "ap_kubi",python -m src.model.infer ^



## License      "confidence": 0.85  --video ".\data\raw_videos\8yang\val\8yang_010.mp4" ^



This software is developed for academic research purposes.     }  --model ".\data\models\rf_8yang.joblib" ^



**Usage Restrictions**:  ]  --report ".\data\annotations\reports\8yang_010_report.xlsx" ^

- Educational and research use permitted

- Commercial use requires explicit authorization}  --config ".\config\default.yaml"

- Cite this work in derivative research

```

**Third-Party Licenses**:

Respect licenses of dependencies:# A partir de landmarks ya calculados

- MediaPipe (Apache 2.0)

- OpenCV (Apache 2.0)### `data/models/stance_classifier_final.pkl`python -m src.model.infer --landmarks ".\data\landmarks\8yang\8yang_010.csv" --model ".\data\models\rf_8yang.joblib" --report ".\data\annotations\reports\8yang_010_report.xlsx"

- scikit-learn (BSD 3-Clause)

- yt-dlp (Unlicense)

- FFmpeg (LGPL/GPL)

Modelo Random Forest serializado (7.3 MB):# Procesa uno o varios videos y emite reportes (offline)

---

- RandomForestClassifierpython -m src.main_offline ^

## Contact & Contribution

- LabelEncoder (3 clases)  --in ".\data\raw_videos\8yang\val" ^

**Author**: David (Thesis Project)  

**Institution**: [Your University]  - Feature names (8 features)  --out ".\data\annotations\reports" ^

**Year**: 2025

  --model ".\data\models\rf_8yang.joblib" ^

### Acknowledgments

### `reports/comparison_baseline_vs_ml.txt`  --config ".\config\default.yaml"

Special thanks to the Taekwondo community for technical guidance and video contributions. This research contributes to the broader goal of objective, technology-assisted martial arts assessment.



---

Reporte comparativo detallado con:# Webcam índice 0 con overlay de deducciones (offline)

**Project Status**: Active Development | Last Updated: November 2025

- Match rates por videopython -m src.main_realtime --camera 0 --model ".\data\models\rf_8yang.joblib" --config ".\config\default.yaml"

- Distribución de posturas

- Análisis de mejora/sesgo# Stream RTSP/archivo (en tiempo real)

python -m src.main_realtime --source "rtsp://usuario:pass@ip/stream" --model ".\data\models\rf_8yang.joblib" --config ".\config\default.yaml"

## Pipeline Completo (Ejemplo)



```powershell

# 1. Descargar videosCréditos y licencia

python -m src.video_get --url-file "urls.txt" --out "data\raw_videos" --alias 8yang --subset train-------------------

Uso académico. Cita y respeta licencias de terceros (yt-dlp, FFmpeg, OpenCV, scikit-learn, etc.).

# 2. Extraer landmarks
python -m src.tools.extract_landmarks --video "data\raw_videos\8yang\train\8yang_001.mp4" --out "data\landmarks\8yang\8yang_001.csv"

# 3. Segmentar movimientos (con ML)
python -m src.tools.capture_moves --landmarks-file "data\landmarks\8yang\8yang_001.csv" --video-file "data\raw_videos\8yang\train\8yang_001.mp4" --spec "config\patterns\8yang_spec.json" --out "data\moves_ml\8yang_001_moves.json" --use-ml-classifier --ml-model "data\models\stance_classifier_final.pkl"

# 4. Validar contra especificación (scoring)
python -m src.tools.score_pal_yang --moves "data\moves_ml\8yang_001_moves.json" --spec "config\patterns\8yang_spec.json"
```

## Herramientas CLI Disponibles

| Script | Función |
|--------|---------|
| `src.video_get` | Descarga y renombrado de videos |
| `src.tools.extract_landmarks` | Extracción de landmarks con MediaPipe |
| `src.tools.capture_moves` | Segmentación de movimientos (un video) |
| `src.tools.capture_moves_batch` | Procesamiento por lotes (múltiples videos) |
| `src.tools.score_pal_yang` | Scoring y validación contra spec |

## Solución de Problemas

### Error: ModuleNotFoundError

```powershell
# Asegúrate de activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# Ejecuta siempre con python -m desde la raíz
python -m src.tools.capture_moves ...
```

### FFmpeg no encontrado

```bash
# Instalar FFmpeg y agregarlo al PATH
# Verificar: ffmpeg -version
```

### MediaPipe no instala en Python 3.13

```bash
# Usar Python 3.12.x
python --version  # Debe mostrar 3.12.x
```

### Videos no descargan

```bash
# Playlist privada → usar cookies
python -m src.video_get --url "URL" --cookies "cookies.txt"

# Verificar conectividad y formato en .env
YTDLP_FORMAT=bv*+ba/b
```

## Trabajo Futuro

- Integración de características temporales para mejorar clasificación
- Etiquetado manual de ground truth confiable (200+ muestras)
- Clasificador híbrido (ML + heurísticas)
- Soporte para más poomsae (Koryo, Keumgang, etc.)
- Análisis de transiciones entre movimientos
- Evaluación de timing y potencia

## Documentación Técnica

Para más detalles sobre la implementación del clasificador ML, consultar:
- `RESUMEN_ML_FINAL.md` - Resumen ejecutivo completo del sistema ML

## Licencia

Uso académico. Respeta las licencias de las dependencias utilizadas (MediaPipe, OpenCV, scikit-learn, yt-dlp, FFmpeg).

---

**Desarrollado como parte de proyecto de tesis - Taekwondo WT Poomsae Analysis**
