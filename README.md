# POOMSAE ACCURACYPOOMSAE ACCURACY — README DE USO Y COMANDOS

==============================================

Sistema automatizado para análisis y evaluación de exactitud técnica en Poomsae (Taekwondo WT) basado en análisis de video y aprendizaje automático.

Resumen del proyecto

## Descripción del Proyecto--------------------

Sistema para apoyar la evaluación de “exactitud” en Poomsae (Taekwondo WT) a partir de video. Flujo general: captura multivista (más adelante 4 cámaras + Kinect), extracción de puntos corporales (MediaPipe cuando corresponda), cálculo de medidas simples (ángulos, alineaciones, profundidad de posturas, estabilidad), reglas de deducción (−0,1 / −0,3) y un modelo clásico de ML (no deep learning) para clasificar movimiento como correcto / error leve / error grave. Incluye utilidades para: descarga masiva de videos, eliminación de fondo, pre-segmentación por actividad y etiquetado rápido, entrenamiento e inferencia offline/tiempo real.

Este sistema permite evaluar la precisión técnica de ejecuciones de poomsae mediante el procesamiento de video y análisis de postura. El proyecto incluye:

Requisitos

- **Detección de postura corporal**: Extracción de landmarks usando MediaPipe Pose----------

- **Segmentación automática de movimientos**: Identificación y captura de movimientos individuales dentro de la ejecución- Windows 10/11 64-bit (probado en PowerShell).

- **Clasificación de posturas (stances)**: Sistema híbrido con clasificador ML (Random Forest) y heurísticas geométricas- Python 3.12.x recomendado.

- **Validación contra especificaciones**: Comparación de movimientos detectados contra patrones de referencia (JSON specs)  - Nota: si más adelante usas MediaPipe, evita 3.13 (no hay wheel oficial estable al momento).

- **Generación de reportes**: Análisis detallado con métricas de exactitud y deducciones- FFmpeg en PATH (requerido por yt-dlp para remux/conversión).

  - Verifica: `ffmpeg -version`

## Requisitos- Git (opcional).



- **Sistema Operativo**: Windows 10/11 64-bit (PowerShell)Creación del entorno e instalación

- **Python**: 3.12.x recomendado (compatible con MediaPipe)----------------------------------

- **FFmpeg**: Requerido para procesamiento de video1) Crear entorno virtual (PowerShell):

  - Verificar: `ffmpeg -version`   python -m venv .venv

- **Git**: Opcional para control de versiones   .\.venv\Scripts\Activate.ps1



## Instalación2) Actualizar herramientas base:

   python -m pip install --upgrade pip setuptools wheel

### 1. Crear entorno virtual (PowerShell)

3) Instalar dependencias principales:

```powershell   pip install -r requirements.txt

python -m venv .venv

.\.venv\Scripts\Activate.ps14) (Opcional) Variables de entorno en `.env`:

```   # proxies (si tu red lo requiere)

   HTTP_PROXY=http://user:pass@host:port

### 2. Actualizar herramientas base   HTTPS_PROXY=http://user:pass@host:port



```powershell   # formato de descarga (yt-dlp)

python -m pip install --upgrade pip setuptools wheel   YTDLP_FORMAT=bv*+ba/b

```

Estructura del repo (carpetas clave)

### 3. Instalar dependencias------------------------------------

C:.

```powershell│  .env

pip install -r requirements.txt│  README.md

```│  requirements.txt

│

### 4. (Opcional) Configurar variables de entorno├─config

│    default.yaml

Crear archivo `.env` en la raíz del proyecto:│

├─data

```bash│  ├─annotations         # CSV de segmentos/labels

# Proxies (si tu red lo requiere)│  ├─landmarks           # puntos corporales exportados (cuando se usen)

HTTP_PROXY=http://user:pass@host:port│  ├─models              # modelos ML entrenados

HTTPS_PROXY=http://user:pass@host:port│  ├─raw_videos          # videos originales

│  └─raw_videos_bg       # videos con fondo eliminado

# Formato de descarga para yt-dlp│

YTDLP_FORMAT=bv*+ba/b└─src

```   │  config.py

   │  main_offline.py        # evaluación desde archivo

## Estructura del Proyecto   │  main_realtime.py       # evaluación desde cámara

   │  video_get.py           # CLI descargas/renombrado

```   │  __init__.py

poomsae-accuracy/   │

├─ config/                    # Configuración del sistema   ├─dataio

│  ├─ default.yaml           # Configuración principal   │    video_downloader.py  # motor de descargas

│  └─ patterns/              # Especificaciones de poomsae   │    video_source.py      # lectura robusta de video

│     ├─ 8yang_spec.json     # Patrón de referencia para 8yang   │    report_generator.py  # generación de informes

│     └─ pose_spec.json      # Configuración de pose   │    __init__.py

│   │

├─ data/                      # Datos del proyecto   ├─features

│  ├─ annotations/           # Anotaciones y segmentos   │    angles.py            # utilidades geométricas de ángulos

│  │  └─ moves/              # Movimientos segmentados (JSON)   │    feature_extractor.py # extracción de features por segmento

│  ├─ labels/                # Datasets de entrenamiento   │

│  │  └─ stance_labels_auto.csv   ├─model

│  ├─ landmarks/             # Landmarks exportados   │    dataset_builder.py   # construye dataset (features + labels)

│  │  └─ 8yang/              # Por poomsae   │    train.py             # entrena modelo ML clásico (p.ej., RandomForest/SVM)

│  ├─ models/                # Modelos entrenados   │    infer.py             # inferencia con modelo entrenado

│  │  └─ stance_classifier_final.pkl   │

│  ├─ moves_ml/              # Predicciones con ML   ├─pose

│  └─ raw_videos/            # Videos originales   │    base.py

│     └─ 8yang/train/        # Organizados por alias/subset   │    csv_backend.py

│   │    mediapipe_backend.py # se integra más adelante (Python 3.12)

├─ reports/                   # Reportes generados   │

│  ├─ comparison_baseline_vs_ml.txt   ├─preprocess

│  └─ feature_analysis.png   │    bg_remove.py         # eliminación de fondo (Rembg/BackgroundMatting o similar)

│   │

└─ src/                       # Código fuente   ├─rules

   ├─ dataio/                # Entrada/salida de datos   │    deduction_rules.py   # mapeo a −0,1 / −0,3 por movimiento

   │  ├─ video_downloader.py # Descarga de videos (yt-dlp)   │

   │  └─ video_source.py     # Lectura de video   ├─segmentation

   │   │    segmenter.py         # segmentación temporal (movimiento)

   ├─ features/              # Extracción de características   │

   │  ├─ angles.py           # Cálculos geométricos   └─utils

   │  └─ stance_features.py  # Features para clasificación        geometry.py

   │        smoothing.py

   ├─ model/                 # Aprendizaje automático        timing.py

   │  └─ stance_classifier.py # Random Forest para posturas

   │Comandos útiles (PowerShell)

   ├─ pose/                  # Detección de postura----------------------------

   │  ├─ mediapipe_backend.py # Backend MediaPipe

   │  └─ csv_backend.py      # Backend desde CSVA) Descarga de videos y normalización de nombres

   │-----------------------------------------------

   ├─ eval/                  # Evaluación y validación1) Descargar una playlist con alias y subset:

   │  ├─ patterns.py         # Comparación con patrones   python -m src.video_get --url "https://youtube.com/playlist?list=PL4sEO8KDGlNXxRjmvUz_a3g8MGDLwv1BG" --out ".\data\raw_videos" --alias 8yang --subset train

   │  ├─ spec_validator.py   # Validación de specs

   │  └─ stance_estimator.py # Estimación de posturas2) Descargar un único video (YouTube u otro sitio soportado por yt-dlp):

   │   python -m src.video_get --url "https://www.youtube.com/watch?v=VIDEO_ID" --out ".\data\raw_videos" --alias 8yang --subset val

   ├─ segmentation/          # Segmentación de movimientos

   │  ├─ segmenter.py        # Detección de segmentos3) Descargar enlaces directos a archivo .mp4/.mkv:

   │  └─ move_capture.py     # Captura de movimientos   python -m src.video_get --url "https://servidor/archivo.mp4" --out ".\data\raw_videos" --alias 8yang --subset test

   │

   └─ tools/                 # Herramientas CLI4) Descargar múltiples URLs desde un .txt (una por línea):

      ├─ extract_landmarks.py      # Extrae landmarks de videos   python -m src.video_get --url-file ".\urls.txt" --out ".\data\raw_videos" --alias 8yang --subset train

      ├─ capture_moves.py          # Segmenta movimientos (un video)

      ├─ capture_moves_batch.py    # Procesamiento por lotes5) Renombrar SOLO lo descargado en la ejecución (con numeración):

      └─ score_pal_yang.py         # Scoring de ejecuciones   python -m src.video_get --rename-downloaded --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}" --out ".\data\raw_videos" --alias 8yang

```

6) Renombrar TODOS los existentes en la carpeta destino alias/subset:

## Funcionalidad Principal   python -m src.video_get --rename-existing --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}" --out ".\data\raw_videos" --alias 8yang --subset train



### 1. Extracción de Landmarks7) Si tus URLs están en Excel (.xlsx), conviértelas a .txt con Python:

   python - << "PY"

Procesa videos para extraer puntos corporales (33 landmarks de MediaPipe):   import pandas as pd

   df = pd.read_excel(r".\links_8yang.xlsx")

```powershell   df.iloc[:,0].dropna().to_csv(r".\urls.txt", index=False, header=False)

python -m src.tools.extract_landmarks `   PY

  --video "data\raw_videos\8yang\train\8yang_001.mp4" `

  --out "data\landmarks\8yang\8yang_001.csv" `B) Eliminación de fondo (pre-procesamiento)

  --fps 30-------------------------------------------

```1) Un solo video:

   python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train\8yang_001.mp4" --out ".\data\raw_videos_bg\8yang\train\8yang_001_bg.mp4"

### 2. Segmentación de Movimientos

2) Carpeta completa (si el script soporta lotes, según implementación):

Detecta y segmenta movimientos individuales dentro de una ejecución:   # ejemplo típico: (si tu bg_remove.py trae modo carpeta)

   python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train" --out ".\data\raw_videos_bg\8yang\train"

```powershell

# Un solo videoC) Segmentación previa y etiquetado rápido

python -m src.tools.capture_moves `------------------------------------------

  --landmarks-file "data\landmarks\8yang\8yang_001.csv" `1) Pre-segmentación por actividad (auto-candidatos):

  --video-file "data\raw_videos\8yang\train\8yang_001.mp4" `   python -m src.annot.auto_seg --in ".\data\raw_videos\8yang\train" --out ".\data\annotations\8yang_cands.csv" --alias 8yang

  --spec "config\patterns\8yang_spec.json" `

  --out "data\annotations\moves\8yang_001_moves.json"2) Etiquetado rápido de un video contra el CSV de candidatos:

```   python -m src.annot.quick_label --video ".\data\raw_videos\8yang\train\8yang_001.mp4" --cands ".\data\annotations\8yang_cands.csv" --out ".\data\annotations\8yang_labels.csv"



#### Procesamiento por lotes (batch)   Atajos típicos en la ventana (según implementación):

   - Espacio: play/pausa

```powershell   - A/D: retroceder/avanzar segmento

# Baseline (heurístico)   - Flechas: ±0.10s / ±0.50s

python -m src.tools.capture_moves_batch `   - 1/2/3: correcto / error_leve / error_grave

  --landmarks-root "data\landmarks" `   - S: guardar

  --videos-root "data\raw_videos" `

  --alias "8yang" --subset "train" `D) Extracción de características y entrenamiento (ML clásico)

  --spec "config\patterns\8yang_spec.json" `-------------------------------------------------------------

  --out-dir "data\annotations\moves"1) Construcción de dataset (features + labels):

```   python -m src.model.dataset_builder --labels ".\data\annotations\8yang_labels.csv" --raw ".\data\raw_videos" --features ".\data\features\8yang" --landmarks ".\data\landmarks\8yang"



```powershell2) Entrenamiento del modelo (ej. RandomForest):

# Con clasificador ML   python -m src.model.train --features ".\data\features\8yang" --out ".\data\models\8yang_rf.pkl" --model rf --params "{'n_estimators':300,'max_depth':20}"

python -m src.tools.capture_moves_batch `

  --landmarks-root "data\landmarks" `3) Inferencia sobre nuevos videos etiquetados o en vivo:

  --videos-root "data\raw_videos" `   # offline (archivo):

  --alias "8yang" --subset "train" `   python -m src.model.infer --model ".\data\models\8yang_rf.pkl" --video ".\data\raw_videos\8yang\val\8yang_021.mp4" --report ".\data\reports\8yang_021.xlsx"

  --spec "config\patterns\8yang_spec.json" `

  --out-dir "data\moves_ml" `   # tiempo real (cámara 0):

  --use-ml-classifier `   python -m src.main_realtime --model ".\data\models\8yang_rf.pkl" --camera 0

  --ml-model "data\models\stance_classifier_final.pkl"

```E) Ejecución de los “mains”

---------------------------

### 3. Descarga de Videos1) Evaluación desde archivo (pipeline completo si ya hay modelo):

   python -m src.main_offline --model ".\data\models\8yang_rf.pkl" --video ".\data\raw_videos\8yang\test\8yang_050.mp4" --report ".\data\reports\8yang_050.xlsx"

Descarga videos desde YouTube (playlists o individuales):

2) Evaluación en vivo (más adelante integrar Kinect y multivista):

```powershell   python -m src.main_realtime --model ".\data\models\8yang_rf.pkl" --camera 0

# Playlist completa

python -m src.video_get `Qué hace cada módulo

  --url "https://youtube.com/playlist?list=PLAYLIST_ID" `--------------------

  --out ".\data\raw_videos" `- src/video_get.py

  --alias 8yang --subset train  CLI de descarga y renombrado. Usa dataio/video_downloader.py. Permite alias (“8yang”), subset (train/val/test), `--url-file`, y normalización de nombres con secuencia.



# Video individual- src/dataio/video_downloader.py

python -m src.video_get `  Motor de descarga. yt-dlp (YouTube/playlist/sites) o HTTP directo. Normaliza alias desde títulos (ej. “Yang 8” → “8yang”) y estructura carpetas de salida.

  --url "https://www.youtube.com/watch?v=VIDEO_ID" `

  --out ".\data\raw_videos" `- src/preprocess/bg_remove.py

  --alias 8yang --subset val  Eliminación de fondo (matting/segmentation). Produce un video con el practicante recortado sobre fondo transparente o plano.



# Múltiples URLs desde archivo .txt- src/segmentation/segmenter.py

python -m src.video_get `  Segmentación temporal por movimiento para cortar ejecuciones en unidades más limpias o detectar inicios/finales.

  --url-file ".\urls.txt" `

  --out ".\data\raw_videos" `- src/features/feature_extractor.py y src/features/angles.py

  --alias 8yang --subset train  A partir de landmarks, calcula medidas: ángulos articulares, alineaciones, profundidad, estabilidad, etc.

```

- src/rules/deduction_rules.py

#### Renombrado automático  Traduce medidas por movimiento a deducciones reglamentarias: −0,1 (leve) y −0,3 (grave).



```powershell- src/model/dataset_builder.py

# Renombrar videos descargados con numeración  Construye el dataset combinando segmentos etiquetados y features, listo para entrenamiento.

python -m src.video_get `

  --out ".\data\raw_videos" `- src/model/train.py

  --alias 8yang `  Entrena un modelo ML clásico (RandomForest, SVM, GradientBoosting) sin deep learning. Guarda `.pkl`.

  --rename-downloaded --seq --seq-start 1 --seq-digits 3 `

  --scheme "{alias}_{n:0Nd}"- src/model/infer.py

  Carga modelo y predice sobre nuevos videos; opcionalmente genera reporte (Excel/JSON).

# Renombrar todos los existentes en carpeta

python -m src.video_get `- src/main_offline.py / src/main_realtime.py

  --out ".\data\raw_videos" `  Entradas de alto nivel para ejecutar el pipeline sobre archivos o cámaras en tiempo real.

  --alias 8yang --subset train `

  --rename-existing --seq --seq-start 1 --seq-digits 3 `- src/pose/*

  --scheme "{alias}_{n:0Nd}"  Backends de pose. `mediapipe_backend.py` se integra cuando el entorno esté listo (Python 3.12). `csv_backend.py` permite reproducir landmarks precomputados.

```

- src/dataio/report_generator.py

## Componentes del Sistema  Ensambla un informe con nota de exactitud, errores y marcas de tiempo (p. ej. Excel).



### Clasificación de Posturas- src/utils/*

  Utilidades transversales (geometría, suavizado temporal, temporización).

El sistema incluye dos métodos de clasificación:

Buenas prácticas asumidas

#### 1. Heurístico (Baseline)-------------------------

- Basado en reglas geométricas (distancias, ángulos)- POO y modularización: componentes separados (descarga, preproceso, features, reglas, modelo).

- Predice 4 posturas: `ap_kubi`, `dwit_kubi`, `beom_seogi`, `moa_seogi`- No hardcodear rutas: usar argumentos de CLI y config/default.yaml.

- **Match rate con spec**: 40.96%- `.env` para proxies y parámetros sensibles.

- **Ventaja**: Mayor diversidad en predicciones- Reproducibilidad: scripts que generan salidas deterministas a partir de insumos claros.

- Sin CNN/Deep Learning: modelos clásicos de scikit-learn.

#### 2. Machine Learning (Random Forest)

- Modelo entrenado con 664 movimientos etiquetadosSolución de problemas comunes

- 8 características geométricas (ankle_dist, hip_offset, knee_angles, etc.)-----------------------------

- **Accuracy**: 56.4% (test), 52.7% ± 2.2% (CV)- “ModuleNotFoundError” al ejecutar `python -m src.algo`:

- **Match rate con spec**: 65.11%  Activa el venv y ejecuta SIEMPRE con `python -m ...` desde la raíz del repo.

- **Limitación**: Sesgo hacia `ap_kubi` (86.7% de predicciones)

- yt-dlp se queja de FFmpeg:

### Feature Extractor  Instala FFmpeg y agrégalo a PATH. Reabre PowerShell.



Extrae características geométricas de landmarks:- MediaPipe no instala en 3.13:

  Usa Python 3.12.x para esa parte del pipeline.

```python

from src.features.stance_features import extract_stance_features_from_dict- Descarga sin archivos:

  Playlist privada / región restringida → usa `--cookies cookies.txt`.

# 8 features: ankle_dist_sw, hip_offset_x, hip_offset_y,  Comprueba la conectividad y el formato `YTDLP_FORMAT` en `.env`.

# knee_angle_left, knee_angle_right, foot_angle_left,

# foot_angle_right, hip_behind_feet- Videos H.265/HEVC no reproducen:

features = extract_stance_features_from_dict(landmarks_dict, frame_idx)  Forzar `--format "bv*+ba/b"` o convertir con FFmpeg a H.264/AAC.

```

Roadmap breve

### Validador de Especificaciones-------------

- Integrar Kinect (profundidad) y multivista.

Compara movimientos detectados con patrones de referencia:- Completar backend de pose con MediaPipe en 3.12.

- Añadir evaluaciones cruzadas con jueces y métricas: F1, MAE, ICC, latencia.

```json- Extender a más poomsae y a Pareja/Equipo.

// config/patterns/8yang_spec.json

{# Playlist YouTube → carpeta data/raw_videos/8yang/train

  "moves": [python -m src.video_get --url "https://youtube.com/playlist?list=PL4sEO8KDGlNXxRjmvUz_a3g8MGDLwv1BG" --out ".\data\raw_videos" --alias 8yang --subset train

    {"name": "joonbi", "expected_stance": "moa_seogi"},

    {"name": "olgul_maki_l", "expected_stance": "ap_kubi"},# Video individual (YouTube)

    ...python -m src.video_get --url "https://www.youtube.com/watch?v=VIDEO_ID" --out ".\data\raw_videos" --alias 8yang --subset val

  ]

}# Enlace directo a .mp4/.mkv

```python -m src.video_get --url "https://servidor/archivo.mp4" --out ".\data\raw_videos" --alias 8yang --subset test



## Análisis de Rendimiento# Varias URLs desde .txt (una por línea)

python -m src.video_get --url-file ".\urls.txt" --out ".\data\raw_videos" --alias 8yang --subset train

### Comparación Baseline vs ML (30 videos)

# Evitar inferencia de alias por título

| Métrica | Baseline | ML | Mejora |python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-normalize-from-title

|---------|----------|----|----|

| **Match rate con spec** | 40.96% | 65.11% | +59% |# No usar yt-dlp primero (preferir HTTP directo si aplica)

| **Diversidad de posturas** | 4 | 3 | -1 |python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-ytdlp-first

| **Predicciones ap_kubi** | 30.8% | 86.7% | +182% |

# No omitir existentes (re-descarga)

**Nota**: El ML mejora el match rate pero muestra sesgo hacia la clase mayoritaria. El clasificador heurístico ofrece mayor diversidad realista.python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-skip-existing



### Resultados del Modelo ML# Cookies (si la playlist lo requiere)

python -m src.video_get --url "URL" --out ".\data\raw_videos" --cookies ".\cookies.txt"

- **Test Accuracy**: 56.4%

- **Cross-Validation**: 52.7% ± 2.2% (5-fold)# Renombrar SOLO lo descargado en esta ejecución (con numeración)

- **Dataset**: 664 movimientos (531 train / 133 test)python -m src.video_get --out ".\data\raw_videos" --alias 8yang --rename-downloaded --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}"

- **Algoritmo**: Random Forest (n_estimators=300, max_depth=20)

# Renombrar TODOS los existentes en alias/subset (sin descargar nada)

**Análisis de características**:python -m src.video_get --out ".\data\raw_videos" --alias 8yang --subset train --rename-existing --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}"

- Todas las 8 features geométricas mostraron **p-value > 0.14** (ANOVA)

- Feature más discriminativa: `knee_angle_left` (F=1.94, p=0.14)# Simulación de renombrado (sin cambios)

- Conclusión: Características simples insuficientes para discriminación robustapython -m src.video_get --out ".\data\raw_videos" --alias 8yang --rename-existing --seq --dry-run



## Archivos de Configuración# Un video → salida con fondo removido

python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train\8yang_001.mp4" --out ".\data\raw_videos_bg\8yang\train\8yang_001_bg.mp4"

### `config/default.yaml`

# Carpeta completa (si el script soporta directorio)

Configuración principal del sistema (thresholds, parámetros de segmentación)python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train" --out ".\data\raw_videos_bg\8yang\train"



### `config/patterns/8yang_spec.json`# Genera dataset tabular a partir de labels + landmarks/features

python -m src.model.dataset_builder ^

Especificación de referencia para el poomsae 8yang:  --labels ".\data\annotations\8yang_labels.csv" ^

- 36 movimientos definidos  --landmarks ".\data\landmarks\8yang" ^

- Postura esperada por movimiento  --features-out ".\data\features\8yang_dataset.parquet" ^

- Usado para validación y scoring  --fps-target 30 --window 0.50 --hop 0.25 --norm zscore



## Datos Generados# RandomForest con validación

python -m src.model.train ^

### `data/annotations/moves/`  --dataset ".\data\features\8yang_dataset.parquet" ^

  --model-out ".\data\models\rf_8yang.joblib" ^

Archivos JSON con movimientos segmentados:  --clf random_forest ^

  --cv 5 --seed 42

```json

{# SVM lineal

  "fps": 30.0,python -m src.model.train --dataset ".\data\features\8yang_dataset.parquet" --model-out ".\data\models\svm_8yang.joblib" --clf svm_linear --cv 5 --seed 42

  "moves": [

    {# Regresión logística

      "name": "olgul_maki_l",python -m src.model.train --dataset ".\data\features\8yang_dataset.parquet" --model-out ".\data\models\logreg_8yang.joblib" --clf logreg --cv 5 --seed 42

      "frame_start": 45,

      "frame_end": 78,# A partir de video con pipeline integrado (pose → features → predicción → reporte)

      "stance_pred": "ap_kubi",python -m src.model.infer ^

      "confidence": 0.85  --video ".\data\raw_videos\8yang\val\8yang_010.mp4" ^

    }  --model ".\data\models\rf_8yang.joblib" ^

  ]  --report ".\data\annotations\reports\8yang_010_report.xlsx" ^

}  --config ".\config\default.yaml"

```

# A partir de landmarks ya calculados

### `data/models/stance_classifier_final.pkl`python -m src.model.infer --landmarks ".\data\landmarks\8yang\8yang_010.csv" --model ".\data\models\rf_8yang.joblib" --report ".\data\annotations\reports\8yang_010_report.xlsx"



Modelo Random Forest serializado (7.3 MB):# Procesa uno o varios videos y emite reportes (offline)

- RandomForestClassifierpython -m src.main_offline ^

- LabelEncoder (3 clases)  --in ".\data\raw_videos\8yang\val" ^

- Feature names (8 features)  --out ".\data\annotations\reports" ^

  --model ".\data\models\rf_8yang.joblib" ^

### `reports/comparison_baseline_vs_ml.txt`  --config ".\config\default.yaml"



Reporte comparativo detallado con:# Webcam índice 0 con overlay de deducciones (offline)

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
