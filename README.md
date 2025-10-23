POOMSAE ACCURACY — README DE USO Y COMANDOS
==============================================

Resumen del proyecto
--------------------
Sistema para apoyar la evaluación de “exactitud” en Poomsae (Taekwondo WT) a partir de video. Flujo general: captura multivista (más adelante 4 cámaras + Kinect), extracción de puntos corporales (MediaPipe cuando corresponda), cálculo de medidas simples (ángulos, alineaciones, profundidad de posturas, estabilidad), reglas de deducción (−0,1 / −0,3) y un modelo clásico de ML (no deep learning) para clasificar movimiento como correcto / error leve / error grave. Incluye utilidades para: descarga masiva de videos, eliminación de fondo, pre-segmentación por actividad y etiquetado rápido, entrenamiento e inferencia offline/tiempo real.

Requisitos
----------
- Windows 10/11 64-bit (probado en PowerShell).
- Python 3.12.x recomendado.
  - Nota: si más adelante usas MediaPipe, evita 3.13 (no hay wheel oficial estable al momento).
- FFmpeg en PATH (requerido por yt-dlp para remux/conversión).
  - Verifica: `ffmpeg -version`
- Git (opcional).

Creación del entorno e instalación
----------------------------------
1) Crear entorno virtual (PowerShell):
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2) Actualizar herramientas base:
   python -m pip install --upgrade pip setuptools wheel

3) Instalar dependencias principales:
   pip install -r requirements.txt

4) (Opcional) Variables de entorno en `.env`:
   # proxies (si tu red lo requiere)
   HTTP_PROXY=http://user:pass@host:port
   HTTPS_PROXY=http://user:pass@host:port

   # formato de descarga (yt-dlp)
   YTDLP_FORMAT=bv*+ba/b

Estructura del repo (carpetas clave)
------------------------------------
C:.
│  .env
│  README.md
│  requirements.txt
│
├─config
│    default.yaml
│
├─data
│  ├─annotations         # CSV de segmentos/labels
│  ├─landmarks           # puntos corporales exportados (cuando se usen)
│  ├─models              # modelos ML entrenados
│  ├─raw_videos          # videos originales
│  └─raw_videos_bg       # videos con fondo eliminado
│
└─src
   │  config.py
   │  main_offline.py        # evaluación desde archivo
   │  main_realtime.py       # evaluación desde cámara
   │  video_get.py           # CLI descargas/renombrado
   │  __init__.py
   │
   ├─dataio
   │    video_downloader.py  # motor de descargas
   │    video_source.py      # lectura robusta de video
   │    report_generator.py  # generación de informes
   │    __init__.py
   │
   ├─features
   │    angles.py            # utilidades geométricas de ángulos
   │    feature_extractor.py # extracción de features por segmento
   │
   ├─model
   │    dataset_builder.py   # construye dataset (features + labels)
   │    train.py             # entrena modelo ML clásico (p.ej., RandomForest/SVM)
   │    infer.py             # inferencia con modelo entrenado
   │
   ├─pose
   │    base.py
   │    csv_backend.py
   │    mediapipe_backend.py # se integra más adelante (Python 3.12)
   │
   ├─preprocess
   │    bg_remove.py         # eliminación de fondo (Rembg/BackgroundMatting o similar)
   │
   ├─rules
   │    deduction_rules.py   # mapeo a −0,1 / −0,3 por movimiento
   │
   ├─segmentation
   │    segmenter.py         # segmentación temporal (movimiento)
   │
   └─utils
        geometry.py
        smoothing.py
        timing.py

Comandos útiles (PowerShell)
----------------------------

A) Descarga de videos y normalización de nombres
-----------------------------------------------
1) Descargar una playlist con alias y subset:
   python -m src.video_get --url "https://youtube.com/playlist?list=PL4sEO8KDGlNXxRjmvUz_a3g8MGDLwv1BG" --out ".\data\raw_videos" --alias 8yang --subset train

2) Descargar un único video (YouTube u otro sitio soportado por yt-dlp):
   python -m src.video_get --url "https://www.youtube.com/watch?v=VIDEO_ID" --out ".\data\raw_videos" --alias 8yang --subset val

3) Descargar enlaces directos a archivo .mp4/.mkv:
   python -m src.video_get --url "https://servidor/archivo.mp4" --out ".\data\raw_videos" --alias 8yang --subset test

4) Descargar múltiples URLs desde un .txt (una por línea):
   python -m src.video_get --url-file ".\urls.txt" --out ".\data\raw_videos" --alias 8yang --subset train

5) Renombrar SOLO lo descargado en la ejecución (con numeración):
   python -m src.video_get --rename-downloaded --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}" --out ".\data\raw_videos" --alias 8yang

6) Renombrar TODOS los existentes en la carpeta destino alias/subset:
   python -m src.video_get --rename-existing --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}" --out ".\data\raw_videos" --alias 8yang --subset train

7) Si tus URLs están en Excel (.xlsx), conviértelas a .txt con Python:
   python - << "PY"
   import pandas as pd
   df = pd.read_excel(r".\links_8yang.xlsx")
   df.iloc[:,0].dropna().to_csv(r".\urls.txt", index=False, header=False)
   PY

B) Eliminación de fondo (pre-procesamiento)
-------------------------------------------
1) Un solo video:
   python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train\8yang_001.mp4" --out ".\data\raw_videos_bg\8yang\train\8yang_001_bg.mp4"

2) Carpeta completa (si el script soporta lotes, según implementación):
   # ejemplo típico: (si tu bg_remove.py trae modo carpeta)
   python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train" --out ".\data\raw_videos_bg\8yang\train"

C) Segmentación previa y etiquetado rápido
------------------------------------------
1) Pre-segmentación por actividad (auto-candidatos):
   python -m src.annot.auto_seg --in ".\data\raw_videos\8yang\train" --out ".\data\annotations\8yang_cands.csv" --alias 8yang

2) Etiquetado rápido de un video contra el CSV de candidatos:
   python -m src.annot.quick_label --video ".\data\raw_videos\8yang\train\8yang_001.mp4" --cands ".\data\annotations\8yang_cands.csv" --out ".\data\annotations\8yang_labels.csv"

   Atajos típicos en la ventana (según implementación):
   - Espacio: play/pausa
   - A/D: retroceder/avanzar segmento
   - Flechas: ±0.10s / ±0.50s
   - 1/2/3: correcto / error_leve / error_grave
   - S: guardar

D) Extracción de características y entrenamiento (ML clásico)
-------------------------------------------------------------
1) Construcción de dataset (features + labels):
   python -m src.model.dataset_builder --labels ".\data\annotations\8yang_labels.csv" --raw ".\data\raw_videos" --features ".\data\features\8yang" --landmarks ".\data\landmarks\8yang"

2) Entrenamiento del modelo (ej. RandomForest):
   python -m src.model.train --features ".\data\features\8yang" --out ".\data\models\8yang_rf.pkl" --model rf --params "{'n_estimators':300,'max_depth':20}"

3) Inferencia sobre nuevos videos etiquetados o en vivo:
   # offline (archivo):
   python -m src.model.infer --model ".\data\models\8yang_rf.pkl" --video ".\data\raw_videos\8yang\val\8yang_021.mp4" --report ".\data\reports\8yang_021.xlsx"

   # tiempo real (cámara 0):
   python -m src.main_realtime --model ".\data\models\8yang_rf.pkl" --camera 0

E) Ejecución de los “mains”
---------------------------
1) Evaluación desde archivo (pipeline completo si ya hay modelo):
   python -m src.main_offline --model ".\data\models\8yang_rf.pkl" --video ".\data\raw_videos\8yang\test\8yang_050.mp4" --report ".\data\reports\8yang_050.xlsx"

2) Evaluación en vivo (más adelante integrar Kinect y multivista):
   python -m src.main_realtime --model ".\data\models\8yang_rf.pkl" --camera 0

Qué hace cada módulo
--------------------
- src/video_get.py
  CLI de descarga y renombrado. Usa dataio/video_downloader.py. Permite alias (“8yang”), subset (train/val/test), `--url-file`, y normalización de nombres con secuencia.

- src/dataio/video_downloader.py
  Motor de descarga. yt-dlp (YouTube/playlist/sites) o HTTP directo. Normaliza alias desde títulos (ej. “Yang 8” → “8yang”) y estructura carpetas de salida.

- src/preprocess/bg_remove.py
  Eliminación de fondo (matting/segmentation). Produce un video con el practicante recortado sobre fondo transparente o plano.

- src/segmentation/segmenter.py
  Segmentación temporal por movimiento para cortar ejecuciones en unidades más limpias o detectar inicios/finales.

- src/features/feature_extractor.py y src/features/angles.py
  A partir de landmarks, calcula medidas: ángulos articulares, alineaciones, profundidad, estabilidad, etc.

- src/rules/deduction_rules.py
  Traduce medidas por movimiento a deducciones reglamentarias: −0,1 (leve) y −0,3 (grave).

- src/model/dataset_builder.py
  Construye el dataset combinando segmentos etiquetados y features, listo para entrenamiento.

- src/model/train.py
  Entrena un modelo ML clásico (RandomForest, SVM, GradientBoosting) sin deep learning. Guarda `.pkl`.

- src/model/infer.py
  Carga modelo y predice sobre nuevos videos; opcionalmente genera reporte (Excel/JSON).

- src/main_offline.py / src/main_realtime.py
  Entradas de alto nivel para ejecutar el pipeline sobre archivos o cámaras en tiempo real.

- src/pose/*
  Backends de pose. `mediapipe_backend.py` se integra cuando el entorno esté listo (Python 3.12). `csv_backend.py` permite reproducir landmarks precomputados.

- src/dataio/report_generator.py
  Ensambla un informe con nota de exactitud, errores y marcas de tiempo (p. ej. Excel).

- src/utils/*
  Utilidades transversales (geometría, suavizado temporal, temporización).

Buenas prácticas asumidas
-------------------------
- POO y modularización: componentes separados (descarga, preproceso, features, reglas, modelo).
- No hardcodear rutas: usar argumentos de CLI y config/default.yaml.
- `.env` para proxies y parámetros sensibles.
- Reproducibilidad: scripts que generan salidas deterministas a partir de insumos claros.
- Sin CNN/Deep Learning: modelos clásicos de scikit-learn.

Solución de problemas comunes
-----------------------------
- “ModuleNotFoundError” al ejecutar `python -m src.algo`:
  Activa el venv y ejecuta SIEMPRE con `python -m ...` desde la raíz del repo.

- yt-dlp se queja de FFmpeg:
  Instala FFmpeg y agrégalo a PATH. Reabre PowerShell.

- MediaPipe no instala en 3.13:
  Usa Python 3.12.x para esa parte del pipeline.

- Descarga sin archivos:
  Playlist privada / región restringida → usa `--cookies cookies.txt`.
  Comprueba la conectividad y el formato `YTDLP_FORMAT` en `.env`.

- Videos H.265/HEVC no reproducen:
  Forzar `--format "bv*+ba/b"` o convertir con FFmpeg a H.264/AAC.

Roadmap breve
-------------
- Integrar Kinect (profundidad) y multivista.
- Completar backend de pose con MediaPipe en 3.12.
- Añadir evaluaciones cruzadas con jueces y métricas: F1, MAE, ICC, latencia.
- Extender a más poomsae y a Pareja/Equipo.

# Playlist YouTube → carpeta data/raw_videos/8yang/train
python -m src.video_get --url "https://youtube.com/playlist?list=PL4sEO8KDGlNXxRjmvUz_a3g8MGDLwv1BG" --out ".\data\raw_videos" --alias 8yang --subset train

# Video individual (YouTube)
python -m src.video_get --url "https://www.youtube.com/watch?v=VIDEO_ID" --out ".\data\raw_videos" --alias 8yang --subset val

# Enlace directo a .mp4/.mkv
python -m src.video_get --url "https://servidor/archivo.mp4" --out ".\data\raw_videos" --alias 8yang --subset test

# Varias URLs desde .txt (una por línea)
python -m src.video_get --url-file ".\urls.txt" --out ".\data\raw_videos" --alias 8yang --subset train

# Evitar inferencia de alias por título
python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-normalize-from-title

# No usar yt-dlp primero (preferir HTTP directo si aplica)
python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-ytdlp-first

# No omitir existentes (re-descarga)
python -m src.video_get --url "URL" --out ".\data\raw_videos" --no-skip-existing

# Cookies (si la playlist lo requiere)
python -m src.video_get --url "URL" --out ".\data\raw_videos" --cookies ".\cookies.txt"

# Renombrar SOLO lo descargado en esta ejecución (con numeración)
python -m src.video_get --out ".\data\raw_videos" --alias 8yang --rename-downloaded --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}"

# Renombrar TODOS los existentes en alias/subset (sin descargar nada)
python -m src.video_get --out ".\data\raw_videos" --alias 8yang --subset train --rename-existing --seq --seq-start 1 --seq-digits 3 --scheme "{alias}_{n:0Nd}"

# Simulación de renombrado (sin cambios)
python -m src.video_get --out ".\data\raw_videos" --alias 8yang --rename-existing --seq --dry-run

# Un video → salida con fondo removido
python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train\8yang_001.mp4" --out ".\data\raw_videos_bg\8yang\train\8yang_001_bg.mp4"

# Carpeta completa (si el script soporta directorio)
python -m src.preprocess.bg_remove --in ".\data\raw_videos\8yang\train" --out ".\data\raw_videos_bg\8yang\train"

# Genera dataset tabular a partir de labels + landmarks/features
python -m src.model.dataset_builder ^
  --labels ".\data\annotations\8yang_labels.csv" ^
  --landmarks ".\data\landmarks\8yang" ^
  --features-out ".\data\features\8yang_dataset.parquet" ^
  --fps-target 30 --window 0.50 --hop 0.25 --norm zscore

# RandomForest con validación
python -m src.model.train ^
  --dataset ".\data\features\8yang_dataset.parquet" ^
  --model-out ".\data\models\rf_8yang.joblib" ^
  --clf random_forest ^
  --cv 5 --seed 42

# SVM lineal
python -m src.model.train --dataset ".\data\features\8yang_dataset.parquet" --model-out ".\data\models\svm_8yang.joblib" --clf svm_linear --cv 5 --seed 42

# Regresión logística
python -m src.model.train --dataset ".\data\features\8yang_dataset.parquet" --model-out ".\data\models\logreg_8yang.joblib" --clf logreg --cv 5 --seed 42

# A partir de video con pipeline integrado (pose → features → predicción → reporte)
python -m src.model.infer ^
  --video ".\data\raw_videos\8yang\val\8yang_010.mp4" ^
  --model ".\data\models\rf_8yang.joblib" ^
  --report ".\data\annotations\reports\8yang_010_report.xlsx" ^
  --config ".\config\default.yaml"

# A partir de landmarks ya calculados
python -m src.model.infer --landmarks ".\data\landmarks\8yang\8yang_010.csv" --model ".\data\models\rf_8yang.joblib" --report ".\data\annotations\reports\8yang_010_report.xlsx"

# Procesa uno o varios videos y emite reportes (offline)
python -m src.main_offline ^
  --in ".\data\raw_videos\8yang\val" ^
  --out ".\data\annotations\reports" ^
  --model ".\data\models\rf_8yang.joblib" ^
  --config ".\config\default.yaml"

# Webcam índice 0 con overlay de deducciones (offline)
python -m src.main_realtime --camera 0 --model ".\data\models\rf_8yang.joblib" --config ".\config\default.yaml"

# Stream RTSP/archivo (en tiempo real)
python -m src.main_realtime --source "rtsp://usuario:pass@ip/stream" --model ".\data\models\rf_8yang.joblib" --config ".\config\default.yaml"



Créditos y licencia
-------------------
Uso académico. Cita y respeta licencias de terceros (yt-dlp, FFmpeg, OpenCV, scikit-learn, etc.).
