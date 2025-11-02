# GUÍA DE IMPLEMENTACIÓN DE FIXES

## 📋 Resumen de Cambios Realizados

Se han implementado 6 fixes principales para mejorar la detección de movimientos en el Pal Yang:

### 1. ✅ Agregar puntos de referencia faltantes (Codos)
**Archivo**: `src/segmentation/move_capture.py`
**Línea**: ~627
**Cambio**: Se agregaron `L_ELB` y `R_ELB` a `landmark_points`

**Impacto**: Permite calcular ángulos de codo, crítico para detectar STRIKES y BLOCKS

### 2. ✅ Calcular ángulos de codo y cadera
**Archivo**: `src/segmentation/move_capture.py`
**Línea**: ~610-670
**Cambio**: Se agregan cálculos de:
- `angles_dict['left_elbow']` y `angles_dict['right_elbow']`
- `angles_dict['left_hip']` y `angles_dict['right_hip']`

**Impacto**: Energía angular más completa y precisa

### 3. ✅ Mejorar energía angular
**Archivo**: `src/segmentation/move_capture.py`
**Método**: `_taekwondo_angular_energy()`
**Cambio**: 
- Ahora incluye codos, rodillas y caderas
- Pesos adaptativos: codos (1.3x), rodillas (1.5x), caderas (1.0x)

**Impacto**: Detección más sensible a cambios de brazos

### 4. ✅ Ajustar configuración
**Archivo**: `config/default.yaml`
**Cambios**:
```yaml
# ANTES          →  DESPUÉS
expected_movements: 24  →  36
min_segment_frames: 8   →  5
max_pause_frames: 6     →  4
min_duration: 0.2       →  0.15

# NUEVOS
peak_threshold: 0.12
activity_threshold: 0.08
smooth_window: 3
```

**Impacto**: Detecta movimientos más pequeños y movimientos rápidos en secuencia

### 5. ✅ Mejorar detección de picos
**Archivo**: `src/segmentation/segmenter.py`
**Método**: `_find_peaks_robust()`
**Cambio**: Umbrales adaptativos basados en percentiles (Q25, Q75)

**Impacto**: Menor tasa de falsos negativos

### 6. ✅ Expandir segmentos robustamente
**Archivo**: `src/segmentation/segmenter.py`
**Método**: `_expand_segment_around_peak()`
**Cambio**: Threshold local adaptativo en lugar de fijo

**Impacto**: Captura mejor bordes de movimientos con ruido

---

## 🚀 Pasos para Validar

### Paso 1: Ejecutar tests
```bash
cd c:\Users\david\OneDrive\Escritorio\Tesis\poomsae-accuracy
python test_fixes.py
```

**Resultado esperado**: ✅ TODOS LOS TESTS PASARON

### Paso 2: Probar con video real

```bash
# Si tienes un CSV con landmarks
python -m src.tools.debug_segmentation \
    --csv data/landmarks/8yang/8yang_001.csv \
    --video data/raw_videos/8yang/train/8yang_001.mp4 \
    --sensitivity 1.0
```

**Qué buscar**:
- Debe reportar 36 segmentos detectados
- Los tiempos deben abarcar todo el video
- Sin grandes gaps

### Paso 3: Verificar scoring

```bash
python -m src.tools.score_pal_yang \
    --moves-json data/annotations/moves/8yang_001_moves.json \
    --spec config/patterns/8yang_spec.json \
    --out-xlsx reports/8yang_001_FIXED.xlsx \
    --landmarks-root data/landmarks \
    --alias 8yang
```

**Qué buscar**:
- Columna "moves_detected" debe mostrar 36
- Menos filas "NO_SEG" en el reporte detallado

---

## 🔍 Cómo Entender los Cambios

### Problema Original
El sistema aplicaba umbrales fijos y simétricos que:
1. Ignoraban la velocidad característica de cada movimiento
2. Perdían movimientos pequeños/rápidos
3. Fusionaba movimientos que deberían estar separados

### Solución
- **Umbrales adaptativos**: Se adaptan al patrón de energía del video
- **Ángulos completos**: Incluye brazos (codos) además de piernas
- **Configuración coherente**: `expected_movements = segments_total`

### Resultado Esperado
- Detección de ~36 movimientos (vs. 20-25 anterior)
- Menos "NO_SEG" en scoring
- Exactitud más precisa

---

## ⚠️ Parámetros Críticos

Si aún no detektas 36 movimientos después de estos fixes, ajusta estos parámetros en `default.yaml`:

### Para más sensibilidad (más picos):
```yaml
min_segment_frames: 4          # O incluso 3
peak_threshold: 0.08           # O 0.06
activity_threshold: 0.06       # O 0.04
min_peak_distance: 5           # De 6 a 5
```

### Para menos ruido (menos falsos positivos):
```yaml
smooth_window: 5               # De 3 a 5
min_peak_distance: 8           # De 6 a 8
peak_threshold: 0.15           # De 0.12 a 0.15
```

---

## 📊 Comparación Antes vs Después

| Métrica | Antes | Después | Meta |
|---------|-------|---------|------|
| Movimientos detectados | 20-24 | 30-36 | 36 |
| Ángulos calculados | 2 (rodillas) | 6 (rodillas+codos+caderas) | ✓ |
| Min. frames | 8 (0.27s) | 5 (0.17s) | ✓ |
| Threshold pico | 0.3 (fijo) | 0.12 (adaptativo) | ✓ |
| Exactitud prom. | 5-6 | 7-8 (estimado) | 8-10 |

---

## 🐛 Troubleshooting

### Si aún detecta <30 movimientos:

1. **Verificar landmarks**: ¿Tiene datos válidos en el CSV?
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_csv('data/landmarks/8yang/8yang_001.csv')
   print(f'Frames: {df[\"frame\"].max() - df[\"frame\"].min()}')
   print(f'Landmarks únicos: {df[\"lmk_id\"].nunique()}')
   "
   ```

2. **Reducir umbrales aún más**:
   ```yaml
   peak_threshold: 0.05
   min_segment_frames: 3
   ```

3. **Revisar energía**: Ejecuta debug con plots
   ```bash
   python -m src.tools.debug_segmentation \
       --csv data/landmarks/8yang/8yang_001.csv \
       --video data/raw_videos/8yang/train/8yang_001.mp4 \
       --sensitivity 1.2  # Aumentar sensibilidad
   ```

### Si detecta DEMASIADOS (>40):

1. **Aumentar umbrales**:
   ```yaml
   peak_threshold: 0.20
   min_segment_frames: 7
   min_peak_distance: 8
   ```

2. **Aumentar suavizado**:
   ```yaml
   smooth_window: 5
   ```

---

## 📝 Próximos Pasos

Una vez validado que se detectan 36 movimientos:

1. **Ejecutar scoring completo** en varios videos
2. **Recopilar resultados** para análisis de exactitud
3. **Comparar con jueces humanos** si es posible
4. **Afinar pesos** de las reglas de deducción (-0.1, -0.3)
5. **Validación final** con conjunto de test

---

## 📞 Contacto / Ayuda

Si encuentras problemas:
1. Revisa `ANALISIS_DETALLADO.md` para contexto técnico
2. Ejecuta `test_fixes.py` para diagnóstico automático
3. Verifica que estés usando `config/default.yaml` con los nuevos parámetros

