# OBS de Evaluación Poomsae

Este módulo implementa una interfaz tipo OBS para la evaluación avanzada de poomsae, integrando múltiples cámaras, métricas biomecánicas y reportes automáticos.

## Características avanzadas

- Selección y visualización de múltiples cámaras en mosaico
- Overlays en tiempo real con métricas biomecánicas (ángulos, errores, nombre)
- Actualización continua de frames mediante QTimer (fluidez tipo streaming)
- Grabación de video y exportación automática de reportes biomecánicos (CSV)
- Visualización gráfica de reportes y métricas (próxima integración con matplotlib)
- Configuración flexible vía archivo `.env`

## Flujo de uso

1. Selecciona las cámaras y configura el nombre del practicante.
2. Inicia la evaluación: se muestran los mosaicos y overlays en tiempo real, con actualización continua.
3. Al detener la evaluación, se exporta automáticamente el reporte biomecánico y se muestra la visualización gráfica.

## Requisitos

- Python 3.12+
- PyQt5
- numpy, opencv-python, pandas, openpyxl, python-dotenv

## Ejecución

```powershell
C:/Users/david/OneDrive/Escritorio/Tesis/poomsae-accuracy/.venv/Scripts/python.exe src/obs/main_obs.py
```


## Personalización y uso avanzado

- **Configuración de refresco:** Puedes ajustar la fluidez de actualización de frames editando el valor `refresh_interval` en tu archivo `.env` (por ejemplo, `refresh_interval=33` para ~30 fps).
- **Manejo de errores:** El sistema muestra mensajes en el mosaico si una cámara no responde o hay errores, sin bloquear la interfaz.
- **Exportación avanzada:** Al finalizar la evaluación, se exportan automáticamente los reportes en formatos CSV, Excel y JSON en el directorio de trabajo.
- **Visualización gráfica:** El reporte incluye gráficos de métricas biomecánicas por cámara usando matplotlib embebido.
- **Sugerencias automáticas:** El sistema analiza los datos y muestra recomendaciones inteligentes en el reporte final.
- **Overlays personalizados:** Puedes modificar el método `draw_metrics` en `OverlayManager` para cambiar colores, métricas mostradas y visualización de errores.

### Ejemplo de flujo avanzado

```python
# Personalizar el overlay para mostrar métricas específicas
overlay_frame = overlay.draw_metrics(frame, metrics={'score': 8.5, 'error_postura': True}, angles={'codo': 45.2}, name='Juan')

# Configurar refresco en .env
refresh_interval=20  # Actualización más rápida (~50 fps)

# Exportar reporte en Excel y JSON automáticamente
# (ya integrado en export_report)

# Visualizar sugerencias automáticas en el reporte
# (se muestran junto al gráfico en ReportViewer)
```

### Recomendaciones
- Para mayor robustez, verifica que todas las cámaras estén conectadas antes de iniciar la evaluación.
- Puedes ampliar el análisis automático en `generate_suggestions` para detectar patrones más complejos.
- Integra nuevos tipos de métricas y overlays según tus necesidades biomecánicas.

## Próximas mejoras

- Integración de visualización gráfica avanzada (matplotlib)
- Exportación a Excel y JSON
- Aprendizaje automático sobre reportes

## Autor

David TKD203
2. Ingresa el nombre del evaluado, selecciona las cámaras y sigue las instrucciones.
3. Marca inicio y final de la evaluación con los botones.
4. Los videos y reportes se exportan automáticamente.

## Mejoras futuras
- Integración de aprendizaje automático para mejorar la evaluación.
- Personalización avanzada de overlays y reportes.
- Soporte para más sensores y cámaras.
- Exportación directa a Excel y visualización gráfica avanzada.

---

Desarrollado por david-tkd203 y GitHub Copilot.
