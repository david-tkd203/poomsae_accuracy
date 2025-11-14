import cv2

class OverlayManager:
    def __init__(self):
        pass

    def draw_metrics(self, frame, metrics: dict, angles: dict, name: str = ""):
        y = 30
        if name:
            cv2.putText(frame, f"Evaluado: {name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,128,255), 2)
            y += 30
        # Títulos visuales
        cv2.putText(frame, "--- Métricas biomecánicas ---", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128,128,128), 2)
        y += 24
        # Métricas biomecánicas
        for k, v in metrics.items():
            if isinstance(v, tuple):
                val, color = v
            else:
                val, color = v, (255,128,0)
            if k.startswith("Postura") or k.startswith("Segmento") or k.startswith("Confianza") or k.startswith("Barra") or k == "Faltan features":
                continue
            cv2.putText(frame, f"{k}: {val}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 22
        # Ángulos
        for k, v in angles.items():
            cv2.putText(frame, f"{k}: {v:.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
            y += 18
        # Título ML
        y += 10
        cv2.putText(frame, "--- ML: Postura y Segmento ---", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,128,255), 2)
        y += 24
        # Postura y segmento
        for k in ["Postura", "Confianza postura", "Barra postura", "Segmento", "Confianza segmento", "Barra segmento"]:
            if k in metrics:
                val, color = metrics[k]
                if "Barra" in k:
                    # Dibujar barra gráfica
                    bar_len = int(val * 2)
                    cv2.rectangle(frame, (10, y+5), (10+bar_len, y+20), color, -1)
                    cv2.putText(frame, f"{k}: {val}%", (10+bar_len+8, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y += 22
                else:
                    cv2.putText(frame, f"{k}: {val}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    y += 22
        # Advertencias
        if "Faltan features" in metrics:
            val, color = metrics["Faltan features"]
            # Fondo rojo
            cv2.rectangle(frame, (5, y), (400, y+28), (0,0,255), -1)
            cv2.putText(frame, f"Faltan datos: {val}", (10, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y += 32
        return frame
