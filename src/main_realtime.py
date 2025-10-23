import cv2, time
from src.config import load_conf, env
from src.model.infer import SegmentInfer
from src.rules.deduction_rules import RuleEngine
# En tiempo real, sin landmarks no hay features. En fase 1, este archivo
# muestra el esqueleto del loop y el overlay (a integrar cuando haya backend online).

def main():
    conf = load_conf()
    cap = cv2.VideoCapture(int(env("CAM_INDEX", 0)))
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara")

    model_path = env("MODEL_PATH", conf["general"]["model_path"])
    try:
        inf = SegmentInfer(model_path)
    except Exception:
        inf = None
    rg = RuleEngine(conf["rules"])

    while True:
        ok, frame = cap.read()
        if not ok: break
        # TODO: integrar backend de landmarks online para extraer features por ventana
        cv2.putText(frame, "Fase 1: integra backend de pose para RT", (16,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
        cv2.putText(frame, "Fase 1: integra backend de pose para RT", (16,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        cv2.imshow("Realtime (placeholder)", frame)
        if cv2.waitKey(1) & 0xFF in (27, ord('q')): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
