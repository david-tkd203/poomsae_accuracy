import cv2

class CameraManager:
    def __init__(self):
        self.cameras = self.detect_cameras()

    def detect_cameras(self):
        # Detecta cámaras disponibles (incluye Kinect si está activa)
        cams = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(f"Webcam {i}")
                cap.release()
        # Detección de Kinect
        try:
            from PyKinect2 import PyKinectRuntime
            kinect = PyKinectRuntime.PyKinectRuntime()
            cams.insert(0, "Kinect")
        except Exception as e:
            print(f"[INFO] Kinect no detectada: {e}")
        if not cams:
            print("[WARN] No se detectaron cámaras disponibles.")
        return cams
