#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de setup y calibración para Kinect y webcam
"""
from src.kinect.kinect_backend_hybrid import KinectPoomsaeCapture, calibrate_poomsae_area

def main():
    print("\n==== Setup y Calibración de Sensor ====")
    capture = KinectPoomsaeCapture()
    if not capture.initialize():
        print("No se pudo inicializar la cámara/sensor.")
        return
    print("Sensor inicializado correctamente.")
    area = calibrate_poomsae_area(capture)
    print(f"Área de calibración: {area}")
    print("Colócate en el centro y presiona cualquier tecla para finalizar.")
    import cv2
    while True:
        frame_data = capture.get_frame()
        if frame_data is None:
            break
        vis_frame = capture.visualize_frame(frame_data)
        cv2.imshow("Calibración", vis_frame)
        if cv2.waitKey(1) & 0xFF != 255:
            break
    capture.cleanup()
    cv2.destroyAllWindows()
    print("Calibración finalizada.")

if __name__ == "__main__":
    main()
