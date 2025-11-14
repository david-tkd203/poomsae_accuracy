from PyQt5.QtWidgets import (QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QComboBox, QLineEdit, QCheckBox, QApplication)
from PyQt5.QtCore import Qt
import sys

class OBSWindow(QMainWindow):
    def __init__(self):
        from PyQt5.QtWidgets import QVBoxLayout, QCheckBox, QHBoxLayout, QPushButton, QLabel, QLineEdit, QWidget
        super().__init__()
        central = QWidget()
        layout = QVBoxLayout()

        # Nombre del evaluado
        self.name_label = QLabel("Nombre del evaluado:")
        self.name_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Escribe el nombre aquí...")
        self.name_input.setStyleSheet("font-size: 16px;")
        layout.addWidget(self.name_label)
        layout.addWidget(self.name_input)

        # Selección de cámaras
        self.cam_label = QLabel("Selecciona las cámaras a habilitar:")
        self.cam_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(self.cam_label)
        self.cam_list = QVBoxLayout()
        self.cam_checks = []
        # Detección dinámica de cámaras
        try:
            from src.obs.camera_manager import CameraManager
            cam_manager = CameraManager()
            cams = cam_manager.cameras
        except Exception as e:
            cams = ["Kinect", "Webcam 0"]
            print(f"[WARN] No se pudo detectar cámaras dinámicamente: {e}")
        if not cams:
            cams = ["Sin cámaras detectadas"]
        for cam in cams:
            chk = QCheckBox(cam)
            chk.setChecked(False)
            self.cam_checks.append(chk)
            self.cam_list.addWidget(chk)
        layout.addLayout(self.cam_list)

        # Botones de control
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Iniciar evaluación")
        self.stop_btn = QPushButton("Finalizar evaluación")
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        # Instrucciones
        self.rules_label = QLabel("Reglamento e instrucciones de poomsae")
        self.rules_label.setStyleSheet("font-size: 15px; color: #0055aa; font-weight: bold;")
        self.rules_text = QLabel("- Marca inicio y final con los botones.\n- Mantén postura y técnica según el reglamento oficial.\n- El sistema grabará video y exportará reportes biomecánicos automáticamente.")
        self.rules_text.setWordWrap(True)
        layout.addWidget(self.rules_label)
        layout.addWidget(self.rules_text)

        central.setLayout(layout)
        self.setCentralWidget(central)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = OBSWindow()
    win.show()
    sys.exit(app.exec_())
