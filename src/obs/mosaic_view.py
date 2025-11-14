from PyQt5.QtWidgets import QWidget, QGridLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
import cv2

class MosaicView(QWidget):
    def __init__(self, cam_names):
        super().__init__()
        self.cam_names = cam_names
        self.setWindowTitle("Visualización de Cámaras")
        self.layout = QGridLayout()
        self.labels = {}
        for idx, name in enumerate(cam_names):
            lbl = QLabel(f"{name}")
            lbl.setFixedSize(320, 240)
            lbl.setStyleSheet("background: #222; color: #fff; font-size: 14px;")
            self.layout.addWidget(lbl, idx//2, idx%2)
            self.labels[name] = lbl
        self.setLayout(self.layout)

    def update_frame(self, cam_name, frame):
        if cam_name in self.labels:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_img)
            self.labels[cam_name].setPixmap(pixmap.scaled(320, 240))
