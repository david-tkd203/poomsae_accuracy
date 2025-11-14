from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTextEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ReportViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reporte Biomecánico")
        self.setGeometry(200, 200, 800, 600)
        layout = QVBoxLayout()
        self.title = QLabel("Reporte de Evaluación Biomecánica")
        self.title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.title)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        self.figure = Figure(figsize=(6, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def show_report(self, report_text, metrics=None):
        self.text.setText(report_text)
        self.figure.clear()
        if metrics:
            ax = self.figure.add_subplot(111)
            # Ejemplo: graficar ángulos de cada cámara
            for cam, vals in metrics.items():
                angles = [v for k, v in vals.items() if 'angle' in k]
                ax.plot(angles, label=f'{cam}')
            ax.set_title('Ángulos biomecánicos por cámara')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Ángulo (°)')
            ax.legend()
        self.canvas.draw()
