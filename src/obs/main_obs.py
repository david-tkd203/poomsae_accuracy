# src/obs/main_obs.py
from PyQt5 import QtCore, QtWidgets
from src.obs.config import ObsSettings
from src.obs.obs_window import ObsWindow

def main():
    import sys
    # HiDPI antes de crear QApplication
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    cfg = ObsSettings.load()
    # Si quieres layout automático por defecto:
    if getattr(cfg, "layout", None) not in ("studio", "grid", "auto"):
        cfg.layout = "auto"
    win = ObsWindow(cfg)
    # win.showFullScreen()  # si prefieres fullscreen duro
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
