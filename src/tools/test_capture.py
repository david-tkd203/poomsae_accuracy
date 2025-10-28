# test_capture.py
import sys
from pathlib import Path

# Agregar el directorio raíz al path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.tools.capture_moves import main

if __name__ == "__main__":
    # Usa paths reales de tu sistema - ajusta estos ejemplos:
    csv_path = "data/landmarks/8yang_001.csv"  # Ajusta a tu archivo real
    video_path = "data/videos/8yang_001.mp4"   # Ajusta a tu archivo real
    
    sys.argv = [
        "capture_moves.py",
        "--csv", csv_path,
        "--video", video_path, 
        "--out-json", "output/movements.json",
        "--vstart", "0.40",
        "--vstop", "0.15", 
        "--min-dur", "0.20",
        "--min-gap", "0.15",
        "--expected-n", "24",
        "--render-out", "output/preview.mp4"
    ]
    
    main()