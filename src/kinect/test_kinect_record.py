import time
from kinect_backend import KinectPoomsaeCapture

# Inicializa el backend Kinect
kinect = KinectPoomsaeCapture(enable_color=True, enable_depth=True, enable_ir=True, enable_body=True)
if not kinect.initialize():
    print("No se pudo inicializar la Kinect.")
    exit(1)

# Inicia la grabación
kinect.start_recording("kinect_test_recordings")
print("Grabando 10 segundos...")
start = time.time()
while time.time() - start < 10:
    frame = kinect.get_frame()
    time.sleep(1.0 / kinect.fps)  # Simula la tasa de frames

# Detiene y guarda la grabación
kinect.stop_recording()
print("Grabación finalizada.")
