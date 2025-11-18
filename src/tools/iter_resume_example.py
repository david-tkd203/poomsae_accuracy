# Ejemplo: iteración que se puede reanudar tras error o corte
import time

# Guardar el índice actual en un archivo
STATE_FILE = "iter_state.txt"
def guardar_estado(indice):
    with open(STATE_FILE, "w") as f:
        f.write(str(indice))

def cargar_estado():
    try:
        with open(STATE_FILE) as f:
            return int(f.read())
    except Exception:
        return 0  # Si no existe, empieza desde cero

# Lista de ejemplo (puedes cambiar por tu flujo real)
lista = ["a", "b", "c", "d", "e", "f"]
indice = cargar_estado()

for i in range(indice, len(lista)):
    try:
        print(f"Procesando: {lista[i]}")
        time.sleep(1)
        # Simula error en "c" y "e"
        if lista[i] in ["c", "e"]:
            raise Exception(f"Error simulado en {lista[i]}")
    except Exception as ex:
        print(f"Error: {ex}. Guardando estado y saliendo...")
        guardar_estado(i)
        break  # Detener ejecución
else:
    print("Iteración completada. Estado borrado.")
    import os
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)
