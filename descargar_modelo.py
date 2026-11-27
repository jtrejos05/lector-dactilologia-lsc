# ============================================================================
#  Descarga del modelo de landmarks de mano
# ----------------------------------------------------------------------------
#  MediaPipe ya no trae los modelos dentro del paquete pip: hay que bajarlos
#  del repositorio oficial de Google. Son ~7.5 MB y solo se hace una vez.
#
#  Este modelo NO es el que reconoce letras. Es el que localiza la mano y
#  devuelve sus 21 puntos. El que reconoce letras lo entrenas tu con
#  entrenar.py a partir de tus propias capturas.
# ============================================================================

import hashlib
import os
import sys
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
       "hand_landmarker/float16/1/hand_landmarker.task")

DESTINO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "modelo", "hand_landmarker.task")


def progreso(bloques, tam_bloque, total):
    """Dibuja la barra de progreso de la descarga, reescribiendo la linea.

    Es el callback que urlretrieve llama periodicamente. El retorno de carro
    sin salto de linea hace que la barra se repinte en el sitio en vez de
    llenar la consola de lineas."""
    if total <= 0:
        return
    hechos = min(bloques * tam_bloque, total)
    pct = 100.0 * hechos / total
    barra = "#" * int(pct / 2.5)
    sys.stdout.write("\r  [%-40s] %5.1f%%  (%.1f/%.1f MB)"
                     % (barra, pct, hechos / 1e6, total / 1e6))
    sys.stdout.flush()


def main():
    """Descarga hand_landmarker.task desde el repositorio oficial de Google.

    El modelo no se guarda en el repositorio: son 7,8 MB que no tiene sentido
    versionar, y es un artefacto de terceros que conviene traer de su origen.

    El archivo .task es un paquete con dos redes encadenadas: un detector de
    palma, que localiza donde hay una mano en el fotograma, y un modelo de
    landmarks que sobre ese recorte devuelve los 21 puntos 3D."""
    os.makedirs(os.path.dirname(DESTINO), exist_ok=True)

    if os.path.exists(DESTINO):
        tam = os.path.getsize(DESTINO) / 1e6
        print("El modelo ya esta descargado (%.1f MB):" % tam)
        print("  " + DESTINO)
        print("Borra el archivo si quieres volver a bajarlo.")
        return 0

    print("Descargando el modelo de landmarks de mano...")
    print("  origen : " + URL)
    print("  destino: " + DESTINO)
    print()

    temporal = DESTINO + ".parcial"
    try:
        urllib.request.urlretrieve(URL, temporal, reporthook=progreso)
    except Exception as e:
        if os.path.exists(temporal):
            os.remove(temporal)
        print("\n\nFallo la descarga: %s: %s" % (type(e).__name__, e))
        print("Revisa tu conexion, o baja el archivo a mano desde la URL de")
        print("arriba y guardalo en la ruta de destino.")
        return 1

    os.replace(temporal, DESTINO)

    with open(DESTINO, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()

    print("\n")
    print("Listo. %.1f MB" % (os.path.getsize(DESTINO) / 1e6))
    print("sha256: " + sha)
    print()
    print("Siguiente paso:  python capturar.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
