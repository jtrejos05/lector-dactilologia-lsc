# ============================================================================
#  Recuperar muestras del CSV a partir de los fotogramas guardados
# ----------------------------------------------------------------------------
#  Si las filas de una letra se pierden del CSV pero sus imagenes siguen en
#  datos/imagenes/<letra>/, este script recalcula las caracteristicas y las
#  vuelve a insertar. El id de rafaga se saca del nombre del archivo, asi que
#  la particion por rafagas de entrenar.py sigue siendo valida.
#
#  Uso:  python recuperar_desde_imagenes.py C [D E ...]
# ============================================================================

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

from senas.manos import DIM_CARACTERISTICAS, DetectorManos

RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(RAIZ, "datos", "muestras.csv")
DIR_IMAGENES = os.path.join(RAIZ, "datos", "imagenes")


def leer_imagen(ruta: str):
    """cv2.imread no abre rutas con enye en Windows; imdecode si."""
    datos = np.fromfile(ruta, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def main(etiquetas: list) -> int:
    import capturar

    filas = capturar.cargar_muestras()
    ya_estan = {f[0] for f in filas}

    with DetectorManos(en_vivo=False) as detector:
        for etiqueta in etiquetas:
            carpeta = os.path.join(DIR_IMAGENES, etiqueta)
            if not os.path.isdir(carpeta):
                print("  '%s': no hay carpeta de imagenes, se omite." % etiqueta)
                continue
            if etiqueta in ya_estan:
                print("  '%s': ya tiene filas en el CSV, se omite." % etiqueta)
                continue

            archivos = sorted(a for a in os.listdir(carpeta) if a.endswith(".jpg"))
            recuperadas, sin_mano = [], 0

            for nombre in archivos:
                img = leer_imagen(os.path.join(carpeta, nombre))
                if img is None:
                    continue
                deteccion = detector.procesar(img)
                vector = detector.caracteristicas(deteccion, img.shape)
                if vector is None:
                    sin_mano += 1
                    continue
                # "C-20261125231530_007.jpg" -> rafaga "C-20261125231530"
                id_rafaga = nombre.rsplit("_", 1)[0]
                recuperadas.append([etiqueta, id_rafaga] + vector.tolist())

            filas.extend(recuperadas)
            n_rafagas = len({f[1] for f in recuperadas})
            print("  '%s': %d muestras recuperadas de %d imagenes (%d rafagas)"
                  % (etiqueta, len(recuperadas), len(archivos), n_rafagas))
            if sin_mano:
                print("        %d imagenes sin mano detectable, descartadas" % sin_mano)

    capturar.guardar_muestras(filas)
    print("\nCSV actualizado: %d muestras en total" % len(filas))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__ or "Uso: python recuperar_desde_imagenes.py C [D ...]")
        raise SystemExit(1)
    raise SystemExit(main([a.upper() for a in sys.argv[1:]]))
