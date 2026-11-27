# ============================================================================
#  Modelo de camara y calibracion
# ----------------------------------------------------------------------------
#  Estima los parametros INTRINSECOS de la camara y los coeficientes de
#  distorsion a partir de varias vistas de un patron de ajedrez.
#
#  El modelo de camara estenopeica (pinhole) dice que un punto 3D del mundo
#  se proyecta al sensor segun:
#
#        s * [u v 1]^T  =  K * [R | t] * [X Y Z 1]^T
#
#  donde K es la matriz intrinseca:
#
#        K = | fx   0  cx |      fx, fy  distancias focales en pixeles
#            |  0  fy  cy |      cx, cy  centro optico (punto principal)
#            |  0   0   1 |
#
#  A eso se le suma la distorsion de la lente, que el modelo recoge en los
#  coeficientes (k1, k2, p1, p2, k3): radial (las rectas se curvan hacia los
#  bordes) y tangencial (el sensor no es perfectamente paralelo a la lente).
#
#  Calibrar sirve para DESDISTORSIONAR la imagen, de modo que las lineas
#  rectas del mundo salgan rectas en la imagen. En este proyecto importa
#  porque la caracterizacion mide formas: si la lente curva el contorno de la
#  mano, los descriptores geometricos quedan sesgados segun en que parte del
#  encuadre este la mano.
# ============================================================================

from __future__ import annotations

import json
import os

import cv2
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_CALIBRACION = os.path.join(RAIZ, "datos", "calibracion.json")

# Esquinas INTERIORES del tablero, no casillas. Un tablero de 10x7 casillas
# tiene 9x6 esquinas interiores.
PATRON = (9, 6)
LADO_CASILLA_MM = 25.0


def generar_tablero(ruta: str, casillas=(10, 7), lado_px: int = 100) -> str:
    """
    Crea la imagen del patron de ajedrez, para imprimir o mostrar en pantalla.

    Asi no hay que buscar un tablero por internet. Si lo muestras en el movil,
    pon el brillo alto y evita los reflejos: el detector necesita contraste.
    """
    cols, filas = casillas
    img = np.zeros((filas * lado_px, cols * lado_px), np.uint8)
    for f in range(filas):
        for c in range(cols):
            if (f + c) % 2 == 0:
                img[f * lado_px:(f + 1) * lado_px,
                    c * lado_px:(c + 1) * lado_px] = 255

    # Margen blanco: el detector necesita ver el borde exterior del tablero.
    margen = lado_px // 2
    lienzo = np.full((img.shape[0] + 2 * margen, img.shape[1] + 2 * margen),
                     255, np.uint8)
    lienzo[margen:margen + img.shape[0], margen:margen + img.shape[1]] = img

    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    ok, buf = cv2.imencode(".png", lienzo)
    if ok:
        # No cv2.imwrite: la ruta del proyecto lleva enye.
        with open(ruta, "wb") as f:
            f.write(buf.tobytes())
    return ruta


def detectar_esquinas(gris: np.ndarray):
    """Busca el patron y refina las esquinas a precision subpixel."""
    ok, esquinas = cv2.findChessboardCorners(
        gris, PATRON,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK)
    if not ok:
        return False, None

    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    esquinas = cv2.cornerSubPix(gris, esquinas, (11, 11), (-1, -1), criterio)
    return True, esquinas


def puntos_objeto() -> np.ndarray:
    """
    Coordenadas 3D del tablero en su propio sistema, con Z=0.

    El tablero es plano, asi que basta con fijar Z=0 y dar las posiciones en
    milimetros reales. Esa escala es la que hace que las focales salgan en
    unidades fisicas coherentes.
    """
    pts = np.zeros((PATRON[0] * PATRON[1], 3), np.float32)
    pts[:, :2] = np.mgrid[0:PATRON[0], 0:PATRON[1]].T.reshape(-1, 2)
    return pts * LADO_CASILLA_MM


def calibrar(lista_esquinas: list, forma_img: tuple) -> dict:
    """
    Resuelve la calibracion y devuelve K, los coeficientes y el error.

    El error de reproyeccion es la medida de calidad: se reproyectan los
    puntos 3D con los parametros estimados y se mide cuanto se desvian de las
    esquinas detectadas. Por debajo de 1 pixel es una buena calibracion.
    """
    objetos = [puntos_objeto() for _ in lista_esquinas]
    alto, ancho = forma_img[:2]

    error, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objetos, lista_esquinas, (ancho, alto), None, None)

    return {"K": K.tolist(),
            "dist": dist.flatten().tolist(),
            "error_reproyeccion": float(error),
            "resolucion": [int(ancho), int(alto)],
            "n_vistas": len(lista_esquinas)}


def guardar(datos: dict, ruta: str = RUTA_CALIBRACION) -> None:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2)


def cargar(ruta: str = RUTA_CALIBRACION):
    """Devuelve (K, dist, datos) o (None, None, None) si no hay calibracion."""
    if not os.path.exists(ruta):
        return None, None, None
    with open(ruta, "r", encoding="utf-8") as f:
        datos = json.load(f)
    return (np.array(datos["K"], np.float64),
            np.array(datos["dist"], np.float64), datos)


def desdistorsionar(img: np.ndarray, K, dist):
    """
    Corrige la distorsion de la lente.

    getOptimalNewCameraMatrix decide cuanto recortar: con alpha=0 se queda
    solo con la zona valida (sin bordes negros) y con alpha=1 conserva todos
    los pixeles pero deja esquinas vacias. Usamos un punto intermedio.
    """
    if K is None or dist is None:
        return img
    alto, ancho = img.shape[:2]
    nueva_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (ancho, alto), 0.3,
                                                 (ancho, alto))
    salida = cv2.undistort(img, K, dist, None, nueva_K)
    x, y, w, h = roi
    if w > 0 and h > 0:
        salida = salida[y:y + h, x:x + w]
        salida = cv2.resize(salida, (ancho, alto))
    return salida


def mapa_distorsion(K, dist, forma, paso: int = 40) -> np.ndarray:
    """
    Dibuja cuanto desplaza la lente cada punto de la imagen.

    Es la forma de VER la distorsion: una rejilla de flechas que muestra a
    donde se mueve cada pixel al corregirla. Util para la sustentacion,
    porque la distorsion de una webcam suele ser demasiado sutil para notarla
    comparando las dos imagenes a ojo.
    """
    alto, ancho = forma[:2]
    lienzo = np.full((alto, ancho, 3), 20, np.uint8)
    if K is None or dist is None:
        return lienzo

    ys, xs = np.mgrid[paso:alto:paso, paso:ancho:paso]
    puntos = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)

    corregidos = cv2.undistortPoints(puntos.reshape(-1, 1, 2), K, dist, P=K)
    corregidos = corregidos.reshape(-1, 2)

    for (x0, y0), (x1, y1) in zip(puntos, corregidos):
        # Se amplifica x5 para que el desplazamiento se aprecie.
        dx, dy = (x1 - x0) * 5, (y1 - y0) * 5
        cv2.arrowedLine(lienzo, (int(x0), int(y0)),
                        (int(x0 + dx), int(y0 + dy)),
                        (80, 200, 255), 1, tipLength=0.35)
        cv2.circle(lienzo, (int(x0), int(y0)), 1, (90, 90, 90), -1)

    return lienzo
