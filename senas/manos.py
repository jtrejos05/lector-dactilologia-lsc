# ============================================================================
#  Deteccion de la mano y extraccion de caracteristicas
# ----------------------------------------------------------------------------
#  Este es el nucleo del proyecto. MediaPipe nos entrega 21 puntos 3D de la
#  mano; el trabajo de vision artificial de verdad esta en convertir esos 21
#  puntos en un vector de caracteristicas que NO dependa de donde esta la mano
#  ni de que tan cerca este de la camara. Si te saltas ese paso, el modelo
#  aprende "la mano estaba arriba a la izquierda" en vez de "los dedos estaban
#  en esta configuracion", y falla en cuanto te mueves.
#
#  IMPORTANTE sobre la version de MediaPipe:
#  A partir de MediaPipe 1.0 el API antiguo (mp.solutions.hands) YA NO EXISTE.
#  Todos los tutoriales que encuentres con "mp.solutions" son para 0.10.x y no
#  funcionan aqui. Este archivo usa el API nuevo (mediapipe.tasks), que ademas
#  necesita descargar el modelo hand_landmarker.task por separado.
# ============================================================================

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Ruta por defecto del modelo de landmarks (lo baja descargar_modelo.py).
RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUTA_MODELO = os.path.join(RAIZ, "modelo", "hand_landmarker.task")

# Indices de los 21 landmarks, segun la convencion de MediaPipe.
MUNECA = 0
NUDILLO_MEDIO = 9                      # base del dedo corazon: referencia de escala
PUNTAS = [4, 8, 12, 16, 20]            # pulgar, indice, corazon, anular, menique

# Dimension del vector de caracteristicas: 21 puntos * 3 coords + 15 distancias.
DIM_CARACTERISTICAS = 21 * 3 + 15


@dataclass
class Deteccion:
    """Resultado de mirar un fotograma."""
    puntos: np.ndarray | None      # (21, 3) coordenadas normalizadas de MediaPipe
    es_izquierda: bool             # mano detectada como izquierda
    confianza: float               # confianza de la lateralidad

    @property
    def hay_mano(self) -> bool:
        """True si MediaPipe encontro una mano en el fotograma."""
        return self.puntos is not None


def extraer_caracteristicas(puntos: np.ndarray, relacion_aspecto: float,
                            es_izquierda: bool) -> np.ndarray:
    """
    Convierte 21 landmarks crudos en un vector invariante a posicion, tamano y
    a que mano se use. Este es el paso que hace que el modelo generalice.

    Parametros
    ----------
    puntos : (21, 3) con x, y en [0,1] (normalizados por ancho y alto) y z
        relativo a la profundidad de la muneca.
    relacion_aspecto : ancho/alto del fotograma. Hace falta porque MediaPipe
        normaliza x e y por separado: en una imagen 16:9 eso ESTIRA la mano
        horizontalmente. Sin corregirlo, la misma sena produce vectores
        distintos en camaras con formatos distintos.
    es_izquierda : si es la mano izquierda reflejamos el eje X, para que zurdos
        y diestros compartan un solo modelo (si no, habria que duplicar el
        dataset capturando cada letra con las dos manos).

    Devuelve
    --------
    Vector de 78 dimensiones (DIM_CARACTERISTICAS).
    """
    p = puntos.astype(np.float64).copy()

    # 1) Corregir la distorsion de aspecto: llevamos x a la misma escala que y.
    p[:, 0] *= relacion_aspecto

    # 2) Reflejar la mano izquierda sobre el eje X.
    if es_izquierda:
        p[:, 0] = -p[:, 0]

    # 3) Invariancia a la posicion: la muneca pasa a ser el origen.
    p -= p[MUNECA]

    # 4) Invariancia a la escala (distancia a la camara). Usamos la distancia
    #    muneca -> nudillo del corazon como "regla" de la mano: es un segmento
    #    rigido, no cambia al doblar los dedos. Por eso es mejor referencia que
    #    el tamano de la caja contenedora, que si se encoge al cerrar el puno.
    escala = float(np.linalg.norm(p[NUDILLO_MEDIO]))
    if escala < 1e-6:
        escala = float(np.abs(p).max()) or 1.0
    p /= escala

    # 5) Distancias entre puntas de dedos, y de cada punta a la muneca.
    #    En teoria son redundantes con las coordenadas, pero le entregan al
    #    clasificador la relacion "dedos juntos vs separados" ya masticada, que
    #    es justo lo que separa los pares dificiles como U/V o M/N.
    distancias = []
    for i in range(len(PUNTAS)):
        for j in range(i + 1, len(PUNTAS)):
            distancias.append(np.linalg.norm(p[PUNTAS[i]] - p[PUNTAS[j]]))
    for t in PUNTAS:
        distancias.append(np.linalg.norm(p[t]))

    return np.concatenate([p.ravel(), np.array(distancias)])


class DetectorManos:
    """
    Envoltorio sobre HandLandmarker. Se usa como context manager:

        with DetectorManos() as det:
            deteccion = det.procesar(frame, marca_ms)
    """

    def __init__(self, ruta_modelo: str = RUTA_MODELO, en_vivo: bool = True,
                 confianza_deteccion: float = 0.5,
                 confianza_seguimiento: float = 0.5):
        """Construye el detector de MediaPipe a partir del modelo descargado.

    El modelo se carga en memoria y se pasan los BYTES, no la ruta. Motivo: la
    capa nativa de MediaPipe (C++) no decodifica bien las rutas con caracteres
    no ASCII en Windows, y la carpeta original de este proyecto lleva enye;
    falla con FileNotFoundError aunque el archivo exista y Python lo vea. Con
    model_asset_buffer la ruta la resuelve Python, que si maneja Unicode.

    En modo en_vivo se usa VIDEO, que mantiene seguimiento entre fotogramas:
    una vez encontrada la mano no vuelve a ejecutar el detector de palma en
    cada imagen, lo que lo hace bastante mas rapido y mas estable. Para
    fotogramas sueltos se usa IMAGE."""
        if not os.path.exists(ruta_modelo):
            raise FileNotFoundError(
                "No encuentro el modelo de landmarks en:\n  " + ruta_modelo +
                "\nEjecuta primero:  python descargar_modelo.py"
            )
        self._en_vivo = en_vivo

        # Cargamos el modelo en memoria y pasamos los BYTES, no la ruta.
        # Motivo: la carpeta de este proyecto se llama "LenguajeDesenas" con
        # enye, y la capa nativa de MediaPipe (C++) no decodifica bien las rutas
        # con caracteres no ASCII en Windows: recibe "LenguajeDese?as" y falla
        # con FileNotFoundError aunque el archivo exista y Python lo vea. Con
        # model_asset_buffer la ruta la resuelve Python, que si maneja Unicode,
        # y la libreria nativa nunca llega a verla.
        with open(ruta_modelo, "rb") as f:
            datos_modelo = f.read()

        opciones = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=datos_modelo),
            # VIDEO aprovecha el seguimiento entre fotogramas: es mas estable y
            # mas rapido que volver a detectar desde cero en cada imagen.
            running_mode=vision.RunningMode.VIDEO if en_vivo else vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=confianza_deteccion,
            min_tracking_confidence=confianza_seguimiento,
        )
        self._detector = vision.HandLandmarker.create_from_options(opciones)
        self._ultima_marca = -1

    def _siguiente_marca(self, marca_ms: int | None) -> int:
        """
        Devuelve una marca de tiempo ESTRICTAMENTE creciente.

        En modo VIDEO, MediaPipe exige que cada fotograma llegue con un
        timestamp mayor que el anterior; si no, lanza ValueError y se cae el
        programa. Dejarlo en manos de quien llama es una trampa: dos fotogramas
        procesados dentro del mismo milisegundo dan la misma marca y revientan.
        Ademas time.time() puede retroceder si el reloj del sistema se ajusta.
        Por eso usamos un reloj monotono y, por si acaso, forzamos el avance.
        """
        if marca_ms is None:
            marca_ms = int(time.monotonic() * 1000)
        if marca_ms <= self._ultima_marca:
            marca_ms = self._ultima_marca + 1
        self._ultima_marca = marca_ms
        return marca_ms

    def __enter__(self) -> "DetectorManos":
        """Permite usar el detector con "with", para que siempre se cierre."""
        return self

    def __exit__(self, *_) -> None:
        """Cierra el detector al salir del bloque "with", incluso si hubo error."""
        self.cerrar()

    def cerrar(self) -> None:
        """Libera el detector de MediaPipe.

    Conviene llamarlo explicitamente: mantiene recursos nativos que no se
    liberan solos cuando el recolector de basura descarta el objeto."""
        self._detector.close()

    def procesar(self, frame_bgr: np.ndarray,
                 marca_ms: int | None = None) -> Deteccion:
        """
        Analiza un fotograma BGR de OpenCV y devuelve la mano encontrada.

        marca_ms es opcional: si no se pasa, el detector lleva su propio reloj
        monotono. Lo normal es no pasarla.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self._en_vivo:
            resultado = self._detector.detect_for_video(
                imagen, self._siguiente_marca(marca_ms))
        else:
            resultado = self._detector.detect(imagen)

        if not resultado.hand_landmarks:
            return Deteccion(None, False, 0.0)

        marcas = resultado.hand_landmarks[0]
        puntos = np.array([[m.x, m.y, m.z] for m in marcas], dtype=np.float64)

        es_izquierda, confianza = False, 0.0
        if resultado.handedness:
            categoria = resultado.handedness[0][0]
            es_izquierda = categoria.category_name == "Left"
            confianza = float(categoria.score)

        return Deteccion(puntos, es_izquierda, confianza)

    def caracteristicas(self, deteccion: Deteccion, forma_frame) -> np.ndarray | None:
        """Atajo: pasa de una Deteccion al vector listo para el clasificador."""
        if not deteccion.hay_mano:
            return None
        alto, ancho = forma_frame[:2]
        return extraer_caracteristicas(deteccion.puntos, ancho / alto,
                                       deteccion.es_izquierda)


# ---------------------------------------------------------------------------
#  Dibujo del esqueleto de la mano sobre el fotograma
# ---------------------------------------------------------------------------

_CONEXIONES = [(c.start, c.end)
               for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def dibujar_mano(frame: np.ndarray, puntos: np.ndarray,
                 color_linea=(0, 255, 0), color_punto=(0, 128, 255)) -> None:
    """Pinta el esqueleto de 21 puntos sobre el fotograma, in place."""
    alto, ancho = frame.shape[:2]
    pix = [(int(x * ancho), int(y * alto)) for x, y, _ in puntos]

    for a, b in _CONEXIONES:
        cv2.line(frame, pix[a], pix[b], color_linea, 2, cv2.LINE_AA)
    for i, punto in enumerate(pix):
        radio = 6 if i in PUNTAS else 4
        cv2.circle(frame, punto, radio, color_punto, -1, cv2.LINE_AA)
