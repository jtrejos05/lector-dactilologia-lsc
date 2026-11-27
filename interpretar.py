# ============================================================================
#  Interprete en vivo: de senas a texto
# ----------------------------------------------------------------------------
#  Lee la webcam, reconoce la letra que estas haciendo y va escribiendo texto.
#
#  El problema que resuelve la parte de arriba de este archivo
#  -----------------------------------------------------------
#  Clasificar fotograma a fotograma da un resultado inservible: a 30 fps el
#  modelo emite 30 letras por segundo, y mientras mueves la mano de una sena a
#  la siguiente pasa por poses intermedias que se parecen a otras letras. El
#  resultado crudo es basura tipo "AAAAWWWSSSSSSDDDD".
#
#  La solucion tiene tres capas:
#    1. SUAVIZADO: no miramos un fotograma, sino la media de las probabilidades
#       de los ultimos N. Un fotograma malo suelto no cambia nada.
#    2. HISTERESIS: una letra solo se escribe si se mantiene estable durante
#       varios fotogramas seguidos y por encima de un umbral de confianza.
#    3. REARME: despues de escribir una letra hay que "soltar" (pasar por la
#       clase _NADA_ o bajar la mano) antes de poder escribir otra. Sin esto,
#       mantener la mano quieta escribiria la misma letra sin parar.
#
#  Teclas
#  ------
#    RETROCESO  borrar el ultimo caracter    C  limpiar todo
#    S          guardar el texto a archivo   D  ver/ocultar diagnostico
#    Q / ESC    salir
# ============================================================================

from __future__ import annotations

import os
import time
from collections import deque

import cv2
import numpy as np

from senas import alfabeto
from senas.manos import DetectorManos, dibujar_mano

RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO = os.path.join(RAIZ, "modelo", "clasificador.joblib")
DIR_SALIDAS = os.path.join(RAIZ, "salidas")

# --- parametros de las tres capas anti-ruido --------------------------------
VENTANA = 10            # fotogramas que se promedian (1)
UMBRAL_CONFIANZA = 0.70  # confianza minima para aceptar una letra (2)
FOTOGRAMAS_ESTABLE = 8   # fotogramas seguidos que hay que aguantar la sena (2)
UMBRAL_REARME = 0.55     # por debajo de esto se considera "mano suelta" (3)

VERDE = (80, 220, 80)
ROJO = (60, 60, 235)
AMARILLO = (60, 220, 240)
BLANCO = (245, 245, 245)
GRIS = (150, 150, 150)


def texto(frame, txt, org, escala=0.6, color=BLANCO, grosor=2):
    """Dibuja texto con contorno negro para que se lea sobre cualquier fondo."""
    cv2.putText(frame, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


class Estabilizador:
    """
    Implementa las tres capas descritas arriba. Se le dan las probabilidades
    de cada fotograma y decide cuando una letra esta lista para escribirse.
    """

    def __init__(self, etiquetas):
        """Prepara el historial y el estado de las tres capas antirruido.

    historial      ventana de probabilidades para el suavizado (capa 1)
    candidato      letra que se esta manteniendo ahora mismo
    seguidos       cuantos fotogramas lleva estable (capa 2)
    armado         si se puede volver a escribir; se desarma al escribir y se
                   rearma al soltar la sena (capa 3)
    ultima_escrita para distinguir repetir letra de cambiar de letra"""
        self.etiquetas = list(etiquetas)
        self.historial = deque(maxlen=VENTANA)
        self.candidato = None
        self.seguidos = 0
        self.armado = True          # False mientras no se suelte la sena anterior
        self.ultima_escrita = None  # para distinguir repetir de cambiar de letra

    def reiniciar(self):
        """Se llama cuando desaparece la mano: el estado deja de tener sentido."""
        self.historial.clear()
        self.candidato, self.seguidos = None, 0
        self.armado = True
        self.ultima_escrita = None

    def actualizar(self, probabilidades):
        """
        Devuelve (etiqueta_estable, confianza, progreso, letra_a_escribir).
        letra_a_escribir es None salvo en el fotograma exacto en que se confirma.
        """
        self.historial.append(probabilidades)
        media = np.mean(self.historial, axis=0)

        i = int(np.argmax(media))
        etiqueta, confianza = self.etiquetas[i], float(media[i])

        # Capa 3: rearme. Si la confianza cae, damos por soltada la sena.
        if confianza < UMBRAL_REARME or etiqueta == alfabeto.NADA:
            self.armado = True

        # Capa 2: histeresis. Contamos fotogramas seguidos del mismo candidato.
        if etiqueta == self.candidato:
            self.seguidos += 1
        else:
            self.candidato, self.seguidos = etiqueta, 1

        escribir = None
        listo = (self.seguidos >= FOTOGRAMAS_ESTABLE
                 and confianza >= UMBRAL_CONFIANZA
                 and etiqueta != alfabeto.NADA)

        # Repetir la MISMA letra exige haber soltado antes (self.armado).
        # Cambiar a OTRA letra se acepta siempre, en cuanto la nueva es estable:
        # no dependemos de que la confianza caiga durante la transicion, porque
        # entre dos senas parecidas puede no caer nunca y la segunda letra se
        # perderia. Un parpadeo de un par de fotogramas no llega a estable, asi
        # que esto no abre la puerta al ruido.
        if listo and (self.armado or etiqueta != self.ultima_escrita):
            escribir = etiqueta
            self.ultima_escrita = etiqueta
            self.armado = False       # hasta que no se suelte, no se repite
            self.seguidos = 0

        puede_escribir = self.armado or self.candidato != self.ultima_escrita
        progreso = 0.0
        if puede_escribir and etiqueta != alfabeto.NADA:
            progreso = min(1.0, self.seguidos / FOTOGRAMAS_ESTABLE)

        return etiqueta, confianza, progreso, escribir


def dibujar_barras(frame, media, etiquetas, x, y):
    """Diagnostico: las 5 clases mas probables del fotograma actual."""
    orden = np.argsort(media)[::-1][:5]
    for k, i in enumerate(orden):
        fila = y + k * 26
        cv2.rectangle(frame, (x, fila), (x + int(180 * media[i]), fila + 18),
                      VERDE if k == 0 else (90, 90, 90), -1)
        texto(frame, "%-10s %4.0f%%" % (alfabeto.mostrar(etiquetas[i]),
                                        100 * media[i]),
              (x + 190, fila + 15), 0.5, BLANCO, 1)


def main() -> int:
    """Bucle del interprete: reconoce senas y va componiendo el texto.

    Carga el clasificador entrenado, lee la camara, extrae el vector de
    caracteristicas de cada fotograma y se lo pasa al Estabilizador, que decide
    cuando una letra esta lo bastante confirmada como para escribirse.

    Sin ese estabilizador el sistema seria inservible: a 30 fotogramas por
    segundo el modelo emitiria 30 letras por segundo, y durante el movimiento
    entre dos senas la mano pasa por poses intermedias que se parecen a otras
    letras."""
    from joblib import load

    if not os.path.exists(RUTA_MODELO):
        print("No encuentro el clasificador en:\n  " + RUTA_MODELO)
        print("Entrenalo primero:  python entrenar.py")
        return 1

    paquete = load(RUTA_MODELO)
    modelo, etiquetas = paquete["modelo"], paquete["etiquetas"]
    print("Modelo cargado: %d clases, %d muestras, %.1f%% en validacion cruzada."
          % (len(etiquetas), paquete["n_muestras"], 100 * paquete["precision_cv"]))

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camara.isOpened():
        print("No pude abrir la camara 0.")
        return 1
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    estabilizador = Estabilizador(etiquetas)
    escrito = ""
    diagnostico = True
    media_actual = np.zeros(len(etiquetas))
    t_anterior, fps = time.time(), 0.0

    with DetectorManos() as detector:
        while True:
            ok, frame = camara.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)     # misma vista espejo que en la captura
            alto, ancho = frame.shape[:2]

            deteccion = detector.procesar(frame)

            etiqueta, confianza, progreso = None, 0.0, 0.0
            if deteccion.hay_mano:
                dibujar_mano(frame, deteccion.puntos)
                vector = detector.caracteristicas(deteccion, frame.shape)
                probabilidades = modelo.predict_proba(vector.reshape(1, -1))[0]

                etiqueta, confianza, progreso, nueva = \
                    estabilizador.actualizar(probabilidades)
                media_actual = np.mean(estabilizador.historial, axis=0)

                if nueva is not None:
                    escrito += " " if nueva == alfabeto.ESPACIO else nueva
                    print("  escrito: " + repr(escrito))
            else:
                estabilizador.reiniciar()

            # --- cabecera: que se esta viendo -------------------------------
            cv2.rectangle(frame, (0, 0), (ancho, 92), (28, 28, 28), -1)

            if deteccion.hay_mano:
                color = VERDE if confianza >= UMBRAL_CONFIANZA else AMARILLO
                texto(frame, alfabeto.mostrar(etiqueta), (24, 62), 1.6, color, 3)
                texto(frame, "%.0f%%" % (100 * confianza), (150, 62), 0.9, color)

                # Barra de "cuanto falta para que se escriba".
                cv2.rectangle(frame, (24, 74), (24 + 260, 84), (70, 70, 70), -1)
                cv2.rectangle(frame, (24, 74), (24 + int(260 * progreso), 84),
                              color, -1)
                bloqueado = (not estabilizador.armado
                             and etiqueta == estabilizador.ultima_escrita)
                if bloqueado:
                    texto(frame, "suelta la sena para repetir esta letra",
                          (300, 84), 0.5, GRIS, 1)
            else:
                texto(frame, "sin mano a la vista", (24, 56), 0.9, ROJO)

            texto(frame, "%.0f fps" % fps, (ancho - 110, 34), 0.6, GRIS, 1)

            if diagnostico and deteccion.hay_mano:
                dibujar_barras(frame, media_actual, etiquetas, 24, 110)

            # --- pie: el texto acumulado ------------------------------------
            cv2.rectangle(frame, (0, alto - 86), (ancho, alto), (28, 28, 28), -1)
            visible = escrito[-38:] if len(escrito) > 38 else escrito
            texto(frame, visible + "|", (24, alto - 42), 1.1, BLANCO, 2)
            texto(frame, "RETROCESO borrar   C limpiar   S guardar   "
                         "D diagnostico   Q salir",
                  (24, alto - 14), 0.5, GRIS, 1)

            cv2.imshow("Interprete de lenguaje de senas", frame)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 / max(ahora - t_anterior, 1e-6)
            t_anterior = ahora

            # --- teclado -----------------------------------------------------
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            elif tecla == 8:                      # retroceso
                escrito = escrito[:-1]
            elif tecla in (ord("c"), ord("C")):
                escrito = ""
            elif tecla in (ord("d"), ord("D")):
                diagnostico = not diagnostico
            elif tecla in (ord("s"), ord("S")) and escrito:
                os.makedirs(DIR_SALIDAS, exist_ok=True)
                ruta = os.path.join(DIR_SALIDAS, "transcripcion_%s.txt"
                                    % time.strftime("%Y%m%d_%H%M%S"))
                with open(ruta, "w", encoding="utf-8") as f:
                    f.write(escrito + "\n")
                print("Guardado en: " + ruta)

    camara.release()
    cv2.destroyAllWindows()
    print("\nTexto final: " + repr(escrito))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
