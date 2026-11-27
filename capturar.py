# ============================================================================
#  Captura del dataset con la webcam
# ----------------------------------------------------------------------------
#  Graba muestras de cada letra directamente desde tu camara. NO guarda
#  imagenes: guarda el vector de 78 caracteristicas ya normalizado. Eso hace
#  que el dataset pese kilobytes en vez de gigas, y que entrenar sea instantaneo.
#
#  Como usarlo
#  -----------
#    1. Pulsa la tecla de la letra que vas a grabar (A-Z).
#       ESPACIO  -> la sena de separacion de palabras (mano abierta al frente)
#       0        -> "nada": mano relajada, en reposo, poses que NO son letras.
#                   Esta clase es la que evita que el interprete escupa letras
#                   al azar mientras piensas. No te la saltes.
#    2. Pulsa ENTER para grabar una rafaga (2 s de cuenta atras + 60 fotogramas).
#    3. Repite la rafaga 3 o 4 veces POR LETRA, cambiando cada vez:
#       distancia a la camara, angulo de la mano, altura, y si puedes la luz.
#       Si grabas las 4 rafagas quieto en la misma pose, el modelo va a sacar
#       99% de acierto en el entrenamiento y va a fallar en la demo.
#
#  Teclas
#  ------
#    A-Z        seleccionar letra      ENTER      grabar rafaga
#    ESPACIO    separador de palabras  RETROCESO  deshacer la ultima rafaga
#    0          "nada"                 * *        borrar las muestras de la letra
#    TAB        ver el recuento        ESC        guardar y salir
#
#  Las letras NUNCA son comandos: pulsar D selecciona la D, no borra nada.
# ============================================================================

from __future__ import annotations

import csv
import os
import time
from collections import Counter

import cv2
import numpy as np

from senas import alfabeto
from senas.manos import DIM_CARACTERISTICAS, DetectorManos, dibujar_mano

RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(RAIZ, "datos", "muestras.csv")
DIR_IMAGENES = os.path.join(RAIZ, "datos", "imagenes")

# Guardar los fotogramas crudos permite calcular DESPUES otras caracteristicas
# (segmentacion de piel, morfologia, momentos de Hu...) sin volver a grabar.
# Es el seguro que deja abierta la via de la tuberia clasica.
GUARDAR_IMAGENES = True
ANCHO_IMAGEN = 640          # se reduce para que el disco no se dispare
CALIDAD_JPEG = 85

FOTOGRAMAS_POR_RAFAGA = 60
SEGUNDOS_CUENTA_ATRAS = 2.0

VERDE = (80, 220, 80)
ROJO = (60, 60, 235)
AMARILLO = (60, 220, 240)
BLANCO = (245, 245, 245)
GRIS = (150, 150, 150)


def cargar_muestras() -> list:
    """Lee el CSV existente para poder seguir ampliando el dataset."""
    if not os.path.exists(RUTA_CSV):
        return []
    filas = []
    with open(RUTA_CSV, "r", newline="", encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector, None)                       # cabecera
        for fila in lector:
            if len(fila) == DIM_CARACTERISTICAS + 2:
                filas.append(fila[:2] + [float(v) for v in fila[2:]])
    return filas


def guardar_muestras(filas: list) -> None:
    """Escribe el dataset completo al CSV.

    La columna "rafaga" identifica la toma de la que sale cada muestra, y no
    es decorativa: los 60 fotogramas de una rafaga son casi identicos, asi que
    si al partir entrenamiento y prueba se repartieran a ambos lados, el modelo
    se examinaria de fotogramas gemelos de los que ya vio y la precision
    saldria inflada. entrenar.py parte POR RAFAGA apoyandose en esta columna."""
    os.makedirs(os.path.dirname(RUTA_CSV), exist_ok=True)
    # La columna "rafaga" identifica la toma de la que sale cada muestra. Es
    # imprescindible para evaluar bien: los 60 fotogramas de una rafaga son casi
    # identicos, asi que si al partir train/test se reparten a ambos lados, el
    # modelo se examina de fotogramas gemelos de los que ya vio y la precision
    # sale inflada. entrenar.py parte POR RAFAGA para evitar esa fuga.
    cabecera = (["etiqueta", "rafaga"] +
                ["f%d" % i for i in range(DIM_CARACTERISTICAS)])
    with open(RUTA_CSV, "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(cabecera)
        escritor.writerows(filas)


def guardar_fotograma(frame, etiqueta: str, id_rafaga: str, indice: int) -> None:
    """
    Escribe el fotograma como JPEG reducido.

    No usamos cv2.imwrite: igual que la capa nativa de MediaPipe, no sabe
    manejar rutas con caracteres no ASCII en Windows, y este proyecto vive en
    una carpeta con enye. Codificamos en memoria con imencode y escribimos el
    archivo desde Python, que si maneja Unicode.
    """
    carpeta = os.path.join(DIR_IMAGENES, etiqueta)
    os.makedirs(carpeta, exist_ok=True)

    alto, ancho = frame.shape[:2]
    if ancho > ANCHO_IMAGEN:
        escala = ANCHO_IMAGEN / ancho
        frame = cv2.resize(frame, (ANCHO_IMAGEN, int(alto * escala)),
                           interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), CALIDAD_JPEG])
    if not ok:
        return
    ruta = os.path.join(carpeta, "%s_%03d.jpg" % (id_rafaga, indice))
    with open(ruta, "wb") as f:
        f.write(buf.tobytes())


def texto(frame, txt, org, escala=0.6, color=BLANCO, grosor=2):
    """Texto con sombra, para que se lea sobre cualquier fondo."""
    cv2.putText(frame, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                escala, color, grosor, cv2.LINE_AA)


def panel_recuento(filas: list) -> None:
    """Imprime en consola cuantas muestras hay de cada etiqueta."""
    cuenta = Counter(f[0] for f in filas)
    rafagas = Counter()
    for etiqueta, rafaga in {(f[0], f[1]) for f in filas}:
        rafagas[etiqueta] += 1

    print("\n--- dataset actual (total %d muestras) ---" % len(filas))
    print("  %-12s %6s %8s" % ("etiqueta", "muestras", "rafagas"))
    faltantes = []
    for etiqueta in alfabeto.ETIQUETAS:
        n, r = cuenta.get(etiqueta, 0), rafagas.get(etiqueta, 0)
        # Menos de 3 rafagas = poca variedad, aunque sobren muestras.
        marca = "  " if (n >= 120 and r >= 3) else ("~ " if n > 0 else "! ")
        if n < 120 or r < 3:
            faltantes.append(alfabeto.mostrar(etiqueta))
        print("  %s%-12s %6d %8d" % (marca, alfabeto.mostrar(etiqueta), n, r))
    if faltantes:
        print("\n  Faltos de muestras o de variedad: " + ", ".join(faltantes))
    print()


def main() -> int:
    """Bucle de captura: camara, maquina de estados y teclado.

    La grabacion es una maquina de tres estados. En "espera" se elige la letra;
    al pulsar ENTER se pasa a "cuenta_atras", que da dos segundos para colocar
    la mano; y luego a "grabando", que acumula 60 fotogramas en un buffer y, al
    llenarlo, los vuelca al CSV como una rafaga nueva.

    Se graba por rafagas y no muestra a muestra por dos motivos: permite variar
    las condiciones entre tomas (distancia, angulo, luz), que es de donde sale
    la variedad que el modelo necesita; y da la unidad de agrupacion con la que
    luego se evalua honestamente.

    Las letras NUNCA son comandos. Los comandos van en teclas que no son
    letras, porque antes estaban en G/U/D/Q y tapaban a esas cuatro: pulsar D
    para elegir la letra D ejecutaba el borrado y se llevaba por delante las
    muestras de la letra activa."""
    filas = cargar_muestras()
    if filas:
        print("Dataset existente: %d muestras. Se van a ampliar." % len(filas))
    panel_recuento(filas)

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camara.isOpened():
        print("No pude abrir la camara 0.")
        return 1
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    etiqueta_actual = alfabeto.ESTATICAS[0]
    estado = "espera"                 # espera | cuenta_atras | grabando
    t_inicio = 0.0
    id_rafaga = ""
    buffer_rafaga: list = []
    historial_rafagas: list = []      # cuantas filas metio cada rafaga (para deshacer)
    pendiente_borrado = False         # primer * armado, segundo * confirma

    print("Ventana abierta. Pulsa una letra y luego G para grabar.")

    with DetectorManos() as detector:
        while True:
            ok, frame = camara.read()
            if not ok:
                print("Se corto el video de la camara.")
                break

            # Vista espejo: es lo natural para el usuario. Ojo, MediaPipe ve la
            # imagen ya volteada, asi que la lateralidad que reporta es la de la
            # imagen espejada. Mientras capturemos y reconozcamos igual, coherente.
            frame = cv2.flip(frame, 1)
            alto, ancho = frame.shape[:2]

            # Copia sin anotaciones: lo que se guarda a disco tiene que ser la
            # imagen cruda, no una con el esqueleto y los rotulos pintados.
            frame_limpio = frame.copy()

            deteccion = detector.procesar(frame)
            if deteccion.hay_mano:
                dibujar_mano(frame, deteccion.puntos)

            # --- maquina de estados de la grabacion ---------------------------
            if estado == "cuenta_atras":
                restante = SEGUNDOS_CUENTA_ATRAS - (time.time() - t_inicio)
                if restante <= 0:
                    estado, buffer_rafaga = "grabando", []
                else:
                    n = int(restante) + 1
                    tam = cv2.getTextSize(str(n), cv2.FONT_HERSHEY_SIMPLEX, 4, 8)[0]
                    texto(frame, str(n), ((ancho - tam[0]) // 2, (alto + tam[1]) // 2),
                          4, AMARILLO, 8)

            elif estado == "grabando":
                vector = detector.caracteristicas(deteccion, frame.shape)
                if vector is not None:
                    if GUARDAR_IMAGENES:
                        guardar_fotograma(frame_limpio, etiqueta_actual,
                                          id_rafaga, len(buffer_rafaga))
                    buffer_rafaga.append([etiqueta_actual, id_rafaga] + vector.tolist())

                cv2.rectangle(frame, (0, 0), (ancho - 1, alto - 1), ROJO, 6)
                progreso = len(buffer_rafaga) / FOTOGRAMAS_POR_RAFAGA
                cv2.rectangle(frame, (0, alto - 12),
                              (int(ancho * progreso), alto), ROJO, -1)
                texto(frame, "GRABANDO  %d/%d" % (len(buffer_rafaga),
                                                  FOTOGRAMAS_POR_RAFAGA),
                      (20, alto - 30), 0.8, ROJO)

                if len(buffer_rafaga) >= FOTOGRAMAS_POR_RAFAGA:
                    filas.extend(buffer_rafaga)
                    historial_rafagas.append(len(buffer_rafaga))
                    guardar_muestras(filas)
                    print("  + %d muestras de '%s'  (total %d)"
                          % (len(buffer_rafaga), alfabeto.mostrar(etiqueta_actual),
                             len(filas)))
                    estado, buffer_rafaga = "espera", []

            # --- interfaz -----------------------------------------------------
            cv2.rectangle(frame, (0, 0), (ancho, 74), (28, 28, 28), -1)
            texto(frame, "Letra: " + alfabeto.mostrar(etiqueta_actual),
                  (20, 34), 0.9, VERDE if deteccion.hay_mano else GRIS)

            n_actual = sum(1 for f in filas if f[0] == etiqueta_actual)
            texto(frame, "muestras de esta letra: %d   |   total: %d"
                  % (n_actual, len(filas)), (20, 62), 0.55, BLANCO, 1)

            estado_mano = "mano detectada" if deteccion.hay_mano else "sin mano"
            texto(frame, estado_mano, (ancho - 250, 34), 0.6,
                  VERDE if deteccion.hay_mano else ROJO)

            if estado == "espera":
                texto(frame, "ENTER grabar   RETROCESO deshacer   TAB recuento   ESC salir",
                      (20, alto - 20), 0.55, GRIS, 1)

            cv2.imshow("Captura de dataset - lenguaje de senas", frame)

            # --- teclado ------------------------------------------------------
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == 255:
                continue

            # ORDEN IMPORTANTE: la seleccion de letra va PRIMERO. Antes los
            # comandos estaban en G/U/D/Q, que son letras del alfabeto y las
            # tapaban: pulsar D para elegir la letra D ejecutaba el borrado y
            # se llevaba por delante las muestras de la letra activa. Ahora
            # todos los comandos usan teclas que no son letras.
            letra = chr(tecla).upper() if 32 < tecla < 127 else ""
            if letra in alfabeto.ESTATICAS:
                etiqueta_actual = letra
                pendiente_borrado = False

            elif letra in alfabeto.DINAMICAS:
                pendiente_borrado = False
                print("  '%s' lleva movimiento (%s): fuera del modelo estatico."
                      % (letra, alfabeto.DINAMICAS[letra]))

            elif tecla == 27:                      # ESC = salir
                break

            elif tecla == ord(" "):
                etiqueta_actual, pendiente_borrado = alfabeto.ESPACIO, False

            elif tecla == ord("0"):
                etiqueta_actual, pendiente_borrado = alfabeto.NADA, False

            elif tecla == ord("	"):
                pendiente_borrado = False
                panel_recuento(filas)

            elif tecla == 13 and estado == "espera":          # ENTER = grabar
                pendiente_borrado = False
                estado, t_inicio = "cuenta_atras", time.time()
                id_rafaga = "%s-%s" % (etiqueta_actual,
                                       time.strftime("%Y%m%d%H%M%S"))

            elif tecla == 8 and estado == "espera":           # RETROCESO = deshacer
                pendiente_borrado = False
                if historial_rafagas:
                    n = historial_rafagas.pop()
                    del filas[-n:]
                    guardar_muestras(filas)
                    print("  - deshecha la ultima rafaga (%d muestras)" % n)
                else:
                    print("  no hay rafagas de esta sesion que deshacer")

            elif tecla == ord("*") and estado == "espera":    # * * = borrar letra
                # Borrado en dos pasos: destruye datos y no tiene deshacer.
                if not pendiente_borrado:
                    pendiente_borrado = True
                    print("  pulsa * otra vez para borrar TODAS las muestras de '%s'"
                          % alfabeto.mostrar(etiqueta_actual))
                else:
                    pendiente_borrado = False
                    antes = len(filas)
                    filas = [f for f in filas if f[0] != etiqueta_actual]
                    historial_rafagas.clear()
                    guardar_muestras(filas)
                    print("  borradas %d muestras de '%s'"
                          % (antes - len(filas), alfabeto.mostrar(etiqueta_actual)))

            else:
                    print("  '%s' lleva movimiento (%s): fuera del modelo estatico."
                          % (letra, alfabeto.DINAMICAS[letra]))

    camara.release()
    cv2.destroyAllWindows()
    guardar_muestras(filas)
    panel_recuento(filas)
    print("Guardado en: " + RUTA_CSV)
    print("Siguiente paso:  python entrenar.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
