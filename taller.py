# ============================================================================
#  Taller interactivo de procesamiento de imagenes
# ----------------------------------------------------------------------------
#  Interfaz para explorar, sobre la camara en vivo, cada tema del curso. Las
#  etapas no son demostraciones sueltas: encadenadas forman la tuberia clasica
#  que localiza la mano sin redes neuronales, y el ultimo modo la enfrenta a
#  la via de MediaPipe sobre el mismo fotograma.
#
#  Modos (teclas 1-8)
#  ------------------
#    1  Modelo de camara y calibracion
#    2  Filtrado
#    3  Histogramas
#    4  Umbralizacion y binarizacion
#    5  Morfologia matematica
#    6  Segmentacion
#    7  Caracterizacion
#    8  Comparacion: tuberia clasica frente a MediaPipe
#
#  Teclas comunes
#  --------------
#    1-8  cambiar de modo      G  guardar captura de la vista
#    TAB  siguiente variante   ESC  salir
#
#  Los deslizadores de la ventana "Controles" cambian segun el modo.
# ============================================================================

from __future__ import annotations

import os
import time

import cv2
import numpy as np

from senas import calibracion as cal
from senas import clasico

RAIZ = os.path.dirname(os.path.abspath(__file__))
DIR_SALIDAS = os.path.join(RAIZ, "salidas")

VENTANA = "Taller de procesamiento de imagenes"
CONTROLES = "Controles"

VERDE = (80, 220, 80)
ROJO = (60, 60, 235)
AMARILLO = (60, 220, 240)
BLANCO = (245, 245, 245)
GRIS = (150, 150, 150)
AZUL = (235, 170, 60)

MODOS = {
    ord("1"): ("Modelo de camara y calibracion", "calibracion"),
    ord("2"): ("Filtrado", "filtro"),
    ord("3"): ("Histogramas", "histograma"),
    ord("4"): ("Umbralizacion y binarizacion", "umbral"),
    ord("5"): ("Morfologia matematica", "morfologia"),
    ord("6"): ("Segmentacion", "segmentacion"),
    ord("7"): ("Caracterizacion", "caracterizacion"),
    ord("8"): ("Clasica frente a MediaPipe", "comparacion"),
}

# Deslizadores de cada modo: (nombre, maximo, valor inicial)
DESLIZADORES = {
    "filtro":         [("nucleo", 31, 5)],
    "histograma":     [],
    "umbral":         [("umbral manual", 255, 127), ("bloque", 99, 31), ("C", 50, 5)],
    "morfologia":     [("nucleo", 31, 5), ("iteraciones", 5, 1)],
    "segmentacion":   [("nucleo morf", 31, 7), ("area minima x100", 100, 15),
                       ("Cr min", 255, 133), ("Cr max", 255, 173),
                       ("Cb min", 255, 77), ("Cb max", 255, 127),
                       ("Y min", 255, 40), ("solo region (ROI)", 1, 1)],
    "caracterizacion": [("nucleo morf", 31, 7)],
    "comparacion":    [("nucleo morf", 31, 7)],
    "calibracion":    [],
}


def texto(img, txt, org, escala=0.55, color=BLANCO, grosor=1):
    cv2.putText(img, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color,
                grosor, cv2.LINE_AA)


def a_color(img):
    """Lleva una imagen de un canal a tres, para poder apilarla con las demas."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def rotular(img, titulo, color=AMARILLO):
    """Pone una banda con el nombre del panel."""
    img = a_color(img).copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (28, 28, 28), -1)
    texto(img, titulo, (8, 18), 0.5, color)
    return img


def rejilla(paneles, cols=2, ancho_celda=480):
    """Compone varios paneles en una cuadricula del mismo tamano."""
    celdas = []
    for p in paneles:
        p = a_color(p)
        alto = int(p.shape[0] * ancho_celda / p.shape[1])
        celdas.append(cv2.resize(p, (ancho_celda, alto)))

    altura = max(c.shape[0] for c in celdas)
    celdas = [cv2.copyMakeBorder(c, 0, altura - c.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
              for c in celdas]

    filas = []
    for i in range(0, len(celdas), cols):
        grupo = celdas[i:i + cols]
        while len(grupo) < cols:
            grupo.append(np.full_like(celdas[0], 20))
        filas.append(np.hstack(grupo))
    return np.vstack(filas)


def crear_controles(modo):
    """Rehace la ventana de deslizadores al cambiar de modo."""
    try:
        cv2.destroyWindow(CONTROLES)
    except cv2.error:
        pass
    deslizadores = DESLIZADORES.get(modo, [])
    if not deslizadores:
        return
    cv2.namedWindow(CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLES, 420, 60 + 50 * len(deslizadores))
    for nombre, maximo, inicial in deslizadores:
        cv2.createTrackbar(nombre, CONTROLES, inicial, maximo, lambda v: None)


def leer(nombre, por_defecto):
    try:
        return cv2.getTrackbarPos(nombre, CONTROLES)
    except cv2.error:
        return por_defecto


def guardar_png(img, nombre):
    os.makedirs(DIR_SALIDAS, exist_ok=True)
    ruta = os.path.join(DIR_SALIDAS, nombre)
    ok, buf = cv2.imencode(".png", img)
    if ok:
        # No cv2.imwrite: la ruta del proyecto lleva enye.
        with open(ruta, "wb") as f:
            f.write(buf.tobytes())
    return ruta


# ============================================================================
#  Un modo por tema
# ============================================================================

def modo_calibracion(frame, est):
    """Captura vistas del tablero, calibra y muestra el efecto."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vista = frame.copy()

    encontrado, esquinas = cal.detectar_esquinas(gris)
    if encontrado:
        cv2.drawChessboardCorners(vista, cal.PATRON, esquinas, True)
        texto(vista, "TABLERO DETECTADO - pulsa C para capturar esta vista",
              (12, vista.shape[0] - 18), 0.6, VERDE, 2)
    else:
        texto(vista, "muestra el patron de ajedrez a la camara",
              (12, vista.shape[0] - 18), 0.6, GRIS)
    est["esquinas_actuales"] = esquinas if encontrado else None

    paneles = [rotular(vista, "vistas capturadas: %d  (hacen falta 8+)"
                       % len(est["vistas"]), VERDE if encontrado else GRIS)]

    if est["K"] is not None:
        corregida = cal.desdistorsionar(frame, est["K"], est["dist"])
        paneles.append(rotular(corregida, "desdistorsionada"))
        paneles.append(rotular(cal.mapa_distorsion(est["K"], est["dist"],
                                                   frame.shape),
                               "desplazamiento por distorsion (x5)"))
        d = est["datos_cal"]
        K = est["K"]
        info = np.full((frame.shape[0], frame.shape[1], 3), 22, np.uint8)
        lineas = [
            "MATRIZ INTRINSECA K",
            "",
            "  fx = %8.2f    cx = %8.2f" % (K[0, 0], K[0, 2]),
            "  fy = %8.2f    cy = %8.2f" % (K[1, 1], K[1, 2]),
            "",
            "COEFICIENTES DE DISTORSION",
            "  k1=%.4f  k2=%.4f  k3=%.4f" % (est["dist"][0], est["dist"][1],
                                             est["dist"][4]),
            "  p1=%.4f  p2=%.4f" % (est["dist"][2], est["dist"][3]),
            "",
            "error de reproyeccion: %.3f px" % d["error_reproyeccion"],
            "  (por debajo de 1.0 es buena)",
            "vistas usadas: %d" % d["n_vistas"],
        ]
        for i, ln in enumerate(lineas):
            texto(info, ln, (20, 40 + i * 26), 0.55,
                  AMARILLO if ln.isupper() and ln else BLANCO)
        paneles.append(rotular(info, "parametros estimados"))
    else:
        ayuda = np.full((frame.shape[0], frame.shape[1], 3), 22, np.uint8)
        for i, ln in enumerate([
                "SIN CALIBRAR",
                "",
                "1. Abre datos/patron_ajedrez.png",
                "   (imprimelo o muestralo en el movil)",
                "2. Ensenaselo a la camara en angulos",
                "   y distancias distintas",
                "3. Pulsa C cada vez que lo detecte",
                "4. Con 8 o mas vistas, pulsa K",
                "",
                "R  descartar las vistas capturadas"]):
            texto(ayuda, ln, (20, 40 + i * 26), 0.55,
                  AMARILLO if i == 0 else BLANCO)
        paneles.append(rotular(ayuda, "como calibrar"))

    return rejilla(paneles, 2)


def modo_filtro(frame, est):
    k = max(1, leer("nucleo", 5))
    paneles = [rotular(frame, "original")]
    for nombre in ["gaussiano", "mediana", "bilateral"]:
        t0 = time.time()
        salida = clasico.aplicar_filtro(frame, nombre, k)
        ms = (time.time() - t0) * 1000
        paneles.append(rotular(salida, "%s  k=%d  (%.1f ms)"
                               % (nombre, k if k % 2 else k + 1, ms)))
    return rejilla(paneles, 2)


def modo_histograma(frame, est):
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    glob = clasico.ecualizar(gris, "global")
    clahe = clasico.ecualizar(gris, "clahe")

    paneles = []
    for img, nombre in [(gris, "gris original"), (glob, "ecualizacion global"),
                        (clahe, "CLAHE (por regiones)")]:
        panel = a_color(img)
        h = clasico.histograma_gris(img, ancho=panel.shape[1] // 2, alto=120)
        panel[panel.shape[0] - h.shape[0]:, :h.shape[1]] = h
        # Media y desviacion: la ecualizacion debe ensanchar la desviacion.
        texto(panel, "media %.0f   desv %.0f" % (img.mean(), img.std()),
              (10, panel.shape[0] - 130), 0.5, AMARILLO)
        paneles.append(rotular(panel, nombre))

    nota = np.full((frame.shape[0], frame.shape[1], 3), 22, np.uint8)
    for i, ln in enumerate([
            "PARA QUE SIRVE AQUI",
            "",
            "Los fotogramas grabados a contraluz",
            "salen oscuros: el histograma se",
            "apelotona en la zona baja y la",
            "umbralizacion no encuentra un corte",
            "limpio.",
            "",
            "La ecualizacion global reparte las",
            "intensidades, pero amplifica el ruido.",
            "CLAHE lo hace por regiones con un",
            "limite, y suele portarse mejor."]):
        texto(nota, ln, (20, 40 + i * 24), 0.5, AMARILLO if i == 0 else BLANCO)
    paneles.append(rotular(nota, "lectura"))
    return rejilla(paneles, 2)


def modo_umbral(frame, est):
    valor = leer("umbral manual", 127)
    bloque = max(3, leer("bloque", 31))
    c = leer("C", 5)

    piel = clasico.mascara_piel(frame)
    gris = cv2.cvtColor(clasico.aplicar_filtro(frame, "gaussiano", 5),
                        cv2.COLOR_BGR2GRAY)

    paneles = [rotular(frame, "original")]
    for metodo in ["manual", "otsu", "adaptativo_gauss"]:
        binaria, usado = clasico.binarizar(gris, metodo, valor, bloque, c)
        etiqueta = metodo if usado < 0 else "%s (umbral=%d)" % (metodo, usado)
        paneles.append(rotular(binaria, etiqueta))

    paneles.append(rotular(piel, "inRange sobre YCrCb (mascara de piel)", AZUL))
    hist = clasico.histograma_gris(gris, ancho=frame.shape[1], alto=frame.shape[0])
    _, otsu_val = clasico.binarizar(gris, "otsu")
    x = int(otsu_val / 255 * hist.shape[1])
    cv2.line(hist, (x, 0), (x, hist.shape[0]), ROJO, 2)
    texto(hist, "Otsu = %d" % otsu_val, (x + 8, 24), 0.55, ROJO)
    paneles.append(rotular(hist, "histograma y corte de Otsu"))
    return rejilla(paneles, 3)


def modo_morfologia(frame, est):
    k = max(1, leer("nucleo", 5))
    it = max(1, leer("iteraciones", 1))

    piel = clasico.mascara_piel(frame)
    paneles = [rotular(piel, "mascara de partida (con ruido)", AZUL)]
    for op in ["erosion", "dilatacion", "apertura", "cierre"]:
        salida = clasico.morfologia(piel, op, "elipse", k, it)
        blancos = int((salida > 0).sum() / max(1, (piel > 0).sum()) * 100)
        paneles.append(rotular(salida, "%s  k=%d  it=%d   (%d%% del area)"
                               % (op, k, it, blancos)))

    n = clasico.nucleo("elipse", k)
    vis = cv2.resize((n * 255).astype(np.uint8), (240, 240),
                     interpolation=cv2.INTER_NEAREST)
    lienzo = np.full((frame.shape[0], frame.shape[1], 3), 22, np.uint8)
    lienzo[30:270, 20:260] = a_color(vis)
    for i, ln in enumerate([
            "erosion  adelgaza, borra motas",
            "dilatacion  engorda, tapa huecos",
            "apertura  erosion+dilatacion:",
            "   quita ruido sin encoger",
            "cierre  dilatacion+erosion:",
            "   rellena sin engordar"]):
        texto(lienzo, ln, (290, 60 + i * 30), 0.5, BLANCO)
    texto(lienzo, "elemento estructurante %dx%d" % (k, k), (20, 300), 0.55, AMARILLO)
    paneles.append(rotular(lienzo, "nucleo y efecto de cada operacion"))
    return rejilla(paneles, 3)


def recorte_roi(frame):
    """
    Region de interes donde se espera la mano.

    Restringir la busqueda a un recuadro es la forma honesta de hacer viable
    la segmentacion por color: evita que la cara y el resto del fondo compitan
    con la mano por ser la region mayor. Es una simplificacion habitual en los
    sistemas clasicos de gestos, y conviene declararla como tal.
    """
    alto, ancho = frame.shape[:2]
    x0, y0 = int(ancho * 0.52), int(alto * 0.08)
    x1, y1 = int(ancho * 0.98), int(alto * 0.92)
    return x0, y0, x1, y1


def aplicar_roi(mascara, frame, activo):
    if not activo:
        return mascara
    x0, y0, x1, y1 = recorte_roi(frame)
    fuera = np.zeros_like(mascara)
    fuera[y0:y1, x0:x1] = mascara[y0:y1, x0:x1]
    return fuera


def rango_de_deslizadores():
    return {"cr_min": leer("Cr min", 133), "cr_max": leer("Cr max", 173),
            "cb_min": leer("Cb min", 77), "cb_max": leer("Cb max", 127),
            "y_min": leer("Y min", 40)}


def modo_segmentacion(frame, est):
    k = max(1, leer("nucleo morf", 7))
    area_min = leer("area minima x100", 15) * 100
    usar_roi = leer("solo region (ROI)", 1) == 1

    filtrada = clasico.aplicar_filtro(frame, "gaussiano", 5)
    piel = clasico.mascara_piel(filtrada, True, rango_de_deslizadores())
    piel = aplicar_roi(piel, frame, usar_roi)
    limpia = clasico.morfologia(piel, "apertura", "elipse", k)
    limpia = clasico.morfologia(limpia, "cierre", "elipse", k)

    contornos, _ = cv2.findContours(limpia, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    todos = frame.copy()
    for cnt in contornos:
        color = VERDE if cv2.contourArea(cnt) >= area_min else (90, 90, 90)
        cv2.drawContours(todos, [cnt], -1, color, 2)
    texto(todos, "%d regiones; en gris las descartadas por area" % len(contornos),
          (12, todos.shape[0] - 16), 0.5, BLANCO)

    mascara, contorno = clasico.region_mayor(limpia)
    recorte = cv2.bitwise_and(frame, frame, mask=mascara)
    if contorno is not None:
        cv2.drawContours(recorte, [contorno], -1, VERDE, 2)
        x, y, w, h = cv2.boundingRect(contorno)
        cv2.rectangle(recorte, (x, y), (x + w, y + h), AMARILLO, 2)

    original = frame.copy()
    if usar_roi:
        x0, y0, x1, y1 = recorte_roi(frame)
        cv2.rectangle(original, (x0, y0), (x1, y1), AMARILLO, 2)
        texto(original, "ROI: pon la mano aqui", (x0 + 8, y0 + 22), 0.5, AMARILLO)

    return rejilla([rotular(original, "original"),
                    rotular(piel, "segmentacion por color de piel", AZUL),
                    rotular(limpia, "tras apertura y cierre  k=%d" % k),
                    rotular(todos, "contornos encontrados"),
                    rotular(mascara, "region mayor"),
                    rotular(recorte, "mano aislada")], 3)


def modo_caracterizacion(frame, est):
    k = max(1, leer("nucleo morf", 7))
    cfg = {"filtro": "gaussiano", "k_filtro": 5, "umbral": "otsu",
           "morfologia": "apertura", "k_morf": k}
    r = clasico.tuberia(frame, cfg)
    contorno, vector = r["contorno"], r["vector"]

    vista = frame.copy()
    if contorno is not None:
        cv2.drawContours(vista, [contorno], -1, VERDE, 2)
        envolvente = cv2.convexHull(contorno)
        cv2.drawContours(vista, [envolvente], -1, AMARILLO, 2)
        dedos, valles = clasico.contar_dedos(contorno)
        for p in valles:
            cv2.circle(vista, p, 9, ROJO, -1)
        texto(vista, "verde contorno | amarillo envolvente | rojo valles",
              (12, vista.shape[0] - 40), 0.5, BLANCO)
        texto(vista, "dedos estimados: %d" % dedos,
              (12, vista.shape[0] - 16), 0.7, VERDE, 2)

    tabla = np.full((frame.shape[0], frame.shape[1], 3), 22, np.uint8)
    texto(tabla, "VECTOR DE CARACTERIZACION (%d)" % clasico.DIM_CLASICAS,
          (18, 32), 0.6, AMARILLO)
    if vector is not None:
        for i, (n, v) in enumerate(zip(clasico.NOMBRES_CARACTERISTICAS, vector)):
            texto(tabla, "%-18s %9.4f" % (n, v), (18, 64 + i * 25), 0.5, BLANCO)
    else:
        texto(tabla, "sin region valida en el fotograma", (18, 70), 0.55, ROJO)

    return rejilla([rotular(vista, "descriptores geometricos"),
                    rotular(tabla, "valores"),
                    rotular(r["limpia"], "mascara usada"),
                    rotular(r["piel"], "segmentacion de partida", AZUL)], 2)


def modo_comparacion(frame, est):
    k = max(1, leer("nucleo morf", 7))

    t0 = time.time()
    r = clasico.tuberia(frame, {"filtro": "gaussiano", "k_filtro": 5,
                                "umbral": "otsu", "morfologia": "apertura",
                                "k_morf": k})
    ms_clasico = (time.time() - t0) * 1000
    vista_c = frame.copy()
    if r["contorno"] is not None:
        cv2.drawContours(vista_c, [r["contorno"]], -1, VERDE, 2)

    ms_mp, vista_m, hay = 0.0, frame.copy(), False
    if est["detector"] is not None:
        from senas.manos import dibujar_mano
        t0 = time.time()
        det = est["detector"].procesar(frame)
        ms_mp = (time.time() - t0) * 1000
        hay = det.hay_mano
        if hay:
            dibujar_mano(vista_m, det.puntos)

    nota = np.full((frame.shape[0], frame.shape[1], 3), 22, np.uint8)
    for i, ln in enumerate([
            "CLASICA frente a MEDIAPIPE",
            "",
            "clasica:   %6.1f ms" % ms_clasico,
            "mediapipe: %6.1f ms" % ms_mp,
            "",
            "La clasica segmenta por COLOR, asi",
            "que toma la cara y cualquier objeto",
            "con tono de piel. Depende de la luz",
            "y del fondo.",
            "",
            "MediaPipe detecta la ESTRUCTURA de",
            "la mano, asi que ignora la cara y",
            "aguanta cambios de iluminacion.",
            "",
            "Ese contraste es el resultado del",
            "proyecto, no un fallo."]):
        texto(nota, ln, (18, 36 + i * 23), 0.48,
              AMARILLO if i == 0 else BLANCO)

    return rejilla([rotular(vista_c, "clasica: contorno por color  (%.1f ms)"
                            % ms_clasico, AZUL),
                    rotular(vista_m, "mediapipe: landmarks  (%.1f ms)%s"
                            % (ms_mp, "" if hay else "  SIN MANO"), VERDE),
                    rotular(r["limpia"], "mascara de la clasica"),
                    rotular(nota, "lectura")], 2)


DESPACHO = {
    "calibracion": modo_calibracion, "filtro": modo_filtro,
    "histograma": modo_histograma, "umbral": modo_umbral,
    "morfologia": modo_morfologia, "segmentacion": modo_segmentacion,
    "caracterizacion": modo_caracterizacion, "comparacion": modo_comparacion,
}


def main() -> int:
    patron = os.path.join(RAIZ, "datos", "patron_ajedrez.png")
    if not os.path.exists(patron):
        cal.generar_tablero(patron)
        print("Patron de ajedrez generado en: " + patron)

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camara.isOpened():
        print("No pude abrir la camara 0.")
        return 1
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    K, dist, datos_cal = cal.cargar()
    if K is not None:
        print("Calibracion cargada (error %.3f px)" % datos_cal["error_reproyeccion"])

    detector = None
    try:
        from senas.manos import DetectorManos
        detector = DetectorManos()
    except Exception as e:
        print("MediaPipe no disponible, el modo 8 ira sin el: %s" % e)

    est = {"vistas": [], "esquinas_actuales": None, "K": K, "dist": dist,
           "datos_cal": datos_cal, "detector": detector}

    modo, titulo = "filtro", "Filtrado"
    cv2.namedWindow(VENTANA, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA, 1280, 760)
    crear_controles(modo)
    print("Taller abierto. Teclas 1-8 para cambiar de tema, ESC para salir.")

    while True:
        ok, frame = camara.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)

        try:
            vista = DESPACHO[modo](frame, est)
        except Exception as e:                   # que un modo falle no tumba todo
            vista = np.full((480, 960, 3), 22, np.uint8)
            texto(vista, "error en el modo '%s': %s" % (modo, e), (20, 60),
                  0.6, ROJO)

        barra = np.full((40, vista.shape[1], 3), 28, np.uint8)
        texto(barra, "[%s]  %s" % (modo, titulo), (14, 26), 0.6, VERDE)
        texto(barra, "1-8 temas   G guardar   ESC salir",
              (vista.shape[1] - 330, 26), 0.5, GRIS)
        cv2.imshow(VENTANA, np.vstack([barra, vista]))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == 255:
            continue

        if tecla == 27:
            break

        elif tecla in MODOS:
            titulo, modo = MODOS[tecla]
            crear_controles(modo)

        elif tecla in (ord("g"), ord("G")):
            ruta = guardar_png(np.vstack([barra, vista]),
                               "taller_%s_%s.png" % (modo, time.strftime("%H%M%S")))
            print("  captura guardada: " + ruta)

        elif tecla in (ord("c"), ord("C")) and modo == "calibracion":
            if est["esquinas_actuales"] is not None:
                est["vistas"].append(est["esquinas_actuales"])
                print("  vista %d capturada" % len(est["vistas"]))
            else:
                print("  no se ve el tablero en este fotograma")

        elif tecla in (ord("k"), ord("K")) and modo == "calibracion":
            if len(est["vistas"]) < 8:
                print("  hacen falta al menos 8 vistas (hay %d)" % len(est["vistas"]))
            else:
                print("  calibrando con %d vistas..." % len(est["vistas"]))
                datos = cal.calibrar(est["vistas"], frame.shape)
                cal.guardar(datos)
                est["K"] = np.array(datos["K"])
                est["dist"] = np.array(datos["dist"])
                est["datos_cal"] = datos
                print("  listo. error de reproyeccion: %.3f px"
                      % datos["error_reproyeccion"])

        elif tecla in (ord("r"), ord("R")) and modo == "calibracion":
            est["vistas"].clear()
            print("  vistas descartadas")

    camara.release()
    if detector is not None:
        detector.cerrar()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
