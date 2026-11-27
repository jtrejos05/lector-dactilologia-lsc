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
    """Dibuja texto con contorno negro para que se lea sobre cualquier fondo.

    Se pinta dos veces: primero en negro con mayor grosor, y encima el
    color real. Es el truco habitual para rotular video, donde el fondo
    cambia en cada fotograma y un texto de un solo color acaba siendo
    ilegible sobre las zonas claras o las oscuras."""
    cv2.putText(img, txt, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX,
                escala, (0, 0, 0), grosor + 2, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, escala, color,
                grosor, cv2.LINE_AA)


def a_color(img):
    """Lleva una imagen de un canal a tres, para poder apilarla con las demas.

    Las mascaras binarias y las imagenes en escala de grises tienen un
    solo canal, mientras que los fotogramas de la camara tienen tres.
    numpy.hstack y numpy.vstack exigen que coincidan, asi que todo pasa
    por aqui antes de componer la rejilla."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def rotular(img, titulo, color=AMARILLO):
    """Pone una banda oscura con el nombre del panel en su borde superior.

    Cada panel de la interfaz muestra una etapa distinta de la tuberia;
    sin rotulo no hay forma de saber cual es cual."""
    img = a_color(img).copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (28, 28, 28), -1)
    texto(img, titulo, (8, 18), 0.5, color)
    return img


def rejilla(paneles, cols=2, ancho_celda=480):
    """Compone varios paneles en una cuadricula uniforme.

    Escala cada panel al mismo ancho conservando su relacion de aspecto,
    rellena por abajo los que queden mas bajos para que todas las celdas
    midan igual, y apila por filas. Sin igualar las alturas, hstack falla.

    Es lo que permite ver varias etapas del procesamiento a la vez, que es
    la idea central de la interfaz: comparar, no mirar una sola salida."""
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
    """Rehace la ventana de deslizadores al cambiar de modo.

    OpenCV no sabe eliminar un deslizador una vez creado: solo permite
    destruir la ventana entera. Por eso cada cambio de modo destruye la
    ventana de controles y la vuelve a crear con los deslizadores que ese
    modo necesita, definidos en el diccionario DESLIZADORES."""
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
    """Lee un deslizador, devolviendo un valor por defecto si no existe.

    Los modos sin deslizadores no crean la ventana de controles, y
    getTrackbarPos lanza cv2.error si se le pregunta por una ventana que
    no esta. Capturarlo aqui permite que las funciones de modo pidan
    cualquier parametro sin comprobar antes si existe."""
    try:
        return cv2.getTrackbarPos(nombre, CONTROLES)
    except cv2.error:
        return por_defecto


def guardar_png(img, nombre):
    """Escribe una captura de la vista actual en la carpeta de salidas.

    No se usa cv2.imwrite: igual que la capa nativa de MediaPipe, no sabe
    manejar rutas con caracteres no ASCII en Windows, y la carpeta original
    del proyecto lleva enye. Se codifica en memoria con imencode y se
    escribe el archivo desde Python, que si maneja Unicode."""
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
    """TEMA: MODELOS DE CAMARA Y CALIBRACION.

    Busca el patron de ajedrez en cada fotograma y, con las vistas que el
    usuario va capturando, estima los parametros de la camara.

    El modelo estenopeico describe la proyeccion de un punto 3D al sensor:

        s * [u v 1]^T = K * [R|t] * [X Y Z 1]^T

    K es la matriz intrinseca (focales fx, fy y centro optico cx, cy) y
    [R|t] la pose del tablero. Como el tablero es plano y conocemos la
    separacion real entre esquinas, varias vistas bastan para despejar K
    y los coeficientes de distorsion de la lente.

    El panel muestra, cuando ya hay calibracion: la imagen corregida, el
    mapa de desplazamiento que introduce la distorsion, y los parametros
    con su error de reproyeccion, que es la medida de calidad."""
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
    """TEMA: FILTRADO.

    Aplica gaussiano, mediana y bilateral al mismo fotograma y los muestra
    en paralelo con su tiempo de ejecucion, para que la comparacion sea
    directa y no de memoria.

    Los tres suavizan, pero de forma distinta: el gaussiano promedia con
    pesos segun la distancia y difumina tambien los bordes; la mediana
    toma el valor central de la vecindad, que elimina el ruido impulsivo
    respetando los bordes; el bilateral promedia solo entre pixeles de
    intensidad parecida, asi que alisa superficies y conserva los bordes,
    a cambio de ser bastante mas caro. Los milisegundos en pantalla lo
    hacen evidente al subir el tamano del nucleo."""
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
    """TEMA: HISTOGRAMAS.

    Muestra la imagen en gris junto a su ecualizacion global y su CLAHE,
    cada una con su histograma superpuesto y su media y desviacion.

    El histograma es la distribucion de intensidades. Ecualizar la
    redistribuye para ocupar todo el rango y ganar contraste: la global
    usa la funcion de distribucion acumulada de la imagen entera, y CLAHE
    lo hace por regiones con un limite de amplificacion, lo que evita que
    el ruido de las zonas planas se dispare.

    Importa en este proyecto porque los fotogramas grabados a contraluz
    tienen el histograma apelotonado en la zona baja, que es justo donde
    la umbralizacion posterior no encuentra un corte limpio."""
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
    """TEMA: UMBRALIZACION Y BINARIZACION.

    Compara umbral manual, Otsu y adaptativo gaussiano sobre el mismo
    fotograma, mas la mascara de piel por rango de color, y dibuja el
    histograma con el corte que Otsu ha elegido marcado en rojo.

    Un umbral global parte la imagen en dos con un unico valor. Otsu
    calcula ese valor automaticamente buscando el que minimiza la varianza
    dentro de cada clase, lo que equivale a separar las dos modas del
    histograma: funciona bien solo si la imagen es realmente bimodal. Los
    adaptativos calculan un umbral distinto por region, que es lo que
    salva las escenas con iluminacion desigual.

    La mascara de piel usa inRange, que es umbralizacion por rango en tres
    canales a la vez en lugar de un corte sobre un solo canal."""
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
    """TEMA: MORFOLOGIA MATEMATICA.

    Aplica erosion, dilatacion, apertura y cierre a la mascara de piel,
    con el elemento estructurante visualizado y el porcentaje de area que
    sobrevive a cada operacion.

    La morfologia opera sobre imagenes binarias haciendo recorrer una
    forma pequena, el elemento estructurante, por toda la imagen:

      erosion     un pixel sobrevive solo si el nucleo cabe entero dentro
                  del objeto; adelgaza y borra las motas sueltas
      dilatacion  la operacion contraria; engorda y tapa los huecos
      apertura    erosion y luego dilatacion: quita ruido SIN encoger
      cierre      dilatacion y luego erosion: rellena SIN engordar

    Apertura y cierre son las que de verdad limpian la mascara de la mano:
    la primera mata las motas del fondo y la segunda rellena los huecos
    que la segmentacion deja entre los dedos."""
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
    """Pone a cero todo lo que quede fuera de la region de interes.

    Se aplica sobre la mascara y no sobre el fotograma a proposito: asi la
    segmentacion solo compite dentro del recuadro, pero la imagen que se
    muestra sigue siendo la completa y se ve el contexto que se descarta."""
    if not activo:
        return mascara
    x0, y0, x1, y1 = recorte_roi(frame)
    fuera = np.zeros_like(mascara)
    fuera[y0:y1, x0:x1] = mascara[y0:y1, x0:x1]
    return fuera


def rango_de_deslizadores():
    """Lee de los deslizadores el rango de piel en YCrCb.

    El rango de piel no es una constante universal: depende del tono de
    piel, de la camara y de la iluminacion. Dejarlo fijo en el codigo
    serviria para una escena y fallaria en la siguiente, asi que se expone
    en la interfaz para ajustarlo en vivo mirando la mascara."""
    return {"cr_min": leer("Cr min", 133), "cr_max": leer("Cr max", 173),
            "cb_min": leer("Cb min", 77), "cb_max": leer("Cb max", 127),
            "y_min": leer("Y min", 40)}


def modo_segmentacion(frame, est):
    """TEMA: SEGMENTACION.

    Recorre la cadena completa que aisla la mano y muestra las seis etapas
    a la vez: original, mascara de piel, mascara limpiada con morfologia,
    contornos encontrados, region mayor y mano recortada.

    Se segmenta en YCrCb y no en BGR porque ese espacio separa la
    crominancia de la luminancia, de modo que el rango de piel aguanta
    mejor los cambios de iluminacion. En BGR la misma piel bajo otra luz
    cae en otra zona del cubo de color y el rango deja de servir.

    La heuristica final es quedarse con la componente conexa mayor. Falla
    en cuanto la cara ocupa mas que la mano, y por eso existe la region de
    interes: es una limitacion real del enfoque clasico, declarada en vez
    de escondida."""
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
    """TEMA: CARACTERIZACION.

    Extrae del contorno segmentado un vector de 14 descriptores y los
    muestra en una tabla junto al contorno, su envolvente convexa y los
    valles detectados entre dedos.

    Los descriptores son adimensionales o estan normalizados por el tamano
    de la imagen, y los momentos de Hu son ademas invariantes a rotacion.
    Esa invariancia es el requisito clave: si una caracteristica dependiera
    de donde esta la mano en el encuadre, el modelo aprenderia el encuadre
    en lugar de la forma de la sena.

    El conteo de dedos sale de los defectos de convexidad: la envolvente
    convexa salta por encima de los valles entre dedos, y filtrando esos
    valles por su angulo se separan los que de verdad dividen dos dedos de
    las simples irregularidades del contorno."""
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
    """Enfrenta la tuberia clasica con MediaPipe sobre el mismo fotograma.

    Es el resultado central del proyecto. Las dos vias resuelven la misma
    tarea y se miden en el mismo fotograma, con sus tiempos en pantalla.

    La clasica segmenta por COLOR: toma cualquier cosa con tono de piel,
    incluidas la cara del usuario y los muebles de madera, y depende de la
    iluminacion y del fondo. MediaPipe detecta la ESTRUCTURA de la mano
    con dos redes encadenadas, asi que ignora la cara y aguanta los
    cambios de luz.

    El contraste no es un fallo del proyecto: es su conclusion, y muestra
    con datos propios por que el campo paso de segmentar pixeles a usar
    landmarks aprendidos."""
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
    """Bucle principal: captura, despacha al modo activo y atiende el teclado.

    El estado vive en el diccionario est, que viaja a cada funcion de modo:
    las vistas del tablero capturadas, los parametros de calibracion y el
    detector de MediaPipe. Se pasa asi, y no en variables globales, para
    que cada modo sea una funcion pura de (fotograma, estado) -> imagen y
    se pueda probar sin camara.

    El despacho va por el diccionario DESPACHO, envuelto en try/except: si
    un modo falla, se dibuja el error en pantalla en lugar de tumbar la
    aplicacion entera, que en mitad de una demostracion importa.

    Teclas: 1-8 cambian de modo, G guarda una captura, ESC sale. En el
    modo de calibracion, ademas, C captura una vista del tablero, K lanza
    el calibrado y R descarta las vistas acumuladas."""
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
