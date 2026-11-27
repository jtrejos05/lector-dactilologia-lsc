# ============================================================================
#  Tuberia de vision clasica
# ----------------------------------------------------------------------------
#  Cada funcion de este modulo corresponde a un tema del curso, y juntas forman
#  una tuberia que localiza la mano SIN usar redes neuronales:
#
#     filtrado -> histograma -> umbralizacion -> morfologia -> segmentacion
#              -> caracterizacion
#
#  El contraste con la via de MediaPipe (senas/manos.py) es justamente lo
#  interesante del proyecto: la misma tarea resuelta con las dos familias de
#  tecnicas, sobre los mismos fotogramas.
# ============================================================================

from __future__ import annotations

import cv2
import numpy as np

# ============================================================================
#  TEMA: FILTRADO
# ============================================================================

# Nombres de los filtros, en el orden en que los recorre la interfaz.
FILTROS = ["ninguno", "gaussiano", "mediana", "bilateral", "nitidez"]


def aplicar_filtro(img: np.ndarray, nombre: str, k: int = 5) -> np.ndarray:
    """
    Suavizado o realce previo a segmentar.

    El tamano de nucleo k debe ser impar; la interfaz lo fuerza, pero lo
    reforzamos aqui porque OpenCV lanza excepcion con nucleos pares.

    Por que importa en esta tuberia: la umbralizacion que viene despues es
    sensible al ruido de la camara. Suavizar antes evita que aparezcan miles
    de motas sueltas que luego hay que limpiar con morfologia.
    """
    if k % 2 == 0:
        k += 1

    if nombre == "gaussiano":
        # Nucleo gaussiano: suaviza fuerte, pero difumina tambien los bordes.
        return cv2.GaussianBlur(img, (k, k), 0)

    if nombre == "mediana":
        # Sustituye cada pixel por la mediana de su vecindad. Es el mejor
        # contra el ruido impulsivo (sal y pimienta) y respeta los bordes
        # mucho mejor que el gaussiano.
        return cv2.medianBlur(img, k)

    if nombre == "bilateral":
        # Promedia solo entre pixeles de intensidad parecida: suaviza las
        # superficies y DEJA los bordes nitidos. Es el mas caro de los tres.
        return cv2.bilateralFilter(img, k, k * 15, k * 15)

    if nombre == "nitidez":
        # Nucleo de realce (laplaciano invertido). Aqui se ve que un filtro
        # no es mas que una convolucion con una matriz pequena.
        nucleo = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]], np.float32)
        return cv2.filter2D(img, -1, nucleo)

    return img.copy()


# ============================================================================
#  TEMA: HISTOGRAMAS
# ============================================================================

def histograma_gris(gris: np.ndarray, ancho=256, alto=140) -> np.ndarray:
    """Dibuja el histograma de intensidades como una imagen para mostrarla."""
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256]).flatten()
    if hist.max() > 0:
        hist = hist / hist.max() * (alto - 10)

    lienzo = np.full((alto, ancho, 3), 25, np.uint8)
    for x in range(256):
        cv2.line(lienzo, (x, alto), (x, alto - int(hist[x])), (180, 180, 180), 1)
    return lienzo


def ecualizar(gris: np.ndarray, metodo: str = "global") -> np.ndarray:
    """
    Redistribuye las intensidades para ganar contraste.

    Esto es directamente util en este proyecto: los fotogramas grabados a
    contraluz salen oscuros y con el histograma apelotonado en la zona baja,
    que es justo donde la umbralizacion falla.
    """
    if metodo == "global":
        # Ecualizacion clasica: usa la CDF del histograma completo.
        return cv2.equalizeHist(gris)
    if metodo == "clahe":
        # Por regiones y con limite de amplificacion: no revienta el ruido de
        # las zonas planas, que es el defecto de la ecualizacion global.
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
    return gris


# ============================================================================
#  TEMA: UMBRALIZACION Y BINARIZACION
# ============================================================================

UMBRALES = ["manual", "otsu", "adaptativo_media", "adaptativo_gauss"]


def binarizar(gris: np.ndarray, metodo: str, valor: int = 127,
              bloque: int = 31, c: int = 5):
    """
    Convierte la imagen en blanco y negro. Devuelve (binaria, umbral_usado).

    'otsu' calcula el umbral solo, buscando el que mejor separa las dos modas
    del histograma: por eso solo funciona bien si la imagen es realmente
    bimodal. Los adaptativos calculan un umbral distinto por region, que es lo
    que salva las escenas con iluminacion desigual.
    """
    if bloque % 2 == 0:
        bloque += 1
    if bloque < 3:
        bloque = 3

    if metodo == "otsu":
        usado, binaria = cv2.threshold(gris, 0, 255,
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaria, int(usado)

    if metodo == "adaptativo_media":
        binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, bloque, c)
        return binaria, -1

    if metodo == "adaptativo_gauss":
        binaria = cv2.adaptiveThreshold(gris, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, bloque, c)
        return binaria, -1

    _, binaria = cv2.threshold(gris, valor, 255, cv2.THRESH_BINARY)
    return binaria, valor


# ============================================================================
#  TEMA: MORFOLOGIA MATEMATICA
# ============================================================================

MORFOLOGIAS = ["ninguna", "erosion", "dilatacion", "apertura", "cierre", "gradiente"]
FORMAS_NUCLEO = {"rect": cv2.MORPH_RECT,
                 "elipse": cv2.MORPH_ELLIPSE,
                 "cruz": cv2.MORPH_CROSS}


def nucleo(forma: str = "elipse", k: int = 5) -> np.ndarray:
    """Elemento estructurante. Su forma y tamano definen toda la operacion."""
    return cv2.getStructuringElement(FORMAS_NUCLEO.get(forma, cv2.MORPH_ELLIPSE),
                                     (max(1, k), max(1, k)))


def morfologia(binaria: np.ndarray, operacion: str,
               forma: str = "elipse", k: int = 5,
               iteraciones: int = 1) -> np.ndarray:
    """
    Operaciones sobre la imagen binaria.

      erosion     adelgaza el objeto blanco; borra motas sueltas
      dilatacion  engorda el objeto; tapa agujeros pequenos
      apertura    erosion y luego dilatacion -> quita ruido SIN encoger
      cierre      dilatacion y luego erosion -> tapa huecos SIN engordar
      gradiente   dilatacion menos erosion -> deja el contorno

    Apertura y cierre son las que de verdad se usan para limpiar la mascara
    de la mano: la apertura mata las motas del fondo y el cierre rellena los
    huecos que deja la segmentacion entre los dedos.
    """
    if operacion == "ninguna":
        return binaria.copy()

    n = nucleo(forma, k)
    if operacion == "erosion":
        return cv2.erode(binaria, n, iterations=iteraciones)
    if operacion == "dilatacion":
        return cv2.dilate(binaria, n, iterations=iteraciones)
    if operacion == "apertura":
        return cv2.morphologyEx(binaria, cv2.MORPH_OPEN, n, iterations=iteraciones)
    if operacion == "cierre":
        return cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, n, iterations=iteraciones)
    if operacion == "gradiente":
        return cv2.morphologyEx(binaria, cv2.MORPH_GRADIENT, n)
    return binaria.copy()


# ============================================================================
#  TEMA: SEGMENTACION
# ============================================================================

# Rango de piel por defecto en YCrCb. Es un punto de partida de la
# literatura, NO una constante universal: depende del tono de piel, de la
# camara y de la luz. Por eso la interfaz lo deja ajustar en vivo.
RANGO_PIEL_YCRCB = {"y_min": 40, "y_max": 255, "cr_min": 133, "cr_max": 173,
                    "cb_min": 77, "cb_max": 127}


def mascara_piel(bgr: np.ndarray, usar_ycrcb: bool = True,
                 rango: dict | None = None) -> np.ndarray:
    """
    Segmenta piel por color.

    Se trabaja en YCrCb (o HSV) y no en BGR a proposito: esos espacios separan
    la CROMINANCIA de la LUMINANCIA, asi que el rango de piel se mantiene mas
    estable cuando cambia la iluminacion. En BGR, la misma piel bajo otra luz
    cae en otra zona del cubo de color y el rango deja de servir.

    Limitacion conocida, y es parte de lo que el proyecto quiere demostrar:
    esto segmenta CUALQUIER cosa con color de piel, incluida la cara del autor
    y los muebles de madera del fondo.
    """
    if usar_ycrcb:
        r = dict(RANGO_PIEL_YCRCB)
        if rango:
            r.update(rango)
        conv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        bajo = np.array([r["y_min"], r["cr_min"], r["cb_min"]], np.uint8)
        alto = np.array([r["y_max"], r["cr_max"], r["cb_max"]], np.uint8)
    else:
        conv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        bajo = np.array([0, 30, 60], np.uint8)
        alto = np.array([25, 160, 255], np.uint8)

    return cv2.inRange(conv, bajo, alto)


def region_mayor(binaria: np.ndarray):
    """
    Se queda con la componente conexa mas grande y devuelve (mascara, contorno).

    Tras la segmentacion por piel suele haber varias regiones (mano, cara,
    brazo, fondo). Nos quedamos con la mayor como heuristica de "esto es la
    mano", que falla en cuanto la cara ocupa mas que la mano: otra limitacion
    que conviene ensenar en lugar de esconder.
    """
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return np.zeros_like(binaria), None

    mayor = max(contornos, key=cv2.contourArea)
    if cv2.contourArea(mayor) < 1500:          # ruido, no es una mano
        return np.zeros_like(binaria), None

    mascara = np.zeros_like(binaria)
    cv2.drawContours(mascara, [mayor], -1, 255, -1)
    return mascara, mayor


# ============================================================================
#  TEMA: CARACTERIZACION
# ============================================================================

NOMBRES_CARACTERISTICAS = (
    ["area_norm", "perimetro_norm", "relacion_aspecto", "extension",
     "solidez", "circularidad", "dedos_estimados"] +
    ["hu%d" % i for i in range(7)]
)
DIM_CLASICAS = len(NOMBRES_CARACTERISTICAS)


def contar_dedos(contorno) -> tuple:
    """
    Cuenta dedos con defectos de convexidad.

    La envolvente convexa de una mano abierta es un poligono que "salta" por
    encima de los valles entre dedos. Cada valle profundo es un defecto de
    convexidad. Filtramos por el angulo del valle (< 90 grados) para quedarnos
    con los que de verdad separan dos dedos, y no con las irregularidades del
    contorno.

    Devuelve (numero_de_dedos, lista_de_puntos_de_valle) para poder dibujarlos.
    """
    if contorno is None or len(contorno) < 5:
        return 0, []

    envolvente = cv2.convexHull(contorno, returnPoints=False)
    if envolvente is None or len(envolvente) < 4:
        return 0, []

    envolvente = np.sort(envolvente.flatten())[::-1].reshape(-1, 1)
    try:
        defectos = cv2.convexityDefects(contorno, envolvente)
    except cv2.error:
        return 0, []
    if defectos is None:
        return 0, []

    valles = []
    for i in range(defectos.shape[0]):
        ini, fin, lejano, prof = defectos[i, 0]
        if prof < 2000:                      # valle poco profundo: no es un dedo
            continue
        a = np.array(contorno[ini][0], float)
        b = np.array(contorno[fin][0], float)
        c = np.array(contorno[lejano][0], float)

        # Angulo en el vertice del valle, por el teorema del coseno.
        la, lb, lc = np.linalg.norm(b - c), np.linalg.norm(a - c), np.linalg.norm(a - b)
        if la * lb == 0:
            continue
        coseno = (la ** 2 + lb ** 2 - lc ** 2) / (2 * la * lb)
        angulo = np.degrees(np.arccos(np.clip(coseno, -1, 1)))
        if angulo < 90:
            valles.append(tuple(contorno[lejano][0]))

    # n valles entre dedos => n+1 dedos, salvo que no haya ninguno.
    return (len(valles) + 1 if valles else 0), valles


def caracterizar(contorno, forma_img) -> np.ndarray | None:
    """
    Vector de 14 descriptores a partir del contorno segmentado.

    Todos son invariantes a la escala (se normalizan por el tamano de la
    imagen o son adimensionales) y los momentos de Hu lo son ademas a la
    rotacion. Ese es el mismo criterio que se aplico al vector de MediaPipe:
    si una caracteristica depende de donde esta la mano en el encuadre, el
    modelo aprende el encuadre en vez de la sena.
    """
    if contorno is None:
        return None

    alto, ancho = forma_img[:2]
    diagonal = float(np.hypot(alto, ancho))

    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    if area <= 0 or perimetro <= 0:
        return None

    x, y, w, h = cv2.boundingRect(contorno)
    envolvente = cv2.convexHull(contorno)
    area_envolvente = cv2.contourArea(envolvente)

    area_norm = area / float(alto * ancho)
    perimetro_norm = perimetro / diagonal
    relacion_aspecto = w / float(h) if h else 0.0
    extension = area / float(w * h) if w * h else 0.0
    # Solidez: que tan "llena" esta la envolvente. Una mano abierta tiene
    # solidez baja (los huecos entre dedos); un puno la tiene alta.
    solidez = area / area_envolvente if area_envolvente else 0.0
    circularidad = 4 * np.pi * area / (perimetro ** 2)

    dedos, _ = contar_dedos(contorno)

    # Momentos de Hu: 7 invariantes a traslacion, escala y rotacion. Abarcan
    # muchos ordenes de magnitud, asi que se comprimen en escala logaritmica.
    hu = cv2.HuMoments(cv2.moments(contorno)).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-30)

    vector = np.array([area_norm, perimetro_norm, relacion_aspecto, extension,
                       solidez, circularidad, dedos / 5.0] + hu.tolist(),
                      dtype=np.float32)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


# ============================================================================
#  La tuberia completa, encadenada
# ============================================================================

def tuberia(bgr: np.ndarray, cfg: dict) -> dict:
    """
    Ejecuta las seis etapas y devuelve todas las imagenes intermedias.

    Devolver los pasos intermedios (y no solo el resultado) es lo que permite
    que la interfaz muestre cada tema por separado.
    """
    filtrada = aplicar_filtro(bgr, cfg.get("filtro", "gaussiano"),
                              cfg.get("k_filtro", 5))

    piel = mascara_piel(filtrada, cfg.get("ycrcb", True))

    gris = cv2.cvtColor(filtrada, cv2.COLOR_BGR2GRAY)
    if cfg.get("ecualizar", "ninguna") != "ninguna":
        gris = ecualizar(gris, cfg["ecualizar"])

    binaria, umbral_usado = binarizar(piel, cfg.get("umbral", "otsu"),
                                      cfg.get("valor_umbral", 127),
                                      cfg.get("bloque", 31), cfg.get("c", 5))

    limpia = morfologia(binaria, cfg.get("morfologia", "apertura"),
                        cfg.get("forma_nucleo", "elipse"),
                        cfg.get("k_morf", 5), cfg.get("iteraciones", 1))

    mascara, contorno = region_mayor(limpia)
    vector = caracterizar(contorno, bgr.shape)

    return {"filtrada": filtrada, "gris": gris, "piel": piel,
            "binaria": binaria, "limpia": limpia, "mascara": mascara,
            "contorno": contorno, "vector": vector,
            "umbral_usado": umbral_usado}
