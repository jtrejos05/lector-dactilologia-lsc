

# ============================================================================
#  Definicion del alfabeto dactilologico a reconocer.
# ----------------------------------------------------------------------------
#  El clasificador de este proyecto es ESTATICO: mira una sola pose de la mano
#  en un instante. Por eso las letras que en lengua de senas se hacen con un
#  movimiento (una trayectoria en el tiempo) quedan fuera: no se pueden
#  distinguir mirando un unico fotograma.
#
#  Configurado para LSC (Lengua de Senas Colombiana). Si trabajas con LSE
#  (Espana) o ASL, la lista de letras es practicamente la misma; lo que cambia
#  es la forma de la mano de cada letra, y eso lo aprende el modelo de TUS
#  capturas, no esta escrito aqui.
# ============================================================================

# Letras con pose fija -> las que el modelo va a aprender.
ESTATICAS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y",
]

# Letras que llevan movimiento -> fuera del alcance del clasificador estatico.
# Se documentan aqui para que quede explicito que es una decision, no un olvido.
DINAMICAS = {
    "J":  "traza un gancho hacia abajo con el menique",
    "Z":  "dibuja una Z en el aire con el indice",
    "N~": "la N con un movimiento ondulado encima",
    "RR": "vibracion / repeticion de la R",
    "LL": "la L con desplazamiento lateral",
}

# Etiquetas especiales que no son letras pero hacen falta para deletrear.
ESPACIO = "_ESPACIO_"   # separador de palabras (sugerencia: mano abierta, palma al frente)
NADA    = "_NADA_"      # mano presente pero sin sena valida; evita falsos positivos

# Conjunto completo de etiquetas que el clasificador puede emitir.
ETIQUETAS = ESTATICAS + [ESPACIO, NADA]


def es_letra(etiqueta: str) -> bool:
    """True si la etiqueta representa una letra imprimible del alfabeto."""
    return etiqueta in ESTATICAS


def mostrar(etiqueta: str) -> str:
    """Convierte una etiqueta interna en algo legible en pantalla."""
    if etiqueta == ESPACIO:
        return "[espacio]"
    if etiqueta == NADA:
        return "-"
    return etiqueta
