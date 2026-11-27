# Lector de dactilología — Lengua de Señas Colombiana (LSC)

**Autor:** Juan Trejos ([@jtrejos05](https://github.com/jtrejos05))
**Asignatura:** Visión Artificial / Procesamiento de Imágenes
**Entrega:** proyecto final

Sistema de visión por computador que reconoce en tiempo real las letras del
alfabeto dactilológico de la LSC desde una webcam y las convierte en texto.

El proyecto implementa **dos tuberías independientes** que resuelven la misma
tarea y se comparan entre sí:

1. **Tubería clásica** — filtrado, histogramas, umbralización, morfología,
   segmentación y caracterización, sin redes neuronales.
2. **Tubería moderna** — detección de 21 puntos de la mano con MediaPipe y
   clasificación con un perceptrón multicapa.

La comparación entre ambas es el resultado central del trabajo.

---

## Precisión terminológica

Esto es un **lector de dactilología**, no un intérprete de lengua de señas.

La dactilología es el deletreo manual letra a letra, y es una parte real de la
LSC: se usa para nombres propios, siglas y palabras sin seña propia. Pero
**señar no es deletrear**. En LSC, «hola» tiene su propia seña —un gesto único
con movimiento— y nadie la deletrea en una conversación.

Reconocer señas de palabras completas exigiría modelar la dimensión temporal,
que queda fuera del alcance de este trabajo.

---

## Arquitectura

### Vista general

```
                      ┌──────────────┐
   webcam ──────────► │  fotograma   │
                      └──────┬───────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
   TUBERÍA CLÁSICA                      TUBERÍA MODERNA
   (senas/clasico.py)                   (senas/manos.py)
          │                                     │
   1. filtrado                          1. detector de palma
      gaussiano / mediana / bilateral      (red convolucional)
          │                                     │
   2. histograma                        2. modelo de landmarks
      ecualización / CLAHE                 21 puntos 3D
          │                                     │
   3. umbralización                     3. normalización
      Otsu / adaptativo / inRange          traslación, escala,
          │                                   lateralidad
   4. morfología                              │
      erosión, dilatación,                    │
      apertura, cierre                        │
          │                                   │
   5. segmentación                            │
      contornos, región mayor                 │
          │                                   │
   6. caracterización                   4. caracterización
      14 descriptores                      78 características
      (Hu, solidez, defectos)                 │
          └──────────────────┬──────────────────┘
                             │
                      ┌──────▼───────┐
                      │ clasificador │  MLP (scikit-learn)
                      └──────┬───────┘
                             │
                      ┌──────▼───────┐
                      │ estabilizador│  suavizado, histéresis, rearme
                      └──────┬───────┘
                             │
                           texto
```

### Los temas del curso, y dónde están

| Tema | Implementación | Archivo |
|---|---|---|
| **Modelos de cámara y calibración** | Modelo estenopeico, matriz intrínseca K, coeficientes de distorsión radial y tangencial, calibración con patrón de ajedrez, desdistorsión, mapa de desplazamiento | [`senas/calibracion.py`](senas/calibracion.py) |
| **Filtros** | Gaussiano, mediana, bilateral y realce por convolución con núcleo explícito | [`senas/clasico.py`](senas/clasico.py) → `aplicar_filtro` |
| **Histogramas** | Cálculo y dibujo del histograma, ecualización global y CLAHE | `clasico.py` → `histograma_gris`, `ecualizar` |
| **Umbralización y binarización** | Manual, Otsu, adaptativo por media y por gaussiana, `inRange` en YCrCb | `clasico.py` → `binarizar`, `mascara_piel` |
| **Morfología matemática** | Erosión, dilatación, apertura, cierre y gradiente, con elemento estructurante configurable en forma y tamaño | `clasico.py` → `morfologia`, `nucleo` |
| **Segmentación** | Segmentación por color de piel en YCrCb, extracción de contornos, selección de región mayor, región de interés | `clasico.py` → `mascara_piel`, `region_mayor` |
| **Caracterización** | 14 descriptores clásicos (momentos de Hu, solidez, circularidad, extensión, defectos de convexidad) y 78 características sobre landmarks | `clasico.py` → `caracterizar`; `manos.py` → `caracteristicas` |

### Por qué YCrCb y no BGR

La segmentación de piel trabaja en YCrCb porque ese espacio **separa la
crominancia de la luminancia**. El rango de piel se mantiene estable cuando
cambia la iluminación, cosa que no ocurre en BGR: la misma piel bajo otra luz
cae en otra zona del cubo de color y el rango deja de servir.

### Por qué las características son invariantes

Ambos vectores están normalizados para no depender de dónde está la mano en el
encuadre ni de la distancia a la cámara. Sin esa normalización el modelo
aprende la posición de la mano en lugar de la forma de la seña.

Las invariancias del vector de MediaPipe están **demostradas con pruebas**
ejecutables: ver [`probar_caracteristicas.py`](probar_caracteristicas.py).

---

## Ramificación

Modelo de ramas por funcionalidad, integrando a `main` mediante *merge* sin
avance rápido (`--no-ff`) para que cada funcionalidad quede identificable en el
historial.

```
main ──────●────────●────────●────────●────────●──────►
            \      / \      / \      / \      /
             ●────●   ●────●   ●────●   ●────●
        feature/   feature/  feature/  feature/
        vision-    calibra-  ui-       dataset-
        clasica    cion      taller    herramientas
```

| Rama | Contenido |
|---|---|
| `main` | Rama estable. Todo lo integrado y funcionando. |
| `feature/vision-clasica` | Filtros, histogramas, umbralización, morfología, segmentación y caracterización clásica. |
| `feature/calibracion` | Modelo de cámara, calibración con tablero, desdistorsión. |
| `feature/ui-taller` | Interfaz interactiva con un modo por tema. |
| `feature/dataset-herramientas` | Captura, entrenamiento, recuperación de muestras. |

---

## Instalación

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe descargar_modelo.py
```

El último paso descarga `hand_landmarker.task` (7,8 MB) desde el repositorio
oficial de Google. No se incluye en el repositorio.

---

## Uso

### Taller interactivo — un modo por tema

```bash
.venv\Scripts\python.exe taller.py
```

| Tecla | Modo |
|---|---|
| `1` | Modelo de cámara y calibración |
| `2` | Filtrado |
| `3` | Histogramas |
| `4` | Umbralización y binarización |
| `5` | Morfología matemática |
| `6` | Segmentación |
| `7` | Caracterización |
| `8` | Comparación: clásica frente a MediaPipe |

Los deslizadores de la ventana «Controles» cambian según el modo. `G` guarda
una captura de la vista actual, `ESC` sale.

**Para calibrar:** abre `datos/patron_ajedrez.png` (se genera solo al primer
arranque), imprímelo o muéstralo en el móvil, enséñaselo a la cámara en
ángulos distintos y pulsa `C` cada vez que lo detecte. Con 8 o más vistas,
pulsa `K`.

### Reconocimiento de señas

```bash
.venv\Scripts\python.exe capturar.py      # grabar el dataset
.venv\Scripts\python.exe entrenar.py      # entrenar y evaluar
.venv\Scripts\python.exe interpretar.py   # reconocimiento en vivo
```

#### Dejar clases fuera del entrenamiento

```bash
.venv\Scripts\python.exe entrenar.py --excluir _ESPACIO_
```

Sirve para cuantificar cuánto cuesta una clase problemática: se entrena con y
sin ella y se comparan las cifras. El resultado en este proyecto:

| Configuración | Clases | Precisión |
|---|---|---|
| Todas las clases | 17 | 83,9 % (± 0,2) |
| **Sin `_ESPACIO_`** | 16 | **90,6 % (± 3,7)** |
| Sin `_ESPACIO_`, A ni B | 14 | 97,7 % (± 1,2) |
| Sin `_ESPACIO_`, A, B, C ni E | 12 | 95,8 % (± 0,1) |

`_ESPACIO_` costaba 7 puntos: el gesto elegido como separador —mano abierta al
frente— coincide con la seña de la B, así que las dos clases son
indistinguibles por construcción. Es un error de diseño, no del modelo.

Obsérvese que la última fila **baja**: quitar más clases no mejora el número.
Son `_ESPACIO_`, A y B concretamente las defectuosas; C y E ya iban bien una
vez eliminada la primera.

En `capturar.py`: las letras `A-Z` seleccionan, `ENTER` graba una ráfaga,
`0` es la clase «nada», `ESPACIO` el separador, `RETROCESO` deshace y `ESC`
guarda y sale.

---

## Metodología de evaluación

`entrenar.py` parte el conjunto **por ráfaga**, no por fotograma, usando
`StratifiedGroupKFold`.

Esto importa: los 60 fotogramas de una misma ráfaga son casi idénticos. Si se
repartieran al azar entre entrenamiento y prueba, el modelo se examinaría de
fotogramas gemelos de los que ya vio y la precisión saldría inflada por encima
del 95 % sin significar nada. Partir por ráfaga obliga a generalizar a una
toma distinta, que es la condición real de uso.

La cifra que reporta el programa es, por tanto, más baja y más honesta que la
que daría una partición ingenua.

Se generan además la matriz de confusión y un listado de las confusiones más
frecuentes, que es lo que guía dónde grabar más datos.

---

## Limitaciones conocidas

- **La segmentación por color captura cualquier cosa con tono de piel**: la
  cara del usuario, paredes beige, muebles de madera. Por eso el modo de
  segmentación incorpora una región de interés y un rango ajustable en vivo.
  Esta fragilidad es precisamente lo que motiva la comparación con MediaPipe.
- **Las letras J, Z y Ñ no se reconocen.** Llevan movimiento, y un
  clasificador que observa un fotograma aislado no puede recuperar una
  trayectoria. Requeriría un modelo secuencial.
- **La iluminación a contraluz degrada el resultado.** Los dedos
  sobreexpuestos se pierden en la segmentación.

---

## Sobre el dataset de este repositorio

`datos/muestras.csv` contiene las muestras grabadas por el autor: para cada
fotograma, el vector de 78 características ya normalizado. Son coordenadas de
puntos de la mano, no imágenes.

Los fotogramas originales **no se incluyen**: son grabaciones de la cámara del
autor y no hacen falta para reproducir el proyecto. `capturar.py` los guarda en
`datos/imagenes/` en tu propia máquina, y `.gitignore` los deja fuera.

---

## Dos trampas de MediaPipe en Windows

Ambas están resueltas en el código, pero conviene conocerlas: cuestan una tarde
cada una y los mensajes de error no apuntan a la causa.

**1. Las rutas con caracteres no ASCII rompen la carga del modelo.** La carpeta
de este proyecto contiene una `ñ`. Pasar la ruta con `model_asset_path` falla
con `FileNotFoundError` *aunque el archivo exista y Python lo vea*: la capa
nativa en C++ no decodifica bien los caracteres no ASCII. La solución es leer
el archivo desde Python y pasar los bytes con `model_asset_buffer`.

El mismo problema afecta a `cv2.imwrite` y `cv2.imread`, que fallan en
silencio devolviendo `False` o `None`. Por eso el proyecto usa `cv2.imencode` /
`cv2.imdecode` con lectura y escritura desde Python.

**2. Los timestamps deben ser estrictamente crecientes.** En modo `VIDEO`, si
un fotograma llega con una marca de tiempo menor o igual que la anterior,
MediaPipe lanza `ValueError`. Usar `time.time() * 1000` es una bomba de
relojería: dos fotogramas procesados dentro del mismo milisegundo dan la misma
marca. `DetectorManos` lleva su propio reloj monótono.

---

## Archivos

| Ruta | Descripción |
|---|---|
| `taller.py` | Interfaz interactiva con un modo por tema del curso |
| `senas/clasico.py` | Tubería de visión clásica completa |
| `senas/calibracion.py` | Modelo de cámara y calibración |
| `senas/manos.py` | Detección con MediaPipe y extracción de características |
| `senas/alfabeto.py` | Definición del alfabeto LSC |
| `capturar.py` | Captura del dataset con la webcam |
| `entrenar.py` | Entrenamiento y evaluación |
| `interpretar.py` | Reconocimiento en vivo |
| `descargar_modelo.py` | Descarga del modelo de MediaPipe |
| `probar_caracteristicas.py` | Pruebas de las invariancias |
| `recuperar_desde_imagenes.py` | Reconstrucción del CSV desde los fotogramas |

---

## Referencias

- INSOR — Instituto Nacional para Sordos. [Alfabeto LSC](https://educativo.insor.gov.co/catdiccionario/alfabeto/)
- INSOR e Instituto Caro y Cuervo. [Diccionario Básico de la Lengua de Señas Colombiana](https://www.colombiaaprende.edu.co/sites/default/files/files_public/2022-04/Diccionario-lengua-de-senas.pdf), 2006.
- Google. [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
