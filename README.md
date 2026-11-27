# Intérprete de lenguaje de señas

Proyecto de visión artificial que lee el alfabeto dactilológico por webcam y lo
convierte en texto. Reconoce las **24 letras de pose estática** del alfabeto,
más una seña de separación de palabras.

---

## Cómo funciona

```
webcam ──▶ MediaPipe ──▶ normalización ──▶ MLP ──▶ estabilizador ──▶ texto
           21 puntos      78 rasgos        letra    3 capas anti-ruido
```

Las dos etapas que hacen el trabajo de verdad son la segunda y la cuarta.

### 1. Detección — `senas/manos.py`

MediaPipe localiza la mano y devuelve **21 puntos 3D** (muñeca, nudillos,
falanges, puntas). Esto sustituye a la segmentación por color de piel de los
enfoques clásicos, que se rompe con cualquier cambio de luz o con un fondo del
color de la mano.

> **Ojo con la versión.** Desde MediaPipe 1.0 el API antiguo `mp.solutions.hands`
> **ya no existe**. Todo tutorial que encuentres con `mp.solutions` es para
> 0.10.x y no funciona aquí. Este proyecto usa el API nuevo, `mediapipe.tasks`.

### 2. Normalización — el paso que decide si el proyecto funciona

Los 21 puntos crudos dependen de dónde está la mano y de cuán cerca esté de la
cámara. Si los metes tal cual en el clasificador, el modelo aprende *"la mano
estaba arriba a la izquierda"* en vez de *"los dedos estaban así"*, y falla en
cuanto te mueves. `extraer_caracteristicas()` hace cuatro cosas:

| Paso | Qué corrige |
|---|---|
| `x *= ancho/alto` | MediaPipe normaliza x e y por separado; en 16:9 eso **estira la mano**. Sin esto, la misma seña da vectores distintos en cámaras con formatos distintos. |
| reflejar eje X si es mano izquierda | Zurdos y diestros comparten un solo modelo. Si no, habría que grabar el dataset dos veces. |
| restar la muñeca | Invariancia a la **posición** en el encuadre. |
| dividir por \|muñeca→nudillo medio\| | Invariancia a la **escala** (distancia a la cámara). Se usa ese segmento porque es rígido: no se encoge al cerrar el puño, cosa que sí le pasa a la caja contenedora. |

Salen **63 coordenadas + 15 distancias** entre puntas de dedos = vector de 78.
Las distancias son redundantes en teoría, pero le dan al clasificador la
relación *"dedos juntos vs separados"* ya masticada, que es justo lo que separa
los pares difíciles (U/V, M/N).

Estas invariancias no son una promesa: están comprobadas numéricamente en
`probar_caracteristicas.py`.

### 3. Clasificación — `entrenar.py`

`StandardScaler` + `MLPClassifier(128, 64)`. La red es pequeña a propósito: con
unos miles de muestras, una más grande se aprende el ruido de tu webcam.

### 4. Estabilización — `interpretar.py`

Clasificar fotograma a fotograma da un resultado inservible: a 30 fps salen 30
letras por segundo, y al mover la mano entre señas se pasa por poses que se
parecen a otras letras. El resultado crudo es `AAAAWWWSSSSSSDDDD`. Tres capas:

1. **Suavizado** — se promedian las probabilidades de los últimos 10 fotogramas.
   Un fotograma malo suelto no cambia nada.
2. **Histéresis** — una letra solo se escribe si aguanta 8 fotogramas seguidos
   por encima del 70 % de confianza.
3. **Rearme** — tras escribir una letra hay que *soltar* la seña para repetir
   esa misma letra. Cambiar a otra letra distinta se acepta de inmediato. Sin
   esta capa, tener la mano quieta escribiría la misma letra sin parar.

---

## Puesta en marcha

```bash
.venv\Scripts\activate
python descargar_modelo.py
```

Baja `hand_landmarker.task` (~7,5 MB) del repositorio oficial de Google. Solo se
hace una vez. Ese modelo **no** reconoce letras: solo localiza la mano. El que
reconoce letras lo entrenas tú.

### Paso 1 — Grabar el dataset

```bash
python capturar.py
```

Pulsa la tecla de una letra, luego `G` para grabar una ráfaga (2 s de cuenta
atrás + 60 fotogramas). No guarda imágenes, solo los vectores de 78 rasgos: el
dataset pesa kilobytes.

**Graba 3 o 4 ráfagas por letra, cambiando cada vez** distancia a la cámara,
ángulo de la mano, altura y, si puedes, la luz. Si grabas las cuatro ráfagas
quieto en la misma pose, vas a sacar 99 % en el entrenamiento y vas a fallar en
la demo.

No te saltes la clase `_NADA_` (tecla `0`): mano relajada, en reposo, poses que
no son ninguna letra. Es la clase que evita que el intérprete escupa letras al
azar mientras piensas.

| Tecla | | Tecla | |
|---|---|---|---|
| `A`–`Z` | seleccionar letra | `G` | grabar ráfaga |
| `ESPACIO` | seña de separador | `U` | deshacer última ráfaga |
| `0` | clase «nada» | `D` | borrar todo de esa letra |
| `TAB` | ver recuento | `Q` / `ESC` | guardar y salir |

### Paso 2 — Entrenar

```bash
python entrenar.py
```

Deja el modelo en `modelo/` y la matriz de confusión en `salidas/`.

> **Sobre el número de precisión que verás.** Los 60 fotogramas de una ráfaga
> son casi el mismo fotograma repetido. Con un `train_test_split` normal caen
> fotogramas gemelos en train y en test a la vez: el modelo se examina de lo que
> ya vio, saca 99,8 % y se cae en la demo. Por eso aquí se parte **por ráfaga**
> (`StratifiedGroupKFold`): ninguna toma se reparte entre train y test. El
> número sale más bajo y es el real. **Espera algo entre 85 % y 95 %.**

Al final imprime las confusiones más frecuentes. Ahí es donde conviene grabar
más ráfagas — o aceptar que dos señas son casi idénticas de verdad.

### Paso 3 — Interpretar

```bash
python interpretar.py
```

| Tecla | |
|---|---|
| `RETROCESO` | borrar el último carácter |
| `C` | limpiar todo |
| `S` | guardar la transcripción |
| `D` | ver/ocultar el diagnóstico de probabilidades |
| `Q` / `ESC` | salir |

Para escribir una letra doble (`LL`, `RR`), baja la mano entre las dos.

---

## Alcance y límites

Configurado para **LSC** (Lengua de Señas Colombiana). Para LSE o ASL no hay que
tocar código: la lista de letras es prácticamente la misma y la forma de cada
una la aprende el modelo de *tus* capturas.

**Quedan fuera las letras con movimiento**, y es una decisión, no un olvido: un
clasificador estático mira una sola pose en un instante, así que no puede
distinguir una trayectoria.

| | Movimiento |
|---|---|
| `J` | gancho hacia abajo con el meñique |
| `Z` | traza una Z en el aire con el índice |
| `Ñ` | la N con un movimiento ondulado encima |
| `RR` | vibración / repetición de la R |
| `LL` | la L con desplazamiento lateral |

Reconocerlas pide un modelo secuencial (LSTM/GRU o DTW) sobre una ventana de
fotogramas, no sobre uno solo. Es la extensión natural del proyecto.

Otras limitaciones honestas: **una sola mano** a la vez (`num_hands=1`), y el
modelo se entrena con *tus* manos y *tu* cámara, así que otra persona tendrá
bastante menos acierto. Es lo esperable con un dataset propio; para mejorarlo
habría que grabar a varias personas.

---

## Sobre el dataset de este repositorio

`datos/muestras.csv` contiene las muestras grabadas por el autor: para cada
fotograma, el vector de 78 caracteristicas ya normalizado. Son coordenadas de
puntos de la mano, no imagenes.

Los fotogramas originales **no se incluyen**: son grabaciones de la camara del
autor y no hacen falta para reproducir el proyecto. `capturar.py` los guarda en
`datos/imagenes/` en tu propia maquina, y `.gitignore` los deja fuera del
repositorio. Si quieres tu propio dataset, grabalo con `capturar.py`.

## Dos trampas de MediaPipe en Windows

Ambas están resueltas en el código, pero conviene conocerlas: cuestan una tarde
cada una y los mensajes de error no apuntan a la causa.

**1. La `ñ` de la ruta rompe la carga del modelo.** La carpeta de este proyecto
se llama `LenguajeDeseñas`. Pasarle a MediaPipe la ruta con `model_asset_path`
falla con `FileNotFoundError` *aunque el archivo exista y Python lo vea*: la capa
nativa en C++ no decodifica bien los caracteres no ASCII y recibe
`LenguajeDese?as`. La solución es leer el archivo desde Python y pasar los bytes
con `model_asset_buffer`, de modo que la librería nativa nunca vea la ruta.

**2. Los timestamps tienen que ser estrictamente crecientes.** En modo `VIDEO`,
si un fotograma llega con una marca de tiempo menor o igual que la anterior,
MediaPipe lanza `ValueError` y tira el programa. Usar `time.time() * 1000` es
una bomba de relojería: dos fotogramas procesados dentro del mismo milisegundo
dan la misma marca, y además `time.time()` puede retroceder si el reloj del
sistema se ajusta. `DetectorManos` lleva su propio reloj monótono y fuerza el
avance, así que quien lo usa no tiene que preocuparse.

## Archivos

| | |
|---|---|
| `senas/manos.py` | detección y extracción de características |
| `senas/alfabeto.py` | qué letras se reconocen y cuáles no, con el porqué |
| `descargar_modelo.py` | baja el modelo de landmarks (una vez) |
| `capturar.py` | graba el dataset con la webcam |
| `entrenar.py` | entrena y evalúa el clasificador |
| `interpretar.py` | intérprete en vivo |
| `probar_caracteristicas.py` | comprueba que las invariancias se cumplen |

## Requisitos

Python 3.14 · MediaPipe 1.0.1 · OpenCV 5 · scikit-learn 1.9 (ver `requirements.txt`)
