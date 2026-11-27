# ============================================================================
#  Entrenamiento del clasificador de letras
# ----------------------------------------------------------------------------
#  Lee datos/muestras.csv y entrena un perceptron multicapa sobre los vectores
#  de 78 caracteristicas. Guarda el modelo en modelo/clasificador.joblib y deja
#  en salidas/ la matriz de confusion, que es lo que de verdad te dice que
#  letras se confunden entre si.
#
#  La decision importante de este archivo es COMO se evalua.
#  Los 60 fotogramas de una rafaga son casi el mismo fotograma repetido. Con un
#  train_test_split normal, fotogramas gemelos caen en train y en test a la vez:
#  el modelo se examina de lo que ya vio y saca 99.8%. Ese numero es mentira y
#  se cae en la demo en vivo. Por eso aqui la particion es POR RAFAGA
#  (StratifiedGroupKFold): ninguna rafaga se parte entre train y test, asi que
#  el modelo se evalua contra tomas que no ha visto nunca. El numero sale mas
#  bajo y es el real.
# ============================================================================

from __future__ import annotations

import csv
import os
import sys
from collections import Counter

import numpy as np

RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_CSV = os.path.join(RAIZ, "datos", "muestras.csv")
RUTA_MODELO = os.path.join(RAIZ, "modelo", "clasificador.joblib")
DIR_SALIDAS = os.path.join(RAIZ, "salidas")


def cargar():
    """Devuelve X (caracteristicas), y (etiquetas) y g (id de rafaga)."""
    if not os.path.exists(RUTA_CSV):
        print("No existe " + RUTA_CSV)
        print("Captura primero el dataset:  python capturar.py")
        sys.exit(1)

    etiquetas, grupos, vectores = [], [], []
    with open(RUTA_CSV, "r", newline="", encoding="utf-8") as f:
        lector = csv.reader(f)
        next(lector, None)
        for fila in lector:
            if len(fila) < 3:
                continue
            etiquetas.append(fila[0])
            grupos.append(fila[1])
            vectores.append([float(v) for v in fila[2:]])

    return np.array(vectores), np.array(etiquetas), np.array(grupos)


def revisar_dataset(y, g) -> None:
    """Avisa de los problemas que hacen que el entrenamiento no sirva."""
    cuenta = Counter(y)
    print("Clases: %d  |  muestras: %d  |  rafagas: %d"
          % (len(cuenta), len(y), len(set(g))))

    problemas = []
    for etiqueta, n in sorted(cuenta.items()):
        n_rafagas = len({gi for gi, yi in zip(g, y) if yi == etiqueta})
        if n_rafagas < 2:
            problemas.append("  '%s': solo %d rafaga(s). Con una sola toma no se"
                             " puede evaluar honestamente." % (etiqueta, n_rafagas))
        elif n < 60:
            problemas.append("  '%s': solo %d muestras." % (etiqueta, n))

    if problemas:
        print("\nAvisos sobre el dataset:")
        print("\n".join(problemas))
        print()

    if len(cuenta) < 2:
        print("Hacen falta al menos 2 clases para entrenar.")
        sys.exit(1)

    # Un desbalance fuerte hace que el modelo se incline por la clase mayoritaria.
    may, men = max(cuenta.values()), min(cuenta.values())
    if may > 3 * men:
        print("Aviso: dataset desbalanceado (%d vs %d muestras entre la clase"
              " mas y menos representada)." % (may, men))


def main() -> int:
    """Entrena el clasificador y lo evalua sin enganarse.

    El orden importa. Primero se evalua con validacion cruzada agrupando por
    rafaga, que es lo que da la cifra que se reporta; y solo despues se
    reentrena con el dataset completo para guardar el modelo que se usara en
    produccion. Al reves, la evaluacion estaria midiendo un modelo que ya vio
    los datos de prueba.

    Imprime ademas el informe por clase y las confusiones mas frecuentes
    ordenadas, que es lo que indica donde grabar mas rafagas y que pares de
    senas son realmente parecidos."""
    from joblib import dump
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, g = cargar()

    # --excluir CLASE [CLASE ...]  deja clases fuera del entrenamiento.
    #
    # Sirve para cuantificar cuanto cuesta una clase problematica: entrenar con
    # y sin ella y comparar. En este proyecto _ESPACIO_ es el caso claro, porque
    # el gesto que se eligio como separador (mano abierta) coincide con la seña
    # de la B, asi que las dos clases son indistinguibles por construccion.
    if "--excluir" in sys.argv:
        fuera = [a.upper() for a in sys.argv[sys.argv.index("--excluir") + 1:]
                 if not a.startswith("-")]
        if fuera:
            mantener = ~np.isin(y, fuera)
            quitadas = len(y) - int(mantener.sum())
            X, y, g = X[mantener], y[mantener], g[mantener]
            print("Excluidas %d muestras de: %s" % (quitadas, ", ".join(fuera)))
            print()

    revisar_dataset(y, g)

    def construir():
        # StandardScaler primero: las 63 coordenadas y las 15 distancias viven en
        # rangos distintos, y una red sin escalar se sesga hacia las de mayor
        # magnitud. El MLP es pequeno a proposito: con unos miles de muestras,
        # una red mas grande se aprende el ruido de tu webcam.
        """Crea la tuberia de clasificacion: escalado y despues el MLP.

    El StandardScaler va primero porque las componentes del vector viven en
    rangos distintos (coordenadas normalizadas frente a distancias entre
    puntos), y una red sin escalar se sesga hacia las de mayor magnitud.

    Se crea una tuberia nueva en cada particion de la validacion cruzada, y no
    una compartida: reutilizarla haria que el escalado se ajustara con datos de
    prueba, que es otra forma sutil de fuga de informacion."""
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                          alpha=1e-3, max_iter=1500, early_stopping=True,
                          n_iter_no_change=25, random_state=0),
        )

    # --- evaluacion honesta: particion por rafaga ---------------------------
    # No podemos pedir mas particiones que rafagas tenga la clase mas pobre:
    # si una letra solo tiene 3 tomas, no hay forma de repartirla en 5 grupos.
    etiqueta_de_rafaga = {gi: y[g == gi][0] for gi in set(g)}
    rafagas_por_clase = Counter(etiqueta_de_rafaga.values())
    n_splits = max(2, min(5, min(rafagas_por_clase.values())))

    print("\nEvaluando con StratifiedGroupKFold (%d particiones, agrupando por"
          " rafaga)..." % n_splits)

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    aciertos, y_real, y_pred = [], [], []

    for i, (tr, te) in enumerate(cv.split(X, y, groups=g), 1):
        modelo = construir()
        modelo.fit(X[tr], y[tr])
        p = modelo.predict(X[te])
        acc = float((p == y[te]).mean())
        aciertos.append(acc)
        y_real.extend(y[te])
        y_pred.extend(p)
        print("  particion %d: %.1f%% sobre %d muestras" % (i, 100 * acc, len(te)))

    print("\n  PRECISION REAL: %.1f%% (+/- %.1f)"
          % (100 * np.mean(aciertos), 100 * np.std(aciertos)))

    etiquetas_orden = sorted(set(y))
    print("\n--- informe por clase ---")
    print(classification_report(y_real, y_pred, labels=etiquetas_orden,
                                zero_division=0))

    # --- que se confunde con que -------------------------------------------
    mc = confusion_matrix(y_real, y_pred, labels=etiquetas_orden)
    confusiones = []
    for i, real in enumerate(etiquetas_orden):
        total = mc[i].sum()
        for j, pred in enumerate(etiquetas_orden):
            if i != j and mc[i, j] > 0 and total:
                confusiones.append((mc[i, j] / total, real, pred, mc[i, j]))
    confusiones.sort(reverse=True)

    if confusiones:
        print("--- confusiones mas frecuentes ---")
        print("  (donde conviene grabar mas rafagas, o revisar si las dos senas"
              " son casi iguales de verdad)")
        for tasa, real, pred, n in confusiones[:10]:
            print("  %-10s -> %-10s  %4.1f%%  (%d veces)"
                  % (real, pred, 100 * tasa, n))
        print()

    guardar_matriz(mc, etiquetas_orden)

    # --- modelo definitivo: se reentrena con TODO el dataset ----------------
    print("Reentrenando con el dataset completo...")
    final = construir()
    final.fit(X, y)

    os.makedirs(os.path.dirname(RUTA_MODELO), exist_ok=True)
    dump({"modelo": final,
          "etiquetas": etiquetas_orden,
          "precision_cv": float(np.mean(aciertos)),
          "n_muestras": int(len(y))}, RUTA_MODELO)

    print("Modelo guardado en: " + RUTA_MODELO)
    print("Siguiente paso:  python interpretar.py")
    return 0


def guardar_matriz(mc, etiquetas) -> None:
    """Dibuja la matriz de confusion normalizada por fila."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(DIR_SALIDAS, exist_ok=True)
    norm = mc.astype(float) / np.maximum(mc.sum(axis=1, keepdims=True), 1)

    n = len(etiquetas)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.45), max(5, n * 0.42)))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(n), etiquetas, rotation=90, fontsize=8)
    ax.set_yticks(range(n), etiquetas, fontsize=8)
    ax.set_xlabel("prediccion")
    ax.set_ylabel("real")
    ax.set_title("Matriz de confusion (normalizada por fila)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # Solo anotamos las celdas con peso, para que no sea una sopa de numeros.
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.02:
                ax.text(j, i, "%d" % round(100 * norm[i, j]),
                        ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] < 0.6 else "black")

    fig.tight_layout()
    ruta = os.path.join(DIR_SALIDAS, "matriz_confusion.png")
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print("Matriz de confusion: " + ruta + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
