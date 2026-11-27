# ============================================================================
#  Pruebas de la extraccion de caracteristicas
# ----------------------------------------------------------------------------
#  Comprueba, sin necesidad de camara ni de modelo, que el vector de
#  caracteristicas es de verdad invariante a lo que decimos que es invariante.
#  Si alguna de estas pruebas falla, el clasificador no va a generalizar por
#  mucho dataset que le eches.
#
#  Ejecutar:  python probar_caracteristicas.py
# ============================================================================

import numpy as np

from senas.manos import DIM_CARACTERISTICAS, extraer_caracteristicas

ASPECTO = 1280 / 720


def mano_de_prueba(semilla=0):
    """21 landmarks plausibles: no es una mano real, pero sirve de referencia."""
    rng = np.random.default_rng(semilla)
    return np.clip(rng.random((21, 3)) * 0.4 + 0.3, 0.0, 1.0)


def comprobar(nombre, a, b, tolerancia=1e-9):
    dif = float(np.abs(a - b).max())
    ok = dif < tolerancia
    print("  [%s] %-42s  dif max = %.2e" % ("OK" if ok else "FALLA", nombre, dif))
    return ok


def main():
    base = mano_de_prueba()
    v0 = extraer_caracteristicas(base, ASPECTO, False)

    print("Dimension del vector: %d (esperada %d)"
          % (len(v0), DIM_CARACTERISTICAS))
    print()
    todo_ok = len(v0) == DIM_CARACTERISTICAS

    # 1) Invariancia a la traslacion: mover la mano por el encuadre no debe
    #    cambiar nada. Es lo que evita que el modelo aprenda "estaba arriba".
    movida = base + np.array([0.15, -0.08, 0.0])
    todo_ok &= comprobar("invariante a la traslacion",
                         v0, extraer_caracteristicas(movida, ASPECTO, False))

    # 2) Invariancia a la escala: acercarse o alejarse de la camara. Escalamos
    #    respecto a la muneca, que es el origen tras centrar.
    escalada = base[0] + (base - base[0]) * 1.8
    todo_ok &= comprobar("invariante a la escala (distancia camara)",
                         v0, extraer_caracteristicas(escalada, ASPECTO, False))

    # 3) Las dos cosas a la vez.
    ambas = base[0] + (base - base[0]) * 0.6 + np.array([-0.1, 0.12, 0.0])
    todo_ok &= comprobar("invariante a traslacion + escala",
                         v0, extraer_caracteristicas(ambas, ASPECTO, False))

    # 4) La mano izquierda reflejada tiene que dar el MISMO vector que la
    #    derecha. Eso es lo que permite que zurdos y diestros compartan modelo.
    espejo = base.copy()
    espejo[:, 0] = 1.0 - espejo[:, 0]
    todo_ok &= comprobar("mano izquierda reflejada == mano derecha",
                         v0, extraer_caracteristicas(espejo, ASPECTO, True))

    # 5) En cambio, doblar un dedo SI tiene que cambiar el vector: si no, el
    #    vector seria invariante a todo, incluida la informacion util.
    otra = base.copy()
    otra[8] += np.array([0.05, 0.05, 0.0])        # punta del indice
    dif = float(np.abs(v0 - extraer_caracteristicas(otra, ASPECTO, False)).max())
    ok = dif > 1e-3
    print("  [%s] %-42s  dif max = %.2e" % ("OK" if ok else "FALLA",
                                            "sensible a mover un dedo", dif))
    todo_ok &= ok

    # 6) Sin valores no finitos, que rompen a sklearn en silencio.
    ok = bool(np.all(np.isfinite(v0)))
    print("  [%s] %-42s" % ("OK" if ok else "FALLA", "sin NaN ni infinitos"))
    todo_ok &= ok

    print()
    print("RESULTADO: " + ("todas las pruebas pasan" if todo_ok
                           else "HAY PRUEBAS QUE FALLAN"))
    return 0 if todo_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
