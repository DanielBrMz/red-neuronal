"""Entrena la red con Iris y guarda los pesos en resultados/pesos.xlsx y resultados/pesos.json.

    python entrenar.py
"""

import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU: mismos pesos en cualquier máquina

import numpy as np
import tensorflow as tf

from red import RedNeuronal, cargar_iris, dividir, one_hot, Estandarizador
from red.datos import CLASES
from red.excel import exportar

CONFIG = {"entradas": 4, "ocultas": 8, "salidas": 3, "tasa": 0.1, "epocas": 2000, "semilla": 42}
SALIDA = Path(__file__).resolve().parent / "resultados"


def fijar_semillas(semilla):
    random.seed(semilla)
    np.random.seed(semilla)
    tf.random.set_seed(semilla)


def exactitud(red, X, y):
    return float(np.mean(red.predecir(X) == y))


def main():
    fijar_semillas(CONFIG["semilla"])
    X, y = cargar_iris()
    idx_ent, idx_pru = dividir(X, y, semilla=CONFIG["semilla"])
    est = Estandarizador(X[idx_ent])
    Z, Y = est(X), one_hot(y)

    red = RedNeuronal(CONFIG["entradas"], CONFIG["ocultas"], CONFIG["salidas"], semilla=CONFIG["semilla"])
    pesos_ini = red.pesos()
    historial = red.entrenar(Z[idx_ent], Y[idx_ent], CONFIG["epocas"], CONFIG["tasa"])
    pesos_fin = red.pesos()

    _, P = red.adelante(Z)
    P = P.numpy()
    metricas = {
        "exactitud_entrenamiento": exactitud(red, Z[idx_ent], y[idx_ent]),
        "exactitud_prueba": exactitud(red, Z[idx_pru], y[idx_pru]),
        "perdida_final": float(red.perdida(P[idx_ent], Y[idx_ent])),
    }

    SALIDA.mkdir(exist_ok=True)
    conjunto = ["prueba" if i in set(idx_pru) else "entrenamiento" for i in range(len(X))]
    exportar(SALIDA / "pesos.xlsx", CONFIG, pesos_ini, pesos_fin, est, X, y, conjunto,
             P, P.argmax(axis=1), historial, metricas)
    with open(SALIDA / "pesos.json", "w") as f:
        json.dump({"config": CONFIG, "metricas": metricas,
                   "estandarizacion": {"media": est.media.tolist(), "desviacion": est.desv.tolist()},
                   "pesos_iniciales": {k: v.tolist() for k, v in pesos_ini.items()},
                   "pesos_finales": {k: v.tolist() for k, v in pesos_fin.items()}}, f, indent=2)

    for epoca, perdida in historial[:: max(1, len(historial) // 10)]:
        print(f"época {epoca:5d}  pérdida {perdida:.6f}")
    print(f"\nexactitud entrenamiento {metricas['exactitud_entrenamiento']:.2%}")
    print(f"exactitud prueba        {metricas['exactitud_prueba']:.2%}")
    pred = red.predecir(Z[idx_pru])
    matriz = np.zeros((3, 3), dtype=int)
    for real, p in zip(y[idx_pru], pred):
        matriz[real, p] += 1
    print("\nmatriz de confusión (prueba), filas = real, columnas = predicción")
    for nombre, fila in zip(CLASES, matriz):
        print(f"  {nombre:16s} {fila}")
    print(f"\npesos guardados en {SALIDA / 'pesos.xlsx'} y {SALIDA / 'pesos.json'}")


if __name__ == "__main__":
    main()
