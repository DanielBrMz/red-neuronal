"""Carga del conjunto Iris, codificación one-hot, división y estandarización."""

import csv
from pathlib import Path

import numpy as np

RUTA_IRIS = Path(__file__).resolve().parent.parent / "data" / "iris.csv"
CLASES = ("Iris-setosa", "Iris-versicolor", "Iris-virginica")
CARACTERISTICAS = ("largo_sepalo", "ancho_sepalo", "largo_petalo", "ancho_petalo")


def cargar_iris(ruta=RUTA_IRIS):
    """Regresa X (150x4, cm) y y (150, índice de clase)."""
    with open(ruta, newline="") as f:
        filas = list(csv.DictReader(f))
    X = np.array([[float(fila[c]) for c in CARACTERISTICAS] for fila in filas])
    y = np.array([CLASES.index(fila["especie"]) for fila in filas])
    return X, y


def one_hot(y, n_clases=len(CLASES)):
    return np.eye(n_clases)[y]


def dividir(X, y, fraccion_prueba=0.2, semilla=42):
    """División estratificada: el mismo número de flores de cada clase en prueba."""
    rng = np.random.default_rng(semilla)
    idx_ent, idx_pru = [], []
    for clase in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == clase))
        n_pru = round(len(idx) * fraccion_prueba)
        idx_pru.extend(idx[:n_pru])
        idx_ent.extend(idx[n_pru:])
    idx_ent = rng.permutation(idx_ent)
    idx_pru = np.sort(idx_pru)
    return idx_ent, idx_pru


class Estandarizador:
    """z = (x - media) / desviación, con media y desviación del conjunto de entrenamiento."""

    def __init__(self, X):
        self.media = X.mean(axis=0)
        self.desv = X.std(axis=0)

    def __call__(self, X):
        return (X - self.media) / self.desv
