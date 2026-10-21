import os
import unittest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf

from red import Estandarizador, RedNeuronal, cargar_iris, dividir, one_hot


class PruebasDatos(unittest.TestCase):
    def test_iris_completo(self):
        X, y = cargar_iris()
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(np.bincount(y).tolist(), [50, 50, 50])

    def test_division_estratificada_y_disjunta(self):
        X, y = cargar_iris()
        ent, pru = dividir(X, y)
        self.assertEqual(len(ent), 120)
        self.assertEqual(np.bincount(y[pru]).tolist(), [10, 10, 10])
        self.assertFalse(set(ent) & set(pru))

    def test_estandarizacion(self):
        X, _ = cargar_iris()
        Z = Estandarizador(X)(X)
        np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-12)


class PruebasRed(unittest.TestCase):
    def setUp(self):
        X, y = cargar_iris()
        self.X = Estandarizador(X)(X)
        self.Y = one_hot(y)
        self.y = y

    def test_probabilidades_suman_uno(self):
        _, P = RedNeuronal(4, 8, 3).adelante(self.X)
        np.testing.assert_allclose(tf.reduce_sum(P, axis=1).numpy(), 1.0, atol=1e-12)

    def test_gradientes_a_mano_igual_a_gradient_tape(self):
        red = RedNeuronal(4, 8, 3, semilla=7)
        _, manuales = red.gradientes(self.X, self.Y)
        with tf.GradientTape() as cinta:
            perdida = red.perdida(red.adelante(self.X)[1], self.Y)
        automaticos = cinta.gradient(perdida, red.parametros)
        for m, a in zip(manuales, automaticos):
            np.testing.assert_allclose(m.numpy(), a.numpy(), atol=1e-10)

    def test_la_perdida_baja(self):
        historial = RedNeuronal(4, 8, 3).entrenar(self.X, self.Y, epocas=300, tasa=0.1)
        self.assertLess(historial[-1][1], historial[0][1] / 5)

    def test_misma_semilla_mismos_pesos(self):
        a, b = RedNeuronal(4, 8, 3, semilla=42), RedNeuronal(4, 8, 3, semilla=42)
        a.entrenar(self.X, self.Y, epocas=50, tasa=0.1)
        b.entrenar(self.X, self.Y, epocas=50, tasa=0.1)
        for k, v in a.pesos().items():
            np.testing.assert_array_equal(v, b.pesos()[k])

    def test_aprende_iris(self):
        red = RedNeuronal(4, 8, 3)
        red.entrenar(self.X, self.Y, epocas=1000, tasa=0.1)
        self.assertGreater(np.mean(red.predecir(self.X) == self.y), 0.95)


if __name__ == "__main__":
    unittest.main()
