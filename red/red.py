"""Red neuronal de dos capas escrita desde cero con operaciones de TensorFlow.

No se usa Keras: los pesos son tf.Variable, la propagación hacia adelante son
multiplicaciones de matrices y los gradientes de la retropropagación se calculan
a mano con la regla de la cadena.

    Z1 = X W1 + b1        H = tanh(Z1)
    Z2 = H W2 + b2        P = softmax(Z2)
    L  = -(1/N) Σ Y ⊙ log P            (entropía cruzada)
"""

import tensorflow as tf

DTYPE = tf.float64  # doble precisión, la misma que usa Excel


def glorot(forma, semilla):
    """Inicialización de Glorot uniforme: U(-r, r) con r = sqrt(6 / (entradas + salidas))."""
    r = (6.0 / (forma[0] + forma[1])) ** 0.5
    return tf.random.stateless_uniform(forma, seed=[semilla, 0], minval=-r, maxval=r, dtype=DTYPE)


class RedNeuronal:
    def __init__(self, n_entradas, n_ocultas, n_salidas, semilla=42):
        self.W1 = tf.Variable(glorot((n_entradas, n_ocultas), semilla), name="W1")
        self.b1 = tf.Variable(tf.zeros(n_ocultas, dtype=DTYPE), name="b1")
        self.W2 = tf.Variable(glorot((n_ocultas, n_salidas), semilla + 1), name="W2")
        self.b2 = tf.Variable(tf.zeros(n_salidas, dtype=DTYPE), name="b2")

    @property
    def parametros(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def pesos(self):
        """Copia de los parámetros como arreglos de NumPy."""
        return {v.name.split(":")[0]: v.numpy().copy() for v in self.parametros}

    def adelante(self, X):
        """Propagación hacia adelante. Regresa (H, P): activaciones ocultas y probabilidades."""
        X = tf.convert_to_tensor(X, dtype=DTYPE)
        H = tf.tanh(tf.matmul(X, self.W1) + self.b1)
        Z2 = tf.matmul(H, self.W2) + self.b2
        Z2 = Z2 - tf.reduce_max(Z2, axis=1, keepdims=True)  # estabilidad numérica
        E = tf.exp(Z2)
        P = E / tf.reduce_sum(E, axis=1, keepdims=True)
        return H, P

    def predecir(self, X):
        return tf.argmax(self.adelante(X)[1], axis=1).numpy()

    @staticmethod
    def perdida(P, Y):
        Y = tf.convert_to_tensor(Y, dtype=DTYPE)
        return -tf.reduce_mean(tf.reduce_sum(Y * tf.math.log(P + 1e-12), axis=1))

    def gradientes(self, X, Y):
        """Retropropagación a mano. Regresa la pérdida y [dW1, db1, dW2, db2]."""
        X = tf.convert_to_tensor(X, dtype=DTYPE)
        Y = tf.convert_to_tensor(Y, dtype=DTYPE)
        n = tf.cast(tf.shape(X)[0], DTYPE)
        H, P = self.adelante(X)

        dZ2 = (P - Y) / n                      # softmax + entropía cruzada
        dW2 = tf.matmul(H, dZ2, transpose_a=True)
        db2 = tf.reduce_sum(dZ2, axis=0)
        dH = tf.matmul(dZ2, self.W2, transpose_b=True)
        dZ1 = dH * (1.0 - H * H)               # derivada de tanh
        dW1 = tf.matmul(X, dZ1, transpose_a=True)
        db1 = tf.reduce_sum(dZ1, axis=0)
        return self.perdida(P, Y), [dW1, db1, dW2, db2]

    def paso(self, X, Y, tasa):
        """Un paso de descenso de gradiente: θ ← θ - tasa · ∂L/∂θ."""
        perdida, grads = self.gradientes(X, Y)
        for v, g in zip(self.parametros, grads):
            v.assign_sub(tasa * g)
        return float(perdida)

    def entrenar(self, X, Y, epocas, tasa, cada=100):
        """Descenso de gradiente con el lote completo. Regresa el historial [(época, pérdida)]."""
        historial = []
        for epoca in range(1, epocas + 1):
            perdida = self.paso(X, Y, tasa)
            if epoca == 1 or epoca % cada == 0:
                historial.append((epoca, perdida))
        return historial
