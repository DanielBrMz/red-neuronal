# Red neuronal desde cero en TensorFlow

Red neuronal de dos capas que clasifica las tres especies de Iris. Está escrita con operaciones básicas
de TensorFlow, sin Keras. Los pesos son `tf.Variable`, la propagación hacia adelante son productos de
matrices, y los gradientes de la retropropagación se derivan a mano con la regla de la cadena.

**Autor:** Daniel Alfredo Barreras Meraz (A01254805) · TC3002B.570, Desarrollo de aplicaciones avanzadas de ciencias computacionales · Tec de Monterrey

**Pesos para la comprobación:** [`resultados/pesos.xlsx`](resultados/pesos.xlsx) (también en [`resultados/pesos.json`](resultados/pesos.json))

## Resultado

| | Valor |
|---|---|
| Arquitectura | 4 entradas → 8 ocultas (tanh) → 3 salidas (softmax) |
| Datos | Iris (UCI), 150 flores, 120 de entrenamiento y 30 de prueba (10 por especie) |
| Entrenamiento | Descenso de gradiente con lote completo, tasa 0.1, 2000 épocas, semilla 42 |
| Pérdida final (entropía cruzada) | 0.0511 |
| Exactitud de entrenamiento | 98.33 % (118 de 120) |
| Exactitud de prueba | 100 % (30 de 30) |

## Cómo funciona

| Paso | Fórmula | Dónde |
|---|---|---|
| Estandarización | `z = (x - media) / desviación`, con media y desviación del entrenamiento | `red/datos.py` |
| Inicialización (Glorot) | `W ~ U(-r, r)`, `r = √(6 / (entradas + salidas))`, `b = 0` | `red/red.py`, `glorot` |
| Capa oculta | `H = tanh(X W1 + b1)` | `RedNeuronal.adelante` |
| Capa de salida | `P = softmax(H W2 + b2)` | `RedNeuronal.adelante` |
| Pérdida | `L = -(1/N) Σ Y ⊙ log P` | `RedNeuronal.perdida` |
| Gradiente de salida | `dZ2 = (P - Y) / N`, `dW2 = Hᵀ dZ2`, `db2 = Σ dZ2` | `RedNeuronal.gradientes` |
| Gradiente oculto | `dZ1 = (dZ2 W2ᵀ) ⊙ (1 - H²)`, `dW1 = Xᵀ dZ1`, `db1 = Σ dZ1` | `RedNeuronal.gradientes` |
| Actualización | `θ ← θ - tasa · ∂L/∂θ` | `RedNeuronal.paso` |

- La derivada de softmax junto con la entropía cruzada se simplifica a `P - Y`, por eso no se calcula el
  jacobiano de softmax por separado.
- La derivada de `tanh` es `1 - tanh²`, y se reutiliza `H`, ya calculada en la propagación hacia adelante.
- Los gradientes a mano se comparan con los de `tf.GradientTape` en las pruebas y coinciden hasta 1e-10.
- Todo se calcula en `float64`, la misma precisión que Excel, para que la hoja de comprobación dé
  exactamente los mismos números.

## El Excel de pesos

| Hoja | Contenido |
|---|---|
| Resumen | Arquitectura, hiperparámetros y exactitud |
| Pesos finales | `W1` (4×8), `b1` (8), `W2` (8×3), `b2` (3), y la media y desviación de la estandarización |
| Pesos iniciales | Los mismos parámetros antes de entrenar |
| Comprobacion | Las 150 flores, recalculadas con fórmulas de Excel a partir de la hoja "Pesos finales" |
| Historial | Pérdida cada 100 épocas |

La hoja **Comprobacion** no copia los resultados de Python. Cada fila calcula `z`, `h = TANH(...)`,
`o` y `p = EXP(o) / Σ EXP(o)` con fórmulas que leen las celdas de "Pesos finales", y al final compara
contra las probabilidades de Python. Las 150 filas coinciden (diferencia máxima 1.9e-15) y la exactitud
calculada en Excel es la misma que en Python. Si se modifica un peso, las predicciones cambian.

## Uso

Requiere Python 3.10 a 3.12 (TensorFlow 2.20 no tiene versión para 3.13 o posterior).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python entrenar.py              # entrena y genera resultados/pesos.xlsx y pesos.json
python -m unittest -v           # 8 pruebas
python verificar_excel.py       # recalcula el Excel con LibreOffice y lo compara con Python
```

El entrenamiento es determinista (semillas fijas, CPU, `TF_DETERMINISTIC_OPS=1`), así que al volver a
correr `entrenar.py` se obtienen los mismos pesos que están en el Excel.

## Estructura

```
red/datos.py         carga de Iris, one-hot, división estratificada, estandarización
red/red.py           la red: pesos, propagación, pérdida, retropropagación a mano, entrenamiento
red/excel.py         exportación de los pesos y la hoja de comprobación con fórmulas
entrenar.py          entrenamiento y generación de resultados
verificar_excel.py   comprobación del Excel
tests/test_red.py    pruebas (gradientes contra GradientTape, determinismo, datos)
data/iris.csv        conjunto Iris (Fisher, 1936; UCI Machine Learning Repository)
```
