from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import tensorflow as tf
# definir parametros.
rnn_units = 50
dense_units = 1
input_shape = (10, 1)  # 10 timesteps, 1 feature per timestep

# Crear el modelo
model = Sequential()
# Añadir capas RNN y Dense
model.add(SimpleRNN(units=rnn_units, input_shape=input_shape, activation='tanh'))

# Añadir una capa densa (FF)
model.add(Dense(units=dense_units, activation='sigmoid'))

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Resumen del modelo
model.summary()

# Datos de ejemplo (aleatorios)
import numpy as np
X_train = np.random.rand(100, 10, 1)  # 100 samples, 10 timesteps, 1 feature
y_train = np.random.randint(0, 2, 100)  # 100 binary labels

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_train, y_train)
print(f'Pérdida: {loss}, Precisión: {accuracy}')
# crear modelo con H5
model.save('rnn_ff.h5')
new_model = tf.keras.models.load_model('rnn_ff.h5')
new_model.summary()

# Explicación del código:
"""Importaciones: Importamos TensorFlow y Keras para construir y entrenar el modelo.

Parámetros del modelo: Definimos la forma de entrada, el número de unidades en la capa RNN y el número de unidades en la capa densa.

Modelo secuencial: Creamos un modelo secuencial y añadimos una capa SimpleRNN seguida de una capa Dense.

Compilación: Compilamos el modelo usando el optimizador Adam y la función de pérdida de entropía cruzada binaria, adecuada para clasificación binaria.

Datos de ejemplo: Generamos datos aleatorios para entrenar y evaluar el modelo.

Entrenamiento y evaluación: Entrenamos el modelo con los datos generados y luego lo evaluamos para obtener la pérdida y precisión."""
