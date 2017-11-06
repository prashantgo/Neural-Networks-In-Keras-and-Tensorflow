import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

variables = [1, 2, 3, 4]
y = [2, 4, 6, 8]
model = Sequential()
model.add(Dense(1, input_shape=(1, )))
model.add(Activation('linear'))

sgd = SGD(0.1)
model.compile(loss="mse", metrics=["mse"], optimizer=sgd)

H = model.fit(variables, y, epochs=100, batch_size=10)
plt.plot(H.history["mean_squared_error"])
plt.show()

w = model.get_weights()
print(w)


