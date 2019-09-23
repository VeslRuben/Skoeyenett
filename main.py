from keras import models
from keras import layers

net = models.Sequential()
net.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
net.add(layers.Dense(10, activation="softmax"))
net.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

