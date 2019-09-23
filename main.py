from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

(train_images, train_target), (test_images, test_target) = mnist.load_data()

train_images = train_images.reshape(60000, 28*28)
test_images = test_images.reshape(10000, 28*28)

train_target = to_categorical(train_target)
test_target = to_categorical(test_target)

train_images = train_images / 255
test_images = test_images / 255


net = models.Sequential()
net.add(layers.Dense(512, activation="relu", input_shape=(28*28,)))
net.add(layers.Dense(10, activation="softmax"))
net.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
net.fit(train_images, train_target, epochs=5, batch_size=128)

# check model performance over testset
test_loss, test_acc = net.evaluate(test_images, test_target)
print('test_accuracy: ', test_acc)

y_pred = net.predict_classes(test_images)

y_true = np.argmax(test_target, axis=1)
cm = confusion_matrix(y_true, y_pred)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
print('cm: ', cm)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, square=True)

ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.show()
