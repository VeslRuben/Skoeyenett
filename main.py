from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def vanligNeural():
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
    net.fit(train_images, train_target, epochs=10, batch_size=128)

    # check model performance over testset
    test_loss, test_acc = net.evaluate(test_images, test_target)
    print('test_accuracy: ', test_acc)

    y_pred = net.predict_classes(test_images)

    y_true = np.argmax(test_target, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print('cm: ', cm)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, square=True, fmt='g')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.show()

def kfoldNeural():
    (train_images, train_target), (test_images, test_target) = mnist.load_data()

    images = np.concatenate([train_images, test_images])
    images = images.reshape(70000, 28*28)
    images = images / 255
    targets = np.concatenate([train_target, test_target])
    targets = to_categorical(targets)

    folds = 2
    evals = []
    for train_index, test_index in KFold(folds).split(images):
        x_train, x_test = images[train_index], images[test_index]
        y_train, y_test = targets[train_index], targets[test_index]
        net = createModel()
        net.fit(x_train, y_train, epochs=2, batch_size=128)
        print('Model Evaluation: ', net.evaluate(x_test, y_test))
        evals.append(net.evaluate(x_test, y_test))

    lossSum = 0
    accSum = 0

    for loss, accuracy in evals:
        lossSum += loss
        accSum += accuracy

    lossSum = lossSum / folds
    accSum = accSum / folds

    print(lossSum)
    print(accSum)
    inn = input('Continue? [Y/N]')
    if inn == 'N':
        return
    elif inn == 'Y' or inn == 'y':
        pass
    else:
        return

    net = createModel()
    net.fit(images, targets, epochs=2, batch_size=128)

    y_pred = net.predict_classes(images)

    y_true = np.argmax(targets, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print('cm: ', cm)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, square=True, fmt='g')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.show()

def pcaNeural():
    (train_images, train_target), (test_images, test_target) = mnist.load_data()

    train_images = train_images.reshape(60000, 28 * 28)
    test_images = test_images.reshape(10000, 28 * 28)

    train_target = to_categorical(train_target)
    test_target = to_categorical(test_target)

    train_images = train_images / 255
    test_images = test_images / 255

    scaler = StandardScaler()
    scaler.fit(train_images)
    X_sc_train = scaler.transform(train_images)
    X_sc_test = scaler.transform(test_images)

    size = 150

    pca = PCA(n_components=size)
    pca.fit(train_images)

    train_images_pca = pca.fit_transform(X_sc_train)
    test_images_pca = pca.fit_transform(X_sc_test)
    pca_std = np.std(train_images_pca)

    model = models.Sequential()
    model.add(layers.Dense(128, activation="relu", input_shape=(size,)))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_images_pca, train_target, epochs=3, batch_size=64, validation_split=0.15)

    # check model performance over testset
    test_loss, test_acc = model.evaluate(test_images_pca, test_target)
    print('test_accuracy: ', test_acc)

    y_pred = model.predict_classes(test_images_pca)

    y_true = np.argmax(test_target, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print('cm: ', cm)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, square=True, fmt='g')

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.show()

def createModel():
    model = models.Sequential()
    model.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation="softmax"))
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    pcaNeural()