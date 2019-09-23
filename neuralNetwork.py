import numpy as np



def sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def sigmoid_derivative(p):
    return p * (1.0 - p)


class NeuralNetwork:

    def __init__(self, learning_rate, lam, mom, number_of_inputs, number_of_hidden_layers, number_of_outputs):
        self.learning_rate = learning_rate
        self.lam = lam
        self.mom = mom
        self.number_of_inputs = number_of_inputs
        self.number_of_hidden_layers = number_of_hidden_layers
        self.number_of_outputs = number_of_outputs
        self.w_h = None
        self.w_y = None
        self.old_dw_y = 0
        self.old_dw_h = 0
        self.y = None

    def train(self, trainX, trainT, epochs):
        self.resetWeights()
        trainError = []
        for c in range(epochs):
            h, h_1, self.y = self.forwardPropagation(trainX)
            er_y = self.backwardPropagation(trainX, trainT, h, h_1, self.y)

            trainError.append(np.mean(np.square(er_y)))

        return trainError

    def test(self, testX, testT):
        testError = []

        _, _, test_y = self.forwardPropagation(testX)
        testError.append(np.mean(np.square(testT - test_y)))

        return testError

    def kfold(self, X, Y, kfold, epochs):
        if X.shape[0] % kfold != 0:
            raise ValueError('Has to be a value')

        incr = int(X.shape[0] / kfold)
        inputArray = []
        outputArray = []

        totalError = []

        for i in range(kfold):
            inputArray.append(X[(i * incr):((i + 1) * incr)])
            outputArray.append(Y[(i * incr):((i + 1) * incr)])

        inputArray = np.array(inputArray)
        outputArray = np.array(outputArray)

        for n in range(kfold):

            self.resetWeights()

            testX = inputArray[n, :, :]
            testT = outputArray[n, :, :]
            trainX = np.ones([1, X.shape[1]])
            trainT = np.ones([1, Y.shape[1]])

            for q in range(kfold):
                if q != n:
                    trainX = np.vstack([trainX, inputArray[q, :, :]])
                    trainT = np.vstack([trainT, outputArray[q, :, :]])

            trainX = np.delete(trainX, 0, axis=0)
            trainT = np.delete(trainT, 0, axis=0)

            testError = []

            for _ in range(epochs):
                h, h_1, y = self.forwardPropagation(trainX)
                er_y = self.backwardPropagation(trainX, trainT, h, h_1, y)

                testError.append(self.test(testX, testT))

            totalError.append(testError)

        return totalError

    def forwardPropagation(self, trainX):
        # Hidden Layer
        net_h = trainX.dot(self.w_h)

        h = sigmoid(net_h)

        h_1 = h

        h = np.append(h, np.ones((trainX.shape[0], 1)), axis=1)

        # Output layer

        net_y = h.dot(self.w_y)

        y = sigmoid(net_y)

        return h, h_1, y

    def backwardPropagation(self, trainX, trainT, h, h_1, y):
        # Output layer

        # Calculating error for y
        er_y = (trainT - y)

        # Calculating change in y
        der_y = sigmoid_derivative(y)

        # Combining for delta_y
        delta_y = er_y * der_y

        # Calculating change in weights for output
        dw_y = np.dot(h.T, delta_y) - (self.lam * np.sum(self.w_y, axis=1, keepdims=True))

        # Hidden layer

        # Calculating error for h

        er_h = np.dot(delta_y, self.w_y[:self.number_of_hidden_layers, :].T)

        delta_h = er_h * sigmoid_derivative(h_1)

        dw_h = np.dot(trainX.T, delta_h) - (self.lam * np.sum(self.w_h, axis=1, keepdims=True))

        # Applying change for weights for output layer
        self.w_y = self.w_y + (self.learning_rate * dw_y) + (self.mom * self.old_dw_y)

        # Applying change for weights for hidden layer
        self.w_h = self.w_h + (self.learning_rate * dw_h) + (self.mom * self.old_dw_h)

        # Momentum
        self.old_dw_y = dw_y
        self.old_dw_h = dw_h

        return er_y

    def resetWeights(self):
        self.w_h = np.random.rand(self.number_of_inputs + 1, self.number_of_hidden_layers)
        self.w_y = np.random.rand(self.number_of_hidden_layers + 1, self.number_of_outputs)