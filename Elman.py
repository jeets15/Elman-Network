import numpy as np
import pandas as pd

def sigmoid(x):
    return np.tanh(x)

def dsigmoid(x):
    return 1.0 - x**2

class Elman:
    def __init__(self, *args):
        self.shape = args
        n = len(args)
        self.layers = []
        self.layers.append(np.ones(self.shape[0]+1+self.shape[1]))

        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))

        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size, self.layers[i + 1].size)))

        self.dw = [0, ] * len(self.weights)
        self.reset()

    def reset(self):
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i+1].size))
            self.weights[i][...] = (2*Z-1)*0.25

    def propagate_forward(self, data):
        self.layers[0][:self.shape[0]] = data
        self.layers[0][self.shape[0]:-1] = self.layers[1]
        for i in range(1, len(self.shape)):
            self.layers[i][...] = sigmoid(np.dot(self.layers[i-1], self.weights[i-1]))
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []
        error = target - self.layers[-1]
        delta = error * dsigmoid(self.layers[-1])
        deltas.append(delta)

        for i in range(len(self.shape)-2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * dsigmoid(self.layers[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

        return (error ** 2).sum()



    def train(self, X, y, epochs=1000, lrate=0.1, momentum=0.1):
        for epoch in range(epochs):
            error = 0.0
            for i in range(X.shape[0]):
                self.propagate_forward(X[i])
                error += self.propagate_backward(y[i], lrate, momentum)
            if epoch % 100 == 0:
                print(f'Epoch {epoch} Error: {error}')

    def predict(self, X):
        return self.propagate_forward(X)





# Data preprocessing
data = pd.read_csv("q_value_dataset.csv")

def state_to_vector(state):
    return np.array(eval(state))

def action_to_vector(action):
    actions = ['up', 'right', 'left']
    vec = np.zeros(len(actions))
    vec[actions.index(action)] = 1
    return vec

X = []
y = []

for i, row in data.iterrows():
    start_state = state_to_vector(row['state'])
    action = action_to_vector(row['action'])
    X.append(start_state)
    y.append(action)

X = np.array(X)
y = np.array(y)

# 1 represents number of tuple
input_size = X.shape[1]
output_size = y.shape[1]
hidden_neurons = 10

net = Elman(input_size, hidden_neurons, output_size)

# Train the network
net.train(X, y, epochs=5000, lrate=0.01, momentum=0.1)

# Predict for given state
test_state = state_to_vector("(9, 0")
predicted_action = net.predict(test_state)
print("Predicted action (raw output):", predicted_action)


predicted_action_label = ['up', 'right', 'left'][np.argmax(predicted_action)]
print("Predicted action:", predicted_action_label)
