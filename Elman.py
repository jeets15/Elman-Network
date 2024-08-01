import numpy as np
import pandas as pd

def sigmoid(x):
    return np.tanh(x)

def dsigmoid(x):
    return 1.0 - x**2 # review

class Elman:

    def __init__(self, *args):

        self.shape = args
        n = len(args)

        self.layers = []

        self.layers.append(np.ones(self.shape[0]+1+self.shape[1]))
        # no. of input neuron + 1(bias) + no. of Hidden neuron

        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))

        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))

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

    def train(self, dataset, epochs=100, lrate=0.1, momentum=0.1):
        for epoch in range(epochs):
            total_error = 0
            for state, action, q_value in dataset:
                output = self.propagate_forward(state)
                target = np.copy(output)
                target[action] = q_value
                error = self.propagate_backward(target, lrate, momentum)
                total_error += error
            print(f"Epoch {epoch+1}/{epochs}, Error: {total_error}")

    def predict(self, state):
        output = self.propagate_forward(state)
        return output



# Example usage
input_size = 2
hidden_size = 5
output_size = 4

network = Elman(input_size, hidden_size, output_size)


csv_file_path = "q_value_dataset.csv"


df = pd.read_csv(csv_file_path)


def parse_state(state_str, target_shape=input_size):
    state_str = state_str.strip('[]')
    state_list = state_str.split()
    state_array = np.array([float(state_list[0]), float(state_list[1])])

    padded_state = np.pad(state_array, (0, target_shape - len(state_array)), 'constant')
    return padded_state


dataset = [(parse_state(row['State']), int(row['Action']), float(row['Q-Value'])) for index, row in df.iterrows()]


network.train(dataset, epochs=100, lrate=0.1, momentum=0.1)


state = np.array([12,12])
q_values = network.predict(state)
print("Predicted Q-values:", q_values)

