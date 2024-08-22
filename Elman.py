import numpy as np
import csv


def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - x**2

class Elman:
    def __init__(self, *args):
        self.shape = args
        n = len(args)
        self.layers = []
        self.layers.append(np.ones(self.shape[0] + 1 + self.shape[1]))
        for i in range(1, n):
            self.layers.append(np.ones(self.shape[i]))
        self.weights = []
        for i in range(n - 1):
            self.weights.append(np.zeros((self.layers[i].size,
                                          self.layers[i + 1].size)))
        self.dw = [0] * len(self.weights)
        self.reset()
        self.context_units = np.zeros(self.shape[1])

    def reset(self):
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * Z - 1) * 0.25

    def propagate_forward(self, data):
        # Set input layer with data
        self.layers[0][:self.shape[0]] = data
        # Set context units (part of the first layer after input data)
        self.layers[0][self.shape[0]:-1] = self.context_units
        self.layers[0][-1] = 1  # Bias unit
        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = tanh(np.dot(self.layers[i - 1], self.weights[i - 1]))
        self.context_units = self.layers[1].copy()
        return self.layers[-1]

    def propagate_backward(self, target, lrate=0.1, momentum=0.1):
        deltas = []
        error = target - self.layers[-1]
        delta = error * dtanh(self.layers[-1])
        deltas.append(delta)

        # Compute error on hidden layers
        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * dtanh(self.layers[i])
            deltas.insert(0, delta)

        # Update weights
        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

        # Return error
        return (error**2).sum()



    def train(self, dataset, epochs=5000, lrate=0.1, momentum=0.1):
        for epoch in range(epochs):
            total_error = 0
            for entry in dataset:
                state_visited, action_taken = entry

                # Encode the state
                state = np.array(state_visited)

                # Forward propagate
                output = self.propagate_forward(state)

                # Back propagate
                total_error += self.propagate_backward(action_taken, lrate, momentum)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Error: {total_error}')

    def predict(self, state):
        # Forward propagate the state
        output = self.propagate_forward(np.array(state))

        # Decode the output to an action
        actions = ["right", "up", "left", "down"]
        return actions[np.argmax(output)]

def load_dataset_from_csv(file_path):
    dataset = []
    action_mapping = {
        "right": np.array([1, 0, 0, 0]),
        "up": np.array([0, 1, 0, 0]),
        "left": np.array([0, 0, 1, 0]),
        "down": np.array([0, 0, 0, 1])
    }
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            state_visited = eval(row['state visited'])
            action_taken = action_mapping[row['action taken']]
            dataset.append((state_visited, action_taken))
    return dataset

# Load the dataset from CSV
file_path = 'q_learning_dataset.csv'
dataset = load_dataset_from_csv(file_path)

# Initialize and train the Elman network
elman_net = Elman(2, 20, 4)
elman_net.train(dataset)

# Predict the action for a given state
state = (5, 5)
predicted_action = elman_net.predict(state)
print(f'Predicted action for state {state}: {predicted_action}')
