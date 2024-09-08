import numpy as np
import matplotlib.pyplot as plt


# activation functions
def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1.0 - x**2


# elman network class
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
        self.hidden_activations = []

    # reset weights
    def reset(self):
        for i in range(len(self.weights)):
            Z = np.random.random((self.layers[i].size, self.layers[i + 1].size))
            self.weights[i][...] = (2 * Z - 1) * 0.25

    # propagate data from input layer to output layer
    def propagate_forward(self, data):
        self.layers[0][:self.shape[0]] = data
        self.layers[0][self.shape[0]:-1] = self.context_units
        self.layers[0][-1] = 1

        for i in range(1, len(self.shape)):
            # Propagate activity
            self.layers[i][...] = tanh(np.dot(self.layers[i - 1], self.weights[i - 1]))
        self.context_units = self.layers[1].copy()
        self.hidden_activations = [layer.copy() for layer in self.layers[1:-1]]
        return self.layers[-1]

    # back propagate error using learning rate
    def propagate_backward(self, target, lrate=0.2, momentum=0.1):
        deltas = []
        error = target - self.layers[-1]
        delta = error * dtanh(self.layers[-1])
        deltas.append(delta)

        for i in range(len(self.shape) - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * dtanh(self.layers[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            layer = np.atleast_2d(self.layers[i])
            delta = np.atleast_2d(deltas[i])
            dw = np.dot(layer.T, delta)
            self.weights[i] += lrate * dw + momentum * self.dw[i]
            self.dw[i] = dw

        return (error**2).sum()

# plot hidden unit activation
def plot_hidden_activations(hidden_activations):
    num_layers = len(hidden_activations)
    fig, axes = plt.subplots(num_layers, 1, figsize=(10, 5 * num_layers))

    if num_layers == 1:
        axes = [axes]

    for i, activation in enumerate(hidden_activations):
        axes[i].imshow(activation.reshape(1, -1), cmap='viridis', aspect='auto')
        axes[i].set_title(f'Hidden Layer {i + 1} Activations')
        axes[i].set_xlabel('Neuron Index')
        axes[i].set_ylabel('Activation')

    plt.tight_layout()
    plt.show()


# plot learning curve
def plot_learning_curve(accuracy_history, mse_history):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)', color='g')
    ax1.plot(accuracy_history, label='Accuracy', color='g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.grid(True)

    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('MSE', color='b')
    ax2.plot(mse_history, label='MSE', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    plt.title('Learning Curve: Accuracy and MSE Over Epochs')
    fig.tight_layout()
    plt.show()


# calculate accuracy
def calculate_accuracy(network, samples):
    correct_predictions = 0
    for i in range(samples.size):
        output = network.propagate_forward(samples['input'][i])
        predicted = (output == output.max()).astype(float)
        if np.array_equal(predicted, samples['output'][i]):
            correct_predictions += 1
    accuracy = (correct_predictions / samples.size) * 100
    return accuracy


# calculate mean square error
def calculate_mse(network, samples):
    mse = 0
    for i in range(samples.size):
        output = network.propagate_forward(samples['input'][i])
        error = samples['output'][i] - output
        mse += np.mean(error ** 2)
    mse /= samples.size
    return mse





if __name__ == '__main__':
    network = Elman(8, 12, 4)
    samples = np.zeros(18, dtype=[('input', float, 8), ('output', float, 4)])

    samples[0] = (1, 0, 0, 1, 0, 0, 0, 0), (1, 0, 0, 0)
    samples[1] = (1, 0, 0, 1, 0, 0, 0, 1), (1, 0, 0, 0)
    samples[2] = (1, 0, 0, 1, 0, 0, 1, 0), (1, 0, 0, 0)
    samples[3] = (1, 0, 0, 1, 0, 0, 1, 1), (0, 1, 0, 0)
    samples[4] = (1, 0, 0, 0, 0, 0, 1, 1), (0, 1, 0, 0)
    samples[5] = (0, 1, 1, 1, 0, 0, 1, 1), (1, 0, 0, 0)
    samples[6] = (0, 1, 1, 1, 0, 1, 0, 0), (1, 0, 0, 0)
    samples[7] = (0, 1, 1, 1, 0, 1, 0, 1), (0, 1, 0, 0)
    samples[8] = (0, 1, 1, 0, 0, 1, 0, 1), (0, 1, 0, 0)
    samples[9] = (0, 1, 0, 1, 0, 1, 0, 1), (1, 0, 0, 0)
    samples[10] = (0, 1, 0, 1, 0, 1, 1, 0), (1, 0, 0, 0)
    samples[11] = (0, 1, 0, 1, 0, 1, 1, 1), (0, 1, 0, 0)
    samples[12] = (0, 1, 0, 0, 0, 1, 1, 1), (0, 1, 0, 0)
    samples[13] = (0, 0, 1, 1, 0, 1, 1, 1), (0, 1, 0, 0)
    samples[14] = (0, 0, 1, 0, 0, 1, 1, 1), (0, 1, 0, 0)
    samples[15] = (0, 0, 0, 1, 0, 1, 1, 1), (0, 0, 1, 0)
    samples[16] = (0, 0, 0, 1, 0, 1, 1, 0), (0, 0, 1, 0)
    samples[17] = (0, 0, 0, 1, 0, 1, 0, 1), (0, 1, 0, 0)

    # store MSE and accuracy
    mse_history = []
    accuracy_history = []



    # Training process
    for i in range(100):
        n = i % samples.size
        network.propagate_forward(samples['input'][n])
        network.propagate_backward(samples['output'][n])
        mse = calculate_mse(network, samples)
        accuracy = calculate_accuracy(network, samples)
        mse_history.append(mse)
        accuracy_history.append(accuracy)

    for i in range(samples.size):
        o = network.propagate_forward(samples['input'][i])
        print(f'Sample {i}: {samples["input"][i]} -> {samples["output"][i]}')
        print(f'               Network output: {(o == o.max()).astype(float)}\n')

    # test for untrained perspective
    test_inputs = np.array([
        (1, 0, 0, 1, 0, 0, 0, 0)
    ])

    for i, test_input in enumerate(test_inputs):
        output = network.propagate_forward(test_input)

        print(f'Test Input {i}: {test_input}')
        print(f'Network Output: {output}')
        print(f'Predicted Output: {(output == output.max()).astype(float)}\n')


    # plotting graphs
    plot_hidden_activations(network.hidden_activations)
    plot_learning_curve(accuracy_history, mse_history)
    accuracy = calculate_accuracy(network, samples)
    mse = calculate_mse(network, samples)
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Mean Squared Error: {mse:.4f}')

