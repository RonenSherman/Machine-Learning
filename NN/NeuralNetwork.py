import math
import random
import time
import numpy as np
from tensorflow.keras.datasets import mnist

"""NN with numpy, sgd and full dataset"""

#  Sigmoid Neuron
class SigmoidNeuron:
    def __init__(self, x, weights, bias, learning_rate=0.05):
        self.x = np.array(x, ndmin=2)
        self.weights = np.array(weights, ndmin=2)
        self.bias = float(bias)
        self.learning_rate = learning_rate

    def output_function(self):
        linear_output = (self.x @ self.weights + self.bias).item()
        return 1 / (1 + math.exp(-linear_output))  # sigmoid

    def cost(self, y_true):
        y_prediction = self.output_function()
        return 0.5 * (y_true - y_prediction) ** 2

    def train_step(self, y_true):
        y_prediction = self.output_function()
        error = y_prediction - y_true
        sigmoid_derivative = y_prediction * (1 - y_prediction)
        d_cost_d_linear = error * sigmoid_derivative

        d_weights = np.array([[self.x[0, i] * d_cost_d_linear] for i in range(self.x.shape[1])])
        d_bias = d_cost_d_linear

        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias

        return y_prediction, self.cost(y_true)


#  Network
class Network:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = [
            SigmoidNeuron(
                x=[0] * self.input_size,
                weights=np.random.uniform(-0.5, 0.5, (self.input_size, 1)),
                bias=random.uniform(-0.5, 0.5)
            )
            for _ in range(self.hidden_size)
        ]

        self.output_layer = [
            SigmoidNeuron(
                x=[0] * self.hidden_size,
                weights=np.random.uniform(-0.5, 0.5, (self.hidden_size, 1)),
                bias=random.uniform(-0.5, 0.5)
            )
            for _ in range(self.output_size)
        ]

    def forward(self, input_vector):
        hidden_outputs = []
        for neuron in self.hidden_layer:
            neuron.x = np.array(input_vector, ndmin=2)
            hidden_outputs.append(neuron.output_function())

        final_outputs = []
        for neuron in self.output_layer:
            neuron.x = np.array(hidden_outputs, ndmin=2)
            final_outputs.append(neuron.output_function())

        return hidden_outputs, final_outputs

    def train_step(self, input_vector, y_true_vector):
        hidden_outputs, final_outputs = self.forward(input_vector)

        # Output layer backprop
        output_deltas = []
        for i, neuron in enumerate(self.output_layer):
            y_prediction = final_outputs[i]
            error = y_prediction - y_true_vector[i]
            delta = error * y_prediction * (1 - y_prediction)
            output_deltas.append(delta)

            for j in range(len(hidden_outputs)):
                neuron.weights[j, 0] -= neuron.learning_rate * delta * hidden_outputs[j]
            neuron.bias -= neuron.learning_rate * delta

        # Hidden layer backprop
        for j, hidden_neuron in enumerate(self.hidden_layer):
            hidden_output = hidden_outputs[j]
            error = sum(output_deltas[k] * self.output_layer[k].weights[j, 0]
                        for k in range(self.output_size))
            delta = error * hidden_output * (1 - hidden_output)
            for i in range(len(input_vector)):
                hidden_neuron.weights[i, 0] -= hidden_neuron.learning_rate * delta * input_vector[i]
            hidden_neuron.bias -= hidden_neuron.learning_rate * delta

        return final_outputs

    def predict(self, input_vector):
        _, outputs = self.forward(input_vector)
        return np.argmax(outputs)


#  Load MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

# Combine into list of tuples for shuffling
training_data = list(zip(X_train, Y_train))

#  Training
NN = Network(784, 64, 10)
EPOCHS = 10
BATCH_SIZE = 32  # typical small batch for SGD

for epoch in range(EPOCHS):
    start_time = time.perf_counter()

    random.shuffle(training_data)  #  SGD randomness

    # Loop through mini-batches
    for batch_start in range(0, len(training_data), BATCH_SIZE):
        batch = training_data[batch_start:batch_start + BATCH_SIZE]
        for x, y_label in batch:
            y_true = [0] * 10
            y_true[y_label] = 1
            NN.train_step(x, y_true)

    # Evaluate after each epoch
    correct = 0
    for i in range(len(X_test)):  # test on 10,000
        prediction = NN.predict(list(X_test[i]))
        if prediction == Y_test[i]:
            correct += 1

    accuracy = (correct / len(X_test)) * 100
    elapsed = time.perf_counter() - start_time
    print(f"Epoch {epoch + 1}/{EPOCHS} — Test Accuracy: {accuracy:.2f}% — Time: {elapsed:.2f}s")