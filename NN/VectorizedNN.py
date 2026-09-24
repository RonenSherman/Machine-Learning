import time
import numpy as np
from tensorflow.keras.datasets import mnist
"""Ronen Sherman - Project 1, learning hand drawn numbers with NN"""
"""Finished NN with vectorization and softmax function instead of sigmoid"""
"""MNIST dataset - all 60,000 samples (had some help from dad on this one) """

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# Network
class Network:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.05):
        self.learning_rate = learning_rate
        # Initialize weights and biases
        self.W1 = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
        self.b1 = np.random.uniform(-0.5, 0.5, (hidden_size, 1))
        self.W2 = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
        self.b2 = np.random.uniform(-0.5, 0.5, (output_size, 1))

    def forward(self, X):
        # Hidden layer
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = sigmoid(Z1)
        # Output layer (softmax)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def backward(self, X, Y, Z1, A1, Z2, A2):
        m = X.shape[1]  # batch size

        # Softmax + Cross-Entropy derivative
        dZ2 = A2 - Y 
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        # Hidden layer
        dZ1 = np.dot(self.W2.T, dZ2) * sigmoid_derivative(A1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def predict(self, X):
        _, _, _, A2 = self.forward(X)
        return np.argmax(A2, axis=0)

# Load MNIST
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784).T / 255.0  # (784, 60000)
X_test = X_test.reshape(-1, 784).T / 255.0    # (784, 10000)

Y_train_oh = np.eye(10)[Y_train].T  # (10, 60000)

# Training
NN = Network(784, 64, 10, learning_rate=3.0)
EPOCHS = 30
BATCH_SIZE = 10

num_samples = X_train.shape[1]

for epoch in range(EPOCHS):
    start_time = time.perf_counter()

    # Shuffle training data
    permutation = np.random.permutation(num_samples)
    X_train_shuffled = X_train[:, permutation]
    Y_train_shuffled = Y_train_oh[:, permutation]

    # Mini-batch training
    for i in range(0, num_samples, BATCH_SIZE):
        X_batch = X_train_shuffled[:, i:i + BATCH_SIZE]
        Y_batch = Y_train_shuffled[:, i:i + BATCH_SIZE]

        Z1, A1, Z2, A2 = NN.forward(X_batch)
        NN.backward(X_batch, Y_batch, Z1, A1, Z2, A2)

    # Printing results
    prediction = NN.predict(X_test)
    accuracy = np.mean(prediction == Y_test) * 100
    elapsed = time.perf_counter() - start_time
    print(f"Epoch {epoch + 1}/{EPOCHS} — Test Accuracy: {accuracy:.2f}% — Time: {elapsed:.2f}s")