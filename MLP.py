import numpy as np

class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.05, alpha=0.01):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.alpha = alpha  # Leaky ReLU slope

        # Initialize weights and biases using He initialization
        self.W = []
        self.b = []

        # Weights and biases from input layer to first hidden layer
        self.W.append(np.random.randn(self.input_size, self.hidden_sizes[0]) * np.sqrt(2 / self.input_size))
        self.b.append(np.zeros((1, self.hidden_sizes[0])))

        # Weights and biases for hidden layers
        for i in range(1, len(self.hidden_sizes)):
            self.W.append(np.random.randn(self.hidden_sizes[i - 1], self.hidden_sizes[i]) * np.sqrt(2 / self.hidden_sizes[i - 1]))
            self.b.append(np.zeros((1, self.hidden_sizes[i])))

        # Weights and biases from last hidden layer to output layer
        self.W.append(np.random.randn(self.hidden_sizes[-1], self.output_size) * np.sqrt(2 / self.hidden_sizes[-1]))
        self.b.append(np.zeros((1, self.output_size)))

    def leaky_relu(self, z):
        return np.maximum(self.alpha * z, z)

    def leaky_relu_derivative(self, z):
        return np.where(z > 0, 1, self.alpha)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, output, y):
        m = y.shape[0]
        return -np.mean(np.sum(y * np.log(output + 1e-8), axis=1))  # Cross-Entropy Loss

    def forward(self, X):
        self.a = [X]  # Store activations
        self.z = []   # Store pre-activations

        # Forward pass through hidden layers
        for i in range(len(self.hidden_sizes)):
            z_i = np.dot(self.a[-1], self.W[i]) + self.b[i]
            a_i = self.leaky_relu(z_i)  # Use Leaky ReLU activation for hidden layers
            self.z.append(z_i)
            self.a.append(a_i)

        # Output layer with softmax
        z_output = np.dot(self.a[-1], self.W[-1]) + self.b[-1]
        a_output = self.softmax(z_output)  # Softmax for output layer
        self.z.append(z_output)
        self.a.append(a_output)

        return a_output

    def backward(self, X, y):
        m = X.shape[0]

        # Initialize gradients for weights and biases
        dW = [None] * len(self.W)
        db = [None] * len(self.b)

        # Compute gradient for output layer
        dz = self.a[-1] - y
        dW[-1] = np.dot(self.a[-2].T, dz) / m
        db[-1] = np.sum(dz, axis=0, keepdims=True) / m

        # Backpropagate through hidden layers
        for i in range(len(self.hidden_sizes) - 1, -1, -1):
            dz = np.dot(dz, self.W[i + 1].T) * self.leaky_relu_derivative(self.z[i])
            dW[i] = np.dot(self.a[i].T, dz) / m
            db[i] = np.sum(dz, axis=0, keepdims=True) / m

        # Update weights and biases with gradient descent
        for i in range(len(self.W)):
            self.W[i] -= self.learning_rate * dW[i]
            self.b[i] -= self.learning_rate * db[i]

    def train(self, X, y, epochs=10, decay_factor=0.99, gradient_clip_value=5.0):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)

            # Clip gradients to prevent exploding gradients
            for i in range(len(self.W)):
                np.clip(self.W[i], -gradient_clip_value, gradient_clip_value, out=self.W[i])
                np.clip(self.b[i], -gradient_clip_value, gradient_clip_value, out=self.b[i])

            # Compute loss and accuracy
            loss = self.compute_loss(output, y)
            predictions = np.argmax(output, axis=1)
            labels = np.argmax(y, axis=1)
            accuracy = np.mean(predictions == labels)

            # Learning rate decay (adjust after every epoch)
            self.learning_rate *= decay_factor

            print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Learning Rate: {self.learning_rate:.6f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)
