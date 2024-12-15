import numpy as np
import os
import json
import logging


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


class MLP:
    def __init__(self, features, hidden_layers, output_size, learning_rate, epoch, callback=None):
        self.features = features
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.callback = callback
        self.weights = []
        self.biases = []

    def initialize_weights(self):
        np.random.seed(42)
        layer_sizes = [self.features] + self.hidden_layers + [self.output_size]
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            bias_vector = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_propagation(self, X):
        self.a = [X]
        self.z = []
        for i in range(len(self.hidden_layers)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = relu(z)
            self.a.append(a)
        z_output = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z_output)
        output = softmax(z_output)
        self.a.append(output)
        return output

    def backward_propagation(self, X, y):
        m = X.shape[0]
        self.d_weights = []
        self.d_biases = []
        delta = (self.a[-1] - y)
        for i in range(len(self.hidden_layers), -1, -1):
            dW = np.dot(self.a[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            self.d_weights.insert(0, dW)
            self.d_biases.insert(0, db)
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * relu_prime(self.z[i - 1])

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(np.clip(y_pred, 1e-10, 1.0))) / m
        return loss

    def compute_accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def train(self, X_train, y_train, batch_size=32):
        try:
            m = X_train.shape[0]
            for epoch in range(self.epoch):
                epoch_loss = 0
                for i in range(0, m, batch_size):
                    try:
                        X_batch = X_train[i:i + batch_size]
                        y_batch = y_train[i:i + batch_size]
                        output = self.forward_propagation(X_batch)
                        loss = self.compute_loss(output, y_batch)
                        epoch_loss += loss
                        self.backward_propagation(X_batch, y_batch)
                        self.update_weights()
                    except Exception as batch_error:
                        logging.error(f"Error processing batch {i // batch_size} in epoch {epoch}: {str(batch_error)}",
                                      exc_info=True)
                        raise

                avg_loss = epoch_loss / (m // batch_size)
                if self.callback:
                    self.callback(f"Epoch {epoch + 1}/{self.epoch}, Loss: {avg_loss:.4f}")
                logging.info(f"Epoch {epoch + 1}/{self.epoch}, Loss: {avg_loss:.4f}")

            self.save_model()
        except Exception as e:
            logging.error(f"Critical error in training process: {str(e)}", exc_info=True)
            raise

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)

    def save_model(self):
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)

            weights_path = os.path.join(data_dir, "model_weights.json")
            biases_path = os.path.join(data_dir, "model_bias.json")

            with open(weights_path, "w") as f:
                json.dump([w.tolist() for w in self.weights], f)

            with open(biases_path, "w") as f:
                json.dump([b.tolist() for b in self.biases], f)

            logging.info(f"Model saved successfully to {data_dir}")
        except Exception as e:
            logging.error(f"Error saving model: {str(e)}", exc_info=True)
            raise

    def load_model(self):
        try:
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            weight_path = os.path.join(data_dir, 'model_weights.json')
            bias_path = os.path.join(data_dir, 'model_bias.json')

            if os.path.exists(weight_path) and os.path.exists(bias_path):
                with open(weight_path, "r") as f:
                    list_weight = json.load(f)
                self.weights = [np.array(weight) for weight in list_weight]

                with open(bias_path, "r") as f:
                    list_bias = json.load(f)
                self.biases = [np.array(bias) for bias in list_bias]

                logging.info(f"Model loaded successfully from {weight_path} and {bias_path}")
            else:
                raise FileNotFoundError(f"Model files not found at {weight_path} or {bias_path}")
        except Exception as e:
            logging.error(f"Error loading model: {str(e)}", exc_info=True)
            raise

    def check_and_train(self, X_train, y_train):
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        weights_path = os.path.join(data_dir, 'model_weights.json')
        bias_path = os.path.join(data_dir, 'model_bias.json')

        if os.path.exists(weights_path) and os.path.exists(bias_path):
            self.load_model()
        else:
            logging.info("Model not found. Starting training...")
            self.train(X_train, y_train)

    def train_again(self, X_train, y_train):
        logging.info("Starting retraining process...")
        self.train(X_train, y_train)

