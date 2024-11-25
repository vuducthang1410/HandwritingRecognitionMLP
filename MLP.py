import numpy as np


def relu_prime(x):
    return np.where(x > 0, 1, 0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def relu(x):
    return np.maximum(0, x)


class MLP:
    def __init__(self, features, hidden_layers, output_size, learning_rate, epoch):
        self.features = features
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epoch = epoch

        # Khởi tạo trọng số và bias
        self.weights = []
        self.biases = []

    def initialize_weights(self):
        np.random.seed(42)  # Đảm bảo khởi tạo ngẫu nhiên tái lập được

        # Xây dựng danh sách kích thước các lớp
        layer_sizes = [self.features] + self.hidden_layers + [self.output_size]

        # Khởi tạo trọng số và bias cho các lớp
        for i in range(len(layer_sizes) - 1):
            weight_matrix = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            bias_vector = np.zeros((1, layer_sizes[i + 1]))  # Bias khởi tạo bằng 0
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def forward_propagation(self, X):
        self.a = []  # danh sách chứa các giá trị của activation
        self.z = []  # danh sách chứa các giá trị của z

        activation = X
        self.a.append(activation)  # Lưu giá trị đầu vào (X) là giá trị đầu tiên của activation

        # Lan truyền xuôi qua các lớp ẩn
        for i in range(len(self.hidden_layers)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            self.z.append(z)
            activation = relu(z)  # Áp dụng hàm ReLU
            self.a.append(activation)  # Lưu giá trị activation

        # Xử lý lớp đầu ra (softmax cho phân loại đa lớp)
        z_output = np.dot(activation, self.weights[-1]) + self.biases[-1]
        self.z.append(z_output)
        output = softmax(z_output)
        self.a.append(output)

        return output

    def backward_propagation(self, X, y):
        m = X.shape[0]
        self.d_weights = []
        self.d_biases = []

        # Tính gradient tại lớp đầu ra
        delta = (self.a[-1] - y)  # Đối với softmax, lỗi là sự khác biệt giữa output và ground truth
        dW = np.dot(self.a[-2].T, delta) / m
        db = np.sum(delta, axis=0, keepdims=True) / m
        self.d_weights.append(dW)
        self.d_biases.append(db)

        # Lan truyền ngược qua các lớp ẩn
        for i in range(len(self.hidden_layers) - 1, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].T) * relu_prime(self.z[i])
            dW = np.dot(self.a[i].T, delta) / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            self.d_weights.insert(0, dW)  # Lưu gradient vào đúng vị trí
            self.d_biases.insert(0, db)

    def update_weights(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]

    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / m  # Cross entropy loss
        return loss

    def compute_accuracy(self, X, y):
        # Dự đoán nhãn
        predictions = self.predict(X)
        # So sánh dự đoán với nhãn thực
        correct_predictions = np.sum(predictions == y)
        accuracy = correct_predictions / X.shape[0]
        return accuracy
    def train(self, X_train, y_train, batch_size=32):
        m = X_train.shape[0]
        for epoch in range(self.epoch):
            for i in range(0, m, batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Lan truyền xuôi
                output = self.forward_propagation(X_batch)

                # Tính mất mát (Loss)
                loss = self.compute_loss(output, y_batch)

                # Lan truyền ngược
                self.backward_propagation(X_batch, y_batch)

                # Cập nhật trọng số và bias
                self.update_weights()

            print(f"Epoch {epoch + 1}/{self.epoch}, Loss: {loss}")

    def predict(self, X):
        output = self.forward_propagation(X)
        return np.argmax(output, axis=1)  # Dự đoán lớp với giá trị xác suất cao nhất
