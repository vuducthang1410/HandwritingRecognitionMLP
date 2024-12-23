import logging
import os
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, \
    QGroupBox

from DATA_SET import DATA_SET
from MLP import MLP
from UI.PaintUI import Paint

# Thiết lập logging
logging.basicConfig(filename='handwriting_recognition.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TrainThread(QThread):
    result_signal = pyqtSignal(str)

    def __init__(self, path_train_file, path_test_file, mlp_model):
        super().__init__()
        self.path_train_file = path_train_file
        self.path_test_file = path_test_file
        self.mlp_model = mlp_model

    def run(self):
        try:
            logging.info("Bắt đầu quá trình huấn luyện")
            data_set = DATA_SET(self.path_train_file, self.path_test_file)
            train_data, train_label = data_set.get_train_data()
            train_labels_one_hot = self.one_hot_encode(train_label)

            self.mlp_model.train_again(train_data, train_labels_one_hot)
            self.result_signal.emit("Huấn luyện lại mô hình thành công!!")
            logging.info("Quá trình huấn luyện hoàn tất thành công")
        except Exception as e:
            error_msg = f"Lỗi khi huấn luyện: {str(e)}"
            self.result_signal.emit(error_msg)
            logging.error(error_msg, exc_info=True)

    def one_hot_encode(self, labels, num_classes=62):
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot


class HandwritingInterface(QMainWindow):
    def __init__(self, path_train_file, path_test_file, path_values_ACII_map, path_map_values_predict):
        super().__init__()
        self.setWindowTitle("Nhận dạng chữ viết tay")
        self.setGeometry(100, 100, 800, 600)
        self.path_values_ACII_map = path_values_ACII_map
        self.path_map_values_predict = path_map_values_predict
        self.canvas = Paint(self)
        self.saveButton = QPushButton("Lưu", self)
        self.saveButton.clicked.connect(self.save)
        self.clearButton = QPushButton("Xóa", self)
        self.clearButton.clicked.connect(self.clear)
        self.checkButton = QPushButton("Kiểm tra", self)
        self.checkButton.clicked.connect(self.check)
        self.trainButton = QPushButton("Huấn luyện lại", self)
        self.trainButton.clicked.connect(lambda: self.start_training(path_train_file, path_test_file))
        self.resultBox = QTextEdit()

        self.setup_ui()
        self.mlp_model = None
        self.initialize_model(path_train_file, path_test_file)

    def setup_ui(self):
        mainLayout = QHBoxLayout()
        canvasGroupBox = QGroupBox("Vùng vẽ")
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.canvas)
        canvasGroupBox.setLayout(canvasLayout)

        buttonGroupBox = QGroupBox("Điều khiển")
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.checkButton)
        buttonLayout.addWidget(self.trainButton)
        buttonLayout.addStretch()
        buttonGroupBox.setLayout(buttonLayout)

        resultGroupBox = QGroupBox("Kết quả")
        resultLayout = QVBoxLayout()
        resultLayout.addWidget(self.resultBox)
        resultGroupBox.setLayout(resultLayout)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(buttonGroupBox)
        rightLayout.addWidget(resultGroupBox)

        mainLayout.addWidget(canvasGroupBox, 3)
        mainLayout.addLayout(rightLayout, 1)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

    def initialize_model(self, path_train_file, path_test_file):
        self.resultBox.setText("Đang cài đặt mô hình...")
        try:
            hidden_sizes = [700, 485, 250, 113, 53]
            self.mlp_model = MLP(features=28 * 28, hidden_layers=hidden_sizes, output_size=62, learning_rate=0.05,
                                 epoch=10,
                                 callback=self.update_training_progress)
            self.mlp_model.initialize_weights()
            data_set = DATA_SET(path_train_file, path_test_file)
            train_data, train_label = data_set.get_train_data()
            train_labels_one_hot = self.one_hot_encode(train_label)
            self.mlp_model.check_and_train(train_data, train_labels_one_hot)
            self.resultBox.setText("Cài đặt mô hình thành công!")
            logging.info("Mô hình được khởi tạo thành công")
        except Exception as e:
            error_msg = f"Lỗi khi cài đặt mô hình: {str(e)}"
            self.resultBox.setText(error_msg)
            logging.error(error_msg, exc_info=True)

    def start_training(self, path_train_file, path_test_file):
        if self.mlp_model is None:
            self.resultBox.setText("Mô hình chưa được khởi tạo. Vui lòng khởi tạo mô hình trước.")
            return

        self.resultBox.setText("Bắt đầu huấn luyện lại...")
        self.train_thread = TrainThread(path_train_file, path_test_file, self.mlp_model)
        self.train_thread.result_signal.connect(self.update_result)
        self.train_thread.start()

    def update_result(self, result):
        self.resultBox.setText(result)

    def update_training_progress(self, message):
        self.resultBox.append(message)
        QApplication.processEvents()  # Cập nhật giao diện người dùng

    def one_hot_encode(self, labels, num_classes=62):
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot

    def save(self):
        folder_path = 'images'
        os.makedirs(folder_path, exist_ok=True)
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f'train_image_{current_time}.png'
        file_path = os.path.join(folder_path, file_name)
        self.canvas.saveImage(file_path)
        self.resultBox.setText(f"Lưu ảnh thành công: {file_name}")

    def clear(self):
        self.canvas.clearImage()
        self.resultBox.setText("Vui lòng vẽ để dự đoán!!!")

    def check(self):
        img = self.convert_img()
        top_5_indices, top_5_probs = self.predict_image(img)
        self.resultBox.setText(f'Nhận dạng thành công! Nhãn dự đoán: ')
        top_5_indices = np.array(top_5_indices).flatten()
        top_5_probs=np.array(top_5_probs).flatten()
        for i in range(len(top_5_indices)):
            self.resultBox.append(f"{self.mappingResultPredictToASCII(top_5_indices[i])}: {top_5_probs[i] * 100:.4f}")

    def convert_img(self):
        img = self.canvas.getImage()
        img = img.convertToFormat(QImage.Format_RGB32)
        img_np = np.array(img.bits().asarray(img.width() * img.height() * 4)).reshape((img.height(), img.width(), 4))
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_cv2 = cv2.resize(img_cv2, (28, 28))
        img_cv2 = np.rot90(img_cv2, k=1)
        img_cv2 = np.flipud(img_cv2)
        return img_cv2

    def preprocess_image(self, img):
        img = 255 - img
        img = img.astype(np.float32) / 255.0
        img_array = img.flatten()
        return img_array

    def predict_image(self, image_path):
        img_data = self.preprocess_image(image_path)
        img_data = img_data.reshape(1, -1)
        top_5_indices, top_5_probs = self.mlp_model.predict(img_data)
        return top_5_indices, top_5_probs

    def closeEvent(self, event):
        logging.info("Ứng dụng đang đóng")
        super().closeEvent(event)

    def mappingResultPredictToASCII(self, keyData):
        result_dict = {}
        with open(self.path_map_values_predict, 'r') as file:
            data = file.readlines()
        for line in data:
            key, value = line.strip().split(' ')
            result_dict[int(key)] = value
        data1 = int(result_dict.get(int(keyData)))
        ascii_dict = {}
        with open(self.path_values_ACII_map, 'r') as file:
            data = file.readlines()
        for line in data:
            key, value = line.strip().split(' ')
            ascii_dict[int(key)] = value
        return ascii_dict.get(data1)
