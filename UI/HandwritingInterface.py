from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit, QHBoxLayout, QGroupBox
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint
import cv2
import numpy as np
from MLP import MLP
from DATA_SET import DATA_SET
from UI.PaintUI import  Paint


class HandwritingInterface(QMainWindow):
    def __init__(self, path_train_file, path_test_file):
        super().__init__()
        self.setWindowTitle("Handwriting Prediction")
        self.setGeometry(100, 100, 800, 600)

        self.canvas = Paint(self)

        self.saveButton = QPushButton("Save", self)
        self.saveButton.clicked.connect(self.save)

        self.clearButton = QPushButton("Reset", self)
        self.clearButton.clicked.connect(self.clear)

        self.checkButton = QPushButton("Check", self)
        self.checkButton.clicked.connect(self.check)

        self.resultBox = QTextEdit()

        # Tạo layout chính
        mainLayout = QHBoxLayout()

        # Tạo QGroupBox cho canvas với tiêu đề
        canvasGroupBox = QGroupBox("Canvas")
        canvasLayout = QVBoxLayout()
        canvasLayout.addWidget(self.canvas)
        canvasGroupBox.setLayout(canvasLayout)

        # Tạo QGroupBox cho các nút với tiêu đề
        buttonGroupBox = QGroupBox("Controls")
        buttonLayout = QVBoxLayout()
        buttonLayout.addWidget(self.saveButton)
        buttonLayout.addWidget(self.clearButton)
        buttonLayout.addWidget(self.checkButton)
        buttonLayout.addStretch()
        buttonGroupBox.setLayout(buttonLayout)

        resultGroupBox = QGroupBox("Results")
        resultLayout = QVBoxLayout()
        resultLayout.addWidget(self.resultBox)
        resultGroupBox.setLayout(resultLayout)

        rightLayout = QVBoxLayout()
        rightLayout.addWidget(buttonGroupBox)
        rightLayout.addWidget(resultGroupBox)

        # Thêm các QGroupBox vào layout chính
        mainLayout.addWidget(canvasGroupBox, 3)  # Tỷ lệ chiếm 3 phần
        mainLayout.addLayout(rightLayout, 1)  # Tỷ lệ chiếm 1 phần

        # Đặt mainLayout làm layout chính của cửa sổ
        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

        self.resultBox.setText("Đang cài đặt mô hình!!")
        hidden_sizes = [700, 485, 250, 113, 53]  # 6 lớp ẩn
        self.mlp_model = MLP(features=28 * 28, hidden_layers=hidden_sizes, output_size=10, learning_rate=0.05, epoch=20)
        self.mlp_model.initialize_weights()
        data_set = DATA_SET(path_train_file, path_test_file)
        train_data, train_label = data_set.get_train_data()
        train_labels_one_hot = self.one_hot_encode(train_label)
        self.mlp_model.check_and_train(train_data, train_labels_one_hot)
        self.resultBox.setText("Cài đặt mô hình thành công!!")

    def one_hot_encode(self, labels, num_classes=10):
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot

    def convert_img(self):
        # Lấy ảnh từ canvas dưới dạng QImage
        img = self.canvas.getImage()

        # Chuyển đổi QImage thành mảng NumPy
        img = img.convertToFormat(QImage.Format_RGB32)
        img_np = np.array(img.bits().asarray(img.width() * img.height() * 4)).reshape((img.height(), img.width(), 4))

        # Chuyển ảnh RGB thành ảnh xám (grayscale)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Thay đổi kích thước ảnh thành 28x28
        img_cv2 = cv2.resize(img_cv2, (28, 28))

        # Nếu cần, có thể xoay hoặc lật ảnh
        img_cv2 = np.rot90(img_cv2, k=1)  # Xoay nếu cần
        img_cv2 = np.flipud(img_cv2)  # Lật nếu cần

        return img_cv2

    def preprocess_image(self, img):
        img = 255 - img
        # Chuẩn hóa giá trị pixel trong khoảng [0, 1]
        img = img.astype(np.float32) / 255.0

        # Chuyển ảnh thành vector 1 chiều (784,)
        img_array = img.flatten()

        return img_array

    def predict_image(self, image_path):
        # Xử lý ảnh
        img_data = self.preprocess_image(image_path)

        # Chuyển ảnh thành batch (1, 784)
        img_data = img_data.reshape(1, -1)

        # Dự đoán
        prediction = self.mlp_model.predict(img_data)
        return prediction[0]  # Trả về nhãn dự đoán

    def save(self):
        img = self.convert_img()
        label_predict = self.predict_image(img)
        self.resultBox.setText(f"Nhận dạng thành công! Nhãn dự đoán: {label_predict}")

    def clear(self):
        self.canvas.clearImage()

    def check(self):
        textOld = self.resultBox.toPlainText()
        self.resultBox.setText(textOld + "1")

