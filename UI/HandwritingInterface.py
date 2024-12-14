import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageDraw
from MLP import MLP
from DATA_SET import DATA_SET


class HandwritingInterface:
    def __init__(self, root, path_train_file, path_test_file):
        self.root = root
        self.root.title("Handwriting Recognition")
        # Kích thước canvas và giao diện
        self.canvas_width = 600
        self.canvas_height = 400

        # Tạo khung chính chia thành 2 phần
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Khung bên trái (Canvas vẽ)
        self.left_frame = tk.Frame(self.main_frame, width=400, height=self.canvas_height)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Khung bên phải (Nội dung khác)
        self.right_frame = tk.Frame(self.main_frame, width=200, height=self.canvas_height, bg="lightgray")
        self.right_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.right_frame.pack_propagate(False)

        # Canvas vẽ (nằm trong khung bên trái)
        self.canvas = tk.Canvas(self.left_frame, width=500, height=self.canvas_height, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Label tiêu đề
        self.label_title = tk.Label(self.right_frame, text="Công cụ vẽ tay", bg="lightgray", font=("Arial", 14, "bold"))
        self.label_title.pack(pady=20)

        self.label_status = tk.Label(self.right_frame, text="Trạng thái: Chưa vẽ", bg="lightgray", font=("Arial", 10))
        self.label_status.pack(pady=10)

        self.button_clear = tk.Button(self.right_frame, text="Xóa", command=self.clear_canvas)
        self.button_clear.pack(pady=10)

        self.image = Image.new("L", (self.canvas_height, self.canvas_height), 255)
        self.draw = ImageDraw.Draw(self.image)

        # Sự kiện vẽ trên canvas
        self.is_drawing = False  # Trạng thái đang vẽ
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.label_status.config(text="Đang cài đặt mô hình!!")
        hidden_sizes = [700, 485, 250, 113, 53]  # 6 lớp ẩn
        self.mlp_model = MLP(features=28 * 28, hidden_layers=hidden_sizes, output_size=10, learning_rate=0.05, epoch=9)
        self.mlp_model.initialize_weights()
        data_set = DATA_SET(path_train_file, path_test_file)
        train_data, train_label = data_set.get_train_data()
        train_labels_one_hot = self.one_hot_encode(train_label)
        self.mlp_model.check_and_train(train_data, train_labels_one_hot)
        self.label_status.config(text="Cài đặt mô hình thành công!!")

    # test lại tỉ lệ kiểm tra xem đọc file đúng chưa
    # test_data, test_label = data_set.get_test_data()
    # test_label_one_hot = self.one_hot_encode(test_label)
    # predictions = self.mlp_model.predict(test_data)
    # accuracy = np.mean(predictions == np.argmax(test_label_one_hot, axis=1))
    # self.label_status.config(text=f'Accuracy: {accuracy * 100:.2f}%')

    def start_drawing(self, event):
        self.clear_canvas()
        self.is_drawing = True
        self.label_status.config(text="Trạng thái: Đang vẽ")
        self.paint(event)

    def paint(self, event):
        if self.is_drawing:
            x, y = event.x, event.y
            r = 6  # Kích thước nét vẽ
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
            self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def stop_drawing(self, event):
        self.is_drawing = False
        self.label_status.config(text="Trạng thái: Kết thúc vẽ")
        img = self.convert_img()
        label_predict = self.predict_image(img)
        self.label_status.config(text=f"Cài đặt mô hình thành công!!{label_predict}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="white")
        self.label_status.config(text="Trạng thái: Chưa vẽ")

    # Hàm xử lý ảnh để đưa vào mô hình
    def preprocess_image(self, img):
        img = 255 - img
        # Chuẩn hóa giá trị pixel trong khoảng [0, 1]
        img = img.astype(np.float32) / 255.0

        # Chuyển ảnh thành vector 1 chiều (784,)
        img_array = img.flatten()

        return img_array

    def one_hot_encode(self, labels, num_classes=10):
        one_hot = np.zeros((labels.size, num_classes))
        one_hot[np.arange(labels.size), labels] = 1
        return one_hot

    # Dự đoán ảnh sử dụng mô hình đã train
    def predict_image(self, image_path):
        # Xử lý ảnh
        img_data = self.preprocess_image(image_path)

        # Chuyển ảnh thành batch (1, 784)
        img_data = img_data.reshape(1, -1)

        # Dự đoán
        prediction = self.mlp_model.predict(img_data)
        return prediction[0]  # Trả về nhãn dự đoán

    def convert_img(self):
        img = self.image.copy()
        img_np = np.array(img)
        img_cv2 = cv2.resize(img_np, (28, 28))
        img_cv2 = np.rot90(img_cv2, k=1)
        img_cv2 = np.flipud(img_cv2)
        return img_cv2
