import numpy as np
import pandas as pd


class DATA_SET(object):
    def __init__(self, train_file, test_file):
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)

        self.train_labels = self.train_data.iloc[:, 0].values  # Cột đầu tiên là nhãn
        self.train_images = self.train_data.iloc[:, 1:].values  # Các cột còn lại là pixel

        self.test_labels = self.test_data.iloc[:, 0].values  # Cột đầu tiên là nhãn
        self.test_images = self.test_data.iloc[:, 1:].values  # Các cột còn lại là pixel


        # Giả sử self.train_images và self.test_images là mảng numpy với giá trị [0, 255]
        self.train_images = (self.train_images >= 128).astype(np.uint8)
        self.test_images = (self.test_images >= 128).astype(np.uint8)

        # # Chuẩn hóa dữ liệu về [0, 1]
        # self.train_images = self.train_images / 255.0
        # self.test_images = self.test_images / 255.0

        # Chuyển đổi thành dạng (số mẫu, 28, 28)
        self.train_images = self.train_images.reshape(-1, 28 * 28)
        self.test_images = self.test_images.reshape(-1, 28 * 28)

    def get_train_data(self):
        return self.train_images, self.train_labels

    def get_test_data(self):
        return self.test_images, self.test_labels
