import numpy as np
from MLP import MLP
from DATA_SET import DATA_SET
import cv2


# Hàm xử lý ảnh để đưa vào mô hình
def preprocess_image(image_path):
    # Đọc ảnh bằng OpenCV và chuyển thành grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Đảm bảo ảnh có kích thước 28x28 (resize nếu cần thiết)
    img = cv2.resize(img, (28, 28))
    img = 255 - img
    # Chuẩn hóa giá trị pixel trong khoảng [0, 1]
    img = img.astype(np.float32) / 255.0

    # Chuyển ảnh thành vector 1 chiều (784,)
    img_array = img.flatten()

    return img_array


def one_hot_encode(labels, num_classes=10):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot


# Dự đoán ảnh sử dụng mô hình đã train
def predict_image(model, image_path):
    # Xử lý ảnh
    img_data = preprocess_image(image_path)

    # Chuyển ảnh thành batch (1, 784)
    img_data = img_data.reshape(1, -1)

    # Dự đoán
    prediction = model.predict(img_data)
    return prediction[0]  # Trả về nhãn dự đoán


if '__main__' == __name__:
    # Đường dẫn đến file dữ liệu
    path_train_file = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\emnist-mnist-train.csv'
    path_test_file = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\emnist-mnist-test.csv'

    # Tạo đối tượng dữ liệu
    data_set = DATA_SET(path_train_file, path_test_file)

    hidden_sizes = [700, 485, 250, 113, 53]  # 6 lớp ẩn

    # Khởi tạo mô hình với 6 lớp ẩn
    model = MLP(features=28 * 28, hidden_layers=hidden_sizes, output_size=10, learning_rate=0.05, epoch=5)
    model.initialize_weights()

    # Lấy dữ liệu huấn luyện và kiểm tra
    train_data, train_label = data_set.get_train_data()
    test_data, test_label = data_set.get_test_data()

    # Mã hóa nhãn thành one-hot
    train_labels_one_hot = one_hot_encode(train_label)
    test_labels_one_hot = one_hot_encode(test_label)

    # Huấn luyện mô hình
    model.train(train_data, train_labels_one_hot)

    # Dự đoán và tính toán độ chính xác
    predictions = model.predict(test_data)
    accuracy = np.mean(predictions == np.argmax(test_labels_one_hot, axis=1))  # so sánh với nhãn thực tế
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Dự đoán một ảnh cụ thể
    image_path_0 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\0.png'
    image_path_1 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\1.png'
    image_path_2 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\2.png'
    image_path_3 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\3.png'
    image_path_4 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\4.png'
    image_path_5 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\5.png'
    image_path_6 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\6.png'
    image_path_7 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\7.png'
    image_path_8 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\8.png'
    image_path_9 = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\9.png'

    # Đổi thành đường dẫn ảnh của bạn
    predicted_label0 = predict_image(model, image_path_0)
    print(f'Dự đoán cho ảnh {image_path_0}: {predicted_label0}')
    predicted_label1 = predict_image(model, image_path_1)
    print(f'Dự đoán cho ảnh {image_path_1}: {predicted_label1}')
    predicted_label2 = predict_image(model, image_path_2)
    print(f'Dự đoán cho ảnh {image_path_2}: {predicted_label2}')
    predicted_label3 = predict_image(model, image_path_3)
    print(f'Dự đoán cho ảnh {image_path_3}: {predicted_label3}')

    predicted_label4 = predict_image(model, image_path_4)
    print(f'Dự đoán cho ảnh {image_path_4}: {predicted_label4}')

    predicted_label5 = predict_image(model, image_path_5)
    print(f'Dự đoán cho ảnh {image_path_5}: {predicted_label5}')

    predicted_label6 = predict_image(model, image_path_6)
    print(f'Dự đoán cho ảnh {image_path_6}: {predicted_label6}')

    predicted_label7 = predict_image(model, image_path_7)
    print(f'Dự đoán cho ảnh {image_path_7}: {predicted_label7}')

    predicted_label8 = predict_image(model, image_path_8)
    print(f'Dự đoán cho ảnh {image_path_8}: {predicted_label8}')

    predicted_label9 = predict_image(model, image_path_9)
    print(f'Dự đoán cho ảnh {image_path_9}: {predicted_label9}')
