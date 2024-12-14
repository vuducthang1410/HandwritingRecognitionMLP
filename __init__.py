import tkinter as tk

from UI.HandwritingInterface import HandwritingInterface

if '__main__' == __name__:
    # # Đường dẫn đến file dữ liệu
    path_train_file = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\emnist-mnist-train.csv'
    path_test_file = 'D:\\DeepLearning\\HandwritingRecognitionMLP\\data\\emnist-mnist-test.csv'

    root = tk.Tk()
    app = HandwritingInterface(root, path_train_file, path_test_file)
    root.mainloop()
