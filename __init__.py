from PyQt5.QtWidgets import QApplication
import sys
from UI.HandwritingInterface import HandwritingInterface

if '__main__' == __name__:
    # Đường dẫn đến file dữ liệu
    path_train_file = 'data/emnist-byclass-train.csv'
    path_test_file = 'data/emnist-byclass-test.csv'
    path_values_ACII_map = 'data/ASCII_values.txt'
    path_map_values_predict='data/emnist-byclass-mapping.txt'
    app = QApplication(sys.argv)
    window = HandwritingInterface(path_train_file, path_test_file,path_values_ACII_map,path_map_values_predict)
    window.show()
    sys.exit(app.exec_())

