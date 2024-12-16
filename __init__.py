from PyQt5.QtWidgets import QApplication
import sys
from UI.HandwritingInterface import HandwritingInterface

if '__main__' == __name__:
    # # Đường dẫn đến file dữ liệu
    path_train_file = 'data/emnist-byclass-train.csv'
    path_test_file = 'data/emnist-byclass-test.csv'

    app = QApplication(sys.argv)
    window = HandwritingInterface(path_train_file, path_test_file)
    window.show()
    sys.exit(app.exec_())
