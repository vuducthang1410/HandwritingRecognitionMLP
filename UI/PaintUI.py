from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPen, QImage
from PyQt5.QtCore import Qt, QPoint

class Paint(QWidget):
    def __init__(self, parent=None):
        super(Paint, self).__init__(parent)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.lastPoint = QPoint()

    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 20, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def clearImage(self):
        self.image.fill(Qt.white)
        self.update()

    def getImage(self):
        return self.crop_to_center(self.image)

    def saveImage(self, filePath):
        imageCrop = self.crop_to_center(self.image)
        imageCrop.save(filePath)

    def resizeEvent(self, event):
        new_image = QImage(self.size(), QImage.Format_RGB32)
        new_image.fill(Qt.white)
        painter = QPainter(new_image)
        painter.drawImage(QPoint(), self.image)
        self.image = new_image
        self.update()
        super().resizeEvent(event)

    def crop_to_center(self, image):
        # Chuyển QImage sang ảnh xám bằng phương thức convertToFormat()
        grayscale_image = image.convertToFormat(QImage.Format_Grayscale8)

        # Tìm vùng chứa các pixel không phải màu trắng
        width = grayscale_image.width()
        height = grayscale_image.height()

        # Tìm bbox (bounding box) chứa các pixel không phải trắng
        left, top, right, bottom = width, height, 0, 0

        for y in range(height):
            for x in range(width):
                pixel_value = grayscale_image.pixel(x, y)

                # Kiểm tra pixel không phải trắng (màu đen hoặc các giá trị thấp hơn)
                if pixel_value != 0xFFFFFFFF:  # Nếu pixel không phải trắng
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)

        # Nếu không tìm thấy bất kỳ pixel không phải trắng nào, trả về ảnh gốc
        if left == width or top == height or right == 0 or bottom == 0:
            return image

        # Cắt ảnh theo bounding box đã xác định
        cropped_image = grayscale_image.copy(left, top, right - left, bottom - top)

        # Tính toán để cắt ảnh sao cho nội dung nằm chính giữa
        crop_width = right - left
        crop_height = bottom - top
        max_dim = max(crop_width, crop_height)
        centered_image = QImage(max_dim, max_dim, QImage.Format_Grayscale8)
        centered_image.fill(Qt.white)

        # Tính toán tọa độ để dán hình vào trung tâm
        offset_x = (max_dim - crop_width) // 2
        offset_y = (max_dim - crop_height) // 2

        painter = QPainter(centered_image)
        painter.drawImage(offset_x, offset_y, cropped_image)
        painter.end()

        return centered_image
