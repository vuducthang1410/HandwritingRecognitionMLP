from PyQt5.QtWidgets import  QWidget
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
        return self.image

    def saveImage(self, filePath):
        self.image.save(filePath)

    def resizeEvent(self, event):
        new_image = QImage(self.size(), QImage.Format_RGB32)
        new_image.fill(Qt.white)
        painter = QPainter(new_image)
        painter.drawImage(QPoint(), self.image)
        self.image = new_image
        self.update()
        super().resizeEvent(event)