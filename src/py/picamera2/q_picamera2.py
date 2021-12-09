from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, QSocketNotifier
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PIL import Image
from PIL.ImageQt import ImageQt


class QPicamera2(QWidget):
    def __init__(self, picam2, parent=None):
        super().__init__(parent=parent)
        self.picamera2 = picam2
        self.label = QLabel(self)
        self.label.resize(640, 480)
        self.label.setText("Camera Preview")
        self.camera_notifier = QSocketNotifier(self.picamera2.camera_manager.efd,
                                               QtCore.QSocketNotifier.Read,
                                               self)
        self.camera_notifier.activated.connect(self.handle_requests)

    @pyqtSlot()
    def handle_requests(self):
        request = self.picamera2.process_requests()
        if not request:
            return

        if self.picamera2.preview_stream >= 0:
            img = request.make_pil_image(self.picamera2.preview_stream)
            qim = ImageQt(img).copy()
            pix = QtGui.QPixmap.fromImage(qim)
            self.label.setPixmap(pix)

        request.release()
