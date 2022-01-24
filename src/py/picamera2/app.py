#!/usr/bin/python3

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from q_picamera2 import *
from picamera2 import *


picam2 = Picamera2()
picam2.open_camera()
picam2.configure({"role": "preview", "format": "BGR888", "size": (800, 600)})

app = QApplication([])
window = QWidget()
window.resize(800, 600)

layout = QVBoxLayout()

button1 = QPushButton("Start Camera")


def on_button1_clicked():
    if not picam2.started:
        picam2.start()


button1.clicked.connect(on_button1_clicked)
layout.addWidget(button1)

button2 = QPushButton("Capture JPEG")


def on_button2_clicked():
    if not picam2.async_operation_in_progress:
        cfg = picam2.generate_configuration({"role": "still"})
        picam2.switch_mode_and_capture("test.jpg", cfg, wait=False, signal_function=None)
    else:
        print("Busy!")


button2.clicked.connect(on_button2_clicked)
layout.addWidget(button2)

qpicamera2 = QPicamera2(picam2)
layout.addWidget(qpicamera2)

window.setLayout(layout)
window.setWindowTitle("Qt Picamera2 App")

window.show()
app.exec()
