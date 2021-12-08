#!/usr/bin/python3

from qt_gl_preview import *
from picamera2 import *
import time

picam2 = Picamera2()
preview = QtGlPreview(picam2)
picam2.open_camera()

cfg = picam2.generate_configuration({"role": "preview"})
cfg["streams"][0]["format"] = "YUYV"
picam2.configure(cfg)

picam2.start()
