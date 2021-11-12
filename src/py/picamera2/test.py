#!/usr/bin/python3

from qt_preview import *
from picamera2 import *
import time

picam2 = Picamera2()
preview = QtPreview(picam2)
picam2.open_camera()

picam2.configure(picam2.generate_configuration({"role": "preview"}))
picam2.start()

time.sleep(2)
capture_config = picam2.generate_configuration({"role": "still"})
picam2.switch_mode_and_capture("test.jpg", capture_config)
