#!/usr/bin/python3

from qt_preview import *
from picamera2 import *

picam2 = Picamera2()
preview = QtPreview(picam2, 640, 480)

picam2.open_camera()
preview_config = picam2.generate_configuration({"role": "preview"})
preview_config["streams"][0]["size"] = (640, 480)
# preview_config["transform"] = pylibcamera.Transform(180)

picam2.configure(preview_config)
picam2.start()

# time.sleep(2)
# capture_config = picam2.generate_configuration({"role": "still"})
# capture_config["transform"] = pylibcamera.Transform(180)
# picam2.switch_mode_and_capture("test.jpg", capture_config)
