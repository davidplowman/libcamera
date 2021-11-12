#!/usr/bin/python3

from drm_preview import *
from picamera2 import *
import time

picam2 = Picamera2()
preview = DrmPreview(picam2)
picam2.open_camera()

cfg = picam2.generate_configuration({"role": "preview"})
cfg[0]["format"] = "RGB888"
picam2.configure(cfg)

picam2.start()
