#!/usr/bin/python3

from h264_encoder import *
from drm_preview import *
from picamera2 import *
import time

picam2 = Picamera2()
picam2.open_camera()

cfg = picam2.generate_configuration({"role": "video"})
cfg["streams"][0]["format"] = "YUV420"
picam2.configure(cfg)

h264 = H264Encoder(picam2)

picam2.start()

time.sleep(10)
