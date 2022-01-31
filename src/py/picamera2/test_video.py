#!/usr/bin/python3

from h264_encoder import *
from drm_preview import *
from picamera2 import *
import time
import os

picam2 = Picamera2()
picam2.open_camera()

cfg = picam2.generate_configuration({"role": "video"})
cfg["streams"][0]["format"] = "YUV420"

preview = NullPreview(picam2)
encoder = H264Encoder(10000000)

picam2.configure(cfg)
encoder.output = open('test.h264', 'wb')
picam2.encoder = encoder
picam2.start_encoder()
picam2.start({"FrameDurationLimits": (33333, 33333)})
time.sleep(10)
picam2.stop()
picam2.stop_encoder()
