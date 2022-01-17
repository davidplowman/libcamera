import picamera2
import selectors
import threading
import atexit
import fcntl
import mmap
from v4l2 import *

class H264Encoder():

    def __init__(self, picam2):
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.thread_func, args=(picam2,))
        self.thread.setDaemon(True)
        self.running = True
        self.thread.start()
        self.event.wait()
        self.bufs = {}
        self.idx = 0
        self.vd = open('/dev/video11', 'rb+', buffering=0)

        cp = v4l2_capability()
        fcntl.ioctl(self.vd, VIDIOC_QUERYCAP, cp)
        print("Driver:", "".join((chr(c) for c in cp.driver)))
        print("Name:", "".join((chr(c) for c in cp.card)))
        print("Is a video capture device?", bool(cp.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        print("Supports read() call?", bool(cp.capabilities &  V4L2_CAP_READWRITE))
        print("Supports streaming?", bool(cp.capabilities & V4L2_CAP_STREAMING))

        ctrl = v4l2_control()
        ctrl.id = V4L2_CID_MPEG_VIDEO_BITRATE
        ctrl.value = 10000000
        fcntl.ioctl(self.vd, VIDIOC_S_CTRL, ctrl)

        V4L2_PIX_FMT_H264 = v4l2_fourcc('H', '2', '6', '4')
        W = 800
        H = 600

        fmt = v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
        fmt.fmt.pix_mp.width = W
        fmt.fmt.pix_mp.height = H
        fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_YUV420
        fmt.fmt.pix_mp.plane_fmt[0].bytesperline = 832
        fmt.fmt.pix_mp.field = V4L2_FIELD_ANY
        fmt.fmt.pix_mp.colorspace = V4L2_COLORSPACE_JPEG
        fmt.fmt.pix_mp.num_planes = 1
        fcntl.ioctl(self.vd, VIDIOC_S_FMT, fmt)

        fmt = v4l2_format()
        fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        fmt.fmt.pix_mp.width = W
        fmt.fmt.pix_mp.height = H
        fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264
        fmt.fmt.pix_mp.field = V4L2_FIELD_ANY
        fmt.fmt.pix_mp.colorspace = V4L2_COLORSPACE_DEFAULT
        fmt.fmt.pix_mp.num_planes = 1
        fmt.fmt.pix_mp.plane_fmt[0].bytesperline = 0
        fmt.fmt.pix_mp.plane_fmt[0].sizeimage = 512 << 10
        fcntl.ioctl(self.vd, VIDIOC_S_FMT, fmt)

        NUM_OUTPUT_BUFFERS = 6
        NUM_CAPTURE_BUFFERS = 12

        reqbufs = v4l2_requestbuffers()
        reqbufs.count = NUM_OUTPUT_BUFFERS
        reqbufs.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
        reqbufs.memory = V4L2_MEMORY_DMABUF
        fcntl.ioctl(self.vd, VIDIOC_REQBUFS, reqbufs)

        reqbufs = v4l2_requestbuffers()
        reqbufs.count = NUM_CAPTURE_BUFFERS
        reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
        reqbufs.memory = V4L2_MEMORY_MMAP
        fcntl.ioctl(self.vd, VIDIOC_REQBUFS, reqbufs)

        for i in range(reqbufs.count):
            planes = v4l2_plane * VIDEO_MAX_PLANES
            planes = planes()
            buffer = v4l2_buffer()
            ctypes.memset(ctypes.byref(buffer), 0, ctypes.sizeof(buffer))
            buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE
            buffer.memory = V4L2_MEMORY_MMAP
            buffer.index = i
            buffer.length = 1
            buffer.m.planes = planes
            ret = fcntl.ioctl(self.vd, VIDIOC_QUERYBUF, buffer)
            self.bufs[i] = ( mmap.mmap(self.vd.fileno(), buffer.m.planes[0].length, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_SHARED,
                               offset=buffer.m.planes[0].m.mem_offset) , buffer.m.planes[0].length)
            ret = fcntl.ioctl(self.vd, VIDIOC_QBUF, buffer)

        typev = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE)
        fcntl.ioctl(self.vd, VIDIOC_STREAMON, typev)
        typev = v4l2_buf_type(V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE)
        fcntl.ioctl(self.vd, VIDIOC_STREAMON, typev)

    def thread_func(self, picam2):
        picam2.asynchronous = True
        sel = selectors.DefaultSelector()
        sel.register(picam2.camera_manager.efd, selectors.EVENT_READ, self.handle_request)
        atexit.register(self.stop)
        self.event.set()

        while self.running:
            events = sel.select()
            for key, mask in events:
                callback = key.data
                callback(picam2)

        atexit.unregister(self.stop)
        picam2.asynchronous = False

    def handle_request(self, picam2):
        completed_request = picam2.process_requests()
        stream = picam2.streams[picam2.video_stream]
        cfg = stream.configuration
        width, height = cfg.size
        
        fb = completed_request.request.buffers[stream]
        fd = fb.fd(0)
        stride = cfg.stride

        buf = v4l2_buffer()
        index = self.idx
        self.idx += 1
        timestamp_us = 0
        planes = v4l2_plane * VIDEO_MAX_PLANES
        planes = planes()

        buf.type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE
        buf.index = index
        buf.field = V4L2_FIELD_NONE
        buf.memory = V4L2_MEMORY_DMABUF
        buf.length = 1
        buf.timestamp.tv_sec = timestamp_us / 1000000
        buf.timestamp.tv_usec = timestamp_us % 1000000
        buf.m.planes = planes
        buf.m.planes[0].m.fd = fd
        buf.m.planes[0].bytesused = cfg.frameSize
        buf.m.planes[0].length = cfg.frameSize
        
        ret = fcntl.ioctl(self.vd, VIDIOC_QBUF, buf)
        completed_request.release()

    def stop(self):
        self.running = False
        self.thread.join()
