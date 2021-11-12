import picamera2
import pykms
from null_preview import *

class DrmPreview(NullPreview):
    FMT_MAP = {
        "RGB888": pykms.PixelFormat.RGB888,
        "YUYV": pykms.PixelFormat.YUYV,
        "ARGB8888": pykms.PixelFormat.ARGB8888,
        "XRGB8888": pykms.PixelFormat.XRGB8888,
    }

    def __init__(self, picam2):
        self.init_drm()
        self.stop_count = 0
        super().__init__(picam2)

    def handle_request(self, picam2):
        completed_request = picam2.process_requests()

        if completed_request:
            self.render_drm(picam2, completed_request)

    def init_drm(self):
        self.card = pykms.Card()
        self.resman = pykms.ResourceManager(self.card)
        conn = self.resman.reserve_connector()
        self.crtc = self.resman.reserve_crtc(conn)

        self.plane = None
        self.drmfbs = {}
        self.current = None
        self.window = (0, 0, 640, 480)

    def render_drm(self, picam2, completed_request):
        stream = picam2.streams[picam2.preview_stream]
        cfg = stream.configuration
        width, height = cfg.size
        fb = completed_request.request.buffers[stream]

        if fb not in self.drmfbs:
            if self.stop_count != picam2.stop_count:
                if picam2.verbose:
                    print("Garbage collecting", len(self.drmfbs), "dmabufs")
                self.drmfbs = {}
                self.stop_count = picam2.stop_count

            fmt = self.FMT_MAP[cfg.fmt]
            if self.plane is None:
                self.plane = self.resman.reserve_overlay_plane(self.crtc, fmt)
                if picam2.verbose:
                    print("Got plane", self.plane, "for format", fmt)
                assert(self.plane)
            fd = fb.fd(0)
            stride = cfg.stride
            drmfb = pykms.DmabufFramebuffer(self.card, width, height, fmt, [fd], [stride], [0])
            self.drmfbs[fb] = drmfb
            if picam2.verbose:
                print("Made drm fb", drmfb, "for request", completed_request.request)

        drmfb = self.drmfbs[fb]
        x, y, w, h = self.window
        self.crtc.set_plane(self.plane, drmfb, x, y, w, h, 0, 0, width, height)

        if self.current:
            self.current.release()
        self.current = completed_request
