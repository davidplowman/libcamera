#!/usr/bin/python3

import os
import pycamera as pylibcamera
import numpy as np
import threading
from PIL import Image
import time


class Picamera2:
    """Picamera2 class"""

    def __init__(self, verbose=True):
        """Initialise camera system and acquire the camera for use."""
        self.camera_manager = pylibcamera.CameraManager.singleton()
        self.verbose = verbose
        self.camera = None
        self.camera_configuration = None
        self.started = False
        self.stop_count = 0
        self.frames = 0
        self.functions = []
        self.event = threading.Event()
        self.asynchronous = False
        self.async_operation_in_progress = False
        self.asyc_result = None
        self.async_error = None
        self.controls_lock = threading.Lock()
        self.controls = {}
        self.options = {}

        if self.verbose:
            print("Camera manager:", self.camera_manager)
            print("Made", self)

    def __del__(self):
        """Free any resources that are held."""
        if self.verbose:
            print("Freeing resources for", self)
        self.close_camera_()

    def open_camera_(self, cam_num):
        """Acquire a single camera for exclusive use."""
        if not self.camera:
            camera = self.camera_manager.cameras[cam_num]
            if camera.acquire() >= 0:
                self.camera = camera
                if self.verbose:
                    print("Opened camera:", self.camera)
            else:
                raise RuntimeError("Failed to acquire camera {} ({})".format(
                    cam_num, self.camera_manager.cameras[cam_num]))

        elif self.verbose:
            print("Camera already open:", self.camera)

    def open_camera(self, cam_num=0):
        self.open_camera_(cam_num)

    def close_camera_(self):
        """Release any camera that we have acquired for use."""
        if self.camera:
            if self.started:
                self.stop_()
            if self.verbose:
                print("Close camera:", self.camera)
            self.camera.release()
            self.camera = None
        elif self.verbose:
            print("No camera to close")

    def close_camera(self):
        """Release this camera for use by others."""
        if self.started:
            self.stop()
        self.close_camera_()

    def lc_camera_config_to_python(self, camera_config, lc_camera_config):
        # Turn a libcamera camera config object into a Python-ified one.
        stream_configs = [self.lc_stream_config_to_python(stream_config, lc_stream_config)
                          for stream_config, lc_stream_config in zip(camera_config["streams"], lc_camera_config)]
        return {"transform": lc_camera_config.transform, "streams": stream_configs}

    def lc_stream_config_to_python(self, stream_config, lc_stream_config):
        # Turn a libcamera stream config to a Python-ified one.
        return {"role": stream_config["role"],
                "format": lc_stream_config.fmt,
                "size": lc_stream_config.size,
                "buffer_count": lc_stream_config.bufferCount,
                "colour_space": lc_stream_config.colorSpace}

    def generate_configuration_(self, camera_config):
        print("generate_configuration:", camera_config)
        return self.lc_camera_config_to_python(camera_config,
                                               self.python_to_lc_camera_config(camera_config))

    def align_stream(self, stream_config):
        # Adjust the image size so that all planes are a mutliple of 32 bytes wide.
        # This matches the hardware behaviour and means we can be more efficient.
        align = 32
        if stream_config["format"] in ("YUV420", "YVU420"):
            align = 64
        size = stream_config["size"]
        stream_config["size"] = (size[0] - size[0] % align, size[1] - size[1] % 2)

    def is_YUV(self, fmt):
        return fmt in ("NV21", "NV12", "YUV420", "YVU420", "YVYU", "YUYV", "UYVY", "VYUY")

    def is_RGB(self, fmt):
        return fmt in ("BGR888", "RGB888", "XBGR8888", "XRGB8888")

    def massage_configuration(self, orig_camera_config, camera_config):
        # Adjust the camera configuration to be more appropriate to Python users.
        raw_stream = None
        # stream1 will be the main output which can be RGB. If present, stream2 must be
        # no larger than stream1 and YUV.
        stream1 = None
        stream2 = None
        for orig_stream_config, stream_config in zip(orig_camera_config["streams"], camera_config["streams"]):
            # If no format was requested by the user, go with XRGB8888 as most things will cope.
            if "format" not in orig_stream_config:
                stream_config["format"] = "XRGB8888"
            if stream_config["role"] == "raw":
                raw_stream = stream_config
            else:
                if not stream1:
                    stream1 = stream_config
                elif not stream2:
                    stream2 = stream_config
                else:
                    raise RuntimeError("Too many streams (maximum permitted is 3)")
                if not self.is_YUV(stream_config["format"]) and not self.is_RGB(stream_config["format"]):
                    raise RuntimeError("Invalid stream format " + stream_config["format"])
                # Images that are not a multiple of 32 bytes across are going to be extra
                # expensive, so try to avoid them.
                self.align_stream(stream_config)

        if stream1 and stream2:
            if stream2["size"][0] > stream1["size"][0] and stream2["size"][1] > stream1["size"][1]:
                stream1, stream2 = stream2, stream1
            if stream2["size"][0] > stream1["size"][0] or stream2["size"][1] > stream1["size"][1]:
                raise RuntimeError("Second stream must be no larger than main stream in both dimensions")
            if stream1["size"] == stream2["size"] and self.is_YUV(stream1["format"]) and self.is_RGB(stream2["format"]):
                stream1, stream2 = stream2, stream1

        if stream2 and not self.is_YUV(stream2["format"]):
            raise RuntimeError("Second stream must be a YUV format")

        if self.verbose:
            print("massage_configuration out:", camera_config)
        return camera_config

    def generate_configuration(self, camera_config):
        """Generate a camera configuration according to the given specification."""
        if self.camera is None:
            raise RuntimeError("Camera has not been opened")
        if isinstance(camera_config, dict):
            if "streams" not in camera_config:
                camera_config = {"transform": pylibcamera.Transform(), "streams": [camera_config]}
        elif isinstance(camera_config, list):
            camera_config = {"transform": pylibcamera.Transform(), "streams": camera_config}
        else:
            raise RuntimeError("Invalid camera configuration")
        return self.massage_configuration(camera_config, self.generate_configuration_(camera_config))

    def python_to_lc_camera_config(self, camera_config):
        # Convert a Python-ified camera config to a libcamera one.
        roles = {"preview": pylibcamera.StreamRole.Viewfinder,
                 "none": pylibcamera.StreamRole.Viewfinder,
                 "still": pylibcamera.StreamRole.StillCapture,
                 "video": pylibcamera.StreamRole.VideoRecording,
                 "raw": pylibcamera.StreamRole.Raw}
        lc_camera_config = self.camera.generateConfiguration([roles[stream_config["role"].lower()]
                                                              for stream_config in camera_config["streams"]])
        lc_camera_config.transform = camera_config["transform"]
        for stream_config, lc_stream_config in zip(camera_config["streams"], lc_camera_config):
            if "format" in stream_config:
                lc_stream_config.fmt = stream_config["format"]
            if "size" in stream_config:
                lc_stream_config.size = stream_config["size"]
            if "buffer_count" in stream_config:
                lc_stream_config.bufferCount = stream_config["buffer_count"]
            if "colour_space" in stream_config:
                lc_stream_config.colorSpace = stream_config["colour_space"]
        return lc_camera_config

    def make_requests(self):
        # Make libcamera request objects. Makes as many as the number of buffers in the
        # stream with the smallest number of buffers.
        num_requests = min([len(self.allocator.buffers(stream)) for stream in self.streams])
        requests = []
        for i in range(num_requests):
            request = self.camera.createRequest()
            if request is None:
                raise RuntimeError("Could not create request")

            for stream in self.streams:
                if request.addBuffer(stream, self.allocator.buffers(stream)[i]) < 0:
                    raise RuntimeError("Failed to set request buffer")

            requests.append(request)

        return requests

    def find_stream(self, camera_configuration, role):
        # Find a stream matching the given role.
        for i, stream_config in enumerate(camera_configuration["streams"]):
            if stream_config["role"] == role:
                return i
        return -1 if role == "raw" else 0

    def configure_(self, camera_config=[{"role": "preview"}]):
        # Configure the camera system with the given configuration.
        if isinstance(camera_config, dict):
            if "streams" not in camera_config:
                camera_config = {"transform": pylibcamera.Transform(), "streams": [camera_config]}
        elif isinstance(camera_config, list):
            camera_config = {"transform": pylibcamera.Transform(), "streams": camera_config}
        else:
            raise RuntimeError("Invalid camera configuration")
        lc_camera_config = self.python_to_lc_camera_config(camera_config)

        status = lc_camera_config.validate()
        camera_config = self.lc_camera_config_to_python(camera_config, lc_camera_config)
        if self.verbose:
            print("Requesting configuration:", camera_config)
        if status == pylibcamera.ConfigurationStatus.Invalid:
            raise RuntimeError("Invalid camera configuration: {}".format(camera_config))
        elif status == pylibcamera.ConfigurationStatus.Adjusted:
            if self.verbose:
                print("Camera configuration has been adjusted!")

        self.camera_configuration = self.lc_camera_config_to_python(camera_config, lc_camera_config)
        if self.verbose:
            print("Final configuration:", self.camera_configuration)

        self.lc_camera_config = lc_camera_config
        if self.camera.configure(lc_camera_config):
            raise RuntimeError("Configuration failed: {}".format(self.camera_configuration))
        if self.verbose:
            print("Configuration successful!")

        # Find the streams that we'll want to display or encode.
        self.preview_stream = self.find_stream(self.camera_configuration, "preview")
        self.still_stream = self.find_stream(self.camera_configuration, "still")
        self.video_stream = self.find_stream(self.camera_configuration, "video")
        self.raw_stream = self.find_stream(self.camera_configuration, "raw")
        if self.verbose:
            print("Streams: preview", self.preview_stream, "still", self.still_stream,
                  "video", self.video_stream, "raw", self.raw_stream)

        # Allocate all the frame buffers.
        self.streams = [lc_stream_config.stream for lc_stream_config in lc_camera_config]
        self.allocator = pylibcamera.FrameBufferAllocator(self.camera)
        for i, stream in enumerate(self.streams):
            if self.allocator.allocate(stream) < 0:
                raise RuntimeError("Failed to allocate buffers")
            if self.verbose:
                print("Allocated", len(self.allocator.buffers(stream)), "buffers for stream", i)

        return True if status == pylibcamera.ConfigurationStatus.Adjusted else False

    def configure(self, camera_config=None):
        """Configure the camera system with the given configuration."""
        if camera_config is None:
            camera_config = {"transform": pylibcamera.Transform(), "streams": [{"role": "preview"}]}
        return self.configure_(camera_config)

    def stream_configuration(self, index):
        """Return the stream configuration for the numbered stream."""
        return self.camera_configuration[index]

    def stream_format(self, index):
        """Return the stream format for the numbered stream."""
        stream = self.lc_camera_config.at(index)
        return {"format": stream.fmt, "frame_size": stream.frameSize, "size": stream.size,
                "stride": stream.stride}

    def list_controls(self):
        return self.camera.controls

    def start_(self, controls={}):
        if self.started:
            raise RuntimeError("Camera already started")
        self.started = True
        self.camera.start(controls)
        for request in self.make_requests():
            self.camera.queueRequest(request)
        if self.verbose:
            print("Camera started")

    def start(self, controls={}):
        """Start the camera system running."""
        self.start_(controls)

    def stop_(self, request=None):
        self.camera.stop()
        self.camera_manager.getReadyRequests()  # Could anything here need flushing?
        self.started = False
        self.stop_count += 1
        if self.verbose:
            print("Camera stopped")
        return True

    def stop(self):
        """Stop the camera system."""
        if not self.started:
            raise RuntimeError("Camera was not started")
        if self.asynchronous:
            self.dispatch_functions([self.stop_], wait=True)
        else:
            self.stop_()

    def set_controls(self, controls):
        """Set camera controls. These will be delivered with the next request that gets submitted."""
        with self.controls_lock:
            for key, value in controls.items():
                self.controls[key] = value

    def get_completed_requests(self):
        # Return all the requests that libcamera has completed.
        data = os.read(self.camera_manager.efd, 8)
        requests = [CompletedRequest(req, self) for req in self.camera_manager.getReadyRequests()
                    if req.status == pylibcamera.RequestStatus.Complete]
        self.frames += len(requests)
        return requests

    def process_requests(self):
        # This is the function that the event loop, which returns externally to us, must
        # call.
        requests = self.get_completed_requests()

        if not requests:
            return

        for request in requests[:-1]:
            request.release()
        request = requests[-1]

        # Once the event loop is running, we don't want picamera2 commands to run in any other
        # thread, so they simply queue up functions for us to call here, in the event loop.
        # Each operation is regarded as completed when it returns True, otherwise it remains
        # in the list to be tried again next time.
        if self.functions:
            i = 0
            function = self.functions[0]
            # if self.verbose:
            #     print("Execute function", function)
            if function(request):
                self.functions = self.functions[1:]
            # Once we've done everything, signal the fact to the thread that requested this work.
            if not self.functions:
                if self.async_signal_function is None:
                    self.async_operation_in_progress = False
                else:
                    self.async_signal_function(self)

        # If one of the functions we ran stopped the camera, then we don't want
        # this going back to the application.
        if request.stop_count != self.stop_count:
            request = None

        return request

    def wait(self, poll=False):
        """Wait for the event loop to finish an operation, or poll to see it it is finished.
        Returns True if the operation is complete."""
        if not self.async_operation_in_progress:
            raise RuntimeError("Waiting for non-existent operation!")
        clear = False
        if not poll:
            self.event.wait()
        clear = self.event.is_set()
        if clear:
            self.event.clear()
            self.async_operation_in_progress = False
        return not self.async_operation_in_progress

    def signal_event(self):
        self.event.set()

    def dispatch_functions(self, functions, wait, signal_function=signal_event):
        """The main thread should use this to dispatch a number of operations for the event
        loop to perform. When there are multiple items each will be processed on a separate
        trip round the event loop, meaning that a single operation could stop and restart the
        camera and the next operation would receive a request from after the restart."""
        if self.async_operation_in_progress:
            raise RuntimeError("Failure to wait for previous operation to finish!")
        self.async_error = None
        self.async_result = None
        self.async_signal_function = signal_function
        self.functions = functions
        self.async_operation_in_progress = True
        if wait:
            self.wait()
            if self.async_error:
                raise self.async_error
            return self.async_result

    def capture_(self, request, filename):
        self.async_result = request.get_metadata()
        request.save(self.still_stream, filename)
        return True

    def capture(self, filename, wait=True, signal_function=signal_event):
        """Capture in image in the current camera mode."""
        return self.dispatch_functions([(lambda r: self.capture_(r, filename))], wait, signal_function)

    def switch_mode_(self, camera_config):
        self.stop_()
        self.configure_(camera_config)
        self.start_()
        self.async_result = self.camera_configuration
        return True

    def switch_mode(self, camera_config, wait=True, signal_function=signal_event):
        """Switch the camera into another mode given by the camera_config."""
        functions = [(lambda r: self.switch_mode_(camera_config))]
        return self.dispatch_functions(functions, wait, signal_function)

    def switch_mode_and_capture(self, filename, camera_config, wait=True, signal_function=signal_event):
        """Switch the camera into a new (capture) mode, capture an image, then return back to
        the initial camera mode."""
        preview_config = self.camera_configuration

        def capture_and_switch_back_(self, request, filename, preview_config):
            self.capture_(request, filename)
            self.switch_mode_(preview_config)
            return True

        functions = [(lambda r: self.switch_mode_(camera_config)),
                     (lambda r: capture_and_switch_back_(self, r, filename, preview_config))]
        return self.dispatch_functions(functions, wait, signal_function)

    def capture_request_(self, request):
        request.acquire()
        self.async_result = request
        return True

    def capture_request(self, wait=True, signal_function=signal_event):
        """Fetch the next completed request from the camera system. You will be holding a
        reference to this request so you must release it again to return it to the camera system."""
        return self.dispatch_functions([self.capture_request_], wait, signal_function)

    def get_metadata_(self, request):
        self.async_result = request.get_metadata()
        return True

    def get_metadata(self, wait=True, signal_function=signal_event):
        """Fetch the metadata from the next camera frame."""
        return self.dispatch_functions([self.get_metadata_], wait, signal_function)

    def capture_buffer_(self, request, index):
        self.async_result = request.make_buffer(index)
        return True

    def capture_buffer(self, index=0, wait=True, signal_function=signal_event):
        """Make a numpy array from the next frame in the numbered stream."""
        return self.dispatch_functions([(lambda r: self.capture_buffer_(r, index))], wait, signal_function)

    def switch_mode_and_capture_buffer(self, camera_config, index=0, wait=True, signal_function=signal_event):
        """Switch the camera into a new (capture) mode, capture the first image buffer, then return
        back to the initial camera mode."""
        preview_config = self.camera_configuration

        def capture_buffer_and_switch_back_(self, request, preview_config, index):
            buffer = request.make_buffer(index)
            self.switch_mode_(preview_config)
            self.async_result = buffer
            return True

        functions = [(lambda r: self.switch_mode_(camera_config)),
                     (lambda r: capture_buffer_and_switch_back_(self, r, preview_config, index))]
        return self.dispatch_functions(functions, wait, signal_function)

    def capture_array_(self, request, index):
        self.async_result = request.make_array(index)
        return True

    def capture_array(self, index=0, wait=True, signal_function=signal_event):
        """Make a 2d image from the next frame in the numbered stream."""
        return self.dispatch_functions([(lambda r: self.capture_array_(r, index))], wait, signal_function)

    def switch_mode_and_capture_array(self, camera_config, index=0, wait=True, signal_function=signal_event):
        """Switch the camera into a new (capture) mode, capture the image array data, then return
        back to the initial camera mode."""
        preview_config = self.camera_configuration

        def capture_array_and_switch_back_(self, request, preview_config, index):
            array = request.make_array(index)
            self.switch_mode_(preview_config)
            self.async_result = array
            return True

        functions = [(lambda r: self.switch_mode_(camera_config)),
                     (lambda r: capture_array_and_switch_back_(self, r, preview_config, index))]
        return self.dispatch_functions(functions, wait, signal_function)

    def capture_image_(self, request, index):
        self.async_result = request.make_image(index)
        return True

    def capture_image(self, index=0, wait=True, signal_function=signal_event):
        """Make a 2d image from the next frame in the numbered stream."""
        return self.dispatch_functions([(lambda r: self.make_image_(r, index))], wait, signal_function)

    def switch_mode_and_capture_image(self, camera_config, index=0, wait=True, signal_function=signal_event):
        """Switch the camera into a new (capture) mode, capture the image, then return
        back to the initial camera mode."""
        preview_config = self.camera_configuration

        def capture_image_and_switch_back_(self, request, preview_config, index):
            image = request.make_image(index)
            self.switch_mode_(preview_config)
            self.async_result = image
            return True

        functions = [(lambda r: self.switch_mode_(camera_config)),
                     (lambda r: capture_image_and_switch_back_(self, r, preview_config, index))]
        return self.dispatch_functions(functions, wait, signal_function)


class CompletedRequest:
    def __init__(self, request, picam2):
        self.request = request
        self.ref_count = 1
        self.lock = threading.Lock()
        self.picam2 = picam2
        self.stop_count = picam2.stop_count

    def acquire(self):
        """Acquire a reference to this completed request, which stops it being recycled back to
        the camera system."""
        with self.lock:
            if self.ref_count == 0:
                raise RuntimeError("CompletedRequest: acquiring lock with ref_count 0")
            self.ref_count += 1

    def release(self):
        """Release this completed frame back to the camera system (once its reference count
        reaches zero."""
        with self.lock:
            self.ref_count -= 1
            if self.ref_count < 0:
                raise RuntimeError("CompletedRequest: lock now has negative ref_count")
            elif self.ref_count == 0:
                # If the camera has been stopped since this request was returned then we
                # can't recycle it.
                if self.stop_count == self.picam2.stop_count:
                    self.request.reuse()
                    with self.picam2.controls_lock:
                        for key, value in self.picam2.controls.items():
                            self.request.set_control(key, value)
                            self.picam2.controls = {}
                        self.request.camera.queueRequest(self.request)
                self.request = None

    def make_buffer(self, index):
        """Make a numpy array from the numbered stream's buffer in this completed request."""
        stream = self.picam2.streams[index]
        fb = self.request.buffers[stream]
        with fb.mmap(0) as b:
            return np.array(b, dtype=np.uint8)

    def get_metadata(self):
        """Fetch the metadata corresponding to this completed request."""
        return self.request.metadata

    def get_info(self, index):
        config = self.picam2.stream_format(index)
        w, h = config["size"]
        return {"width": w, "height": h, "stride": config["stride"], "format": config["format"]}

    def make_array(self, index):
        """Make a PIL image from the numbered stream's buffer in this completed request."""
        array = self.make_buffer(index)
        config = self.picam2.stream_format(index)
        fmt = config["format"]
        w, h = config["size"]
        stride = config["stride"]

        # Turning the 1d array into a 2d image-like array only works if the
        # image stride (which is in bytes) is a whole number of pixels. Even
        # then, if they don't match exactly you will get "padding" down the RHS.
        # Working around this would require another expensive copy of all the data,
        # but we can think it over whether we want to do that.
        if fmt in ("BGR888", "RGB888"):
            if stride % 3:
                raise RuntimeError("Bad width for 3 channel image")
            image = array.reshape((h, w, 3))
        elif fmt in ("XBGR8888", "XRGB8888"):
            if stride % 4:
                raise RuntimeError("Bad width for 4 channel image")
            image = array.reshape((h, w, 4))
        elif fmt[0] == 'S': # raw formats
            image = array.reshape((h, stride))
        else:
            raise RuntimeError("Format " + fmt + " not supported")
        return image

    def make_image(self, index, width=None, height=None):
        rgb = self.make_array(index)
        fmt = self.picam2.streams[index].configuration.fmt
        mode_lookup = {"RGB888": "BGR", "BGR888": "RGB", "XBGR8888": "RGBA", "XRGB8888": "BGRX"}
        mode = mode_lookup[fmt]
        pil_img = Image.frombuffer("RGB", (rgb.shape[1], rgb.shape[0]), rgb, "raw", mode, 0, 1)
        if width is None:
            width = rgb.shape[1]
        if height is None:
            height = rgb.shape[0]
        if width != rgb.shape[1] or height != rgb.shape[0]:
            # This will be slow. Consider requesting camera images of this size in the first place!
            pil_img = pil_img.resize((width, height))
        return pil_img

    def save(self, index, filename):
        """Save a JPEG or PNG image of the numbered stream's buffer in this completed request."""
        # This is probably a hideously expensive way to do a capture.
        start_time = time.time()
        img = self.make_image(index)
        if img.mode == "RGBA":
            # Nasty hack. Qt doesn't understand RGBX so we have to use RGBA. But saving a JPEG
            # doesn't like RGBA to we have to bodge that to RGBX.
            img.mode = "RGBX"
        # compress_level=1 saves pngs much faster, and still gets most of the compression.
        png_compress_level = self.picam2.options.get("compress_level", 1)
        jpeg_quality = self.picam2.options.get("quality", 90)
        img.save(filename, compress_level=png_compress_level, quality=jpeg_quality)
        if self.picam2.verbose:
            end_time = time.time()
            print("Saved", self, "to file", filename)
            print("Time taken for encode:", (end_time - start_time) * 1000, "ms")
