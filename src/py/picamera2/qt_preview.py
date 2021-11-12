import picamera2
import threading
import atexit

class QtPreview:
    def thread_func(self, picam2):
        # Running Qt in a thread other than the main thread is a bit tricky...
        from q_picamera2 import QApplication, QPicamera2

        self.app = QApplication([])
        self.qpicamera2 = QPicamera2(picam2)
        self.qpicamera2.setWindowTitle("QtPreview")
        self.qpicamera2.show()
        picam2.asynchronous = True
        atexit.register(self.stop)
        self.event.set()

        self.app.exec()

        atexit.unregister(self.stop)
        self.qpicamera2.picamera2.asynchronous = False
        del self.qpicamera2.label
        del self.qpicamera2.camera_notifier
        del self.qpicamera2
        del self.app

    def __init__(self, picam2):
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.thread_func, args=(picam2,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.event.wait()

    def stop(self):
        self.app.quit()
        self.thread.join()
