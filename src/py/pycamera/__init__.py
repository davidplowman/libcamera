from .pycamera import *
import mmap

def __FrameBuffer__mmap(self, plane):
	return mmap.mmap(self.fd(plane), self.length(plane), mmap.MAP_SHARED, mmap.PROT_WRITE)

FrameBuffer.mmap = __FrameBuffer__mmap
