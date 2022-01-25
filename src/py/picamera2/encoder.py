import io

class Encoder:
    
    def __init__(self, width, height, bitrate):
        if not (isinstance(width, int) and isinstance(height, int) and isinstance(bitrate, int)):
            raise RuntimeError("Must pass width, height, bitrate as integers")
        self._width = width
        self._height = height
        self._bitrate = bitrate
        self._output = None

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def bitrate(self):
        return self._bitrate

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        if not isinstance(value, io.BufferedIOBase):
            raise RuntimeError("Must pass BufferedIOBase")
        self._output = value
