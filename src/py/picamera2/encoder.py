class Encoder:
    
    def __init__(self, width, height, bitrate):
        if not (isinstance(width, int) and isinstance(height, int) and isinstance(bitrate, int)):
            raise RuntimeError("Must pass width, height, bitrate as integers")
        self._width = width
        self._height = height
        self._bitrate = bitrate

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def bitrate(self):
        return self._bitrate
