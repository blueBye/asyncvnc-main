from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import re
from typing import Dict, List, Set, Tuple

from calmjs.parse import es5
from calmjs.parse.unparsers.extractor import ast_to_dict


# Common screen aspect ratios
screen_ratios: Set[Fraction] = {
    Fraction(3, 2), 
    Fraction(4, 3), 
    Fraction(16, 10), 
    Fraction(16, 9), 
    Fraction(32, 9), 
    Fraction(64, 27),
}

class UpdateType(Enum):
    """
    Update from server to client.
    """

    #: Video update.
    VIDEO = 0

# Colour channel orders
video_modes: Dict[bytes, str] = {
     b'\x20\x18\x00\x01\x00\xff\x00\xff\x00\xff\x10\x08\x00': 'bgra',
     b'\x20\x18\x00\x01\x00\xff\x00\xff\x00\xff\x00\x08\x10': 'rgba',
     b'\x20\x18\x01\x01\x00\xff\x00\xff\x00\xff\x10\x08\x00': 'argb',
     b'\x20\x18\x01\x01\x00\xff\x00\xff\x00\xff\x00\x08\x10': 'abgr',
}

class Reader():
    def __init__(self, data: bytearray):
        self.data = data

    def readexactly(self, n):
        data = self.data[:n]
        self.data = self.data[n:]
        return data


def read_int(reader: Reader, length: int) -> int:
    """
    Reads, unpacks, and returns an integer of *length* bytes.
    """

    return int.from_bytes(reader.readexactly(length), 'big')


def read_text(reader: Reader, encoding: str) -> str:
    """
    Reads, unpacks, and returns length-prefixed text.
    """

    length = read_int(reader, 4)
    data: bytearray = reader.readexactly(length)
    return data.decode(encoding)

@dataclass
class Screen:
    """
    Computer screen.
    """

    #: Horizontal position in pixels.
    x: int

    #: Vertical position in pixels.
    y: int

    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    @property
    def slices(self) -> Tuple[slice, slice]:
        """
        Object that can be used to crop the video buffer to this screen.
        """

        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)

    @property
    def score(self) -> float:
        """
        A measure of our confidence that this represents a real screen. For screens with standard aspect ratios, this
        is proportional to its pixel area. For non-standard aspect ratios, the score is further multiplied by the ratio
        or its reciprocal, whichever is smaller.
        """

        value = float(self.width * self.height)
        ratios = {Fraction(self.width, self.height).limit_denominator(64),
                  Fraction(self.height, self.width).limit_denominator(64)}
        if not ratios & screen_ratios:
            value *= min(ratios) * 0.5
        return value







with open(file="./record.js", mode="r") as f:
    # parse js file and extract the VNC_frame_data from it
    program = es5("\n".join(f.readlines()))
    VNC_frame_data: List[str] = ast_to_dict(ast=program)['VNC_frame_data']

    # extract the server buffer frames from the VNC_frame_data 
    server_frame_data: List[str] = list(filter(lambda x: x.startswith("{"), VNC_frame_data))
    
    # seperate the timestamp from the server frame data and store it as a tuple
    timestamps = list(map(lambda x: int(re.search("^{(\d+){", x).group(1)), server_frame_data))
    frame_data = list(map(lambda x: re.sub("^{\d+{", "", x).encode(), server_frame_data))
    data = list(zip(timestamps, frame_data))
    
    # process data
    RFB_VERSION_FRAME = data[0]
    SECURITY_TYPES_FRAME = data[1]
    SECURITY_RESULT_FRAME = data[3]
    SCREEN_DETAILS_FRAME = data[4]

    print(SCREEN_DETAILS_FRAME)

    # for d in data:
    #     reader = Reader(d[1])
