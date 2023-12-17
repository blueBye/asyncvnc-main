from asyncio import StreamReader, StreamWriter
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
from itertools import product
from os import urandom
from typing import Callable, Dict, List, Optional, Set, Tuple
from zlib import decompressobj

import numpy as np


# Common screen aspect ratios
screen_ratios: Set[Fraction] = {
    Fraction(3, 2), Fraction(4, 3), Fraction(16, 10), Fraction(16, 9), Fraction(32, 9), Fraction(64, 27)}

# Colour channel orders
video_modes: Dict[bytes, str] = {
     b'\x20\x18\x00\x01\x00\xff\x00\xff\x00\xff\x10\x08\x00': 'bgra',
     b'\x20\x18\x00\x01\x00\xff\x00\xff\x00\xff\x00\x08\x10': 'rgba',
     b'\x20\x18\x01\x01\x00\xff\x00\xff\x00\xff\x10\x08\x00': 'argb',
     b'\x20\x18\x01\x01\x00\xff\x00\xff\x00\xff\x00\x08\x10': 'abgr',
}


async def read_int(reader: StreamReader, length: int) -> int:
    """
    Reads, unpacks, and returns an integer of *length* bytes.
    """

    return int.from_bytes(await reader.readexactly(length), 'big')


async def read_text(reader: StreamReader, encoding: str) -> str:
    """
    Reads, unpacks, and returns length-prefixed text.
    """

    length = await read_int(reader, 4)
    data = await reader.readexactly(length)
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


@dataclass
class Video:
    """
    Video buffer.
    """

    reader: StreamReader = field(repr=False)
    writer: StreamWriter = field(repr=False)
    decompress: Callable[[bytes], bytes] = field(repr=False)

    #: Desktop name.
    name: str

    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    #: Colour channel order.
    mode: str

    #: 3D numpy array of colour data.
    data: Optional[np.ndarray] = None

    @classmethod
    async def create(cls, reader: StreamReader, writer: StreamWriter) -> 'Video':
        width = await read_int(reader, 2)
        height = await read_int(reader, 2)
        mode_data = bytearray(await reader.readexactly(13))
        mode_data[2] &= 1  # set big endian flag to 0 or 1
        mode_data[3] &= 1  # set true colour flag to 0 or 1
        mode = video_modes.get(bytes(mode_data))
        await reader.readexactly(3)  # padding
        name = await read_text(reader, 'utf-8')

        if mode is None:
            mode = 'rgba'
            writer.write(b'\x00\x00\x00\x00\x20\x18\x00\x01\x00\xff'
                         b'\x00\xff\x00\xff\x00\x08\x10\x00\x00\x00')
        writer.write(b'\x02\x00\x00\x01\x00\x00\x00\x06')
        decompress = decompressobj().decompress
        return cls(reader, writer, decompress, name, width, height, mode)

    def refresh(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        """
        Sends a video buffer update request to the server.
        """

        incremental = self.data is not None
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        self.writer.write(
            b'\x03' +
            incremental.to_bytes(1, 'big') +
            x.to_bytes(2, 'big') +
            y.to_bytes(2, 'big') +
            width.to_bytes(2, 'big') +
            height.to_bytes(2, 'big'))

    async def read(self):
        x = await read_int(self.reader, 2)
        y = await read_int(self.reader, 2)
        width = await read_int(self.reader, 2)
        height = await read_int(self.reader, 2)
        encoding = await read_int(self.reader, 4)

        if encoding == 0:  # Raw
            data = await self.reader.readexactly(height * width * 4)
        elif encoding == 6:  # ZLib
            length = await read_int(self.reader, 4)
            data = await self.reader.readexactly(length)
            data = self.decompress(data)
        else:
            raise ValueError(encoding)

        if self.data is None:
            self.data = np.zeros((self.height, self.width, 4), 'B')
        self.data[y:y + height, x:x + width] = np.ndarray((height, width, 4), 'B', data)
        self.data[y:y + height, x:x + width, self.mode.index('a')] = 255

    def as_rgba(self) -> np.ndarray:
        """
        Returns the video buffer as a 3D RGBA array.
        """

        if self.data is None:
            return np.zeros((self.height, self.width, 4), 'B')
        if self.mode == 'rgba':
            return self.data
        if self.mode == 'abgr':
            return self.data[:, :, ::-1]
        return np.dstack((
            self.data[:, :, self.mode.index('r')],
            self.data[:, :, self.mode.index('g')],
            self.data[:, :, self.mode.index('b')],
            self.data[:, :, self.mode.index('a')]))

    def is_complete(self):
        """
        Returns true if the video buffer is entirely opaque.
        """

        if self.data is None:
            return False
        return self.data[:, :, self.mode.index('a')].all()

    def detect_screens(self) -> List[Screen]:
        """
        Detect physical screens by inspecting the alpha channel.
        """

        if self.data is None:
            return []

        mask = self.data[:, :, self.mode.index('a')]
        mask = np.pad(mask // 255, ((1, 1), (1, 1))).astype(np.int8)
        mask_a = mask[1:, 1:]
        mask_b = mask[1:, :-1]
        mask_c = mask[:-1, 1:]
        mask_d = mask[:-1, :-1]

        screens = []
        while True:
            # Detect corners by ANDing perpendicular pairs of differences.
            corners = product(
                np.argwhere(mask_b - mask_a & mask_c - mask_a == -1),  # top left
                np.argwhere(mask_a - mask_b & mask_d - mask_b == -1),  # top right
                np.argwhere(mask_d - mask_c & mask_a - mask_c == -1),  # bottom left
                np.argwhere(mask_c - mask_d & mask_b - mask_d == -1))  # bottom right

            # Find cases where 3 corners align, forming an  'L' shape.
            rects = set()
            for a, b, c, d in corners:
                ab = a[0] == b[0] and a[1] < b[1]  # top
                cd = c[0] == d[0] and c[1] < d[1]  # bottom
                ac = a[1] == c[1] and a[0] < c[0]  # left
                bd = b[1] == d[1] and b[0] < d[0]  # right
                if ab and ac:
                    rects.add((a[1], a[0], b[1], c[0]))
                if ab and bd:
                    rects.add((a[1], a[0], d[1], d[0]))
                if cd and ac:
                    rects.add((a[1], a[0], d[1], d[0]))
                if cd and bd:
                    rects.add((c[1], b[0], d[1], d[0]))

            # Create screen objects and sort them by their scores.
            candidates = [Screen(int(x0), int(y0), int(x1 - x0), int(y1 - y0)) for x0, y0, x1, y1 in rects]
            candidates.sort(key=lambda screen: screen.score, reverse=True)

            # Find a single fully-opaque screen
            for screen in candidates:
                if mask_a[screen.slices].all():
                    mask_a[screen.slices] = 0
                    screens.append(screen)
                    break

            # Finish up if no screens remain
            else:
                return screens


class UpdateType(Enum):
    """
    Update from server to client.
    """

    #: Video update.
    VIDEO = 0


@dataclass
class Client:
    """
    VNC client.
    """

    reader: StreamReader = field(repr=False)

    #: The video buffer.
    video: Video

    async def read(self) -> UpdateType:
        """
        Reads an update from the server and returns its type.
        """

        update_type = UpdateType(await read_int(self.reader, 1))

        if update_type is UpdateType.VIDEO:
            await self.reader.readexactly(1)  # padding
            for _ in range(await read_int(self.reader, 2)):
                await self.video.read()

        return update_type

    async def screenshot(self, x: int = 0, y: int = 0, width: Optional[int] = None, height: Optional[int] = None):
        """
        Takes a screenshot and returns a 3D RGBA array.
        """

        self.video.data = None
        self.video.refresh(x, y, width, height)
        while True:
            update_type = await self.read()
            if update_type is UpdateType.VIDEO:
                if self.video.is_complete():
                    return self.video.as_rgba()
