# TPLLib - A Python library for decoding Nintendo image formats
# Version 0.2
# Copyright (C) 2009-2014 Tempus, RoadrunnerWMC, 2021-2026 Nin0

# This file was part of TPLLib.

# TPLLib is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# TPLLib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with TPLLib.  If not, see <http://www.gnu.org/licenses/>.



################################################################
################################################################


class Decoder():
    """
    Object that decodes a texture.

    Decoder results use the byte order expected by the original TPLLib API:
    blue, green, red, alpha (BGRA).  Wii texture data is tiled and every tile
    is stored in full, even when the image dimensions end part-way through a
    tile.
    """
    def __init__(self, tex, width, height, palette):
        """
        Initializes the decoder
        """
        self.tex = tex
        self.size = [width, height]
        self.palette = palette
        self.result = None


    def run(self):
        """
        Runs the algorithm
        """
        raise NotImplementedError('You cannot run an abstract decoder')


# Texture layout and pixel conversion are based on Dolphin's reference GX
# decoder and the format documentation maintained by the Mario Kart Wii wiki:
# https://github.com/dolphin-emu/dolphin/blob/master/Source/Core/VideoCommon/TextureDecoder_Generic.cpp
# https://mkwiiki.org/wiki/Image_Formats


def _prepare_decode(tex, width, height, block_width, block_height,
                    bytes_per_block):
    """Validate a tiled source and allocate an exact-size BGRA result."""
    if not isinstance(width, int) or not isinstance(height, int):
        raise TypeError('Width and height must be ints')
    if width < 0 or height < 0:
        raise ValueError('Width and height cannot be negative')

    blocks_wide = (width + block_width - 1) // block_width
    blocks_high = (height + block_height - 1) // block_height
    required = blocks_wide * blocks_high * bytes_per_block
    if len(tex) < required:
        raise ValueError(
            'Texture data is too short: expected at least {} bytes, got {}'.format(
                required, len(tex)))

    return bytearray(width * height * 4)


def _write_pixel(output, width, height, x, y, pixel):
    """Write a BGRA pixel, discarding the padded part of an edge tile."""
    if x >= width or y >= height:
        return
    offset = (y * width + x) * 4
    output[offset:offset + 4] = bytes(pixel)


def _decode_tiled(tex, width, height, block_width, block_height,
                  bytes_per_pixel, pixel_decoder):
    """Decode an uncompressed format whose pixels are contiguous in a tile."""
    bytes_per_block = block_width * block_height * bytes_per_pixel
    output = _prepare_decode(tex, width, height, block_width, block_height,
                             bytes_per_block)
    block_offset = 0

    for tile_y in range(0, height, block_height):
        for tile_x in range(0, width, block_width):
            for pixel_y in range(block_height):
                for pixel_x in range(block_width):
                    pixel_number = pixel_y * block_width + pixel_x
                    source_offset = block_offset + pixel_number * bytes_per_pixel
                    pixel = pixel_decoder(tex, source_offset)
                    _write_pixel(output, width, height,
                                 tile_x + pixel_x, tile_y + pixel_y, pixel)
            block_offset += bytes_per_block

    return bytes(output)


def _decode_nibbles(tex, width, height, pixel_decoder):
    """Decode an 8x8 tiled format containing two 4-bit pixels per byte."""
    output = _prepare_decode(tex, width, height, 8, 8, 32)
    block_offset = 0

    for tile_y in range(0, height, 8):
        for tile_x in range(0, width, 8):
            for pixel_y in range(8):
                for pair_x in range(4):
                    value = tex[block_offset + pixel_y * 4 + pair_x]
                    x = tile_x + pair_x * 2
                    _write_pixel(output, width, height, x, tile_y + pixel_y,
                                 pixel_decoder(value >> 4))
                    _write_pixel(output, width, height, x + 1, tile_y + pixel_y,
                                 pixel_decoder(value & 0x0F))
            block_offset += 32

    return bytes(output)


def _palette_pixel(palette, index):
    """Return one BGRA palette entry with a useful error for bad indexes."""
    if palette is None:
        raise TypeError('Palette for decoding seems to be missing!')
    try:
        pixel = palette[index]
    except IndexError as error:
        raise ValueError('Palette has no entry for index {}'.format(index)) from error
    try:
        if len(pixel) != 4:
            raise ValueError
        return bytes(pixel)
    except (TypeError, ValueError) as error:
        raise ValueError('Palette entries must contain four BGRA bytes') from error


def _expand3(value):
    return (value << 5) | (value << 2) | (value >> 1)


def _expand5(value):
    return (value << 3) | (value >> 2)


def _expand6(value):
    return (value << 2) | (value >> 4)


def _rgb565(value):
    """Convert a GX RGB565 word to BGRA."""
    red = _expand5((value >> 11) & 0x1F)
    green = _expand6((value >> 5) & 0x3F)
    blue = _expand5(value & 0x1F)
    return blue, green, red, 0xFF


def _rgb5a3(value):
    """Convert a GX RGB5A3 word to BGRA."""
    if value & 0x8000:
        red = _expand5((value >> 10) & 0x1F)
        green = _expand5((value >> 5) & 0x1F)
        blue = _expand5(value & 0x1F)
        return blue, green, red, 0xFF

    alpha = _expand3((value >> 12) & 0x07)
    red = ((value >> 8) & 0x0F) * 0x11
    green = ((value >> 4) & 0x0F) * 0x11
    blue = (value & 0x0F) * 0x11
    return blue, green, red, alpha


def _cmpr_blend(first, second):
    """GX's CMPR interpolation: 3/8 of first plus 5/8 of second."""
    return (first * 3 + second * 5) >> 3


class I4Decoder(Decoder):
    """
    Decodes an I4 texture
    """
    # Format:
    # IIII
    bytesPerPixel = .5

    def run(self):
        """
        Runs the algorithm
        """
        def decode(nibble):
            intensity = nibble * 0x11
            return intensity, intensity, intensity, 0xFF

        self.result = _decode_nibbles(
            self.tex, self.size[0], self.size[1], decode)
        return self.result


class I8Decoder(Decoder):
    """
    Decodes an I8 texture
    """
    # Format:
    # IIIIIIII
    bytesPerPixel = 1

    def run(self):
        """
        Runs the algorithm
        """
        def decode(tex, offset):
            intensity = tex[offset]
            return intensity, intensity, intensity, 0xFF

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 8, 4, 1, decode)
        return self.result


class IA4Decoder(Decoder):
    """
    Decodes an IA4 texture
    """
    # Format:
    # AAAAIIII
    bytesPerPixel = 1

    def run(self):
        """
        Runs the algorithm
        """
        def decode(tex, offset):
            value = tex[offset]
            alpha = (value >> 4) * 0x11
            intensity = (value & 0x0F) * 0x11
            return intensity, intensity, intensity, alpha

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 8, 4, 1, decode)
        return self.result


class IA8Decoder(Decoder):
    """
    Decodes an IA8 texture
    """
    # Format:
    # IIIIIIII AAAAAAAA
    bytesPerPixel = 2

    def run(self):
        """
        Runs the algorithm
        """
        def decode(tex, offset):
            alpha = tex[offset]
            intensity = tex[offset + 1]
            return intensity, intensity, intensity, alpha

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 4, 4, 2, decode)
        return self.result


class RGB565Decoder(Decoder):
    """
    Decodes an RGB565 texture
    """
    # Format:
    # RRRRRGGG GGGBBBBB
    bytesPerPixel = 2

    def run(self):
        """
        Runs the algorithm
        """
        def decode(tex, offset):
            value = (tex[offset] << 8) | tex[offset + 1]
            return _rgb565(value)

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 4, 4, 2, decode)
        return self.result



class RGB5A3Decoder(Decoder):
    """
    Decodes an RGB5A3 texture
    """
    # Formats:
    # 1RRRRRGG GGGBBBBB
    # 0AAARRRR GGGGBBBB
    bytesPerPixel = 2

    def run(self):
        """
        Runs the algorithm
        """
        def decode(tex, offset):
            value = (tex[offset] << 8) | tex[offset + 1]
            return _rgb5a3(value)

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 4, 4, 2, decode)
        return self.result


class RGBA8Decoder(Decoder):
    """
    Decodes an RGBA8 texture
    """
    # Format:
    # RRRRRRRR GGGGGGGG BBBBBBBB AAAAAAAA
    bytesPerPixel = 4

    def run(self):
        """
        Runs the algorithm
        """
        tex, w, h = self.tex, self.size[0], self.size[1]
        output = _prepare_decode(tex, w, h, 4, 4, 64)
        block_offset = 0

        for tile_y in range(0, h, 4):
            for tile_x in range(0, w, 4):
                for pixel_number in range(16):
                    ar_offset = block_offset + pixel_number * 2
                    gb_offset = block_offset + 32 + pixel_number * 2
                    alpha = tex[ar_offset]
                    red = tex[ar_offset + 1]
                    green = tex[gb_offset]
                    blue = tex[gb_offset + 1]
                    _write_pixel(output, w, h,
                                 tile_x + pixel_number % 4,
                                 tile_y + pixel_number // 4,
                                 (blue, green, red, alpha))
                block_offset += 64

        self.result = bytes(output)
        return self.result



class CI4Decoder(Decoder):
    """
    Decodes a CI4 texture
    """
    def run(self):
        """
        Runs the algorithm
        """
        if self.palette is None:
            raise TypeError("Palette for decoding seems to be missing!")

        self.result = _decode_nibbles(
            self.tex, self.size[0], self.size[1],
            lambda index: _palette_pixel(self.palette, index))
        return self.result


class CI8Decoder(Decoder):
    """
    Decodes a CI8 texture
    """
    def run(self):
        """
        Runs the algorithm
        """
        if self.palette is None:
            raise TypeError("Palette for decoding seems to be missing!")

        def decode(tex, offset):
            return _palette_pixel(self.palette, tex[offset])

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 8, 4, 1, decode)
        return self.result


class CI14X2Decoder(Decoder):
    """
    Decodes a CI14X2 texture.

    Each big-endian 16-bit value contains a 14-bit palette index; the top two
    bits are padding and must be ignored.
    """
    bytesPerPixel = 2

    def run(self):
        if self.palette is None:
            raise TypeError("Palette for decoding seems to be missing!")

        def decode(tex, offset):
            index = ((tex[offset] << 8) | tex[offset + 1]) & 0x3FFF
            return _palette_pixel(self.palette, index)

        self.result = _decode_tiled(
            self.tex, self.size[0], self.size[1], 4, 4, 2, decode)
        return self.result

class CMPRDecoder(Decoder):
    """
    Decodes a Wii CMPR texture (GX's tiled variant of BC1/DXT1).
    """
    
    bytesPerPixel = 0.5 #, or 4 bits
    blockWidth   = 8
    blockHeight  = 8
    
    def run(self):
        """
        Runs the algorithm
        """
        tex, w, h = self.tex, self.size[0], self.size[1]
        output = _prepare_decode(tex, w, h, 8, 8, 32)
        block_offset = 0

        for tile_y in range(0, h, 8):
            for tile_x in range(0, w, 8):
                # Sub-block order is top-left, top-right, bottom-left,
                # bottom-right. Each sub-block is one 8-byte BC1 block.
                for subblock in range(4):
                    source = block_offset + subblock * 8
                    color_0 = (tex[source] << 8) | tex[source + 1]
                    color_1 = (tex[source + 2] << 8) | tex[source + 3]
                    endpoint_0 = _rgb565(color_0)
                    endpoint_1 = _rgb565(color_1)

                    if color_0 > color_1:
                        color_2 = tuple(
                            _cmpr_blend(endpoint_1[channel], endpoint_0[channel])
                            for channel in range(3)) + (0xFF,)
                        color_3 = tuple(
                            _cmpr_blend(endpoint_0[channel], endpoint_1[channel])
                            for channel in range(3)) + (0xFF,)
                    else:
                        average = tuple(
                            (endpoint_0[channel] + endpoint_1[channel]) // 2
                            for channel in range(3))
                        color_2 = average + (0xFF,)
                        # The GX decoder retains the average RGB for the
                        # transparent entry, unlike ordinary PC DXT1.
                        color_3 = average + (0x00,)

                    palette = (endpoint_0, endpoint_1, color_2, color_3)
                    subblock_x = tile_x + (subblock & 1) * 4
                    subblock_y = tile_y + (subblock >> 1) * 4
                    for row in range(4):
                        selectors = tex[source + 4 + row]
                        for column in range(4):
                            index = (selectors >> (6 - column * 2)) & 0x03
                            _write_pixel(output, w, h,
                                         subblock_x + column,
                                         subblock_y + row,
                                         palette[index])

                block_offset += 32

        self.result = bytes(output)
        return self.result






# Enums
I4 = 0
I8 = 1
IA4 = 2
IA8 = 3
RGB565 = 4
RGB5A3 = 5
RGBA8 = 6
CI4 = 8
CI8 = 9
CI14X2 = 10
CMPR = 14



def getDecoder(type):
    """
    Returns the appropriate decoding algorithm based on the type specified
    """
    if not isinstance(type, int):
        raise TypeError('Type is not an int')

    if type == I4:
        return I4Decoder
    elif type == I8:
        return I8Decoder
    elif type == IA4:
        return IA4Decoder
    elif type == IA8:
        return IA8Decoder
    elif type == RGB565:
        return RGB565Decoder
    elif type == RGB5A3:
        return RGB5A3Decoder
    elif type == RGBA8:
        return RGBA8Decoder
    elif type == CI4:
        return CI4Decoder
    elif type == CI8:
        return CI8Decoder
    elif type == CI14X2:
        return CI14X2Decoder
    elif type == CMPR:
        return CMPRDecoder
    else:
        raise ValueError('Unrecognized type')


