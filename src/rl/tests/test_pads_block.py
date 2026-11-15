#!/usr/bin/env python3
"""Layout tests for the two-player synthetic pads block (no game)."""

from __future__ import annotations

import os
import struct
import tempfile
import unittest

from rl.input.pads_block import (HEADER, MAGIC, REC, SIZE, PadsBlock, XINPUT)


class PadsBlockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "pads.bin")

    def test_create_layout_and_write(self):
        with PadsBlock(self.path) as pads:
            pads.set(0, stick=(1.0, 0.0), buttons=["A"])
            pads.set(1, stick=(-1.0, 0.5), cstick=(0.0, -1.0), trigger=1.0)
            pads.write()
        data = open(self.path, "rb").read()
        self.assertEqual(len(data), SIZE)
        self.assertEqual(data[:4], MAGIC)
        version, seq, players = struct.unpack_from("<III", data, 4)
        self.assertEqual(version, 1)
        self.assertEqual(players, 2)
        self.assertEqual(seq & 1, 0)                       # even: complete
        b0, lx0, ly0, rx0, ry0, lt0, rt0, flags0, _, _ = \
            struct.unpack_from("<IhhhhBBHII", data, HEADER)
        self.assertEqual(b0, XINPUT["A"])
        self.assertEqual((lx0, ly0), (32767, 0))
        self.assertEqual(flags0 & 1, 1)
        b1, lx1, ly1, rx1, ry1, lt1, rt1, flags1, _, _ = \
            struct.unpack_from("<IhhhhBBHII", data, HEADER + REC)
        self.assertEqual((lx1, ly1, rx1, ry1), (-32767, 16384, 0, -32767))
        self.assertEqual((lt1, rt1), (255, 0))
        self.assertEqual(flags1 & 1, 1)

    def test_disconnected_by_default(self):
        with PadsBlock(self.path) as pads:
            pads.write()
        data = open(self.path, "rb").read()
        for i in range(2):
            flags = struct.unpack_from("<H", data, HEADER + REC * i + 14)[0]
            self.assertEqual(flags & 1, 0)

    def test_button_names_and_masks(self):
        with PadsBlock(self.path) as pads:
            pads.set(0, buttons=["A", "B", "LEFT", 0x8000])
            pads.write()
        data = open(self.path, "rb").read()
        mask = struct.unpack_from("<I", data, HEADER)[0]
        self.assertEqual(mask, XINPUT["A"] | XINPUT["B"] | XINPUT["LEFT"] | 0x8000)


if __name__ == "__main__":
    unittest.main()
