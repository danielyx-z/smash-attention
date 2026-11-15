#!/usr/bin/env python3
"""Two-player synthetic XInput pads for self-play (no uinput, no SDL).

The recomp reads ``/dev/shm/ssbu_pads.bin`` once per game frame when it is
started with ``SSBU_PAD_FROM_CTL=1``: two XInput pads on slots 0 and 1, i.e.
players 1 and 2.  This module writes that block.  It is the reliable,
low-latency path for self-play: no kernel device, no hot-plug, no controller
assignment, and both players come from one consistent snapshot.

Layout (little-endian)::

    0   char4  magic "SSBP"
    4   u32    version (1)
    8   u32    seq      (odd while the writer updates the records)
    12  u32    players  (2)
    16  ...    reserved
    24  Pad[2], 24 bytes each:
        0   u32   buttons (XInput mask; see XINPUT)
        4   s16   lx
        6   s16   ly
        8   s16   rx
        10  s16   ry
        12  u8    lt
        13  u8    rt
        14  u16   flags (bit 0 = connected)
        16  ...   reserved

Usage::

    with PadsBlock() as pads:
        pads.set(0, stick=(1.0, 0.0))
        pads.set(1, buttons=["A"], stick=(-1.0, 0.0))
        pads.write()

Start the game with ``SSBU_PAD_FROM_CTL=1`` and a human/human match.  A
scripted menu walk (``SSBU_SEQ``) still drives player 1 while both pads are
disconnected; after the fight starts, connect both and write the actions.
"""

from __future__ import annotations

import mmap
import os
import struct

MAGIC = b"SSBP"
VERSION = 1
HEADER = 24
REC = 24
PLAYERS = 2
SIZE = HEADER + REC * PLAYERS
DEFAULT_PATH = "/dev/shm/ssbu_pads.bin"

#: XInput button masks, as the recomp's synthetic pad expects them (the same
#: mapping the SSBU_XINPUT_SYNTH sequence uses: console A is XInput B).
XINPUT = {
    "A": 0x2000, "B": 0x1000, "X": 0x8000, "Y": 0x4000,
    "L": 0x0100, "R": 0x0200, "ZL": 0x0000, "ZR": 0x0000,
    "PLUS": 0x0010, "MINUS": 0x0020,
    "UP": 0x0001, "DOWN": 0x0002, "LEFT": 0x0004, "RIGHT": 0x0008,
}

_U32 = struct.Struct("<I")
_RECORD = struct.Struct("<IhhhhBBHII")


def _i16(value: float) -> int:
    return max(-32768, min(32767, int(round(float(value) * 32767))))


def _u8(value: float) -> int:
    return max(0, min(255, int(round(float(value) * 255))))


def _mask(buttons) -> int:
    if isinstance(buttons, int):
        return buttons & 0xFFFF
    mask = 0
    for name in buttons or ():
        if isinstance(name, int):
            mask |= name & 0xFFFF
        else:
            mask |= XINPUT.get(str(name).upper(), 0)
    return mask


class PadsBlock:
    """One writer for the two synthetic XInput pads."""

    def __init__(self, path: str = DEFAULT_PATH, players: int = PLAYERS):
        self.path = str(path)
        self.players = max(1, min(PLAYERS, int(players)))
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self._seq = 0
        self._state = [
            {"buttons": 0, "stick": (0.0, 0.0), "cstick": (0.0, 0.0),
             "trigger": 0.0, "connected": False}
            for _ in range(self.players)
        ]

    # -- lifecycle -----------------------------------------------------
    def open(self) -> "PadsBlock":
        if self.fd is not None:
            return self
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        self.fd = fd
        if os.fstat(fd).st_size < SIZE:
            os.ftruncate(fd, SIZE)
            header = MAGIC + _U32.pack(VERSION) + _U32.pack(0) + \
                _U32.pack(PLAYERS) + b"\0" * 8
            os.pwrite(fd, header, 0)
            os.pwrite(fd, b"\0" * (REC * PLAYERS), HEADER)
        self.mm = mmap.mmap(fd, SIZE, access=mmap.ACCESS_WRITE)
        if self.mm[:4] != MAGIC or _U32.unpack_from(self.mm, 4)[0] < VERSION:
            raise ValueError(f"{self.path}: not an SSBP pads block")
        self._seq = _U32.unpack_from(self.mm, 8)[0]
        return self

    def close(self) -> None:
        if self.mm is not None:
            try:
                self.mm.close()
            except (BufferError, ValueError):
                pass
            self.mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None

    def __enter__(self) -> "PadsBlock":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writer --------------------------------------------------------
    def set(self, player: int, buttons=0, stick=(0.0, 0.0),
            cstick=(0.0, 0.0), trigger: float = 0.0,
            connected: bool = True) -> None:
        if not 0 <= int(player) < self.players:
            raise ValueError(f"player must be 0..{self.players - 1}")
        self._state[int(player)] = {
            "buttons": _mask(buttons), "stick": tuple(stick),
            "cstick": tuple(cstick), "trigger": float(trigger),
            "connected": bool(connected),
        }

    def neutral(self, player: int, connected: bool = True) -> None:
        self.set(player, connected=connected)

    def write(self) -> None:
        """Publish both records atomically (odd/even seq)."""
        if self.mm is None:
            raise RuntimeError("PadsBlock is not open")
        body = bytearray(REC * self.players)
        for player, state in enumerate(self._state):
            lx, ly = state["stick"]
            rx, ry = state["cstick"]
            _RECORD.pack_into(
                body, player * REC, state["buttons"], _i16(lx), _i16(ly),
                _i16(rx), _i16(ry), _u8(state["trigger"]), 0,
                1 if state["connected"] else 0, 0, 0)
        self._seq = (self._seq + 1) & 0xFFFFFFFF          # odd: writing
        _U32.pack_into(self.mm, 8, self._seq)
        self.mm[HEADER:HEADER + len(body)] = body
        self._seq = (self._seq + 1) & 0xFFFFFFFF          # even: stable
        _U32.pack_into(self.mm, 8, self._seq)
