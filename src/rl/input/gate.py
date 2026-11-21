"""Client for the runtime's frame gate (lockstep stepping).

The game, started with ``SSBU_GATE=1`` and ``SSBU_SPEED != 1`` (one simulation
step per presented frame), publishes ``/dev/shm/ssbu_gate.bin`` (48 bytes)::

    0   char4  magic "SGAT"
    4   u32    version (1)
    8   u32    enable   client: park the game thread when frame >= limit
    12  u32    epoch    runtime: game pid
    16  u64    limit    client: the game may run until this many frames
    24  u64    frame    runtime: frames presented so far
    32  u64    blocked  runtime: the frame it is parked at, all ones while running
    40  u32    flags
    44  u32    pad

While parked at frame F the fighter memory holds exactly the result of frame F,
so a reader sees a consistent, exactly-labelled snapshot.  ``advance(n)`` lets
the game run n more frames and parks it again; every decision of the agent is
therefore exactly ``frame_skip`` game frames, independent of wall-clock speed
and of how long the trainer takes between decisions.

A client that exits while the gate is enabled leaves the game parked.  ``close``
disables the gate; ``release`` does it explicitly.
"""

from __future__ import annotations

import mmap
import os
import struct
import time
from dataclasses import dataclass

GATE_PATH = "/dev/shm/ssbu_gate.bin"
MAGIC = b"SGAT"
SIZE = 48
_FMT = struct.Struct("<4sIIIQQQII")
NOT_BLOCKED = 0xFFFFFFFFFFFFFFFF
_ENABLE_OFF = 8
_LIMIT_OFF = 16
_FLAGS_OFF = 40
_SEED_OFF = 44
FLAG_ARM = 1 << 0          # park after the next reset
FLAG_SEED = 1 << 1         # `seed` is valid: re-seed the RNG inside the reset


class GateError(RuntimeError):
    pass


@dataclass(frozen=True)
class GateState:
    enable: int
    epoch: int
    limit: int
    frame: int
    blocked: int          # NOT_BLOCKED while the game runs

    @property
    def parked(self) -> bool:
        return self.blocked != NOT_BLOCKED


class FrameGate:
    def __init__(self, path: str = GATE_PATH):
        self.path = path
        self._fd: int | None = None
        self._mm: mmap.mmap | None = None

    # -- lifecycle --------------------------------------------------------
    def open(self, timeout: float = 0.0) -> "FrameGate":
        """Map the gate file; waits up to ``timeout`` s for the game to make it."""
        if self._mm is not None:
            return self
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                fd = os.open(self.path, os.O_RDWR)
                try:
                    if os.fstat(fd).st_size >= SIZE:
                        mm = mmap.mmap(fd, SIZE, mmap.MAP_SHARED,
                                       mmap.PROT_READ | mmap.PROT_WRITE)
                        if bytes(mm[0:4]) == MAGIC:
                            self._fd, self._mm = fd, mm
                            return self
                        mm.close()
                finally:
                    if self._mm is None:
                        os.close(fd)
            except FileNotFoundError:
                pass
            if time.monotonic() >= deadline:
                raise GateError(
                    f"no frame gate at {self.path}: start the game with "
                    "SSBU_GATE=1 (and SSBU_SPEED != 1)")
            time.sleep(0.05)

    def close(self) -> None:
        if self._mm is not None:
            try:
                self.release()
            except Exception:                      # noqa: BLE001
                pass
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FrameGate":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._mm is not None

    # -- state ------------------------------------------------------------
    def read(self) -> GateState:
        mm = self._need()
        _magic, _ver, enable, epoch, limit, frame, blocked, _f, _p = \
            _FMT.unpack_from(mm, 0)
        return GateState(enable, epoch, limit, frame, blocked)

    def _need(self) -> mmap.mmap:
        if self._mm is None:
            raise GateError("gate is not open")
        return self._mm

    # -- control ----------------------------------------------------------
    def _write_limit(self, limit: int) -> None:
        struct.pack_into("<Q", self._need(), _LIMIT_OFF, int(limit))

    def _write_enable(self, on: bool) -> None:
        struct.pack_into("<I", self._need(), _ENABLE_OFF, 1 if on else 0)

    def release(self) -> None:
        """Let the game run freely (gate off)."""
        self._write_enable(False)

    def arm_reset(self, park_after: int = 1, seed: int | None = None) -> None:
        """Make the *next* reset park the game at reset frame + ``park_after``.

        The runtime applies this inside the reset itself, so the parked frame
        does not depend on how fast this client is; with ``seed`` it also
        re-seeds the game's RNG there, so the first observation of every
        episode is reproducible.
        """
        k = max(1, min(255, int(park_after)))
        flags = FLAG_ARM | (k << 8)
        mm = self._need()
        if seed is not None:
            struct.pack_into("<I", mm, _SEED_OFF, int(seed) & 0xFFFFFFFF)
            flags |= FLAG_SEED
        self._write_enable(False)
        struct.pack_into("<I", mm, _FLAGS_OFF, flags)

    def hold(self, timeout: float = 10.0) -> int:
        """Engage the gate on a running game; returns the frame it parks at."""
        st = self.read()
        self._write_limit(st.frame + 1)            # a strictly future frame
        self._write_enable(True)
        return self.wait_parked(timeout)

    def advance(self, n: int) -> int:
        """Let the parked game run exactly ``n`` more frames; returns the target."""
        if n < 1:
            raise ValueError("n must be >= 1")
        st = self.read()
        if not st.enable or not st.parked:
            raise GateError("advance() needs a parked game; call hold() first")
        target = st.blocked + int(n)
        self._write_limit(target)
        return target

    def wait_parked(self, timeout: float = 10.0, at: int | None = None) -> int:
        """Wait until the game is parked (at frame ``at`` if given)."""
        deadline = time.monotonic() + timeout
        spins = 0
        while True:
            st = self.read()
            if st.enable and st.parked and (at is None or st.blocked == at):
                return st.blocked
            if time.monotonic() >= deadline:
                raise GateError(
                    f"game not parked within {timeout:.1f}s (enable={st.enable}"
                    f" limit={st.limit} frame={st.frame} "
                    f"blocked={'-' if not st.parked else st.blocked}"
                    f"{'' if at is None else f', wanted {at}'})")
            spins += 1
            if spins > 200:
                time.sleep(0.0002)
