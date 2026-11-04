#!/usr/bin/env python3
"""Reset control block shared with the runtime: /dev/shm/ssbu_control.bin.

Layout (little-endian, 24 bytes, exactly CONTRACTS.md):

    offset  type   field
    0       char4  magic       "SSBC"
    4       u32    version     1
    8       u32    reset_seq   the client increments it to request a reset
    12      u32    ack_seq     the runtime copies reset_seq here when done
    16      u32    state       0 idle, 1 requested, 2 running, 3 done, 4 error
    20      u32    pad         zero, keeps the block 8-byte friendly

Protocol (single client, single runtime):

    client: read, reset_seq += 1, state = REQUESTED, whole 24 bytes written
    runtime: sees reset_seq != ack_seq, sets state = RUNNING, resets the match,
             sets ack_seq = reset_seq and state = DONE (or ERROR)
    client: waits for ack_seq == reset_seq, then for state == DONE, then for
             fight_on == 1 in the state block for a few stable frames
             (gym_reset.ResetClient does the last part).

The file is a plain tmpfs file; the 24-byte struct is written with a single
pread/pwrite.  Readers in the runtime poll it.  Create it with mode 0666 so a
game started from another account/session can still see it.

This module has no dependencies outside the standard library.
"""

from __future__ import annotations

import os
import struct
import time
from dataclasses import dataclass

MAGIC = b"SSBC"
VERSION = 1
SHM_PATH = "/dev/shm/ssbu_control.bin"

STATE_IDLE = 0
STATE_REQUESTED = 1
STATE_RUNNING = 2
STATE_DONE = 3
STATE_ERROR = 4

STATE_NAMES = {
    STATE_IDLE: "idle",
    STATE_REQUESTED: "requested",
    STATE_RUNNING: "running",
    STATE_DONE: "done",
    STATE_ERROR: "error",
}

# magic, version, reset_seq, ack_seq, state, pad
_STRUCT = struct.Struct("<4sIIIII")
SIZE = _STRUCT.size

RESET_SEQ_OFFSET = 8
ACK_SEQ_OFFSET = 12
STATE_OFFSET = 16
PAD_OFFSET = 20

assert SIZE == 24


class ControlBlockError(RuntimeError):
    """The control block is missing, malformed, or reported an error."""


class ControlBlockTimeout(TimeoutError):
    """The runtime did not acknowledge a reset in time."""


@dataclass(frozen=True)
class ControlBlockSnapshot:
    """One consistent read of the control block."""

    magic: bytes
    version: int
    reset_seq: int
    ack_seq: int
    state: int
    pad: int

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, f"unknown({self.state})")

    @property
    def pending(self) -> bool:
        """True while the runtime has not acknowledged the latest request."""
        return self.ack_seq != self.reset_seq

    @property
    def done(self) -> bool:
        return self.state == STATE_DONE and not self.pending

    @property
    def failed(self) -> bool:
        return self.state == STATE_ERROR


def pack(reset_seq: int = 0, ack_seq: int = 0, state: int = STATE_IDLE,
         pad: int = 0, version: int = VERSION, magic: bytes = MAGIC) -> bytes:
    """Serialize a control block.  Exposed for unit tests and the runtime mock."""
    return _STRUCT.pack(magic, version, reset_seq & 0xFFFFFFFF,
                        ack_seq & 0xFFFFFFFF, state & 0xFFFFFFFF,
                        pad & 0xFFFFFFFF)


def unpack(data: bytes) -> ControlBlockSnapshot:
    if len(data) < SIZE:
        raise ControlBlockError(f"short control block: {len(data)} < {SIZE} bytes")
    magic, version, reset_seq, ack_seq, state, pad = _STRUCT.unpack_from(data)
    if magic != MAGIC:
        raise ControlBlockError(f"bad magic {magic!r}; expected {MAGIC!r}")
    if version != VERSION:
        raise ControlBlockError(f"unsupported control block version {version}; expected {VERSION}")
    return ControlBlockSnapshot(magic, version, reset_seq, ack_seq, state, pad)


class ControlBlock:
    """Owner-side view of the control block (the environment / gym)."""

    def __init__(self, path: str = SHM_PATH, create: bool = True,
                 reinit: bool = False, timeout: float = 5.0,
                 poll: float = 0.002):
        self.path = path
        self.create = create
        self.reinit = reinit
        self.timeout = timeout
        self.poll = poll
        self.fd: int | None = None

    # -- lifecycle -----------------------------------------------------
    def open(self) -> "ControlBlock":
        if self.fd is not None:
            return self
        flags = os.O_RDWR
        if self.create:
            flags |= os.O_CREAT
        try:
            fd = os.open(self.path, flags, 0o666)
        except FileNotFoundError:
            raise ControlBlockError(
                f"{self.path} does not exist (start the runtime first)") from None
        try:
            try:
                os.fchmod(fd, 0o666)
            except OSError:
                pass
            if os.fstat(fd).st_size < SIZE:
                os.ftruncate(fd, SIZE)
            if self.reinit:
                self._init_fd(fd)
            else:
                data = os.pread(fd, SIZE, 0)
                if len(data) < SIZE or data[:4] != MAGIC:
                    if not self.create:
                        raise ControlBlockError(
                            f"{self.path} is not an SSBC control block")
                    self._init_fd(fd)
                else:
                    snap = unpack(data)
                    if snap.version != VERSION and not self.create:
                        raise ControlBlockError(
                            f"unsupported version {snap.version}")
        except Exception:
            os.close(fd)
            raise
        self.fd = fd
        return self

    @staticmethod
    def _init_fd(fd: int) -> None:
        os.pwrite(fd, pack(), 0)

    def close(self) -> None:
        if self.fd is not None:
            try:
                os.close(self.fd)
            finally:
                self.fd = None

    def __enter__(self) -> "ControlBlock":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self.fd is None

    def _require_fd(self) -> int:
        if self.fd is None:
            raise ControlBlockError("control block is not open")
        return self.fd

    # -- reads / writes ------------------------------------------------
    def read(self) -> ControlBlockSnapshot:
        fd = self._require_fd()
        data = os.pread(fd, SIZE, 0)
        return unpack(data)

    snapshot = read

    def write(self, reset_seq: int, ack_seq: int, state: int, pad: int = 0) -> ControlBlockSnapshot:
        fd = self._require_fd()
        data = pack(reset_seq, ack_seq, state, pad)
        written = os.pwrite(fd, data, 0)
        if written != SIZE:
            raise ControlBlockError(f"short write: {written} != {SIZE}")
        return unpack(data)

    def set_state(self, state: int) -> ControlBlockSnapshot:
        snap = self.read()
        return self.write(snap.reset_seq, snap.ack_seq, state, snap.pad)

    # -- request / acknowledge ----------------------------------------
    def request_reset(self) -> int:
        """Increment reset_seq and mark the request.  Returns the new sequence."""
        snap = self.read()
        seq = (snap.reset_seq + 1) & 0xFFFFFFFF
        self.write(seq, snap.ack_seq, STATE_REQUESTED, snap.pad)
        return seq

    # The runtime side of the protocol, useful for the mock and for tests.
    def acknowledge(self, seq: int | None = None,
                    state: int = STATE_DONE) -> ControlBlockSnapshot:
        snap = self.read()
        if seq is None:
            seq = snap.reset_seq
        return self.write(snap.reset_seq, seq, state, snap.pad)

    ack = acknowledge

    def wait_for_ack(self, seq: int, timeout: float | None = None,
                     poll: float | None = None) -> ControlBlockSnapshot:
        """Wait until ack_seq == seq; raises on timeout or state ERROR."""
        timeout = self.timeout if timeout is None else timeout
        poll = self.poll if poll is None else poll
        deadline = time.monotonic() + timeout
        while True:
            snap = self.read()
            if snap.failed:
                raise ControlBlockError(
                    f"runtime reported an error while resetting (seq {seq})")
            if snap.ack_seq == seq:
                return snap
            if snap.ack_seq > seq:
                # A newer request already got acked; the caller lost the race.
                return snap
            if time.monotonic() >= deadline:
                raise ControlBlockTimeout(
                    f"reset seq {seq} not acknowledged within {timeout:.2f}s "
                    f"(ack {snap.ack_seq}, state {snap.state_name})")
            time.sleep(poll)

    def wait_for_done(self, seq: int, timeout: float | None = None,
                      poll: float | None = None) -> ControlBlockSnapshot:
        """Wait for the ack and then for state DONE."""
        snap = self.wait_for_ack(seq, timeout, poll)
        timeout = self.timeout if timeout is None else timeout
        poll = self.poll if poll is None else poll
        deadline = time.monotonic() + timeout
        while snap.state in (STATE_REQUESTED, STATE_RUNNING):
            if time.monotonic() >= deadline:
                raise ControlBlockTimeout(
                    f"reset seq {seq} acknowledged but not done within "
                    f"{timeout:.2f}s (state {snap.state_name})")
            time.sleep(poll)
            snap = self.read()
        if snap.state == STATE_ERROR:
            raise ControlBlockError(f"runtime reported an error for reset seq {seq}")
        return snap


# ---- module-level convenience helpers --------------------------------

def open_control(path: str = SHM_PATH, create: bool = True,
                 reinit: bool = False) -> ControlBlock:
    return ControlBlock(path, create=create, reinit=reinit).open()


def create_control_block(path: str = SHM_PATH) -> ControlBlock:
    """Create (or truncate) a fresh control block at path."""
    return ControlBlock(path, create=True, reinit=True).open()


def read_control(path: str = SHM_PATH) -> ControlBlockSnapshot:
    with ControlBlock(path, create=False) as cb:
        return cb.read()


def request_reset(path: str = SHM_PATH) -> int:
    with ControlBlock(path) as cb:
        return cb.request_reset()


def wait_for_ack(seq: int, path: str = SHM_PATH,
                 timeout: float = 5.0) -> ControlBlockSnapshot:
    with ControlBlock(path, create=False) as cb:
        return cb.wait_for_ack(seq, timeout)


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=SHM_PATH)
    ap.add_argument("--init", action="store_true", help="create/reinitialize the block")
    ap.add_argument("--read", action="store_true", help="print the block")
    ap.add_argument("--request", action="store_true", help="increment reset_seq")
    ap.add_argument("--wait", type=int, metavar="SEQ",
                    help="wait for the ack of SEQ")
    ap.add_argument("--timeout", type=float, default=5.0)
    args = ap.parse_args()
    try:
        if args.init:
            with create_control_block(args.path) as cb:
                print(cb.read())
        elif args.request:
            with ControlBlock(args.path) as cb:
                seq = cb.request_reset()
                print(f"requested reset seq {seq}")
        elif args.wait is not None:
            with ControlBlock(args.path, create=False) as cb:
                print(cb.wait_for_ack(args.wait, args.timeout))
        else:
            print(read_control(args.path))
    except (ControlBlockError, ControlBlockTimeout) as e:
        print(f"control_block: {e}", file=sys.stderr)
        raise SystemExit(1)
