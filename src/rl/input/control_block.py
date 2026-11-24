#!/usr/bin/env python3
"""Reset/command control block shared with the runtime: /dev/shm/ssbu_control.bin.

Current layout: v2, little-endian, 80 bytes.

    offset  type   field
    0       char4  magic        "SSBC"
    4       u32    version      2
    8       u32    reset_seq    the client increments it to request a reset
    12      u32    ack_seq      the runtime copies reset_seq here when done
    16      u32    state        0 idle, 1 requested, 2 running, 3 done, 4 error
    20      u32    flags        bit0 fight_on       bit1 reset_busy
                                bit2 pad_active     bit3 turbo
                                bit4 pad_from_ctl   bit5 fixed_seed
                                bit6 hash_valid     bits 8..31 epoch
                                (game pid low 24 bits; a block from another
                                process is stale)
    24      u32    game_frame   runtime-written, presented frame
    28      u32    state_hash   runtime-written, FNV-1a of the fighter sim
    32      u32    seed         runtime echo of the match seed
    36      u32    cmd_seq      the client increments it for each command
    40      u32    cmd          0 none, 1 set_seed, 2 set_pad, 3 clear_pad
    44      u32    arg0         seed, or target game frame for set_pad
                                (0 = next frame)
    48      u32    arg1         XInput button mask for the synthetic pad
    52      u32    arg2         stick: int16 x in low 16 bits, int16 y in high 16
    56      u32    arg3         c-stick, same packing
    60      u32    cmd_ack      runtime echoes cmd_seq when the command ran
    64      u32    ack_frame    game_frame when the command ran
    68      u32    speed_milli  runtime echo: 1000 = 1x, 0 = uncapped
                                (parsed from SSBU_SPEED)
    72      u32    pad0         set_pad triggers: lt | rt << 8 (0..255 each);
                                otherwise reserved and preserved
    76      u32    pad1         zero, reserved
Backward compatibility: v1 was a 24-byte block with the same first six words.
``unpack`` accepts both sizes and versions 1 and 2; a 24-byte (v1) read reports
zero for every v2 field.  ``pack`` remains the v1 packer (same signature as
before) and is still used by the older unit tests; new code should use
``pack_v2`` / ``pack_snapshot``.  Reading a v1 file is supported, but the first
write from this client upgrades it to the full 80-byte v2 layout.

Protocols (single client, single runtime):

    reset:   client: read, reset_seq += 1, state = REQUESTED, whole struct written
             runtime: sees reset_seq != ack_seq, sets state = RUNNING, resets the
                      match, sets ack_seq = reset_seq and state = DONE (or ERROR)
             client: waits for ack_seq == reset_seq, then for state == DONE, then
                      for fight_on == 1 in the state block for a few stable frames
                      (gym_reset.ResetClient does the last part).

    command: client: read, cmd_seq += 1, cmd/args written, whole struct written
             runtime: runs the command, sets cmd_ack = cmd_seq and ack_frame
             client: waits for cmd_ack == cmd_seq (``send_command``)

``set_pad`` arg1 goes straight to the runtime's synthetic XInput pad
(nnhid.c), so it uses XInput bits rather than the canonical 12-button order:

    A 0x1000  B 0x2000  X 0x4000  Y 0x8000
    LB 0x0100 RB 0x0200 START/PLUS 0x0010 BACK/MINUS 0x0020
    DPAD_UP 0x0001 DPAD_DOWN 0x0002 DPAD_LEFT 0x0004 DPAD_RIGHT 0x0008

``set_pad(buttons=...)`` accepts that integer mask or canonical action names
(``L -> LB``, ``R -> RB``; ``ZL``/``ZR`` become triggers); the trigger bytes go
to pad0 as ``lt | rt << 8``.

The file is a plain tmpfs file; the struct is written with a single
``struct.pack_into`` buffer plus one pwrite, and read with one pread.  Readers in
the runtime poll it.  Create it with mode 0666 so a game started from another
account/session can still see it.

This module has no dependencies outside the standard library.
"""

from __future__ import annotations

import numbers
import os
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace

MAGIC = b"SSBC"

VERSION_V1 = 1
VERSION_V2 = 2
# ``VERSION`` is kept for v1 compatibility: it is the version emitted by the
# legacy ``pack()`` helper and used by old tests/mocks.
VERSION = VERSION_V1
CURRENT_VERSION = VERSION_V2

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

# flags bits
FLAG_FIGHT_ON = 1 << 0
FLAG_RESET_BUSY = 1 << 1
FLAG_PAD_ACTIVE = 1 << 2
FLAG_TURBO = 1 << 3
FLAG_PAD_FROM_CTL = 1 << 4
FLAG_FIXED_SEED = 1 << 5
FLAG_HASH_VALID = 1 << 6

# commands
CMD_NONE = 0
CMD_SET_SEED = 1
CMD_SET_PAD = 2
CMD_CLEAR_PAD = 3
#: DAgger: hand player 1 to the CPU for ``arg0`` presented frames.  The CPU's
#: own input for player 1 lands in the control module, which the state block
#: publishes as that fighter's buttons/stick: the expert label for the state the
#: policy produced.  ``flags`` bit 4 = forcing right now, bit 5 = P1 is CPU.
CMD_HANDOVER = 4
#: Drill drop: the NEXT rematch places a fighter airborne at (x, y) with a
#: damage value instead of on its spawn at 0 %.  arg0/1/2 = float32 bits of
#: x, y, damage; arg3 bit0 = helpless fall instead of fall, bit1 = slot.
CMD_DRILL_DROP = 5
#: Cancel a pending drill drop.
CMD_DRILL_CLEAR = 6
#: Launch for the next drill drop: arg0/arg1 = float32 bits of the x / y speed
#: (game units per frame, applied one frame after the reset), arg2 = status id
#: to start in (0 = fall / helpless fall), arg3 = 1 to enable.  One-shot.
CMD_DRILL_LAUNCH = 8

CMD_NAMES = {
    CMD_NONE: "none",
    CMD_SET_SEED: "set_seed",
    CMD_SET_PAD: "set_pad",
    CMD_CLEAR_PAD: "clear_pad",
    CMD_HANDOVER: "handover",
    CMD_DRILL_DROP: "drill_drop",
    CMD_DRILL_CLEAR: "drill_clear",
    CMD_DRILL_LAUNCH: "drill_launch",
}

# ``set_pad`` arg1 is fed straight to the synthetic XInput pad in the runtime
# (nnhid.c), so it uses XInput button bits, not the canonical 12-button order.
XINPUT_A = 0x1000
XINPUT_B = 0x2000
XINPUT_X = 0x4000
XINPUT_Y = 0x8000
XINPUT_LB = 0x0100
XINPUT_RB = 0x0200
XINPUT_START = 0x0010
XINPUT_BACK = 0x0020
XINPUT_DPAD_UP = 0x0001
XINPUT_DPAD_DOWN = 0x0002
XINPUT_DPAD_LEFT = 0x0004
XINPUT_DPAD_RIGHT = 0x0008

# Canonical action names (plus common aliases) -> XInput bits.
XINPUT_BUTTONS = {
    "A": XINPUT_A,
    "B": XINPUT_B,
    "X": XINPUT_X,
    "Y": XINPUT_Y,
    "L": XINPUT_LB,
    "R": XINPUT_RB,
    "LB": XINPUT_LB,
    "RB": XINPUT_RB,
    "PLUS": XINPUT_START,
    "START": XINPUT_START,
    "MINUS": XINPUT_BACK,
    "BACK": XINPUT_BACK,
    "DPAD_U": XINPUT_DPAD_UP,
    "DPAD_UP": XINPUT_DPAD_UP,
    "DPAD_D": XINPUT_DPAD_DOWN,
    "DPAD_DOWN": XINPUT_DPAD_DOWN,
    "DPAD_L": XINPUT_DPAD_LEFT,
    "DPAD_LEFT": XINPUT_DPAD_LEFT,
    "DPAD_R": XINPUT_DPAD_RIGHT,
    "DPAD_RIGHT": XINPUT_DPAD_RIGHT,
}

# Canonical ZL/ZR (and LT/RT aliases) are analog triggers: pad0, not arg1.
XINPUT_TRIGGER_NAMES = {
    "ZL": "lt",
    "ZR": "rt",
    "LT": "lt",
    "RT": "rt",
}

# magic, version, reset_seq, ack_seq, state, pad
_V1_STRUCT = struct.Struct("<4sIIIII")
# magic + 19 u32 words: version .. pad1 (see the table above)
_V2_STRUCT = struct.Struct("<4s19I")

SIZE_V1 = _V1_STRUCT.size
SIZE_V2 = _V2_STRUCT.size
# Legacy aliases: ``SIZE`` is the v1 struct size used by ``pack()``.
SIZE = SIZE_V1
CURRENT_SIZE = SIZE_V2

assert (SIZE_V1, SIZE_V2) == (24, 80)

# v2 field offsets (bytes from the start of the block)
VERSION_OFFSET = 4
RESET_SEQ_OFFSET = 8
ACK_SEQ_OFFSET = 12
STATE_OFFSET = 16
FLAGS_OFFSET = 20
GAME_FRAME_OFFSET = 24
STATE_HASH_OFFSET = 28
SEED_OFFSET = 32
CMD_SEQ_OFFSET = 36
CMD_OFFSET = 40
ARG0_OFFSET = 44
ARG1_OFFSET = 48
ARG2_OFFSET = 52
ARG3_OFFSET = 56
CMD_ACK_OFFSET = 60
ACK_FRAME_OFFSET = 64
SPEED_MILLI_OFFSET = 68
PAD0_OFFSET = 72
PAD1_OFFSET = 76

# Legacy name for the trailing v1 word at offset 20.
PAD_OFFSET = FLAGS_OFFSET

_M32 = 0xFFFFFFFF


class ControlBlockError(RuntimeError):
    """The control block is missing, malformed, or reported an error."""


class ControlBlockTimeout(TimeoutError):
    """The runtime did not acknowledge a reset/command in time."""


def _s16(value: int) -> int:
    """Clamp/coerce a Python int to a signed 16-bit value."""
    value = int(value)
    return max(-32768, min(32767, value))


def pack_stick(x: int, y: int) -> int:
    """Pack an int16 (x, y) pair into the u32 used by arg2/arg3."""
    return (_s16(x) & 0xFFFF) | ((_s16(y) & 0xFFFF) << 16)


def unpack_stick(value: int) -> tuple[int, int]:
    """Unpack the u32 used by arg2/arg3 into a signed int16 (x, y) pair."""
    value = int(value) & _M32
    x = value & 0xFFFF
    y = (value >> 16) & 0xFFFF
    if x >= 0x8000:
        x -= 0x10000
    if y >= 0x8000:
        y -= 0x10000
    return x, y


# Explicit aliases for callers that prefer the longer name.
pack_stick_pair = pack_stick
unpack_stick_pair = unpack_stick


def trigger_byte(value: float | int) -> int:
    """Coerce a trigger value to 0..255.

    Floats (non-integer numbers) are treated as a 0..1 fraction, like
    ``gym_input.trigger_to_abs``, and scaled to 0..255; integers are treated as
    raw 0..255 values.
    """
    if isinstance(value, numbers.Integral):
        return max(0, min(255, int(value)))
    return int(round(max(0.0, min(1.0, float(value))) * 255))


def pack_triggers(lt: float | int = 0, rt: float | int = 0) -> int:
    """Pack (lt, rt) into the pad0 word: ``lt | rt << 8`` (0..255 each)."""
    return (trigger_byte(lt) & 0xFF) | ((trigger_byte(rt) & 0xFF) << 8)


def unpack_triggers(value: int) -> tuple[int, int]:
    """Unpack the pad0 word into raw (lt, rt) bytes."""
    value = int(value) & _M32
    return value & 0xFF, (value >> 8) & 0xFF


def encode_buttons(buttons: int | str | Iterable[str | int]) -> tuple[int, int, int]:
    """Encode a button spec as ``(xinput_mask, lt, rt)``.

    ``buttons`` is either an integer XInput mask (0xFFFF max, used as-is) or a
    canonical action name/iterable of names.  Names map to XInput bits
    (``L -> LB``, ``R -> RB``, ``PLUS -> START``, ``MINUS -> BACK``); ``ZL`` /
    ``ZR`` set the left/right trigger to 255 instead of setting a button bit.
    """
    if isinstance(buttons, numbers.Integral):
        return int(buttons) & 0xFFFF, 0, 0
    if isinstance(buttons, str):
        names = (buttons,)
    else:
        names = buttons
    mask = 0
    lt = rt = 0
    for item in names:
        if isinstance(item, numbers.Integral):
            mask |= int(item) & 0xFFFF
            continue
        name = str(item).strip().upper()
        if not name:
            continue
        if name in XINPUT_TRIGGER_NAMES:
            if XINPUT_TRIGGER_NAMES[name] == "lt":
                lt = 255
            else:
                rt = 255
            continue
        try:
            mask |= XINPUT_BUTTONS[name]
        except KeyError:
            raise ValueError(f"unknown button name {item!r}") from None
    return mask & 0xFFFF, lt, rt


@dataclass(frozen=True)
class ControlBlockSnapshot:
    """One consistent read of the control block.

    ``pad`` is the legacy v1 trailing word at offset 20.  In v2 that slot is
    ``flags``; ``pad`` is reported as 0 and is not used by writers.
    """

    magic: bytes = MAGIC
    version: int = VERSION_V1
    reset_seq: int = 0
    ack_seq: int = 0
    state: int = STATE_IDLE
    pad: int = 0
    # ---- v2 fields (zero for a 24-byte v1 block) ----
    flags: int = 0
    game_frame: int = 0
    state_hash: int = 0
    seed: int = 0
    cmd_seq: int = 0
    cmd: int = CMD_NONE
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    arg3: int = 0
    cmd_ack: int = 0
    ack_frame: int = 0
    speed_milli: int = 0
    pad0: int = 0
    pad1: int = 0

    @property
    def state_name(self) -> str:
        return STATE_NAMES.get(self.state, f"unknown({self.state})")

    @property
    def pending(self) -> bool:
        """True while the runtime has not acknowledged the latest reset."""
        return self.ack_seq != self.reset_seq

    @property
    def done(self) -> bool:
        return self.state == STATE_DONE and not self.pending

    @property
    def failed(self) -> bool:
        return self.state == STATE_ERROR

    # -- flag accessors -------------------------------------------------
    def _flag(self, bit: int) -> bool:
        return bool(self.flags & bit)

    @property
    def fight_on(self) -> bool:
        return self._flag(FLAG_FIGHT_ON)

    @property
    def reset_busy(self) -> bool:
        return self._flag(FLAG_RESET_BUSY)

    @property
    def pad_active(self) -> bool:
        return self._flag(FLAG_PAD_ACTIVE)

    @property
    def turbo(self) -> bool:
        return self._flag(FLAG_TURBO)

    @property
    def pad_from_ctl(self) -> bool:
        return self._flag(FLAG_PAD_FROM_CTL)

    @property
    def fixed_seed(self) -> bool:
        return self._flag(FLAG_FIXED_SEED)

    @property
    def hash_valid(self) -> bool:
        return self._flag(FLAG_HASH_VALID)

    # -- command accessors ---------------------------------------------
    @property
    def cmd_name(self) -> str:
        return CMD_NAMES.get(self.cmd, f"unknown({self.cmd})")

    @property
    def cmd_pending(self) -> bool:
        """True while the runtime has not acknowledged the latest command."""
        return self.cmd_ack != self.cmd_seq

    @property
    def stick(self) -> tuple[int, int]:
        return unpack_stick(self.arg2)

    @property
    def cstick(self) -> tuple[int, int]:
        return unpack_stick(self.arg3)

    @property
    def triggers(self) -> tuple[int, int]:
        """Raw (lt, rt) bytes from the pad0 word."""
        return unpack_triggers(self.pad0)

    @property
    def lt(self) -> int:
        return unpack_triggers(self.pad0)[0]

    @property
    def rt(self) -> int:
        return unpack_triggers(self.pad0)[1]


def _check_header(magic: bytes, version: int) -> None:
    if magic != MAGIC:
        raise ControlBlockError(f"bad magic {magic!r}; expected {MAGIC!r}")
    if version not in (VERSION_V1, VERSION_V2):
        raise ControlBlockError(
            f"unsupported control block version {version}; "
            f"expected {VERSION_V1} or {VERSION_V2}")


def pack(reset_seq: int = 0, ack_seq: int = 0, state: int = STATE_IDLE,
         pad: int = 0, version: int = VERSION, magic: bytes = MAGIC) -> bytes:
    """Serialize a legacy 24-byte (v1) control block.

    Kept with its original signature for existing tests and v1 mocks.  New code
    should use :func:`pack_v2` (or :func:`pack_snapshot`).
    """
    return _V1_STRUCT.pack(magic, version, reset_seq & _M32, ack_seq & _M32,
                           state & _M32, pad & _M32)


def pack_v2(*, reset_seq: int = 0, ack_seq: int = 0, state: int = STATE_IDLE,
            flags: int = 0, game_frame: int = 0, state_hash: int = 0,
            seed: int = 0, cmd_seq: int = 0, cmd: int = CMD_NONE,
            arg0: int = 0, arg1: int = 0, arg2: int = 0, arg3: int = 0,
            cmd_ack: int = 0, ack_frame: int = 0, speed_milli: int = 0,
            pad0: int = 0, pad1: int = 0,
            version: int = CURRENT_VERSION, magic: bytes = MAGIC) -> bytes:
    """Serialize the full 80-byte v2 control block."""
    buf = bytearray(SIZE_V2)
    _V2_STRUCT.pack_into(
        buf, 0, magic, version & _M32, reset_seq & _M32, ack_seq & _M32,
        state & _M32, flags & _M32, game_frame & _M32, state_hash & _M32,
        seed & _M32, cmd_seq & _M32, cmd & _M32, arg0 & _M32, arg1 & _M32,
        arg2 & _M32, arg3 & _M32, cmd_ack & _M32, ack_frame & _M32,
        speed_milli & _M32, pad0 & _M32, pad1 & _M32)
    return bytes(buf)


def pack_snapshot(snap: ControlBlockSnapshot) -> bytes:
    """Serialize a snapshot as a v2 block (magic/version are normalized)."""
    return pack_v2(
        reset_seq=snap.reset_seq, ack_seq=snap.ack_seq, state=snap.state,
        flags=snap.flags, game_frame=snap.game_frame,
        state_hash=snap.state_hash, seed=snap.seed, cmd_seq=snap.cmd_seq,
        cmd=snap.cmd, arg0=snap.arg0, arg1=snap.arg1, arg2=snap.arg2,
        arg3=snap.arg3, cmd_ack=snap.cmd_ack, ack_frame=snap.ack_frame,
        speed_milli=snap.speed_milli, pad0=snap.pad0, pad1=snap.pad1)


def unpack(data: bytes) -> ControlBlockSnapshot:
    """Parse a 24-byte (v1) or 80-byte (v2) block.

    A v1 block yields zero for every v2 field.  Bad magic and versions other
    than 1 or 2 raise :class:`ControlBlockError`.
    """
    if len(data) >= SIZE_V2:
        fields = _V2_STRUCT.unpack_from(data)
        (magic, version, reset_seq, ack_seq, state, flags, game_frame,
         state_hash, seed, cmd_seq, cmd, arg0, arg1, arg2, arg3, cmd_ack,
         ack_frame, speed_milli, pad0, pad1) = fields
        _check_header(magic, version)
        return ControlBlockSnapshot(
            magic=magic, version=version, reset_seq=reset_seq, ack_seq=ack_seq,
            state=state, pad=0, flags=flags, game_frame=game_frame,
            state_hash=state_hash, seed=seed, cmd_seq=cmd_seq, cmd=cmd,
            arg0=arg0, arg1=arg1, arg2=arg2, arg3=arg3, cmd_ack=cmd_ack,
            ack_frame=ack_frame, speed_milli=speed_milli, pad0=pad0, pad1=pad1)
    if len(data) >= SIZE_V1:
        magic, version, reset_seq, ack_seq, state, pad = _V1_STRUCT.unpack_from(data)
        _check_header(magic, version)
        return ControlBlockSnapshot(
            magic=magic, version=version, reset_seq=reset_seq, ack_seq=ack_seq,
            state=state, pad=pad)
    raise ControlBlockError(
        f"short control block: {len(data)} < {SIZE_V1} bytes")


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
            size = os.fstat(fd).st_size
            if size == 0:
                # Newly created (or empty) file: lay down the full v2 block.
                if not self.create:
                    raise ControlBlockError(f"{self.path} is an empty control block")
                os.ftruncate(fd, SIZE_V2)
                self._init_fd(fd)
            elif self.reinit:
                os.ftruncate(fd, SIZE_V2)
                self._init_fd(fd)
            else:
                data = os.pread(fd, SIZE_V2, 0)
                if len(data) < 4 or data[:4] != MAGIC:
                    if not self.create:
                        raise ControlBlockError(
                            f"{self.path} is not an SSBC control block")
                    os.ftruncate(fd, SIZE_V2)
                    self._init_fd(fd)
                elif len(data) < SIZE_V1:
                    # Partial file from a crashed/older writer: only safe to
                    # reinitialize when we own creation.
                    if not self.create:
                        raise ControlBlockError(
                            f"short control block: {len(data)} < {SIZE_V1} bytes")
                    os.ftruncate(fd, SIZE_V2)
                    self._init_fd(fd)
                else:
                    # Accepts v1 (24) and v2 (80); raises on bad magic/version.
                    unpack(data)
        except Exception:
            os.close(fd)
            raise
        self.fd = fd
        return self

    @staticmethod
    def _init_fd(fd: int) -> None:
        written = os.pwrite(fd, pack_v2(), 0)
        if written != SIZE_V2:
            raise ControlBlockError(f"short write: {written} != {SIZE_V2}")

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
        data = os.pread(fd, SIZE_V2, 0)
        return unpack(data)

    snapshot = read

    def _commit(self, snap: ControlBlockSnapshot) -> ControlBlockSnapshot:
        """Write the whole 80-byte struct in one pwrite (single writer)."""
        fd = self._require_fd()
        snap = replace(snap, magic=MAGIC, version=CURRENT_VERSION, pad=0)
        buf = bytearray(SIZE_V2)
        _V2_STRUCT.pack_into(
            buf, 0, snap.magic, snap.version, snap.reset_seq & _M32,
            snap.ack_seq & _M32, snap.state & _M32, snap.flags & _M32,
            snap.game_frame & _M32, snap.state_hash & _M32, snap.seed & _M32,
            snap.cmd_seq & _M32, snap.cmd & _M32, snap.arg0 & _M32,
            snap.arg1 & _M32, snap.arg2 & _M32, snap.arg3 & _M32,
            snap.cmd_ack & _M32, snap.ack_frame & _M32,
            snap.speed_milli & _M32, snap.pad0 & _M32, snap.pad1 & _M32)
        written = os.pwrite(fd, buf, 0)
        if written != SIZE_V2:
            raise ControlBlockError(f"short write: {written} != {SIZE_V2}")
        return snap

    def write_snapshot(self, snap: ControlBlockSnapshot) -> ControlBlockSnapshot:
        """Write an arbitrary snapshot as a v2 block (runtime mock / tests)."""
        return self._commit(snap)

    def write(self, reset_seq: int, ack_seq: int, state: int,
              pad: int = 0) -> ControlBlockSnapshot:
        """Legacy reset writer: updates the reset fields, keeps everything else.

        ``pad`` is accepted for signature compatibility but is ignored; in v2
        offset 20 is ``flags`` and is preserved from the current snapshot.
        Writing upgrades a v1 file to the full 80-byte layout.
        """
        snap = self.read()
        return self._commit(replace(snap, reset_seq=reset_seq, ack_seq=ack_seq,
                                    state=state))

    def set_state(self, state: int) -> ControlBlockSnapshot:
        snap = self.read()
        return self._commit(replace(snap, state=state))

    # -- reset request / acknowledge -----------------------------------
    def request_reset(self) -> int:
        """Increment reset_seq and mark the request.  Returns the new sequence."""
        snap = self.read()
        seq = (snap.reset_seq + 1) & _M32
        self._commit(replace(snap, reset_seq=seq, state=STATE_REQUESTED))
        return seq

    # The runtime side of the protocol, useful for the mock and for tests.
    def acknowledge(self, seq: int | None = None,
                    state: int = STATE_DONE) -> ControlBlockSnapshot:
        snap = self.read()
        if seq is None:
            seq = snap.reset_seq
        return self._commit(replace(snap, ack_seq=seq, state=state))

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

    # -- commands ------------------------------------------------------
    def wait_for_command(self, seq: int, timeout: float | None = None,
                         poll: float | None = None) -> ControlBlockSnapshot:
        """Wait until cmd_ack == seq; raises on timeout or state ERROR."""
        timeout = self.timeout if timeout is None else timeout
        poll = self.poll if poll is None else poll
        deadline = time.monotonic() + timeout
        while True:
            snap = self.read()
            if snap.failed:
                raise ControlBlockError(
                    f"runtime reported an error while running "
                    f"{snap.cmd_name} (cmd_seq {seq})")
            if snap.cmd_ack == seq:
                return snap
            if time.monotonic() >= deadline:
                raise ControlBlockTimeout(
                    f"command seq {seq} not acknowledged within {timeout:.2f}s "
                    f"(cmd_ack {snap.cmd_ack}, cmd {snap.cmd_name})")
            time.sleep(poll)

    def send_command(self, cmd: int, arg0: int = 0, arg1: int = 0,
                     arg2: int = 0, arg3: int = 0,
                     timeout: float | None = None,
                     poll: float | None = None,
                     *, pad0: int | None = None) -> ControlBlockSnapshot:
        """Issue a v2 command and wait for the runtime to echo cmd_seq.

        ``pad0`` (keyword-only) is an optional extra word, used by ``set_pad``
        for the trigger bytes; when None the current value is preserved.
        """
        if cmd == CMD_NONE:
            raise ValueError("cmd must not be CMD_NONE")
        snap = self.read()
        seq = (snap.cmd_seq + 1) & _M32
        sent = replace(snap, cmd_seq=seq, cmd=cmd, arg0=arg0, arg1=arg1,
                       arg2=arg2, arg3=arg3)
        if pad0 is not None:
            sent = replace(sent, pad0=pad0 & _M32)
        self._commit(sent)
        return self.wait_for_command(seq, timeout=timeout, poll=poll)

    def set_seed(self, seed: int, timeout: float | None = None,
                 poll: float | None = None) -> ControlBlockSnapshot:
        """Ask the runtime to use ``seed`` for the next match."""
        return self.send_command(CMD_SET_SEED, arg0=seed, timeout=timeout,
                                 poll=poll)

    def set_pad(self, buttons: int | str | Iterable[str | int] = 0,
                stick: tuple[int, int] = (0, 0),
                cstick: tuple[int, int] = (0, 0), frame: int = 0,
                trigger: float | int | None = None,
                triggers: tuple[float | int, float | int] | None = None,
                timeout: float | None = None,
                poll: float | None = None) -> ControlBlockSnapshot:
        """Override the synthetic XInput pad for a game frame.

        ``buttons`` is either an integer XInput mask or canonical action names
        (``A``, ``B``, ``X``, ``Y``, ``L``, ``R``, ``ZL``, ``ZR``, ``PLUS``,
        ``MINUS``, ``DPAD_L``, ``DPAD_R``; also ``DPAD_U``/``DPAD_D`` and the
        aliases ``LB``, ``RB``, ``START``, ``BACK``).  ``L`` maps to LB and
        ``R`` to RB; ``ZL``/``ZR`` drive the analog triggers, not arg1.

        ``trigger`` sets the right trigger (``ZR``, matching ``gym_input``);
        ``triggers=(lt, rt)`` sets both and overrides trigger names.  Trigger
        values are floats in 0..1 (scaled to 0..255) or ints in 0..255.  They
        are packed into ``pad0`` as ``lt | rt << 8``.

        ``stick``/``cstick`` are int16 (x, y) pairs packed into arg2/arg3 with
        x in the low 16 bits and y in the high 16 bits.  ``frame`` is the
        target game frame (0 = next frame).
        """
        mask, lt, rt = encode_buttons(buttons)
        if triggers is not None:
            lt, rt = trigger_byte(triggers[0]), trigger_byte(triggers[1])
        if trigger is not None:
            rt = trigger_byte(trigger)
        x, y = stick
        cx, cy = cstick
        return self.send_command(
            CMD_SET_PAD, arg0=frame, arg1=mask,
            arg2=pack_stick(x, y), arg3=pack_stick(cx, cy),
            timeout=timeout, poll=poll, pad0=pack_triggers(lt, rt))

    def clear_pad(self, timeout: float | None = None,
                  poll: float | None = None) -> ControlBlockSnapshot:
        """Drop the pad override and return to the runtime's own input path."""
        return self.send_command(CMD_CLEAR_PAD, timeout=timeout, poll=poll)


# ---- module-level convenience helpers --------------------------------

def open_control(path: str = SHM_PATH, create: bool = True,
                 reinit: bool = False) -> ControlBlock:
    return ControlBlock(path, create=create, reinit=reinit).open()


def create_control_block(path: str = SHM_PATH) -> ControlBlock:
    """Create (or truncate) a fresh 80-byte v2 control block at path."""
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
