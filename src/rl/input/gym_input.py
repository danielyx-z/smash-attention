#!/usr/bin/env python3
"""Raw uinput Xbox 360 virtual pad for the SSBU gym.

No python-evdev: the device is created with /dev/uinput, struct and fcntl.ioctl.
It presents the capabilities and USB ids of a wired Xbox 360 pad (045e:028e,
version 0x0114) so SDL3 (runtime/nnpad_sdl.c) opens it as a gamepad, or at
worst as a joystick that controllers.ini can map by its GUID.

Canonical action (CONTRACTS.md) -> Linux event codes:

    A      BTN_SOUTH (0x130)        L      BTN_TL (0x136)
    B      BTN_EAST  (0x131)        R      BTN_TR (0x137)
    X      BTN_WEST  (0x134)        ZL     ABS_Z  (0..255, analog only)
    Y      BTN_NORTH (0x133)        ZR     ABS_RZ (0..255, analog only)
    PLUS   BTN_START (0x13b)        MINUS  BTN_TL2 (0x138)
    DPAD_L BTN_DPAD_LEFT (0x222)    DPAD_R BTN_DPAD_RIGHT (0x223)
    (also DPAD_U BTN_DPAD_UP, DPAD_D BTN_DPAD_DOWN; every direction is also
     sent as the standard 360 hat ABS_HAT0X/Y)

    left stick  ABS_X (-32768..32767, right positive)
                ABS_Y (-32768..32767, UP negative: evdev convention;
                       API y is +1 up, so ABS_Y = -y)
    C-stick     ABS_RX / ABS_RY, same convention
    trigger     ABS_RZ (right trigger), 0..255.  The digital ZR button also
                drives ABS_RZ to full, so both mappings see it.

BTN_TL2 and BTN_START are the codes used for MINUS and PLUS because SDL's
stock Xbox 360 map turns BTN_SELECT into Guide (which opens the runtime's
settings menu) and BTN_TL2/TR2 into Back/Start.  BTN_SELECT and BTN_TR2 are
advertised by the descriptor but not driven.

A neutral pad sends ABS_X/Y/RX/RY/HAT0X/HAT0Y = 0 and ABS_Z/RZ = 0.

controllers.ini has a [device <SDL_GUID>] section for this pad (see
build-linux/config/controllers.ini and FINDINGS.md): SDL's stock Xbox 360 map
would make south = Pro B, send TL2/TR2 to Back/Start and look for a hat for
the D-pad, so the explicit raw-button map is what makes the canonical names
land on the right Pro Controller buttons.

Permissions: /dev/uinput must be writable by the user.  On this machine it is
root:input with ACL user:daniel:rw-, and daniel is not in the input group, so
the ACL (or `sudo setfacl -m u:daniel:rw /dev/uinput`) is what makes it work.
The kernel creates the resulting /dev/input/eventN as root:root 0600, so
create() also grants this user a read ACL on that node (setfacl, through
passwordless `sudo -n` when needed).  Without that ACL SDL enumerates the pad
and the game logs "connected", but every read fails and the pad never moves.

One virtual pad may exist at a time: the pad lock /tmp/ssbu_pad.lock is held
from UI_DEV_CREATE until UI_DEV_DESTROY.  Always use the context manager or
call destroy() in a finally block; closing the fd also removes the device.

    python3 tools/mem/gym_input.py --test            # print the event node and exit
    python3 tools/mem/gym_input.py --test --seconds 5  # keep it alive for 5 s
"""

from __future__ import annotations

import argparse
import array
import fcntl
import glob
import os
import signal
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.utils.locks import PAD_LOCK, FileLock  # noqa: E402

# ---- Linux input constants -------------------------------------------
EV_SYN = 0x00
EV_KEY = 0x01
EV_ABS = 0x03
SYN_REPORT = 0

# event-codes.h
BTN_SOUTH = 0x130          # BTN_A / BTN_GAMEPAD
BTN_EAST = 0x131           # BTN_B
BTN_NORTH = 0x133          # BTN_Y (Xbox Y is the top face button)
BTN_WEST = 0x134           # BTN_X
BTN_TL = 0x136
BTN_TR = 0x137
BTN_TL2 = 0x138
BTN_TR2 = 0x139
BTN_SELECT = 0x13A
BTN_START = 0x13B
BTN_DPAD_UP = 0x220
BTN_DPAD_DOWN = 0x221
BTN_DPAD_LEFT = 0x222
BTN_DPAD_RIGHT = 0x223

ABS_X = 0x00
ABS_Y = 0x01
ABS_Z = 0x02
ABS_RX = 0x03
ABS_RY = 0x04
ABS_RZ = 0x05
ABS_HAT0X = 0x10
ABS_HAT0Y = 0x11

BUS_USB = 0x03
VENDOR_ID = 0x045E             # Microsoft
PRODUCT_ID = 0x028E            # Xbox 360 wired controller
DEVICE_VERSION = 0x0114
DEVICE_NAME = "SSBU Virtual Pad"
# SDL3 overrides the name from its controller database, and folds a CRC16 of
# that name into GUID byte 2.  With SDL 3.2.6 both are deterministic:
SDL_NAME = "Xbox 360 Controller"
SDL_GUID = "030049205e0400008e02000014010000"
EXPECTED_GUID = SDL_GUID       # kept for older callers/tests

# Abs axis and their proper ranges: sticks are full signed 16-bit, triggers 8-bit,
# the standard Xbox 360 D-pad is a hat whose two axes are -1..1.
ABS_INFO = {
    ABS_X: (-32768, 32767),
    ABS_Y: (-32768, 32767),
    ABS_RX: (-32768, 32767),
    ABS_RY: (-32768, 32767),
    ABS_Z: (0, 255),
    ABS_RZ: (0, 255),
    ABS_HAT0X: (-1, 1),
    ABS_HAT0Y: (-1, 1),
}

# ---- ioctl numbers ----------------------------------------------------
UINPUT_IOCTL_BASE = ord("U")
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2


def _IOC(dir_: int, type_: int, nr: int, size: int) -> int:
    return ((dir_ << _IOC_DIRSHIFT) | (type_ << _IOC_TYPESHIFT) |
            (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT))


def _IO(type_: int, nr: int) -> int:
    return _IOC(_IOC_NONE, type_, nr, 0)


def _IOW(type_: int, nr: int, size: int) -> int:
    return _IOC(_IOC_WRITE, type_, nr, size)


def _IOC_READ_REQ(type_: int, nr: int, size: int) -> int:
    return _IOC(_IOC_READ, type_, nr, size)


UI_DEV_CREATE = _IO(UINPUT_IOCTL_BASE, 1)
UI_DEV_DESTROY = _IO(UINPUT_IOCTL_BASE, 2)
UI_DEV_SETUP = _IOW(UINPUT_IOCTL_BASE, 3, 92)        # struct uinput_setup
UI_ABS_SETUP = _IOW(UINPUT_IOCTL_BASE, 4, 28)        # struct uinput_abs_setup
UI_SET_EVBIT = _IOW(UINPUT_IOCTL_BASE, 100, 4)
UI_SET_KEYBIT = _IOW(UINPUT_IOCTL_BASE, 101, 4)
UI_SET_ABSBIT = _IOW(UINPUT_IOCTL_BASE, 103, 4)
UI_GET_SYSNAME = lambda n: _IOC_READ_REQ(UINPUT_IOCTL_BASE, 44, n)  # noqa: E731
UI_GET_VERSION = _IOC_READ_REQ(UINPUT_IOCTL_BASE, 45, 4)

_INPUT_ID = struct.Struct("<HHHH")                   # bustype, vendor, product, version
_UINPUT_SETUP = struct.Struct("<HHHH80sI")           # input_id, name[80], ff_effects_max
_INPUT_ABSINFO = struct.Struct("<iiiiii")            # value, min, max, fuzz, flat, resolution
_UINPUT_ABS_SETUP = struct.Struct("<HHiiiiii")       # code, pad, absinfo
_INPUT_EVENT = struct.Struct("<qqHHi")               # timeval(sec,usec), type, code, value

assert _UINPUT_SETUP.size == 92
assert _UINPUT_ABS_SETUP.size == 28
assert _INPUT_EVENT.size == 24

# legacy uinput_user_dev, only used if UI_DEV_SETUP is not available
# name[80] input_id ff_effects_max absmax[64] absmin[64] absfuzz[64] absflat[64]
_UINPUT_USER_DEV = struct.Struct("<80sHHHHI" + "i" * 256)
assert _UINPUT_USER_DEV.size == 1116

# ---- canonical action mapping -----------------------------------------
CANONICAL_BUTTONS = (
    "A", "B", "X", "Y", "L", "R", "ZL", "ZR",
    "PLUS", "MINUS", "DPAD_L", "DPAD_R",
)

BUTTON_CODES = {
    # Natural Xbox face-button codes.  controllers.ini maps this device's raw
    # indices explicitly, so A is Pro A even though the stock Xbox 360 SDL
    # mapping would make south = Pro B (positional layout).
    "A": BTN_SOUTH,
    "B": BTN_EAST,
    "X": BTN_WEST,
    "Y": BTN_NORTH,
    "L": BTN_TL,
    "R": BTN_TR,
    "PLUS": BTN_START,
    # BTN_SELECT is not used: SDL's stock 360 map turns it into GUIDE, and the
    # runtime opens the settings menu on Guide.  BTN_TL2 is SDL "back" and is
    # safe for MINUS; BTN_TL2 therefore cannot also be ZL.
    "MINUS": BTN_TL2,
    "DPAD_L": BTN_DPAD_LEFT,
    "DPAD_R": BTN_DPAD_RIGHT,
    # aliases and the two dpad directions the canonical vector omits
    "DPAD_LEFT": BTN_DPAD_LEFT,
    "DPAD_RIGHT": BTN_DPAD_RIGHT,
    "DPAD_U": BTN_DPAD_UP,
    "DPAD_UP": BTN_DPAD_UP,
    "DPAD_D": BTN_DPAD_DOWN,
    "DPAD_DOWN": BTN_DPAD_DOWN,
    "START": BTN_START,
    "SELECT": BTN_SELECT,
}

# ZL / ZR are analog triggers: ABS_Z / ABS_RZ only.  A digital BTN_TR2 would
# be SDL "start" (and BTN_TL2 is MINUS above), so they stay purely analog.
ANALOG_BUTTONS = {
    "ZL": ABS_Z,
    "ZR": ABS_RZ,
}

# D-pad is sent twice: as BTN_DPAD_* (the task's descriptor) and as the
# standard Xbox 360 hat ABS_HAT0X/Y (which SDL's built-in 360 mapping reads).
DPAD_HAT = {
    BTN_DPAD_LEFT: (ABS_HAT0X, -1),
    BTN_DPAD_RIGHT: (ABS_HAT0X, 1),
    BTN_DPAD_UP: (ABS_HAT0Y, -1),
    BTN_DPAD_DOWN: (ABS_HAT0Y, 1),
}

# every key the device advertises, in a stable order
KEY_CODES = (
    BTN_SOUTH, BTN_EAST, BTN_NORTH, BTN_WEST,
    BTN_TL, BTN_TR, BTN_TL2, BTN_TR2,
    BTN_SELECT, BTN_START,
    BTN_DPAD_UP, BTN_DPAD_DOWN, BTN_DPAD_LEFT, BTN_DPAD_RIGHT,
)

TRIGGER_THRESHOLD = 0.33      # the game's controllers.ini trigger_threshold


def clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def stick_to_abs(x: float, y: float) -> tuple[int, int]:
    """API stick (x right +, y up +) -> ABS_X, ABS_Y (evdev: up is negative)."""
    ax = int(round(clamp(x) * 32767.0))
    ay = int(round(-clamp(y) * 32767.0))
    return ax, ay


def trigger_to_abs(t: float) -> int:
    return int(round(clamp(t, 0.0, 1.0) * 255.0))


@dataclass
class Action:
    """A decoded canonical action.

    ``buttons`` is any iterable of names from CANONICAL_BUTTONS; ``stick`` and
    ``cstick`` are (x, y) with x right positive / y up positive; ``trigger`` is
    the analog right trigger (ZR) in [0, 1].
    """

    buttons: Iterable[str] = field(default_factory=tuple)
    stick: tuple[float, float] = (0.0, 0.0)
    cstick: tuple[float, float] = (0.0, 0.0)
    trigger: float = 0.0

    def button_set(self) -> set[str]:
        if isinstance(self.buttons, str):
            names = self.buttons.replace(",", " ").split()
        else:
            names = list(self.buttons)
        # gym_action/gym_env pass the 12-slot canonical bit vector (numpy int8
        # or a list of 0/1), not names.  A list of 12 strings is names.
        if len(names) == len(CANONICAL_BUTTONS) and all(
                not isinstance(v, str) for v in names):
            return {name for name, on in zip(CANONICAL_BUTTONS, names) if on}
        out: set[str] = set()
        for name in names:
            key = str(name).strip().upper()
            if key not in BUTTON_CODES and key not in ANALOG_BUTTONS:
                raise ValueError(f"unknown button {name!r}; expected one of {CANONICAL_BUTTONS}")
            out.add(key)
        return out


def decode_action(action) -> Action:
    """Accept an Action, a mapping, an object with attributes, or a flat vector.

    A flat vector has 17 values: 12 booleans in CANONICAL_BUTTONS order, then
    stick x, stick y, cstick x, cstick y, trigger.
    """
    if isinstance(action, Action):
        return action
    if isinstance(action, Mapping):
        return Action(
            buttons=action.get("buttons", ()),
            stick=tuple(action.get("stick", (0.0, 0.0))),
            cstick=tuple(action.get("cstick", (0.0, 0.0))),
            trigger=float(action.get("trigger", 0.0)),
        )
    if hasattr(action, "buttons"):
        return Action(
            buttons=getattr(action, "buttons", ()),
            stick=tuple(getattr(action, "stick", (0.0, 0.0))),
            cstick=tuple(getattr(action, "cstick", (0.0, 0.0))),
            trigger=float(getattr(action, "trigger", 0.0)),
        )
    seq = list(action)
    if len(seq) == 17:
        return Action(
            buttons=[name for name, on in zip(CANONICAL_BUTTONS, seq[:12]) if on],
            stick=(float(seq[12]), float(seq[13])),
            cstick=(float(seq[14]), float(seq[15])),
            trigger=float(seq[16]),
        )
    raise TypeError("action must be Action, mapping, object, or a 17-value vector")


class VirtualPad:
    """One uinput Xbox 360 pad.

    Use as a context manager::

        with VirtualPad() as pad:
            pad.apply({"buttons": ["A"], "stick": (1.0, 0.0)})

    The uinput device exists only between create() and destroy(); the pad lock
    is held for exactly the same interval.
    """

    def __init__(self, name: str = DEVICE_NAME, path: str = "/dev/uinput",
                 lock: bool = True, lock_timeout: float = 120.0,
                 vendor: int = VENDOR_ID, product: int = PRODUCT_ID,
                 version: int = DEVICE_VERSION, bus: int = BUS_USB,
                 auto_create: bool = True):
        self.name = name
        self.path = path
        self.want_lock = lock
        self.lock_timeout = lock_timeout
        self.auto_create = auto_create
        self.vendor = vendor
        self.product = product
        self.version = version
        self.bus = bus
        self.fd: int | None = None
        self.event_path: str | None = None
        self.sysname: str | None = None
        self._lock: FileLock | None = None
        self._keys: dict[int, int] = {code: 0 for code in KEY_CODES}
        self._abs: dict[int, int] = {code: 0 for code in ABS_INFO}

    def _ensure_alive(self) -> None:
        """Create on first use (gym_env adapters call apply() directly)."""
        if self.fd is None:
            if self.auto_create:
                self.create()
            else:
                raise RuntimeError("virtual pad is not created")

    # -- lifecycle -----------------------------------------------------
    def create(self) -> "VirtualPad":
        if self.fd is not None:
            return self
        if self.want_lock:
            self._lock = FileLock(PAD_LOCK, self.lock_timeout).acquire()
        try:
            self._open_device()
            self._create_device()
            self.event_path = self._find_event_path()
            self._grant_read(self.event_path)
            self.release_all()
        except BaseException:
            self.destroy()
            raise
        return self

    def _open_device(self) -> None:
        try:
            self.fd = os.open(self.path, os.O_WRONLY | os.O_NONBLOCK)
        except PermissionError as e:
            raise PermissionError(
                f"cannot open {self.path}: {e}. The user needs write access "
                f"(ACL user:<you>:rw- or membership in the input group).") from None

    def _create_device(self) -> None:
        fd = self.fd
        assert fd is not None
        fcntl.ioctl(fd, UI_SET_EVBIT, EV_KEY)
        fcntl.ioctl(fd, UI_SET_EVBIT, EV_ABS)
        for code in KEY_CODES:
            fcntl.ioctl(fd, UI_SET_KEYBIT, code)
        for code, (lo, hi) in ABS_INFO.items():
            fcntl.ioctl(fd, UI_SET_ABSBIT, code)
            fcntl.ioctl(fd, UI_ABS_SETUP,
                        _UINPUT_ABS_SETUP.pack(code, 0, 0, lo, hi, 0, 0, 0))
        setup = _UINPUT_SETUP.pack(
            self.bus, self.vendor, self.product, self.version,
            self.name.encode("utf-8")[:79], 0)
        try:
            fcntl.ioctl(fd, UI_DEV_SETUP, setup)
        except OSError:
            # pre-4.5 kernel: use the legacy struct instead
            self._legacy_setup()
        fcntl.ioctl(fd, UI_DEV_CREATE)
        self._read_sysname()

    def _legacy_setup(self) -> None:
        fd = self.fd
        assert fd is not None
        absmax = [0] * 64
        absmin = [0] * 64
        for code, (lo, hi) in ABS_INFO.items():
            absmin[code] = lo
            absmax[code] = hi
        data = _UINPUT_USER_DEV.pack(
            self.name.encode("utf-8")[:79], self.bus, self.vendor,
            self.product, self.version, 0, *absmax, *absmin,
            *([0] * 128))
        os.write(fd, data)

    def _read_sysname(self) -> None:
        fd = self.fd
        assert fd is not None
        buf = array.array("B", b"\0" * 64)
        try:
            fcntl.ioctl(fd, UI_GET_SYSNAME(64), buf, False)
            name = bytes(buf).split(b"\0", 1)[0].decode("ascii", "replace")
            self.sysname = name or None
        except OSError:
            self.sysname = None

    def _find_event_path(self) -> str | None:
        # sysfs: /sys/class/input/<sysname>/event*/dev, then /dev/input/<event>
        if self.sysname:
            for dev in glob.glob(f"/sys/class/input/{self.sysname}/event*/dev"):
                ev = os.path.basename(os.path.dirname(dev))
                path = f"/dev/input/{ev}"
                if os.path.exists(path):
                    return path
        # fallback: newest event node whose device name matches
        for _ in range(50):
            matches = []
            for namef in glob.glob("/sys/class/input/event*/device/name"):
                try:
                    with open(namef, "r", encoding="utf-8", errors="replace") as f:
                        if f.read().strip() == self.name:
                            ev = namef.split("/")[4]
                            matches.append(f"/dev/input/{ev}")
                except OSError:
                    pass
            if matches:
                return sorted(matches, key=lambda p: int(p.rsplit("event", 1)[1]))[-1]
            time.sleep(0.02)
        return None

    def _grant_read(self, path: str | None) -> bool:
        """Make the created event node readable by this user.

        The kernel creates ``/dev/input/eventN`` as ``root:root 0600``.  SDL
        still enumerates the pad (and ``nn::pad`` prints ``connected``), but
        every read fails, so the game sees a connected controller that never
        sends a button or a stick value.  Grant a user ACL on the node;
        passwordless sudo is used when the node is not ours.
        """
        if not path:
            return False
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            os.close(fd)
            return True
        except OSError:
            pass
        uid = os.getuid()
        commands = [
            ["setfacl", "-m", f"u:{uid}:rw", path],
            ["sudo", "-n", "setfacl", "-m", f"u:{uid}:rw", path],
            ["sudo", "-n", "chmod", "a+rw", path],
        ]
        for cmd in commands:
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=5)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.returncode != 0:
                continue
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                os.close(fd)
                return True
            except OSError:
                continue
        print(
            f"warning: {path} is not readable by this user; SDL will see the "
            "virtual pad but receive no events. Run "
            f"'sudo setfacl -m u:{uid}:rw {path}' or add a udev rule "
            "(MODE=\"0666\") for the 'SSBU Virtual Pad' input device.",
            file=sys.stderr,
        )
        return False

    def destroy(self) -> None:
        if self.fd is not None:
            try:
                try:
                    self.release_all()
                except OSError:
                    pass
                try:
                    fcntl.ioctl(self.fd, UI_DEV_DESTROY)
                except OSError:
                    pass
            finally:
                try:
                    os.close(self.fd)
                finally:
                    self.fd = None
                    self.event_path = None
                    self.sysname = None
        if self._lock is not None:
            self._lock.release()
            self._lock = None

    # gym_env's input adapter looks for close() and calls it on env.close()
    close = destroy

    def __enter__(self) -> "VirtualPad":
        return self.create()

    def __exit__(self, *exc) -> None:
        self.destroy()

    @property
    def alive(self) -> bool:
        return self.fd is not None

    # -- event emission -------------------------------------------------
    def _emit(self, type_: int, code: int, value: int) -> None:
        fd = self.fd
        if fd is None:
            raise RuntimeError("virtual pad is not created")
        os.write(fd, _INPUT_EVENT.pack(0, 0, type_, code, value))

    def _syn(self) -> None:
        self._emit(EV_SYN, SYN_REPORT, 0)

    def _set_key(self, code: int, down: bool, syn: bool = True) -> None:
        value = 1 if down else 0
        if self._keys.get(code, 0) == value:
            return
        self._keys[code] = value
        self._emit(EV_KEY, code, value)
        if syn:
            self._syn()

    def _set_abs(self, code: int, value: int, syn: bool = True) -> None:
        lo, hi = ABS_INFO[code]
        value = int(clamp(value, lo, hi))
        if self._abs.get(code) == value:
            return
        self._abs[code] = value
        self._emit(EV_ABS, code, value)
        if syn:
            self._syn()

    # -- public controls -------------------------------------------------
    def press(self, button: str) -> None:
        self._ensure_alive()
        name = str(button).strip().upper()
        if name in ANALOG_BUTTONS:
            self._set_abs(ANALOG_BUTTONS[name], 255)
        else:
            self._set_key(self._code(name), True)

    def release(self, button: str) -> None:
        self._ensure_alive()
        name = str(button).strip().upper()
        if name in ANALOG_BUTTONS:
            self._set_abs(ANALOG_BUTTONS[name], 0)
        else:
            self._set_key(self._code(name), False)

    def _code(self, button: str) -> int:
        key = str(button).strip().upper()
        try:
            return BUTTON_CODES[key]
        except KeyError:
            raise ValueError(f"unknown button {button!r}; expected one of {CANONICAL_BUTTONS}") from None

    def set_stick(self, x: float, y: float) -> None:
        self._ensure_alive()
        ax, ay = stick_to_abs(x, y)
        self._set_abs(ABS_X, ax, syn=False)
        self._set_abs(ABS_Y, ay)

    def set_cstick(self, x: float, y: float) -> None:
        self._ensure_alive()
        ax, ay = stick_to_abs(x, y)
        self._set_abs(ABS_RX, ax, syn=False)
        self._set_abs(ABS_RY, ay)

    def set_trigger(self, value: float) -> None:
        """Analog right trigger (ZR): 0..1 -> ABS_RZ (no digital button)."""
        self._ensure_alive()
        self._set_abs(ABS_RZ, trigger_to_abs(clamp(value, 0.0, 1.0)))

    def release_all(self) -> None:
        """Neutral pad: no keys, sticks centered, triggers released, hat centered."""
        if self.fd is None and self.auto_create:
            return
        for code in KEY_CODES:
            self._set_key(code, False, syn=False)
        for code in ABS_INFO:
            self._set_abs(code, 0, syn=False)
        self._syn()

    def apply(self, action) -> None:
        """Apply a whole canonical action with a single SYN_REPORT at the end."""
        self._ensure_alive()
        act = decode_action(action)
        buttons = act.button_set()
        down_codes = {BUTTON_CODES[name] for name in buttons if name in BUTTON_CODES}
        # keys in canonical order (plus dpad up/down if supplied)
        for code in KEY_CODES:
            self._set_key(code, code in down_codes, syn=False)
        sx, sy = stick_to_abs(*act.stick)
        self._set_abs(ABS_X, sx, syn=False)
        self._set_abs(ABS_Y, sy, syn=False)
        cx, cy = stick_to_abs(*act.cstick)
        self._set_abs(ABS_RX, cx, syn=False)
        self._set_abs(ABS_RY, cy, syn=False)
        self._set_abs(ABS_Z, 255 if "ZL" in buttons else 0, syn=False)
        rz = 255 if "ZR" in buttons else trigger_to_abs(act.trigger)
        self._set_abs(ABS_RZ, rz, syn=False)
        # the standard 360 hat mirrors the d-pad buttons
        hat_x = hat_y = 0
        for code in down_codes:
            if code in DPAD_HAT:
                axis, value = DPAD_HAT[code]
                if axis == ABS_HAT0X:
                    hat_x = value
                else:
                    hat_y = value
        self._set_abs(ABS_HAT0X, hat_x, syn=False)
        self._set_abs(ABS_HAT0Y, hat_y, syn=False)
        self._syn()

    # -- introspection ---------------------------------------------------
    def describe(self) -> dict:
        return {
            "name": self.name,
            "event": self.event_path,
            "sysname": self.sysname,
            "vendor": f"0x{self.vendor:04x}",
            "product": f"0x{self.product:04x}",
            "version": f"0x{self.version:04x}",
            "bus": self.bus,
            "sdl_name": SDL_NAME,
            "guid": SDL_GUID,
            "expected_guid": EXPECTED_GUID,
            "buttons": list(CANONICAL_BUTTONS),
            "abs_ranges": {str(k): v for k, v in ABS_INFO.items()},
        }


def _demo(pad: VirtualPad, seconds: float = 0.0) -> None:
    """A short visual check: A, then right stick, then neutral."""
    pad.apply({"buttons": ["A"]})
    time.sleep(0.2)
    pad.apply({"buttons": [], "stick": (1.0, 0.0)})
    time.sleep(0.2)
    pad.apply({"buttons": [], "stick": (0.0, 1.0)})
    time.sleep(0.2)
    pad.release_all()
    if seconds:
        time.sleep(seconds)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SSBU raw uinput virtual pad")
    ap.add_argument("--test", action="store_true",
                    help="create the pad, print its event device, and exit")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="keep the pad alive this long (with --test)")
    ap.add_argument("--demo", action="store_true",
                    help="press a few controls while the pad is alive")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--name", default=DEVICE_NAME)
    args = ap.parse_args(argv)

    if not args.test:
        ap.print_help()
        return 2

    pad = VirtualPad(name=args.name)
    stop = {"now": False}

    def _on_signal(signum, frame):
        stop["now"] = True

    old = signal.signal(signal.SIGINT, _on_signal)
    try:
        pad.create()
        info = pad.describe()
        if args.json:
            import json
            print(json.dumps(info, indent=2))
        else:
            print(f"pad: {info['name']}")
            print(f"event device: {info['event']}")
            print(f"ids: {info['vendor']}:{info['product']} version {info['version']} bus {info['bus']}")
            print(f"expected SDL GUID: {info['expected_guid']}")
            print(f"pad lock: {PAD_LOCK}")
        if args.demo:
            _demo(pad)
        deadline = time.monotonic() + max(0.0, args.seconds)
        while not stop["now"] and time.monotonic() < deadline:
            time.sleep(0.02)
    except (OSError, RuntimeError) as e:
        print(f"gym_input: {e}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGINT, old)
        pad.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
