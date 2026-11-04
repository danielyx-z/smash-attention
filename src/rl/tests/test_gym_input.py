#!/usr/bin/env python3
"""Tests for control_block.py, gym_input.py and gym_reset.py.

    python3 tools/mem/test_gym_input.py                 # unit tests, no locks
    python3 tools/mem/test_gym_input.py --live           # + live pad-test
    python3 tools/mem/game_session.py run --owner D -- \
        python3 tools/mem/test_gym_input.py --live

The live test creates the virtual pad, runs
``build-linux/work/ssbu_native --pad-test`` and checks that SDL sees
"Xbox 360 Controller" at GUID 030049205e0400008e02000014010000 with a mapping,
and that every canonical button reaches exactly its Pro Controller button.
It must run under the game lock (game_session.py sets SSBU_GAME_LOCK_HELD=1).
If SDL reports no mapping for the pad, the test appends a [device <GUID>]
section to controllers.ini after backing it up.
"""

from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "ssbu-recomp"))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.input import control_block as cb  # noqa: E402
from rl.input import gym_input as gi  # noqa: E402
from rl.input import gym_reset as gr  # noqa: E402

CONTROLLERS_INI = os.path.join(REPO, "build-linux", "config", "controllers.ini")
BINARY = os.path.join(REPO, "build-linux", "work", "ssbu_native")
BACKUP_ROOT = "/tmp/opencode/ssbu_backups"


# ---------------------------------------------------------------------------
# helpers


class RecordingPad(gi.VirtualPad):
    """VirtualPad without /dev/uinput: records the events it would write."""

    def __init__(self):
        super().__init__(lock=False, auto_create=False)
        self.events: list[tuple[int, int, int]] = []

    def _ensure_alive(self) -> None:
        pass

    def _emit(self, type_: int, code: int, value: int) -> None:
        self.events.append((type_, code, value))

    def _syn(self) -> None:
        self.events.append((gi.EV_SYN, gi.SYN_REPORT, 0))

    def key_events(self) -> dict[int, int]:
        return {code: value for t, code, value in self.events if t == gi.EV_KEY}

    def abs_events(self) -> dict[int, int]:
        return {code: value for t, code, value in self.events if t == gi.EV_ABS}

    def clear(self) -> None:
        self.events.clear()


def _fresh() -> RecordingPad:
    return RecordingPad()


# ---------------------------------------------------------------------------
# control block


class ControlBlockTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ssbu_cb_", dir="/tmp/opencode")
        self.path = os.path.join(self.dir, "ssbu_control.bin")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_layout(self):
        self.assertEqual(cb.SIZE, 24)
        self.assertEqual(cb.MAGIC, b"SSBC")
        self.assertEqual(cb.VERSION, 1)
        data = cb.pack(reset_seq=0x11223344, ack_seq=0x55667788, state=2, pad=9)
        self.assertEqual(len(data), 24)
        self.assertEqual(struct.unpack_from("<I", data, 8)[0], 0x11223344)
        self.assertEqual(struct.unpack_from("<I", data, 12)[0], 0x55667788)
        self.assertEqual(struct.unpack_from("<I", data, 16)[0], 2)
        self.assertEqual(struct.unpack_from("<I", data, 20)[0], 9)
        snap = cb.unpack(data)
        self.assertEqual((snap.reset_seq, snap.ack_seq, snap.state, snap.pad),
                         (0x11223344, 0x55667788, 2, 9))
        self.assertFalse(snap.done)
        self.assertTrue(snap.pending)

    def test_bad_magic_and_version(self):
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(cb.pack(magic=b"XXXX"))
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(cb.pack(version=99))
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(b"SSBC")

    def test_file_roundtrip(self):
        with cb.create_control_block(self.path) as control:
            snap = control.read()
            self.assertEqual((snap.reset_seq, snap.ack_seq, snap.state), (0, 0, 0))
            seq = control.request_reset()
            self.assertEqual(seq, 1)
            snap = control.read()
            self.assertEqual((snap.reset_seq, snap.ack_seq, snap.state),
                             (1, 0, cb.STATE_REQUESTED))
            control.acknowledge(seq)
            done = control.wait_for_done(seq, timeout=0.5)
            self.assertEqual(done.ack_seq, 1)
            self.assertEqual(done.state, cb.STATE_DONE)
            self.assertTrue(done.done)
            self.assertEqual(control.request_reset(), 2)
            self.assertEqual(control.read().reset_seq, 2)

    def test_wait_ack_timeout(self):
        with cb.ControlBlock(self.path, create=True) as control:
            seq = control.request_reset()
            with self.assertRaises(cb.ControlBlockTimeout):
                control.wait_for_ack(seq, timeout=0.05, poll=0.005)

    def test_wait_ack_error(self):
        with cb.ControlBlock(self.path, create=True) as control:
            seq = control.request_reset()
            control.acknowledge(seq, state=cb.STATE_ERROR)
            with self.assertRaises(cb.ControlBlockError):
                control.wait_for_ack(seq, timeout=0.2)

    def test_helpers(self):
        cb.create_control_block(self.path).close()
        seq = cb.request_reset(self.path)
        self.assertEqual(seq, 1)
        with cb.ControlBlock(self.path, create=False) as control:
            control.acknowledge(seq)
        self.assertEqual(cb.wait_for_ack(seq, self.path, timeout=0.2).ack_seq, seq)
        self.assertEqual(cb.read_control(self.path).state, cb.STATE_DONE)

    def test_ack_ahead_is_accepted(self):
        with cb.ControlBlock(self.path, create=True) as control:
            for _ in range(3):
                control.request_reset()
            control.acknowledge(3)
            self.assertEqual(control.wait_for_ack(1, timeout=0.1).ack_seq, 3)


# ---------------------------------------------------------------------------
# action decoding / struct mapping


class ActionTests(unittest.TestCase):
    def test_canonical_order(self):
        self.assertEqual(gi.CANONICAL_BUTTONS,
                         ("A", "B", "X", "Y", "L", "R", "ZL", "ZR",
                          "PLUS", "MINUS", "DPAD_L", "DPAD_R"))

    def test_digital_button_codes(self):
        expected = {
            "A": gi.BTN_SOUTH, "B": gi.BTN_EAST,
            "X": gi.BTN_WEST, "Y": gi.BTN_NORTH,
            "L": gi.BTN_TL, "R": gi.BTN_TR,
            "PLUS": gi.BTN_START, "MINUS": gi.BTN_TL2,
            "DPAD_L": gi.BTN_DPAD_LEFT, "DPAD_R": gi.BTN_DPAD_RIGHT,
        }
        for name, code in expected.items():
            self.assertEqual(gi.BUTTON_CODES[name], code, name)
        self.assertEqual(gi.ANALOG_BUTTONS, {"ZL": gi.ABS_Z, "ZR": gi.ABS_RZ})

    def test_stick_conversion(self):
        self.assertEqual(gi.stick_to_abs(0.0, 0.0), (0, 0))
        self.assertEqual(gi.stick_to_abs(1.0, 1.0), (32767, -32767))
        self.assertEqual(gi.stick_to_abs(-1.0, -1.0), (-32767, 32767))
        self.assertEqual(gi.stick_to_abs(2.0, -2.0), (32767, 32767))  # clamped

    def test_trigger_conversion(self):
        self.assertEqual(gi.trigger_to_abs(0.0), 0)
        self.assertEqual(gi.trigger_to_abs(1.0), 255)
        self.assertEqual(gi.trigger_to_abs(0.5), 128)
        self.assertEqual(gi.trigger_to_abs(-1.0), 0)
        self.assertEqual(gi.trigger_to_abs(2.0), 255)

    def test_legacy_uinput_struct_size(self):
        # the pre-4.5 fallback: name, input_id, ff_effects_max, 4 x 64 ints
        data = gi._UINPUT_USER_DEV.pack(
            b"SSBU Virtual Pad", gi.BUS_USB, gi.VENDOR_ID, gi.PRODUCT_ID,
            gi.DEVICE_VERSION, 0,
            *([0] * 64), *([0] * 64), *([0] * 128))
        self.assertEqual(len(data), 1116)
        self.assertEqual(gi._UINPUT_SETUP.size, 92)
        self.assertEqual(gi._UINPUT_ABS_SETUP.size, 28)

    def test_decode_action_button_vector(self):
        # gym_action.decode() returns a 12-slot int8 vector in canonical order
        vec = [0] * 12
        vec[0] = 1   # A
        vec[4] = 1   # L
        a = gi.decode_action({"buttons": vec, "stick": (0.5, -0.5)})
        self.assertEqual(a.button_set(), {"A", "L"})
        try:
            import numpy as np
        except ImportError:
            return
        b = gi.decode_action({"buttons": np.array(vec, dtype=np.int8)})
        self.assertEqual(b.button_set(), {"A", "L"})
        pad = _fresh()
        pad.apply({"buttons": np.array(vec, dtype=np.int8)})
        self.assertEqual(pad.key_events(), {gi.BTN_SOUTH: 1, gi.BTN_TL: 1})

    def test_decode_action_forms(self):
        a = gi.decode_action({"buttons": ["X", "y"], "stick": (1, 0),
                              "cstick": (0, -1), "trigger": 0.5})
        self.assertEqual(a.button_set(), {"X", "Y"})
        self.assertEqual(a.stick, (1, 0))
        self.assertEqual(a.cstick, (0, -1))
        self.assertEqual(a.trigger, 0.5)

        vec = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
               0.5, 0.0, 0.0, -0.5, 1.0]
        b = gi.decode_action(vec)
        self.assertEqual(b.button_set(), {"A", "DPAD_R"})
        self.assertEqual(b.stick, (0.5, 0.0))
        self.assertEqual(b.cstick, (0.0, -0.5))
        self.assertEqual(b.trigger, 1.0)

        with self.assertRaises(ValueError):
            gi.decode_action({"buttons": ["NOPE"]}).button_set()
        with self.assertRaises(TypeError):
            gi.decode_action([1, 2, 3])

    def test_apply_each_digital_button(self):
        cases = {
            "A": gi.BTN_SOUTH, "B": gi.BTN_EAST, "X": gi.BTN_WEST,
            "Y": gi.BTN_NORTH, "L": gi.BTN_TL, "R": gi.BTN_TR,
            "PLUS": gi.BTN_START, "MINUS": gi.BTN_TL2,
        }
        for name, code in cases.items():
            pad = _fresh()
            pad.apply({"buttons": [name]})
            self.assertEqual(pad.key_events(), {code: 1}, name)
            self.assertEqual(pad.abs_events(), {}, f"{name} emitted abs")
            self.assertEqual(pad.events[-1], (gi.EV_SYN, gi.SYN_REPORT, 0))
        for name, code, axis, value in (
            ("DPAD_L", gi.BTN_DPAD_LEFT, gi.ABS_HAT0X, -1),
            ("DPAD_R", gi.BTN_DPAD_RIGHT, gi.ABS_HAT0X, 1),
        ):
            pad = _fresh()
            pad.apply({"buttons": [name]})
            self.assertEqual(pad.key_events(), {code: 1}, name)
            self.assertEqual(pad.abs_events(), {axis: value}, name)

    def test_apply_zl_zr_are_analog(self):
        pad = _fresh()
        pad.apply({"buttons": ["ZL"]})
        self.assertEqual(pad.abs_events(), {gi.ABS_Z: 255})
        self.assertEqual(pad.key_events(), {})
        pad = _fresh()
        pad.apply({"buttons": ["ZR"]})
        self.assertEqual(pad.abs_events(), {gi.ABS_RZ: 255})
        self.assertEqual(pad.key_events(), {})

    def test_apply_sticks_cstick_and_trigger(self):
        pad = _fresh()
        pad.apply({"stick": (1.0, 1.0), "cstick": (-1.0, 0.5),
                   "trigger": 0.5})
        self.assertEqual(pad.abs_events()[gi.ABS_X], 32767)
        self.assertEqual(pad.abs_events()[gi.ABS_Y], -32767)
        self.assertEqual(pad.abs_events()[gi.ABS_RX], -32767)
        self.assertEqual(pad.abs_events()[gi.ABS_RY], -16384)  # -round(0.5*32767)
        self.assertEqual(pad.abs_events()[gi.ABS_RZ], 128)
        self.assertEqual(pad.key_events(), {})

    def test_apply_dpad_also_sets_hat(self):
        for name, key, axis, value in (
            ("DPAD_L", gi.BTN_DPAD_LEFT, gi.ABS_HAT0X, -1),
            ("DPAD_R", gi.BTN_DPAD_RIGHT, gi.ABS_HAT0X, 1),
            ("DPAD_U", gi.BTN_DPAD_UP, gi.ABS_HAT0Y, -1),
            ("DPAD_D", gi.BTN_DPAD_DOWN, gi.ABS_HAT0Y, 1),
        ):
            pad = _fresh()
            pad.apply({"buttons": [name]})
            self.assertEqual(pad.key_events(), {key: 1}, name)
            self.assertEqual(pad.abs_events(), {axis: value}, name)

    def test_release_all(self):
        pad = _fresh()
        pad.apply({"buttons": ["A", "ZR"], "stick": (1, 1)})
        pad.clear()
        pad.release_all()
        # A was down (key -> 0); ZR was analog.  Only changed values emit.
        self.assertEqual(pad.key_events(), {gi.BTN_SOUTH: 0})
        self.assertEqual(pad.abs_events(), {
            gi.ABS_X: 0, gi.ABS_Y: 0, gi.ABS_RZ: 0,
        })

    def test_press_release_analog(self):
        pad = _fresh()
        pad.press("ZL")
        self.assertEqual(pad.abs_events(), {gi.ABS_Z: 255})
        pad.clear()
        pad.release("ZL")
        self.assertEqual(pad.abs_events(), {gi.ABS_Z: 0})


# ---------------------------------------------------------------------------
# reset client


class ResetTests(unittest.TestCase):
    def test_mock_reset(self):
        with gr.ResetClient(mock=True, timeout=1.0) as client:
            result = client.reset()
        self.assertTrue(result.ok, result)
        self.assertEqual(result.seq, 1)
        self.assertEqual(result.ack_seq, 1)
        self.assertEqual(result.state, cb.STATE_DONE)
        self.assertGreaterEqual(result.stable_frames, 3)
        self.assertIsNotNone(result.last)
        self.assertTrue(result.last.ready)

    def test_mock_reset_twice(self):
        with gr.ResetClient(mock=True, timeout=1.0) as client:
            first = client.reset()
            second = client.reset()
        self.assertEqual((first.seq, second.seq), (1, 2))

    def test_mock_reset_failure(self):
        with gr.ResetClient(mock=True, timeout=0.5, mock_fail=True) as client:
            with self.assertRaises(gr.ResetError):
                client.reset()

    def test_mock_reset_fight_on_timeout(self):
        client = gr.ResetClient(mock=True, timeout=0.15, stable_frames=100)
        with client:
            with self.assertRaises(gr.ResetTimeout):
                client.reset()

    def test_protocol_only(self):
        with gr.ResetClient(mock=True, state_reader=None, timeout=1.0) as client:
            result = client.reset()
        self.assertTrue(result.ok)
        self.assertEqual(result.stable_frames, 3)


class StateSampleTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ssbu_state_", dir="/tmp/opencode")
        self.path = os.path.join(self.dir, "ssbu_state.bin")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, seq: int, fight_on: int, match_state: int) -> None:
        data = bytearray(40)
        data[0:4] = b"SSBU"
        struct.pack_into("<I", data, 4, 7)
        struct.pack_into("<I", data, 8, 2)
        struct.pack_into("<I", data, 12, seq)
        struct.pack_into("<Q", data, 16, 123456789)
        struct.pack_into("<I", data, 32, fight_on)
        struct.pack_into("<I", data, 36, match_state)
        with open(self.path, "wb") as f:
            f.write(data)

    def test_read_state(self):
        self._write(7, 1, 5)
        sample = gr.read_state(self.path)
        self.assertIsNotNone(sample)
        self.assertEqual((sample.seq, sample.fight_on, sample.match_state),
                         (7, 1, 5))
        self.assertTrue(sample.ready)
        self.assertEqual(sample.ts_ns, 123456789)

    def test_read_state_missing_or_bad(self):
        self.assertIsNone(gr.read_state(self.path))
        with open(self.path, "wb") as f:
            f.write(b"nope" * 20)
        self.assertIsNone(gr.read_state(self.path))

    def test_wait_fight_on_stable(self):
        samples = [
            gr.StateSample(1, 0, 0, 0),
            gr.StateSample(2, 1, 1, 0),
            gr.StateSample(2, 1, 1, 0),   # repeated seq must not count twice
            gr.StateSample(3, 1, 1, 0),
            gr.StateSample(4, 1, 1, 0),
        ]

        def reader():
            return samples.pop(0) if samples else gr.StateSample(5, 1, 1, 0)

        last, frames, ready = gr.wait_fight_on(reader, timeout=0.5,
                                               stable_frames=3, poll=0)
        self.assertTrue(last.ready)
        self.assertEqual(ready, 3)
        self.assertGreaterEqual(frames, 4)

    def test_wait_fight_on_timeout(self):
        sample = gr.StateSample(1, 0, 0, 0)
        with self.assertRaises(gr.ResetTimeout):
            gr.wait_fight_on(lambda: sample, timeout=0.05, stable_frames=2,
                             poll=0.001)

    def test_wait_fight_on_no_reader(self):
        with self.assertRaises(gr.ResetTimeout):
            gr.wait_fight_on(lambda: None, timeout=0.05, stable_frames=1,
                             poll=0.001)


# ---------------------------------------------------------------------------
# controllers.ini device section


class ControllersIniTests(unittest.TestCase):
    def test_expected_device_section(self):
        if not os.path.exists(CONTROLLERS_INI):
            self.skipTest("controllers.ini not present")
        with open(CONTROLLERS_INI, "r", encoding="utf-8") as f:
            text = f.read()
        if f"[device {gi.SDL_GUID}]" not in text:
            self.skipTest("no virtual-pad [device] section")
        section = text.split(f"[device {gi.SDL_GUID}]", 1)[1].split("[", 1)[0]
        for key in ("map.A = button0", "map.B = button1", "map.X = button3",
                    "map.Y = button2", "map.ZL = lefttrigger",
                    "map.ZR = righttrigger", "map.PLUS = button9",
                    "map.MINUS = button6", "map.STICK_L = none"):
            self.assertIn(key, section)


# ---------------------------------------------------------------------------
# live test


def _romfs() -> str | None:
    env = os.environ.get("SSBU_ROMFS")
    if env:
        return env
    for cand in (os.path.join(REPO, "..", "dump", "romfs"),
                 "/home/daniel/Projects/SSBU/dump/romfs"):
        if os.path.isdir(cand):
            return cand
    return None


def _backup_controllers() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(BACKUP_ROOT, f"{stamp}_test_gym_input")
    os.makedirs(dest, exist_ok=True)
    shutil.copy2(CONTROLLERS_INI, os.path.join(dest, "controllers.ini"))
    return dest


def ensure_device_section(guid: str) -> bool:
    """Append a [device guid] section if it is missing.  Returns True if changed."""
    with open(CONTROLLERS_INI, "r", encoding="utf-8") as f:
        text = f.read()
    if f"[device {guid}]" in text:
        return False
    backup = _backup_controllers()
    section = f"""
; added by test_gym_input.py: SDL did not map the SSBU Virtual Pad
[device {guid}]
name = SSBU Virtual Pad
map.A = button0
map.B = button1
map.X = button3
map.Y = button2
map.L = button4
map.R = button5
map.ZL = axis2
map.ZR = axis5
map.PLUS = button9
map.MINUS = button6
map.UP = button10, hat0up
map.DOWN = button11, hat0down
map.LEFT = button12, hat0left
map.RIGHT = button13, hat0right
map.STICK_L = none
map.STICK_R = none
axis.LX = axis0
axis.LY = axis1
axis.RX = axis3
axis.RY = axis4
"""
    with open(CONTROLLERS_INI, "a", encoding="utf-8") as f:
        f.write(section)
    print(f"test_gym_input: no SDL mapping; backed up controllers.ini to {backup} "
          f"and appended [device {guid}]")
    return True


PRO_RE = re.compile(
    r"Pro buttons:(?P<buttons>.*?)\s+L\((?P<lx>-?\d+),(?P<ly>-?\d+)\) "
    r"R\((?P<rx>-?\d+),(?P<ry>-?\d+)\)")
GUID_RE = re.compile(r'GUID ([0-9a-fA-F]{32}).*?SDL mapping: (yes|NO)')


def run_pad_test(seconds: int = 14) -> str:
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":0")
    romfs = _romfs()
    if romfs:
        env.setdefault("SSBU_ROMFS", romfs)
    proc = subprocess.Popen(
        [BINARY, "--pad-test", "--seconds", str(seconds)],
        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    time.sleep(2.0)                 # tool_init sleeps 800 ms, then polls
    expected = ["A", "B", "X", "Y", "L", "R", "ZL", "ZR",
                "PLUS", "MINUS", "DPAD_L", "DPAD_R"]
    pad = gi.VirtualPad()
    outputs = []
    try:
        pad.create()
        for name in expected:
            outputs.append(("button", name))
            print(f"test_gym_input: press {name}", flush=True)
            pad.apply({"buttons": [name]})
            time.sleep(0.25)
            pad.apply({"buttons": []})
            time.sleep(0.12)
        for label, action in (
            ("stick right", {"stick": (1.0, 0.0)}),
            ("stick up", {"stick": (0.0, 1.0)}),
            ("stick left", {"stick": (-1.0, 0.0)}),
            ("stick down", {"stick": (0.0, -1.0)}),
            ("cstick right", {"cstick": (1.0, 0.0)}),
            ("cstick up", {"cstick": (0.0, 1.0)}),
            ("trigger", {"trigger": 1.0}),
        ):
            print(f"test_gym_input: {label}", flush=True)
            pad.apply(action)
            time.sleep(0.3)
            pad.release_all()
            time.sleep(0.12)
    finally:
        pad.destroy()
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
    return out


class LivePadTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("SSBU_LIVE") != "1":
            raise unittest.SkipTest("set SSBU_LIVE=1 or pass --live")
        if os.environ.get("SSBU_GAME_LOCK_HELD") != "1":
            raise unittest.SkipTest(
                "live test needs the game lock: run it through "
                "tools/mem/game_session.py")
        if not os.path.exists(BINARY):
            raise unittest.SkipTest(f"{BINARY} not built")
        if not os.path.exists("/dev/uinput"):
            raise unittest.SkipTest("/dev/uinput not present")
        cls.output = run_pad_test()
        # If SDL has no mapping, add an explicit [device GUID] section and retry.
        matches = GUID_RE.findall(cls.output)
        if matches and matches[0][1] == "NO":
            if ensure_device_section(matches[0][0]):
                cls.output = run_pad_test()
        print("\n----- pad-test output -----")
        print(cls.output)
        print("----- end pad-test output -----")

    def test_sdl_sees_mapped_xbox360(self):
        matches = GUID_RE.findall(self.output)
        self.assertTrue(matches, "no GUID / SDL mapping line in pad-test output")
        guid, mapped = matches[0]
        self.assertEqual(guid.lower(), gi.SDL_GUID)
        self.assertEqual(mapped, "yes",
                         "SDL did not map the virtual pad; see ensure_device_section")
        self.assertIn("Xbox 360 Controller", self.output)

    def test_every_canonical_button_reaches_the_pro_pad(self):
        rows = PRO_RE.findall(self.output)
        self.assertTrue(rows, "pad-test printed no Pro button states")
        observed = set()
        for buttons, *_ in rows:
            observed.update(buttons.split())
        expected = {"A", "B", "X", "Y", "L", "R", "ZL", "ZR",
                    "PLUS", "MINUS", "LEFT", "RIGHT"}
        self.assertEqual(observed, expected,
                         f"observed {sorted(observed)} != {sorted(expected)}; "
                         f"spurious {sorted(observed - expected)}")

    def test_sticks_and_trigger(self):
        vectors = [(int(a), int(b), int(c), int(d))
                   for *_x, a, b, c, d in PRO_RE.findall(self.output)]
        self.assertIn((32767, 0, 0, 0), vectors, "left stick right")
        self.assertIn((0, 32767, 0, 0), vectors, "left stick up")
        self.assertIn((-32767, 0, 0, 0), vectors, "left stick left")
        self.assertIn((0, -32767, 0, 0), vectors, "left stick down")
        self.assertIn((0, 0, 32767, 0), vectors, "c-stick right")
        self.assertIn((0, 0, 0, 32767), vectors, "c-stick up")

    def test_no_unmapped_device(self):
        self.assertNotIn("SDL mapping: NO", self.output)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    live = "--live" in argv
    if live:
        argv.remove("--live")
        os.environ["SSBU_LIVE"] = "1"
        if os.environ.get("SSBU_GAME_LOCK_HELD") != "1":
            print("test_gym_input: --live must run under the game lock, e.g.\n"
                  "  python3 tools/mem/game_session.py run --owner D -- "
                  "python3 tools/mem/test_gym_input.py --live",
                  file=sys.stderr)
            return 2
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
