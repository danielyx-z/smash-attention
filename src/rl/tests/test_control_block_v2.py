#!/usr/bin/env python3
"""Tests for the v2 (80-byte) control block and v1 compatibility.

    .venv/bin/python -m pytest src/rl/tests/test_control_block_v2.py -q

No game and no live shared memory: every test uses a tempfile path.  A small
fake-runtime thread plays the runtime side of the command protocol by patching
cmd_ack in the same file.
"""

from __future__ import annotations

import os
import shutil
import stat
import struct
import sys
import tempfile
import threading
import time
import unittest
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.input import control_block as cb  # noqa: E402
from rl.input import gym_reset as gr  # noqa: E402


class FakeCommandRuntime(threading.Thread):
    """Acknowledge the next v2 command by patching the block in place."""

    def __init__(self, path: str, *, error: bool = False,
                 timeout: float = 2.0):
        super().__init__(daemon=True)
        self.path = path
        self.error = error
        self.timeout = timeout
        self.snapshot = None
        self._halt = threading.Event()

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        deadline = time.monotonic() + self.timeout
        while not self._halt.is_set() and time.monotonic() < deadline:
            try:
                with open(self.path, "rb") as f:
                    data = f.read()
            except OSError:
                time.sleep(0.002)
                continue
            if len(data) < cb.SIZE_V2:
                time.sleep(0.002)
                continue
            snap = cb.unpack(data)
            if snap.cmd_seq == 0 or snap.cmd_ack == snap.cmd_seq:
                time.sleep(0.002)
                continue
            self.snapshot = snap
            out = replace(snap, cmd_ack=snap.cmd_seq,
                          ack_frame=snap.game_frame + 1)
            if self.error:
                out = replace(out, state=cb.STATE_ERROR)
            fd = os.open(self.path, os.O_RDWR)
            try:
                os.pwrite(fd, cb.pack_snapshot(out), 0)
            finally:
                os.close(fd)
            # keep serving further commands until stopped or timed out
            time.sleep(0.002)


class ControlBlockV2Tests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="ssbu_cb_v2_", dir="/tmp/opencode")
        self.path = os.path.join(self.dir, "ssbu_control.bin")
        self._threads: list[FakeCommandRuntime] = []

    def tearDown(self):
        for thread in self._threads:
            thread.stop()
            thread.join(timeout=1.0)
        shutil.rmtree(self.dir, ignore_errors=True)

    def _start_runtime(self, **kwargs) -> FakeCommandRuntime:
        thread = FakeCommandRuntime(self.path, **kwargs)
        thread.start()
        self._threads.append(thread)
        return thread

    # -- layout / packing ------------------------------------------------
    def test_sizes_and_offsets(self):
        self.assertEqual((cb.SIZE_V1, cb.SIZE_V2), (24, 80))
        self.assertEqual(cb.SIZE_V2, 80)
        self.assertEqual(cb.VERSION_V1, 1)
        self.assertEqual(cb.VERSION_V2, 2)
        self.assertEqual(cb.CURRENT_VERSION, 2)
        # Legacy names still point at the v1 struct used by pack().
        self.assertEqual((cb.SIZE, cb.VERSION), (24, 1))
        data = cb.pack_v2()
        self.assertEqual(len(data), 80)
        for offset in (4, 8, 12, 16, 20, 24, 36, 40, 56, 60, 68, 76):
            self.assertEqual(len(data[offset:offset + 4]), 4)

    def test_v2_roundtrip(self):
        data = cb.pack_v2(
            reset_seq=0x11223344, ack_seq=0x55667788, state=cb.STATE_RUNNING,
            flags=cb.FLAG_FIGHT_ON | cb.FLAG_TURBO | cb.FLAG_HASH_VALID,
            game_frame=1234, state_hash=0xDEADBEEF, seed=99, cmd_seq=7,
            cmd=cb.CMD_SET_PAD, arg0=1200, arg1=0b101, arg2=cb.pack_stick(-3, 4),
            arg3=cb.pack_stick(5, -6), cmd_ack=6, ack_frame=1199,
            speed_milli=1000)
        self.assertEqual(len(data), 80)
        self.assertEqual(struct.unpack_from("<I", data, 4)[0], 2)
        self.assertEqual(struct.unpack_from("<I", data, 24)[0], 1234)
        self.assertEqual(struct.unpack_from("<I", data, 68)[0], 1000)
        snap = cb.unpack(data)
        self.assertEqual(snap.magic, cb.MAGIC)
        self.assertEqual(snap.version, 2)
        self.assertEqual((snap.reset_seq, snap.ack_seq, snap.state),
                         (0x11223344, 0x55667788, cb.STATE_RUNNING))
        self.assertEqual(snap.flags,
                         cb.FLAG_FIGHT_ON | cb.FLAG_TURBO | cb.FLAG_HASH_VALID)
        self.assertTrue(snap.fight_on)
        self.assertTrue(snap.turbo)
        self.assertTrue(snap.hash_valid)
        self.assertFalse(snap.reset_busy)
        self.assertEqual((snap.game_frame, snap.state_hash, snap.seed),
                         (1234, 0xDEADBEEF, 99))
        self.assertEqual((snap.cmd_seq, snap.cmd, snap.cmd_ack),
                         (7, cb.CMD_SET_PAD, 6))
        self.assertEqual(snap.cmd_name, "set_pad")
        self.assertEqual((snap.arg0, snap.arg1), (1200, 0b101))
        self.assertEqual(snap.stick, (-3, 4))
        self.assertEqual(snap.cstick, (5, -6))
        self.assertEqual((snap.ack_frame, snap.speed_milli), (1199, 1000))
        self.assertEqual(snap.pad, 0)  # v1 slot is flags in v2
        self.assertTrue(snap.cmd_pending)

    def test_v1_read(self):
        data = cb.pack(reset_seq=3, ack_seq=2, state=cb.STATE_REQUESTED, pad=9)
        self.assertEqual(len(data), 24)
        snap = cb.unpack(data)
        self.assertEqual(snap.version, 1)
        self.assertEqual((snap.reset_seq, snap.ack_seq, snap.state, snap.pad),
                         (3, 2, cb.STATE_REQUESTED, 9))
        # every v2 field defaults to zero
        for name in ("flags", "game_frame", "state_hash", "seed", "cmd_seq",
                     "cmd", "arg0", "arg1", "arg2", "arg3", "cmd_ack",
                     "ack_frame", "speed_milli", "pad0", "pad1"):
            self.assertEqual(getattr(snap, name), 0, name)

    def test_v1_file_read_and_upgrade(self):
        with open(self.path, "wb") as f:
            f.write(cb.pack(reset_seq=4, ack_seq=4, state=cb.STATE_DONE))
        self.assertEqual(os.path.getsize(self.path), 24)
        with cb.ControlBlock(self.path, create=False) as control:
            snap = control.read()
            self.assertEqual((snap.version, snap.reset_seq, snap.ack_seq),
                             (1, 4, 4))
            self.assertEqual(snap.state_hash, 0)
            # A reset write upgrades the file to the full v2 layout.
            seq = control.request_reset()
            self.assertEqual(seq, 5)
        self.assertEqual(os.path.getsize(self.path), 80)
        snap = cb.read_control(self.path)
        self.assertEqual((snap.version, snap.reset_seq, snap.state),
                         (2, 5, cb.STATE_REQUESTED))

    def test_version_and_magic_rejection(self):
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(cb.pack_v2(version=3))
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(cb.pack(version=99))
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(cb.pack_v2(magic=b"XXXX"))
        with self.assertRaises(cb.ControlBlockError):
            cb.unpack(b"SSBC")
        # version 1 is accepted in either block size
        self.assertEqual(cb.unpack(cb.pack(version=1)).version, 1)
        self.assertEqual(cb.unpack(cb.pack_v2(version=1)).version, 1)

    # -- file lifecycle --------------------------------------------------
    def test_create_writes_80_byte_v2_block(self):
        with cb.create_control_block(self.path) as control:
            self.assertEqual(os.path.getsize(self.path), 80)
            snap = control.read()
            self.assertEqual((snap.magic, snap.version, snap.state),
                             (cb.MAGIC, 2, cb.STATE_IDLE))
            self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o666)

    def test_partial_file_is_reinitialized_when_creating(self):
        with open(self.path, "wb") as f:
            f.write(b"SSBC")
        with cb.ControlBlock(self.path, create=True) as control:
            self.assertEqual(os.path.getsize(self.path), 80)
            self.assertEqual(control.read().version, 2)
        with open(self.path, "wb") as f:
            f.write(b"SSBC")
        with self.assertRaises(cb.ControlBlockError):
            cb.ControlBlock(self.path, create=False).open()

    def test_reset_on_80_byte_file(self):
        with cb.create_control_block(self.path) as control:
            seq = control.request_reset()
            self.assertEqual(seq, 1)
            snap = control.read()
            self.assertEqual((snap.reset_seq, snap.ack_seq, snap.state),
                             (1, 0, cb.STATE_REQUESTED))
            self.assertTrue(snap.pending)
            self.assertEqual(os.path.getsize(self.path), 80)
            # runtime: running, then done
            control.write_snapshot(replace(control.read(),
                                           state=cb.STATE_RUNNING,
                                           ack_seq=seq))
            control.write_snapshot(replace(control.read(),
                                           state=cb.STATE_DONE))
            done = control.wait_for_done(seq, timeout=0.5)
            self.assertTrue(done.done)
            self.assertEqual(done.ack_seq, 1)
            self.assertEqual(control.request_reset(), 2)

    # -- commands --------------------------------------------------------
    def test_command_ack(self):
        runtime = self._start_runtime()
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            snap = control.send_command(cb.CMD_SET_SEED, arg0=1234,
                                        timeout=1.0, poll=0.002)
            self.assertEqual(snap.cmd_seq, 1)
            self.assertEqual((snap.cmd, snap.arg0), (cb.CMD_SET_SEED, 1234))
            self.assertEqual(snap.cmd_ack, 1)
            self.assertFalse(snap.cmd_pending)
            self.assertEqual(snap.ack_frame, 1)
            self.assertIsNotNone(runtime.snapshot)
            self.assertEqual(runtime.snapshot.cmd, cb.CMD_SET_SEED)

    def test_command_timeout(self):
        with cb.ControlBlock(self.path, timeout=0.5) as control:
            with self.assertRaises(cb.ControlBlockTimeout):
                control.send_command(cb.CMD_CLEAR_PAD, timeout=0.05,
                                     poll=0.005)
            snap = control.read()
            # the command was still written for the runtime to pick up
            self.assertEqual(snap.cmd, cb.CMD_CLEAR_PAD)
            self.assertEqual(snap.cmd_seq, 1)
            self.assertEqual(snap.cmd_ack, 0)

    def test_command_error_state(self):
        self._start_runtime(error=True)
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            with self.assertRaises(cb.ControlBlockError):
                control.send_command(cb.CMD_SET_SEED, arg0=7, timeout=1.0)

    def test_send_command_rejects_none(self):
        with cb.ControlBlock(self.path) as control:
            with self.assertRaises(ValueError):
                control.send_command(cb.CMD_NONE)

    def test_set_seed_and_clear_pad(self):
        self._start_runtime(timeout=3.0)
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            seed_snap = control.set_seed(0xCAFEBABE, timeout=1.0)
            self.assertEqual((seed_snap.cmd, seed_snap.arg0),
                             (cb.CMD_SET_SEED, 0xCAFEBABE))
            self.assertEqual(seed_snap.cmd_seq, 1)
            clear_snap = control.clear_pad(timeout=1.0)
            self.assertEqual(clear_snap.cmd, cb.CMD_CLEAR_PAD)
            self.assertEqual(clear_snap.cmd_seq, 2)
            self.assertEqual(clear_snap.cmd_ack, 2)

    def test_set_pad_args(self):
        runtime = self._start_runtime()
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            snap = control.set_pad(0b101, stick=(-100, 200),
                                   cstick=(300, -400), frame=42,
                                   timeout=1.0)
            self.assertEqual(snap.cmd, cb.CMD_SET_PAD)
            self.assertEqual(snap.arg0, 42)
            self.assertEqual(snap.arg1, 0b101)
            self.assertEqual(snap.arg2, cb.pack_stick(-100, 200))
            self.assertEqual(snap.arg3, cb.pack_stick(300, -400))
            self.assertEqual(snap.stick, (-100, 200))
            self.assertEqual(snap.cstick, (300, -400))
            self.assertEqual(runtime.snapshot.arg0, 42)

    # -- XInput buttons / triggers --------------------------------------
    def test_encode_buttons(self):
        self.assertEqual(cb.encode_buttons(0x1000), (0x1000, 0, 0))
        self.assertEqual(cb.encode_buttons([]), (0, 0, 0))
        self.assertEqual(cb.encode_buttons(["A", "R"]),
                         (cb.XINPUT_A | cb.XINPUT_RB, 0, 0))
        self.assertEqual(cb.encode_buttons("plus"), (cb.XINPUT_START, 0, 0))
        self.assertEqual(cb.encode_buttons(["ZL"]), (0, 255, 0))
        self.assertEqual(cb.encode_buttons(["zl", "ZR"]), (0, 255, 255))
        self.assertEqual(cb.encode_buttons([cb.XINPUT_X, "B"]),
                         (cb.XINPUT_X | cb.XINPUT_B, 0, 0))
        with self.assertRaises(ValueError):
            cb.encode_buttons(["NOPE"])

    def test_trigger_helpers(self):
        self.assertEqual(cb.trigger_byte(0.0), 0)
        self.assertEqual(cb.trigger_byte(1.0), 255)
        self.assertEqual(cb.trigger_byte(0.5), 128)
        self.assertEqual(cb.trigger_byte(2.0), 255)
        self.assertEqual(cb.trigger_byte(-1.0), 0)
        self.assertEqual(cb.trigger_byte(128), 128)
        self.assertEqual(cb.trigger_byte(300), 255)
        self.assertEqual(cb.pack_triggers(0x12, 0x34), 0x3412)
        self.assertEqual(cb.unpack_triggers(0x3412), (0x12, 0x34))

    def test_set_pad_button_names(self):
        self._start_runtime(timeout=3.0)
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            snap = control.set_pad(["A", "B", "L", "PLUS", "DPAD_L"],
                                   frame=7, timeout=1.0)
            expected = (cb.XINPUT_A | cb.XINPUT_B | cb.XINPUT_LB |
                        cb.XINPUT_START | cb.XINPUT_DPAD_LEFT)
            self.assertEqual(snap.cmd, cb.CMD_SET_PAD)
            self.assertEqual(snap.arg1, expected)
            self.assertEqual(snap.arg1, 0x3114)
            self.assertEqual(snap.pad0, 0)
            self.assertEqual(snap.triggers, (0, 0))

    def test_set_pad_trigger_names_and_values(self):
        self._start_runtime(timeout=3.0)
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            snap = control.set_pad(["ZL", "ZR"], timeout=1.0)
            self.assertEqual(snap.arg1, 0)
            self.assertEqual(snap.pad0, 0xFFFF)
            self.assertEqual(snap.triggers, (255, 255))
            self.assertEqual((snap.lt, snap.rt), (255, 255))

            snap = control.set_pad("A", trigger=0.5, timeout=1.0)
            self.assertEqual(snap.arg1, cb.XINPUT_A)
            self.assertEqual(snap.pad0, 128 << 8)
            self.assertEqual(snap.triggers, (0, 128))

            snap = control.set_pad(0, triggers=(1.0, 0.0), timeout=1.0)
            self.assertEqual(snap.pad0, 255)
            self.assertEqual(snap.triggers, (255, 0))

            # explicit triggers override the ZL/ZR names
            snap = control.set_pad(["ZR"], triggers=(0.0, 0.0), timeout=1.0)
            self.assertEqual(snap.pad0, 0)

    def test_commands_preserve_pad0(self):
        self._start_runtime()
        with cb.ControlBlock(self.path, timeout=1.0) as control:
            control.write_snapshot(replace(control.read(), pad0=0x1234))
            snap = control.set_seed(5, timeout=1.0)
            self.assertEqual(snap.pad0, 0x1234)
            self.assertEqual(snap.triggers, (0x34, 0x12))

    # -- stick packing ---------------------------------------------------
    def test_stick_packing(self):
        self.assertEqual(cb.pack_stick(0, 0), 0)
        self.assertEqual(cb.unpack_stick(0), (0, 0))
        # positive x in the low half, negative y in the high half
        self.assertEqual(cb.pack_stick(1, -1), 0xFFFF0001)
        self.assertEqual(cb.unpack_stick(0xFFFF0001), (1, -1))
        # both negative
        self.assertEqual(cb.pack_stick(-2, -3), 0xFFFDFFFE)
        self.assertEqual(cb.unpack_stick(0xFFFDFFFE), (-2, -3))
        self.assertEqual(cb.pack_stick(32767, -32768), 0x80007FFF)
        self.assertEqual(cb.unpack_stick(0x80007FFF), (32767, -32768))
        # out-of-range values are clamped, not wrapped
        self.assertEqual(cb.pack_stick(40000, -40000),
                         cb.pack_stick(32767, -32768))
        for x, y in ((-32768, 0), (0, 32767), (-1, 1), (1234, -5678)):
            self.assertEqual(cb.unpack_stick(cb.pack_stick(x, y)), (x, y))

    # -- reset client compatibility --------------------------------------
    def test_gym_reset_mock_snapshot_shape(self):
        with gr.ResetClient(mock=True, timeout=1.0) as client:
            result = client.reset()
        self.assertTrue(result.ok, result)
        self.assertEqual(result.state, cb.STATE_DONE)
        self.assertEqual(result.ack_seq, 1)


if __name__ == "__main__":
    unittest.main()
