"""Lockstep stepping: gate client, reader sampling and the env, against a fake
runtime that follows the C gate semantics (no game needed)."""
from __future__ import annotations

import os
import struct
import tempfile
import threading
import time
import unittest

import numpy as np

from rl.env import gym_env
from rl.input import gate as gate_mod
from rl.input.gate import FrameGate, GateError, NOT_BLOCKED
from rl.memory.reader import gated_sample

FRAME_OFF, BLOCKED_OFF = 24, 32


def _new_gate_file():
    fd, path = tempfile.mkstemp(prefix="ssbu_gate_", suffix=".bin")
    os.write(fd, struct.pack("<4sIIIQQQII", b"SGAT", 1, 0, 4242, 0, 0,
                             NOT_BLOCKED, 0, 0))
    os.close(fd)
    return path


class FakeRuntime:
    """The game thread (park logic of rl_gate_frame) plus the reader.

    Every tick the "game" runs one frame unless parked.  State of frame F is
    ``state_fn(F)``; a paused game keeps returning the frame it paused at.
    """

    def __init__(self, gate_path, fixture, state_fn=None, tick=0.0003):
        self.path = gate_path
        self.fix = fixture
        self.frame = 100
        self.tick = tick
        self.paused = False
        self._shown = self.frame
        self.state_fn = state_fn or self._default_state
        self._mm = FrameGate(gate_path).open()._mm
        self._client = FrameGate(gate_path).open()
        self._stop = threading.Event()
        self.rematches = []
        self.reader_state: dict = {}
        self.published = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _default_state(frame):
        return [dict(x=float(frame), y=0.0, action_state=0, action_frame=float(frame),
                     on_ground=1, is_actionable=1, is_on_rebirth=0),
                dict(x=-float(frame), y=0.0, action_state=0, action_frame=float(frame),
                     on_ground=1, is_actionable=1, is_on_rebirth=0)]

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2)

    def _gate(self):
        return self._client.read()

    def rematch(self, timeout=None):                 # what rl_poll does at a reset
        st = self._gate()
        flags = struct.unpack_from("<I", self._mm, 40)[0]
        seed = struct.unpack_from("<I", self._mm, 44)[0]
        self.rematches.append((flags, seed))
        if flags & 1:
            k = (flags >> 8) & 0xFF or 1
            struct.pack_into("<Q", self._mm, 16, self.frame + k)
            struct.pack_into("<I", self._mm, 8, 1)
            struct.pack_into("<I", self._mm, 40, flags & ~3)

    def _read(self):
        return self.frame

    def _publish(self, tag, frame):
        shown = self._shown if self.paused else frame
        for slot, fields in enumerate(self.state_fn(shown)):
            self.fix.set_fighter(slot, **fields)
        self.fix.publish(game_frame=tag)
        self.published.append(tag)

    def _loop(self):
        while not self._stop.is_set():
            st = self._gate()
            parked = st.enable and self.frame >= st.limit
            if not parked:
                self.frame += 1
                if not self.paused:
                    self._shown = self.frame
                struct.pack_into("<Q", self._mm, FRAME_OFF, self.frame)
                st = self._gate()
                if st.enable and self.frame >= st.limit:
                    struct.pack_into("<Q", self._mm, BLOCKED_OFF, self.frame)
                else:
                    struct.pack_into("<Q", self._mm, BLOCKED_OFF, NOT_BLOCKED)
            else:
                struct.pack_into("<Q", self._mm, BLOCKED_OFF, self.frame)
            if not gated_sample(self._client, self.reader_state,
                                self._read, self._publish):
                self.fix.publish()                  # free-running reader
            time.sleep(self.tick)


class _Adapter:
    name = "fake"

    def __init__(self, runtime=None):
        self.actions = []
        self.runtime = runtime

    def apply(self, action):
        self.actions.append(action)
        buttons = np.asarray(action["buttons"])
        if self.runtime is not None and buttons[8]:        # PLUS toggles pause
            self.runtime.paused = not self.runtime.paused

    def close(self):
        pass


class GateClientTest(unittest.TestCase):
    def setUp(self):
        self.path = _new_gate_file()
        self.fix = gym_env.ShmFixture()
        self.rt = FakeRuntime(self.path, self.fix).start()
        self.gate = FrameGate(self.path).open()

    def tearDown(self):
        self.gate.release()
        self.rt.stop()
        self.gate.close()
        self.fix.close()
        os.unlink(self.path)

    def test_hold_parks_and_advance_is_exact(self):
        f = self.gate.hold(timeout=2)
        for _ in range(30):
            target = self.gate.advance(2)
            self.assertEqual(self.gate.wait_parked(2, at=target), target)
            self.assertEqual(target, f + 2)
            f = target
        self.assertEqual(self.gate.read().blocked, f)

    def test_advance_requires_a_parked_game(self):
        with self.assertRaises(GateError):
            self.gate.advance(1)

    def test_release_lets_the_game_run(self):
        f = self.gate.hold(timeout=2)
        self.gate.release()
        time.sleep(0.05)
        st = self.gate.read()
        self.assertGreater(st.frame, f + 5)
        self.assertFalse(st.parked)

    def test_arm_reset_writes_flags_and_seed(self):
        self.gate.arm_reset(park_after=3, seed=77)
        mm = self.gate._mm
        flags = struct.unpack_from("<I", mm, 40)[0]
        self.assertEqual(flags & 3, 3)
        self.assertEqual((flags >> 8) & 0xFF, 3)
        self.assertEqual(struct.unpack_from("<I", mm, 44)[0], 77)

    def test_wait_parked_times_out_without_a_parked_game(self):
        with self.assertRaises(GateError):
            self.gate.wait_parked(timeout=0.05)

    def test_open_fails_clearly_without_a_gate(self):
        with self.assertRaises(GateError):
            FrameGate("/nonexistent/ssbu_gate.bin").open(timeout=0.0)


class ReaderSamplingTest(unittest.TestCase):
    def test_one_tagged_publish_per_parked_frame_and_race_is_retried(self):
        path = _new_gate_file()
        try:
            gate = FrameGate(path).open()
            mm = gate._mm
            state, out = {}, []

            def park(frame):
                struct.pack_into("<I", mm, 8, 1)
                struct.pack_into("<Q", mm, 16, frame)
                struct.pack_into("<Q", mm, 32, frame)

            park(10)
            for _ in range(3):      # polled repeatedly: published only once
                gated_sample(gate, state, lambda: "data", lambda t, d: out.append(t))
            self.assertEqual(out, [10])
            park(12)
            gated_sample(gate, state, lambda: "data", lambda t, d: out.append(t))
            self.assertEqual(out, [10, 12])

            # the game moves on while the memory is being read: no publish
            def racing_read():
                struct.pack_into("<Q", mm, 32, NOT_BLOCKED)
                return "torn"
            park(14)
            gated_sample(gate, state, racing_read, lambda t, d: out.append(t))
            self.assertEqual(out, [10, 12])
            # gate off: the caller runs the timer path
            struct.pack_into("<I", mm, 8, 0)
            self.assertFalse(gated_sample(gate, state, lambda: 1, lambda t, d: 1))
            gate.close()
        finally:
            os.unlink(path)


class LockstepEnvTest(unittest.TestCase):
    def _env(self, state_fn=None, paused=False, frame_skip=2):
        self.path = _new_gate_file()
        self.fix = gym_env.ShmFixture()
        self.rt = FakeRuntime(self.path, self.fix, state_fn)
        self.rt.paused = paused
        self.adapter = _Adapter(self.rt)
        self.rt.start()
        env = gym_env.SSBUGymEnv(
            shm_path=self.fix.path, frame_skip=frame_skip, max_steps=10_000,
            reward_mode="none", stock_termination=False,
            require_positive_stock=False, input_adapter=self.adapter,
            frame_timeout=3.0, reset_timeout=5.0)
        env.enable_lockstep(seed=99, gate_path=self.path)
        env._lock_reset = type("R", (), {"reset": staticmethod(
            lambda timeout=None: self.rt.rematch())})()
        self.env = env
        return env

    def tearDown(self):
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
        rt = getattr(self, "rt", None)
        if rt is not None:
            rt.stop()
        fix = getattr(self, "fix", None)
        if fix is not None:
            fix.close()
        path = getattr(self, "path", None)
        if path and os.path.exists(path):
            os.unlink(path)

    def _idle(self):
        return {"buttons": np.zeros(12, np.float32),
                "stick": np.zeros(2, np.float32),
                "cstick": np.zeros(2, np.float32), "trigger": 0.0}

    def test_every_step_advances_exactly_frame_skip_frames(self):
        for skip in (1, 2, 5):
            self.tearDown()
            env = self._env(frame_skip=skip)
            env.reset()
            last = env._last_snapshot.game_frame
            for _ in range(25):
                _o, _r, term, trunc, info = env.step(self._idle())
                self.assertEqual(info["frames_advanced"], skip)
                self.assertEqual(info["game_frame"] - last, skip)
                last = info["game_frame"]
                self.assertFalse(term or trunc)

    def test_observation_belongs_to_the_tagged_frame(self):
        env = self._env()
        env.reset()
        for _ in range(10):
            _o, _r, _t, _u, info = env.step(self._idle())
            # the fake publishes x == the frame number it was parked at
            self.assertEqual(info["raw_fighters"][0]["x"],
                             float(info["game_frame"]))

    def test_a_slow_trainer_changes_nothing(self):
        env = self._env()
        env.reset()
        a = []
        for i in range(12):
            if i % 4 == 3:
                time.sleep(0.2)                    # PPO update / GPU busy
            _o, _r, _t, _u, info = env.step(self._idle())
            a.append(info["game_frame"] - env._last_snapshot.game_frame
                     + info["frames_advanced"])
        self.assertEqual(set(a), {2})

    def test_reset_arms_the_gate_with_seed_and_parks_deterministically(self):
        env = self._env()
        env.reset()
        flags, seed = self.rt.rematches[-1]
        self.assertEqual(flags & 3, 3)
        self.assertEqual(seed, 99)
        _o, info = env.reset()
        self.assertTrue(info["lockstep"])

    def test_reset_waits_frame_by_frame_for_a_clean_start(self):
        # the ego is on the revival platform for 6 frames after the reset
        def state(frame, _start=[None]):
            base = FakeRuntime._default_state(frame)
            if _start[0] is None:
                _start[0] = frame
            if frame - _start[0] < 6:
                base[0]["is_on_rebirth"] = 1
                base[0]["on_ground"] = 0
            return base
        env = self._env(state_fn=state)
        _o, info = env.reset()
        self.assertGreaterEqual(info["reset_frames_advanced"], 4)

    def test_a_paused_game_is_resumed_with_plus_at_reset(self):
        env = self._env(paused=True)
        _o, info = env.reset()
        pressed = [a for a in self.adapter.actions if np.asarray(a["buttons"])[8]]
        self.assertGreaterEqual(len(pressed), 1)
        self.assertFalse(self.rt.paused)

    def test_close_releases_the_game(self):
        env = self._env()
        env.reset()
        env.step(self._idle())
        env.close()
        self.assertEqual(FrameGate(self.path).open().read().enable, 0)
        self.env = None

    def test_inputs_are_neutralised_before_a_reset(self):
        env = self._env()
        env.reset()
        held = self._idle()
        held["stick"] = np.array([1.0, 0.0], np.float32)
        env.step(held)
        n = len(self.adapter.actions)
        env.reset()
        first = self.adapter.actions[n]
        self.assertEqual(float(np.asarray(first["stick"])[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
