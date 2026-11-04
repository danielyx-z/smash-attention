#!/usr/bin/env python3
"""Unit tests for the SSBU gym environment (no game, no locks, no sudo).

Run with either of::

    .venv/bin/python tools/mem/test_gym_env.py -v
    .venv/bin/python -m unittest discover -s tools/mem -p 'test_gym_env.py' -v

Everything runs against :class:`gym_env.ShmFixture`, an in-process mock of
``/dev/shm/ssbu_state.bin`` that speaks the same v7 layout and odd/even seq
protocol.  ``gym_input``/``gym_reset`` are injected with a recorder and a no-op
so the tests never touch uinput, the game, or the real reset control block.
"""

from __future__ import annotations

import os
import sys
import time
import types
import unittest
import warnings
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
from gymnasium.envs.registration import EnvSpec
from gymnasium.utils.env_checker import check_env

from rl.env import gym_action
from rl.env import gym_env as ge
from rl.env import gym_obs
from rl.env import gym_reward


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class RecordingAdapter:
    """Captures the canonical action dicts the env would send to gym_input."""

    name = "recording"

    def __init__(self):
        self.calls: list[dict] = []

    def apply(self, action: dict) -> None:
        snapshot = dict(action)
        for key in ("buttons", "stick", "cstick"):
            snapshot[key] = np.array(action[key], copy=True)
        self.calls.append(snapshot)

    def close(self) -> None:
        self.closed = True


def noop_reset() -> None:
    return None


def fighter(**kw):
    """Minimal fighter dict for reward tests (raw_flags marks it readable)."""
    base = {"stock": 1, "damage": 0.0, "raw_flags": 0xFFFFFFFF}
    base.update(kw)
    return base


def snap(fighters, fight_on=1):
    return SimpleNamespace(fighters=list(fighters), fight_on=fight_on)


class EnvCase(unittest.TestCase):
    """Creates a mock shm + a fast publisher for environment tests."""

    hz = 500.0

    def setUp(self):
        self.fix = ge.ShmFixture()
        self.fix.start_publisher(hz=self.hz)
        self.adapter = RecordingAdapter()
        self.env = None

    def make_env(self, **kwargs):
        opts = dict(
            shm_path=self.fix.path,
            input_adapter=self.adapter,
            reset_fn=noop_reset,
            frame_timeout=2.0,
            reset_timeout=3.0,
            spawn_stable_frames=1,
        )
        opts.update(kwargs)
        self.env = ge.SSBUGymEnv(**opts)
        return self.env

    def tearDown(self):
        if self.env is not None:
            self.env.close()
        self.fix.close()


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
class TestLayout(unittest.TestCase):
    def test_sizes_match_contract(self):
        self.assertEqual(ge.HEADER_SIZE, 80)
        self.assertEqual(ge.FIGHTER_SIZE, 128)
        self.assertEqual(ge.BLOCK_SIZE, 336)
        self.assertEqual(ge.MAX_FIGHTERS, 2)
        self.assertEqual(ge._HEADER.size, 80)
        self.assertEqual(ge._FIGHTER.size, 128)
        self.assertEqual(len(ge.HEADER_FIELDS), 19)
        self.assertEqual(len(ge.FIGHTER_FIELDS), 32)
        self.assertEqual(len(ge.FIGHTER_NAMES), 32)
        # seq sits where reader.py/contract says it does
        self.assertEqual(ge.SEQ_OFFSET, 12)
        self.assertEqual(ge.HEADER_FMT, "<4sIIIQfIIIffffiIIIII")
        self.assertEqual(ge.FIGHTER_NAMES[30], "raw_flags")
        self.assertEqual(ge.FIGHTER_NAMES[24], "buttons")

    def test_header_field_order(self):
        names = [n for n, _f in ge.HEADER_FIELDS]
        self.assertEqual(names[:10], [
            "magic", "version", "n_fighters", "seq", "ts_ns", "fps",
            "reader_pid", "fight_on", "match_state", "blast_left"])
        self.assertEqual(names[-6:], ["stage_id", "game_frame", "flags",
                                      "pad0", "pad1", "pad2"])
        self.assertEqual([n for n, _f in ge.FIGHTER_FIELDS[24:31]], [
            "buttons", "stick_x", "stick_y", "trigger", "cstick_x",
            "cstick_y", "raw_flags"])

    def test_obs_profile_constants(self):
        self.assertEqual(gym_obs.PER_FIGHTER_SIZE, 50)
        self.assertEqual(gym_obs.OBS_SIZE, 100)
        self.assertEqual(len(gym_obs.FEATURE_NAMES), 100)
        self.assertEqual(gym_obs.FEATURE_NAMES[0], "p1_x")
        self.assertEqual(gym_obs.FEATURE_NAMES[-1], "p2_facing_rel")
        self.assertEqual(gym_obs.ACTION_ONEHOT_SIZE, 32)
        self.assertEqual(len(gym_obs.CURATED_ACTION_STATES), 31)
        self.assertEqual(len(set(gym_obs.ACTION_STATE_ONEHOT.values())), 31)


# --------------------------------------------------------------------------
# raw snapshot reader
# --------------------------------------------------------------------------
class TestReader(unittest.TestCase):
    def setUp(self):
        self.fix = ge.ShmFixture()
        self.reader = ge.RawSnapshotReader(self.fix.path, timeout=0.5)

    def tearDown(self):
        self.reader.close()
        self.fix.close()

    def test_roundtrip_fields(self):
        self.fix.set_fighter(0, x=12.5, y=-3.0, damage=33.0,
                             action_state=2, stock=1, facing=-1.0)
        self.fix.set_fighter(1, x=-12.5, damage=7.5, kind=1)
        self.fix.set_state(fight_on=1, match_state=1, stage_id=454)
        seq = self.fix.publish()

        s = self.reader.read_latest()
        self.assertEqual(s.version, 7)
        self.assertEqual(s.seq, seq)
        self.assertEqual(s.seq % 2, 0)
        self.assertEqual(s.fight_on, 1)
        self.assertEqual(s.match_state, 1)
        self.assertEqual(s.stage_id, 454)
        self.assertEqual(s.n_fighters, 2)
        self.assertEqual(len(s.fighters), 2)
        self.assertEqual(tuple(s.fighters[0]), ge.FIGHTER_NAMES)
        self.assertAlmostEqual(s.fighters[0]["x"], 12.5)
        self.assertAlmostEqual(s.fighters[0]["y"], -3.0)
        self.assertAlmostEqual(s.fighters[0]["damage"], 33.0)
        self.assertEqual(s.fighters[0]["action_state"], 2)
        self.assertAlmostEqual(s.fighters[0]["facing"], -1.0)
        self.assertEqual(s.fighters[1]["kind"], 1)
        self.assertAlmostEqual(s.fighters[1]["damage"], 7.5)
        self.assertEqual(s.raw.shape, (2, 32))
        self.assertEqual(s.raw.dtype, np.float64)
        self.assertAlmostEqual(float(s.raw[0, 4]), 12.5)
        self.assertEqual(s.blast, (-240.0, 240.0, 180.0, -140.0))
        self.assertEqual(s.header_dict()["seq"], seq)
        info = s.to_info()
        self.assertEqual(info["raw"].shape, (2, 32))
        self.assertEqual(info["header"]["stage_id"], 454)

    def test_wait_frames_advances_and_times_out(self):
        start = self.reader.read_latest()
        self.fix.publish_frames(5)
        s = self.reader.wait_frames(5, timeout=1.0, start=start)
        self.assertGreaterEqual(s.seq - start.seq, 10)
        with self.assertRaises(TimeoutError):
            self.reader.wait_frames(5, timeout=0.2, start=s)

    def test_torn_odd_frame_is_retried(self):
        self.fix.publish()
        # leave the writer "mid-write": seq odd with an otherwise valid body
        self.fix.write_seq(5)
        import threading
        threading.Timer(0.05, self.fix.publish).start()
        t0 = time.monotonic()
        s = self.reader.read_latest(timeout=1.0)
        self.assertGreaterEqual(time.monotonic() - t0, 0.04)
        self.assertEqual(s.seq % 2, 0)
        self.assertGreaterEqual(s.seq, 6)

    def test_version_mismatch_is_fatal(self):
        self.fix.set_state(version=6)
        self.fix.publish()
        with self.assertRaises(ge.SchemaMismatch):
            self.reader.read_latest(timeout=0.5)

    def test_missing_file_times_out(self):
        reader = ge.RawSnapshotReader(
            os.path.join(HERE, "test_artifacts", "does_not_exist.bin"),
            timeout=0.1)
        with self.assertRaises(TimeoutError):
            reader.read_latest()
        reader.close()

    def test_short_file_times_out(self):
        path = self.fix.path + ".short"
        with open(path, "wb") as fh:
            fh.write(b"\0" * 16)
        reader = ge.RawSnapshotReader(path, timeout=0.1)
        try:
            with self.assertRaises(TimeoutError):
                reader.read_latest()
        finally:
            reader.close()
            os.unlink(path)

    def test_short_v6_block_reports_schema_mismatch(self):
        # The pre-v7 bridge wrote 224-byte blocks; name the problem, don't
        # just time out.
        import struct
        path = self.fix.path + ".v6"
        with open(path, "wb") as fh:
            fh.write(b"SSBU" + struct.pack("<I", 6) + b"\0" * (224 - 8))
        reader = ge.RawSnapshotReader(path, timeout=0.1)
        try:
            with self.assertRaises(ge.SchemaMismatch) as ctx:
                reader.read_latest()
            self.assertIn("v6", str(ctx.exception))
            self.assertIn("v7", str(ctx.exception))
        finally:
            reader.close()
            os.unlink(path)


# --------------------------------------------------------------------------
# observation profile
# --------------------------------------------------------------------------
class TestObservation(unittest.TestCase):
    def setUp(self):
        self.fix = ge.ShmFixture()
        self.reader = ge.RawSnapshotReader(self.fix.path, timeout=0.5)

    def tearDown(self):
        self.reader.close()
        self.fix.close()

    def _snapshot(self):
        self.fix.publish()
        return self.reader.read_latest()

    def test_shape_dtype_and_order(self):
        s = self._snapshot()
        obs = gym_obs.build_observation(s)
        self.assertEqual(obs.shape, (100,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(gym_obs.SLICES["p1_scalars"], slice(0, 14))
        self.assertEqual(gym_obs.SLICES["p1_action"], slice(14, 46))
        self.assertEqual(gym_obs.SLICES["p1_relative"], slice(46, 50))
        self.assertEqual(gym_obs.SLICES["p2_scalars"], slice(50, 64))

    def test_raw_values_and_one_hot(self):
        self.fix.set_fighter(
            0, x=100.0, y=-50.0, vx=25.0, vy=-10.0, damage=42.0, facing=-1.0,
            on_ground=0, is_actionable=1, jumps_left=1, action_frame=30.0,
            shield=1, dist_to_blast=60.0, hitstun=7, is_on_rebirth=1,
            action_state=2, dx=40.0, dy=30.0, facing_rel=-1)
        self.fix.set_fighter(1, x=-20.0, action_state=999, dx=-40.0, dy=-30.0,
                             facing_rel=0)
        s = self._snapshot()
        obs = gym_obs.build_observation(s, normalize=False)

        # P1 scalars in contract order
        expected = [100.0, -50.0, 25.0, -10.0, 42.0, -1.0, 0.0, 1.0,
                    1.0, 30.0, 1.0, 60.0, 7.0, 1.0]
        np.testing.assert_allclose(obs[:14], expected)

        # RUN (id 2) is curated index 2
        p1_oh = obs[14:46]
        self.assertEqual(p1_oh.sum(), 1.0)
        self.assertEqual(int(p1_oh.argmax()), 2)
        # dx, dy, distance=50, facing_rel
        np.testing.assert_allclose(obs[46:50], [40.0, 30.0, 50.0, -1.0])

        # P2: unknown status -> OTHER (index 31 of its one-hot block)
        p2_oh = obs[64:96]
        self.assertEqual(p2_oh.sum(), 1.0)
        self.assertEqual(int(p2_oh.argmax()), 31)
        np.testing.assert_allclose(obs[96:100], [-40.0, -30.0, 50.0, 0.0])

    def test_normalized_divisors(self):
        self.fix.set_fighter(0, x=100.0, y=-50.0, vx=25.0, vy=-10.0,
                             damage=42.0, jumps_left=1, action_frame=30.0,
                             dist_to_blast=60.0, action_state=0,
                             dx=40.0, dy=0.0, facing_rel=1)
        s = self._snapshot()
        obs = gym_obs.build_observation(s, normalize=True)
        self.assertAlmostEqual(float(obs[0]), 1.0)        # x/100
        self.assertAlmostEqual(float(obs[1]), -0.5)       # y/100
        self.assertAlmostEqual(float(obs[2]), 0.5)        # vx/50
        self.assertAlmostEqual(float(obs[3]), -0.2)       # vy/50
        self.assertAlmostEqual(float(obs[4]), 0.42)       # damage/100
        self.assertAlmostEqual(float(obs[8]), 0.5)        # jumps_left/2
        self.assertAlmostEqual(float(obs[9]), 0.5)        # action_frame/60
        self.assertAlmostEqual(float(obs[11]), 0.2)       # dist/300
        self.assertAlmostEqual(float(obs[48]), 0.2)       # distance/200

    def test_empty_slots_are_all_zero(self):
        self.fix.set_state(n_fighters=0)
        s = self._snapshot()
        obs = gym_obs.build_observation(s)
        np.testing.assert_array_equal(obs, np.zeros(100, dtype=np.float32))
        # no fake WAIT one-hot for empty slots
        self.assertEqual(float(obs[14:46].sum()), 0.0)

    def test_curated_one_hot_mapping(self):
        for _name, state_id in gym_obs.CURATED_ACTION_STATES:
            vec = gym_obs._action_onehot(state_id)
            self.assertEqual(vec.shape, (32,))
            self.assertEqual(vec.sum(), 1.0)
            self.assertEqual(int(vec.argmax()), gym_obs.ACTION_STATE_ONEHOT[state_id])
        for unknown in (-1, 999, 2 ** 31 - 1, None):
            vec = gym_obs._action_onehot(unknown)
            self.assertEqual(int(vec.argmax()), gym_obs.OTHER_INDEX)
            self.assertEqual(vec.sum(), 1.0)

    def test_nonfinite_fields_are_sanitized(self):
        f = {"x": float("nan"), "y": float("inf"), "raw_flags": 1}
        obs = gym_obs.build_observation([f, {}])
        self.assertTrue(np.all(np.isfinite(obs)))
        self.assertEqual(float(obs[0]), 0.0)
        self.assertEqual(float(obs[1]), 0.0)

    def test_metadata_documents_exclusions(self):
        meta = gym_obs.observation_metadata()
        self.assertEqual(meta["profile"], "v1")
        self.assertEqual(meta["size"], 100)
        for item in ("stock", "dead", "stage", "match_state", "opponent_id",
                     "controls", "projectiles"):
            self.assertIn(item, meta["excluded"])


# --------------------------------------------------------------------------
# action space
# --------------------------------------------------------------------------
class TestAction(unittest.TestCase):
    def test_space_types(self):
        space = gym_action.make_action_space()
        self.assertIsInstance(space, __import__("gymnasium").spaces.Dict)
        self.assertEqual(set(space.spaces), {"buttons", "stick", "cstick"})
        self.assertEqual(space["buttons"].n, 12)
        self.assertEqual(space["stick"].shape, (2,))
        self.assertEqual(space["cstick"].shape, (2,))
        self.assertEqual(float(space["stick"].low[0]), -1.0)
        self.assertEqual(float(space["stick"].high[0]), 1.0)
        self.assertEqual(gym_action.BUTTONS[0], "A")
        self.assertEqual(gym_action.BUTTONS[-1], "DPAD_R")
        self.assertEqual(len(gym_action.BUTTONS), 12)
        macro = gym_action.make_action_space(macro=True)
        self.assertIsInstance(macro, __import__("gymnasium").spaces.Discrete)
        self.assertEqual(int(macro.n), len(gym_action.MACROS))

    def test_roundtrip(self):
        space = gym_action.make_action_space()
        space.seed(7)
        for _ in range(20):
            native = space.sample()
            can = gym_action.decode_action(native)
            self.assertEqual(can["buttons"].shape, (12,))
            self.assertEqual(can["buttons"].dtype, np.int8)
            self.assertEqual(can["stick"].shape, (2,))
            self.assertEqual(can["cstick"].shape, (2,))
            self.assertEqual(can["trigger"], 0.0)
            self.assertEqual(can["buttons_mask"],
                             gym_action.buttons_mask(native["buttons"]))
            native2 = gym_action.encode_action(can)
            can2 = gym_action.decode_action(native2)
            np.testing.assert_array_equal(can["buttons"], can2["buttons"])
            np.testing.assert_allclose(can["stick"], can2["stick"])
            np.testing.assert_allclose(can["cstick"], can2["cstick"])
            self.assertEqual(can["button_names"], can2["button_names"])

    def test_button_order_and_mask(self):
        can = gym_action.decode_action({
            "buttons": np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], np.int8),
            "stick": np.zeros(2, np.float32),
            "cstick": np.zeros(2, np.float32),
        })
        self.assertEqual(can["button_names"], ("A", "DPAD_R"))
        self.assertEqual(can["buttons_mask"], (1 << 0) | (1 << 11))
        self.assertEqual(can["buttons"].tolist(), [1, 0, 0, 0, 0, 0, 0, 0,
                                                  0, 0, 0, 1])

    def test_stick_clamping_and_validation(self):
        can = gym_action.decode_action({
            "buttons": np.zeros(12, np.int8),
            "stick": np.array([2.0, -3.0], np.float32),
            "cstick": np.array([-0.5, 0.5], np.float32),
        })
        np.testing.assert_allclose(can["stick"], [1.0, -1.0])
        np.testing.assert_allclose(can["cstick"], [-0.5, 0.5])
        with self.assertRaises(ValueError):
            gym_action.decode_action({
                "buttons": np.zeros(12, np.int8),
                "stick": np.array([np.inf, 0.0], np.float32),
                "cstick": np.zeros(2, np.float32),
            })
        with self.assertRaises(ValueError):
            gym_action.decode_action({"buttons": np.zeros(12, np.int8)})
        with self.assertRaises(ValueError):
            gym_action.decode_action({"buttons": np.zeros(5), "stick": [0, 0],
                                      "cstick": [0, 0]})

    def test_flat_vector(self):
        vec = np.zeros(16, dtype=np.float32)
        vec[0] = 1.0
        vec[12] = 0.25
        vec[13] = -0.25
        can = gym_action.decode_action(vec)
        self.assertEqual(can["button_names"], ("A",))
        np.testing.assert_allclose(can["stick"], [0.25, -0.25])
        with self.assertRaises(ValueError):
            gym_action.decode_action(np.zeros(7, dtype=np.float32))

    def test_macros(self):
        for index, macro in enumerate(gym_action.MACROS):
            can = gym_action.decode_action(index, macro=True)
            self.assertEqual(can["macro"], macro.name)
            self.assertEqual(can["button_names"], macro.buttons)
            np.testing.assert_allclose(can["stick"], macro.stick)
            np.testing.assert_allclose(can["cstick"], macro.cstick)
        noop = gym_action.decode_action(0, macro=True)
        self.assertEqual(noop["buttons_mask"], 0)
        np.testing.assert_array_equal(noop["stick"], [0.0, 0.0])
        smash = gym_action.decode_action(
            gym_action.MACRO_INDEX["SMASH_RIGHT"], macro=True)
        self.assertIn("A", smash["button_names"])
        np.testing.assert_allclose(smash["stick"], [1.0, 0.0])
        with self.assertRaises(ValueError):
            gym_action.decode_action(len(gym_action.MACROS), macro=True)
        with self.assertRaises(ValueError):
            gym_action.decode_action(-1, macro=True)


# --------------------------------------------------------------------------
# reward
# --------------------------------------------------------------------------
class TestReward(unittest.TestCase):
    def test_damage_deltas(self):
        prev = snap([fighter(damage=10.0), fighter(damage=20.0)])
        curr = snap([fighter(damage=15.0), fighter(damage=30.0)])
        r = gym_reward.compute_reward(prev, curr)
        self.assertAlmostEqual(r, 0.01 * (10.0 - 5.0))

    def test_ko_and_win(self):
        prev = snap([fighter(stock=1, damage=20.0),
                     fighter(stock=1, damage=60.0)])
        curr = snap([fighter(stock=1, damage=20.0),
                     fighter(stock=0, damage=0.0)])
        r = gym_reward.compute_reward(prev, curr)
        self.assertAlmostEqual(r, 1.0 + 1.0)   # KO bonus + win, no heal penalty

    def test_fight_off_win(self):
        prev = snap([fighter(stock=1), fighter(stock=1)], fight_on=1)
        curr = snap([fighter(stock=1), fighter(stock=0)], fight_on=0)
        self.assertAlmostEqual(gym_reward.compute_reward(prev, curr), 2.0)

    def test_stock_loss(self):
        prev = snap([fighter(stock=1), fighter(stock=1)])
        curr = snap([fighter(stock=0), fighter(stock=1)])
        self.assertAlmostEqual(gym_reward.compute_reward(prev, curr), -2.0)

    def test_double_ko_is_draw(self):
        # ko_bonus (+1) and stock_loss (-1) cancel; draw_reward is 0 by default
        prev = snap([fighter(stock=1), fighter(stock=1)])
        curr = snap([fighter(stock=0), fighter(stock=0)])
        self.assertAlmostEqual(gym_reward.compute_reward(prev, curr), 0.0)
        cfg = gym_reward.RewardConfig(draw_reward=0.5)
        self.assertAlmostEqual(gym_reward.compute_reward(prev, curr, cfg), 0.5)

    def test_ego_slot_swap(self):
        prev = snap([fighter(damage=0.0), fighter(damage=0.0)])
        curr = snap([fighter(damage=10.0), fighter(damage=40.0)])
        r0 = gym_reward.compute_reward(prev, curr, ego_slot=0)
        r1 = gym_reward.compute_reward(prev, curr, ego_slot=1)
        self.assertAlmostEqual(r0, 0.01 * (40.0 - 10.0))
        self.assertAlmostEqual(r1, 0.01 * (10.0 - 40.0))

    def test_custom_config(self):
        cfg = gym_reward.RewardConfig(damage_scale=0.1, damage_dealt=2.0,
                                      damage_taken=0.5, ko_bonus=5.0,
                                      stock_loss=-5.0, win_bonus=10.0,
                                      loss_penalty=-10.0)
        prev = snap([fighter(stock=1, damage=0.0),
                     fighter(stock=1, damage=0.0)])
        curr = snap([fighter(stock=1, damage=10.0),
                     fighter(stock=0, damage=100.0)])
        r = gym_reward.compute_reward(prev, curr, cfg)
        # damage: 0.1 * (2*100 - 0.5*10) = 19.5, plus KO 5 + win 10
        self.assertAlmostEqual(r, 19.5 + 5.0 + 10.0)

    def test_invalid_inputs_are_zero(self):
        empty = SimpleNamespace(fighters=[], fight_on=1)
        self.assertEqual(gym_reward.compute_reward(None, empty), 0.0)
        self.assertEqual(gym_reward.compute_reward(empty, empty), 0.0)
        blank = snap([{}, {}])
        self.assertEqual(gym_reward.compute_reward(blank, blank), 0.0)
        cfg = gym_reward.RewardConfig()
        self.assertIn("damage_scale", cfg.as_dict())
        with self.assertRaises(ValueError):
            gym_reward.compute_reward(blank, blank, ego_slot=2)


# --------------------------------------------------------------------------
# environment (mock shm + publisher)
# --------------------------------------------------------------------------
class TestEnv(EnvCase):
    def test_check_env(self):
        env = self.make_env()
        # A live game is inherently non-deterministic; mark the spec so
        # check_env skips only its same-action determinism assertion.
        env.spec = EnvSpec(id="SSBU-v0", entry_point="gym_env:SSBUGymEnv",
                           nondeterministic=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            check_env(env, skip_render_check=True, skip_close_check=True)
        self.assertEqual([str(w.message) for w in caught], [])

    def test_100_steps(self):
        env = self.make_env()
        obs, info = env.reset(seed=0)
        self.assertIn(obs, env.observation_space)
        self.assertEqual(info["phase"], "reset")
        self.assertEqual(env.step_count, 0)

        t0 = time.monotonic()
        for i in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            self.assertIn(obs, env.observation_space)
            self.assertEqual(obs.shape, (100,))
            self.assertEqual(obs.dtype, np.float32)
            self.assertIsInstance(reward, float)
            self.assertIsInstance(terminated, bool)
            self.assertIsInstance(truncated, bool)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertEqual(reward, 0.0)          # reward_mode="none" default
            self.assertEqual(info["phase"], "step")
            self.assertEqual(info["step"], i + 1)
            self.assertGreaterEqual(info["frames_advanced"], 5)
            self.assertGreaterEqual(info["seq_delta"], 10)
            self.assertEqual(info["seq_delta"] % 2, 0)
            self.assertGreaterEqual(info["age_ns"], 0)
        self.assertEqual(env.step_count, 100)
        self.assertEqual(len(self.adapter.calls), 100)
        self.assertLess(time.monotonic() - t0, 20.0)

    def test_action_reaches_adapter(self):
        env = self.make_env()
        env.reset()
        action = {
            "buttons": np.zeros(12, np.int8),
            "stick": np.zeros(2, np.float32),
            "cstick": np.zeros(2, np.float32),
        }
        action["buttons"][0] = 1                  # A
        action["stick"][:] = [1.0, 0.0]
        env.step(action)
        self.assertEqual(len(self.adapter.calls), 1)
        sent = self.adapter.calls[0]
        self.assertEqual(sent["button_names"], ("A",))
        self.assertEqual(sent["buttons_mask"], 1)
        np.testing.assert_allclose(sent["stick"], [1.0, 0.0])
        self.assertEqual(sent["trigger"], 0.0)
        self.assertIsNone(sent["macro"])

    def test_macro_mode(self):
        env = self.make_env(macro_mode=True)
        self.assertEqual(int(env.action_space.n), len(gym_action.MACROS))
        env.reset()
        env.step(gym_action.MACRO_INDEX["UP"])
        sent = self.adapter.calls[-1]
        self.assertEqual(sent["macro"], "UP")
        np.testing.assert_allclose(sent["stick"], [0.0, 1.0])
        self.assertEqual(sent["buttons_mask"], 0)

    def test_info_carries_raw_extras_and_timing(self):
        env = self.make_env()
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        self.assertEqual(tuple(info["raw_fighters"][0]), ge.FIGHTER_NAMES)
        self.assertEqual(info["raw"].shape, (2, 32))
        self.assertEqual(info["header"]["version"], 7)
        for key in ("decode_s", "write_s", "read_s", "total_s"):
            self.assertGreaterEqual(info["timing"][key], 0.0)
        self.assertEqual(info["input_adapter"], "recording")
        self.assertIsNone(info["input_error"])
        self.assertEqual(info["stocks"], [1, 1])
        self.assertTrue(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [True, True])
        self.assertIn("blast", info)

    def test_reset_calls_reset_fn(self):
        calls = []

        def reset_fn():
            calls.append(time.monotonic())

        env = self.make_env(reset_fn=reset_fn)
        _, info = env.reset()
        self.assertEqual(len(calls), 1)
        self.assertTrue(info["reset_fn_used"])
        self.assertGreaterEqual(info["timing"]["reset_s"], 0.0)

    def test_reset_timeout_when_no_match(self):
        self.fix.set_state(fight_on=0)
        env = self.make_env(reset_timeout=0.3)
        t0 = time.monotonic()
        with self.assertRaises(TimeoutError) as ctx:
            env.reset()
        self.assertLess(time.monotonic() - t0, 3.0)
        self.assertIn("fight_on==1", str(ctx.exception))

    def test_reset_timeout_with_one_fighter(self):
        self.fix.set_state(n_fighters=1)
        self.fix.set_fighter(1, raw_flags=0)
        env = self.make_env(reset_timeout=0.3)
        with self.assertRaises(TimeoutError):
            env.reset()

    def test_reset_requires_positive_stocks_once_signal_is_usable(self):
        # Once a positive stock has been observed, a real stock signal is
        # strict: stock 0 means "not spawned / dead", so reset waits.
        env = self.make_env(reset_timeout=0.3)
        _, info = env.reset()
        self.assertTrue(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [True, True])
        self.fix.set_fighter(0, stock=0)
        self.fix.set_fighter(1, stock=0)
        with self.assertRaises(TimeoutError):
            env.reset()

    def test_reset_requires_fresh_frames(self):
        # Stop the writer: the last frame may say fight_on=1, but reset must
        # see spawn_stable_frames *fresh* frames, so it times out.
        self.fix.stop_publisher()
        env = self.make_env(reset_timeout=0.3, spawn_stable_frames=2)
        with self.assertRaises(TimeoutError):
            env.reset()

    def test_termination_on_fight_off(self):
        env = self.make_env()
        env.reset()
        self.fix.set_state(fight_on=0)
        _, _, terminated, truncated, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(info["end_reason"], "fight_off")
        self.assertEqual(info["fight_on"], 0)

    def test_termination_on_zero_stock(self):
        env = self.make_env()
        env.reset()
        self.fix.set_fighter(1, stock=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "stock")
        self.assertTrue(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [True, True])

    def test_one_stock_signal_positive_then_drop(self):
        # 1-stock match: stock 1 is observed -> usable; 1->0 terminates.
        env = self.make_env()
        _, info = env.reset()
        self.assertEqual(info["stocks"], [1, 1])
        self.assertTrue(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [True, True])
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        self.assertFalse(terminated)

        self.fix.set_fighter(1, stock=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "stock")

    def test_stock_always_zero_falls_back_to_fight_on(self):
        # The debug rule can report stock 0 even in a 1-stock match.  After
        # stock_zero_frames it is declared unusable, reset succeeds, and only
        # fight_on 1->0 may terminate.
        self.fix.set_fighter(0, stock=0)
        self.fix.set_fighter(1, stock=0)
        env = self.make_env()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, info = env.reset()
        self.assertFalse(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [False, False])
        messages = [str(w.message) for w in caught]
        self.assertTrue(any("unusable" in m and "stock" in m for m in messages))

        for _ in range(3):
            _, _, terminated, truncated, step_info = env.step(
                env.action_space.sample())
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertFalse(step_info["stock_usable"])

        self.fix.set_state(fight_on=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "fight_off")

    def test_mixed_stock_signals(self):
        # Slot 1 never reports stock: it becomes unusable and is ignored,
        # while slot 0's real stock signal still ends the episode.
        self.fix.set_fighter(1, stock=0)
        env = self.make_env()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _, info = env.reset()
        self.assertTrue(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [True, False])
        self.assertTrue(any("slot 1" in str(w.message) for w in caught))

        # slot 1 staying at 0 must not terminate
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        self.assertFalse(terminated)

        self.fix.set_fighter(0, stock=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "stock")

    def test_stock_signal_recovers_after_late_init(self):
        self.fix.set_fighter(0, stock=0)
        self.fix.set_fighter(1, stock=0)
        env = self.make_env()
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            _, info = env.reset()
        self.assertFalse(info["stock_usable"])

        # Stock initialises late: a positive value re-enables the signal.
        self.fix.set_fighter(0, stock=1)
        self.fix.set_fighter(1, stock=1)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertFalse(terminated)
        self.assertTrue(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [True, True])

        self.fix.set_fighter(1, stock=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "stock")

    def test_require_positive_stock_false_allows_zero_stock_reset(self):
        self.fix.set_fighter(0, stock=0)
        self.fix.set_fighter(1, stock=0)
        env = self.make_env(require_positive_stock=False)
        _, info = env.reset()                 # one fresh frame is enough
        self.assertIsNone(info["stock_usable"])
        self.assertEqual(info["stock_usable_slots"], [None, None])
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        self.assertFalse(terminated)

    def test_stock_termination_false_uses_fight_on(self):
        env = self.make_env(stock_termination=False)
        env.reset()
        self.fix.set_fighter(1, stock=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertFalse(terminated)
        self.assertIsNone(info["end_reason"])

        self.fix.set_state(fight_on=0)
        _, _, terminated, _, info = env.step(env.action_space.sample())
        self.assertTrue(terminated)
        self.assertEqual(info["end_reason"], "fight_off")

    def test_truncation_on_max_steps(self):
        env = self.make_env(max_steps=1)
        env.reset()
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["end_reason"], "max_steps")
        self.assertEqual(reward, 0.0)

    def test_combat_reward_wired(self):
        env = self.make_env(reward_mode="combat")
        env.reset()
        self.fix.set_fighter(1, damage=50.0)
        _, reward, terminated, truncated, info = env.step(env.action_space.sample())
        self.assertAlmostEqual(reward, 0.5, places=6)
        self.assertEqual(info["reward_mode"], "combat")
        self.assertFalse(terminated or truncated)

    def test_none_reward_ignores_damage(self):
        env = self.make_env()                      # default reward_mode="none"
        env.reset()
        self.fix.set_fighter(1, damage=90.0)
        _, reward, _, _, info = env.step(env.action_space.sample())
        self.assertEqual(reward, 0.0)
        self.assertEqual(info["reward_mode"], "none")

    def test_normalized_observation(self):
        env = self.make_env(normalize_obs=True)
        obs, _ = env.reset()
        self.assertAlmostEqual(float(obs[0]), -0.2)     # P1 x=-20 / 100
        self.assertAlmostEqual(float(obs[50]), 0.2)     # P2 x=20 / 100
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        self.assertIn(obs2, env.observation_space)

    def test_step_timeout_when_writer_stops(self):
        env = self.make_env(frame_timeout=0.2)
        env.reset()
        self.fix.stop_publisher()
        with self.assertRaises(TimeoutError):
            env.step(env.action_space.sample())

    def test_render_close_and_reuse(self):
        env = self.make_env()
        env.reset()
        self.assertIsNone(env.render())
        self.assertIsNotNone(env.last_snapshot)
        env.close()
        env.close()                               # idempotent
        with self.assertRaises(RuntimeError):
            env.step(env.action_space.sample())

    def test_lazy_gym_input_and_gym_reset_imports(self):
        calls = []
        resets = []
        input_mod = types.ModuleType("gym_input")
        input_mod.apply_action = calls.append
        reset_mod = types.ModuleType("gym_reset")
        reset_mod.reset = lambda: resets.append(1)

        old = {k: sys.modules.get(k) for k in ("rl.input.gym_input", "rl.input.gym_reset")}
        sys.modules["rl.input.gym_input"] = input_mod
        sys.modules["rl.input.gym_reset"] = reset_mod
        try:
            env = self.make_env(input_adapter=None, reset_fn=None)
            env.reset()
            self.assertEqual(len(resets), 1)
            env.step(gym_action.neutral_action())
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["buttons"].shape, (12,))
            _, _, _, _, info = env.step(gym_action.neutral_action())
            self.assertEqual(info["input_adapter"], "module.apply_action")
        finally:
            env.close()
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_default_adapters_fall_back_to_noop(self):
        old = {k: sys.modules.get(k) for k in ("rl.input.gym_input", "rl.input.gym_reset")}
        try:
            sys.modules["rl.input.gym_input"] = None       # type: ignore[assignment]
            sys.modules["rl.input.gym_reset"] = None       # type: ignore[assignment]
            adapter, error = ge.default_input_adapter()
            self.assertIsInstance(adapter, ge.NoopInputAdapter)
            self.assertIn("gym_input unavailable", error)
            self.assertIsNone(ge.default_reset_fn())
            adapter.apply(gym_action.neutral_action())   # no-op, no error
            adapter.close()
        finally:
            for key, value in old.items():
                if value is None:
                    sys.modules.pop(key, None)
                else:
                    sys.modules[key] = value

    def test_fighter_valid(self):
        self.assertTrue(ge.fighter_valid(ge.make_fighter()))
        self.assertFalse(ge.fighter_valid({"raw_flags": 0}))
        self.assertFalse(ge.fighter_valid({}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
