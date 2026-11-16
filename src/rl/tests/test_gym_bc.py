#!/usr/bin/env python3
"""Unit tests for gym_record.py / gym_bc.py / gym_eval.py (no game needed).

    .venv/bin/python tools/mem/test_gym_bc.py

Everything is synthetic: the recorder is exercised against a temporary v7
block, the BC pipeline against generated episodes.  The live path (real reader,
real env) is the coordinator's acceptance run.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rl.bc import gym_bc as bc          # noqa: E402
from rl.bc import gym_eval as ge        # noqa: E402
from rl.record import gym_record as rec     # noqa: E402

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def make_obs_actions(n: int = 30, seed: int = 7):
    """Observation vectors whose value encodes the frame index."""
    rng = np.random.default_rng(seed)
    frame = np.arange(n, dtype=np.float32)[:, None]
    feat = np.arange(bc.OBS_DIM, dtype=np.float32)[None, :]
    obs = frame * 1000.0 + feat
    buttons = rng.integers(0, 2, size=(n, bc.N_BUTTONS)).astype(np.float32)
    acts = {
        "buttons": buttons,
        "stick": rng.uniform(-1, 1, size=(n, 2)).astype(np.float32),
        "cstick": rng.uniform(-1, 1, size=(n, 2)).astype(np.float32),
        "trigger": rng.uniform(0, 1, size=(n,)).astype(np.float32),
    }
    return obs, acts


def episode_from_obs_actions(obs, acts, index=0) -> bc.EpisodeData:
    n = len(obs)
    raw = np.zeros((n, 2, len(bc.FIELDS)), np.float32)
    buttons_raw = bc.encode_buttons(acts["buttons"])
    buttons_raw2 = np.stack([buttons_raw, buttons_raw], axis=1)
    stick = np.stack([acts["stick"], acts["stick"]], axis=1)
    cstick = np.stack([acts["cstick"], acts["cstick"]], axis=1)
    trigger = np.stack([acts["trigger"], acts["trigger"]], axis=1)
    meta = {"index": index, "p1_char": "mario", "p2_char": "donkey_kong",
            "stage": "final_destination", "build": "test"}
    return bc.EpisodeData(index, f"mem_{index}", meta, raw, buttons_raw2,
                          stick, cstick, trigger, player=0, obs=obs, acts=acts)


def write_episode(path: str, ep: bc.EpisodeData, frame_skip: int = 5) -> None:
    n = len(ep.raw)
    np.savez(
        path,
        seq=np.arange(2, 2 * n + 2, 2, dtype=np.uint64),
        ts_ns=np.arange(n, dtype=np.uint64) * 16_666_667,
        raw_fighters=ep.raw,
        buttons_raw=ep.buttons_raw,
        stick=ep.stick,
        cstick=ep.cstick,
        trigger=ep.trigger,
        frame_skip=np.asarray(frame_skip, dtype=np.int64),
        meta_json=np.asarray(json.dumps(ep.meta)),
        result_json=np.asarray(json.dumps({"test": True})),
    )


def fake_frame(seq: int, fight: int = 1, kind=(0, 1)) -> dict:
    fighters = np.zeros((2, len(bc.FIELDS)), np.float32)
    fighters[:, bc.FIELD_INDEX["kind"]] = kind
    fighters[:, bc.FIELD_INDEX["stock"]] = 1
    fighters[:, bc.FIELD_INDEX["damage"]] = [12.0, 34.0]
    fighters[:, bc.FIELD_INDEX["x"]] = [10.0, -10.0]
    fighters[:, bc.FIELD_INDEX["action_state"]] = 0
    fighters[0, bc.FIELD_INDEX["buttons"]] = 1
    fighters[0, bc.FIELD_INDEX["stick_x"]] = 0.5
    fighters[0, bc.FIELD_INDEX["trigger"]] = 0.25
    fighters[1, bc.FIELD_INDEX["cstick_y"]] = -0.5
    return {
        "seq": int(seq),
        "ts_ns": int(seq) * 16_666_667,
        "fps": 60.0,
        "fight_on": int(fight),
        "match_state": 1 if fight else 2,
        "stage_id": 0,
        "game_frame": int(seq) // 2,
        "flags": 0,
        "bounds": (-240.0, 240.0, 180.0, -140.0),
        "fighters": fighters,
        "buttons_raw": fighters[:, bc.FIELD_INDEX["buttons"]].astype(np.uint32),
        "stick": fighters[:, [bc.FIELD_INDEX["stick_x"],
                              bc.FIELD_INDEX["stick_y"]]].astype(np.float32),
        "cstick": fighters[:, [bc.FIELD_INDEX["cstick_x"],
                               bc.FIELD_INDEX["cstick_y"]]].astype(np.float32),
        "trigger": fighters[:, bc.FIELD_INDEX["trigger"]].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# dataset builder
# ---------------------------------------------------------------------------
class WindowAlignmentTest(unittest.TestCase):
    def test_window_prev_target_alignment(self):
        window, horizon = 8, 4
        obs, acts = make_obs_actions(30)
        s = bc.window_samples(obs, acts, window, horizon, include_cstick=True,
                              include_trigger=True)
        n_expected = len(obs) - window - horizon + 2
        self.assertEqual(s["windows"].shape, (n_expected, window, bc.OBS_DIM))
        self.assertEqual(s["prev"].shape, (n_expected, 17))
        self.assertEqual(s["targets"]["buttons"].shape,
                         (n_expected, horizon, bc.N_BUTTONS))
        vec = bc.prev_action_vectors(acts, True, True)
        t_values = np.arange(window - 1, len(obs) - horizon + 1)
        for i, t in enumerate(t_values):
            np.testing.assert_allclose(s["windows"][i], obs[t - window + 1:t + 1])
            np.testing.assert_allclose(s["prev"][i], vec[t])
            np.testing.assert_allclose(s["targets"]["buttons"][i],
                                       acts["buttons"][t:t + horizon])
            np.testing.assert_array_equal(s["targets"]["stick"][i],
                                          bc.stick_to_bin(acts["stick"][t:t + horizon]))
            np.testing.assert_allclose(s["targets"]["trigger"][i],
                                       acts["trigger"][t:t + horizon])

    def test_short_episode_yields_no_windows(self):
        obs, acts = make_obs_actions(9)   # T=8, H=4 needs 10 decisions
        s = bc.window_samples(obs, acts, 8, 4)
        self.assertEqual(len(s["windows"]), 0)

    def test_no_cross_episode_windows(self):
        obs1, acts1 = make_obs_actions(30, seed=1)
        obs2, acts2 = make_obs_actions(30, seed=2)
        obs1 = obs1.copy()
        obs2 = obs2.copy()
        obs1[:, 0] = 111.0
        obs2[:, 0] = 222.0
        ep1 = episode_from_obs_actions(obs1, acts1, index=0)
        ep2 = episode_from_obs_actions(obs2, acts2, index=1)
        cfg = bc.BCConfig(window=8, horizon=4)
        samples = bc.samples_for([ep1, ep2], cfg, mirror=False)
        marker = samples["windows"][:, :, 0]
        self.assertTrue(np.all(np.isin(marker, (111.0, 222.0))))
        # every window is constant within one episode
        self.assertTrue(np.all(marker == marker[:, :1]))
        self.assertEqual(int((marker == 111.0).any(axis=1).sum()),
                         len(obs1) - 8 - 4 + 2)
        self.assertEqual(int((marker == 222.0).any(axis=1).sum()),
                         len(obs2) - 8 - 4 + 2)

    def test_split_is_episode_level(self):
        for n in (1, 2, 3, 10, 20):
            tr, va, te = bc.split_episodes(n, seed=3)
            self.assertGreaterEqual(len(tr), 1)
            self.assertGreaterEqual(len(va), 1)
            self.assertGreaterEqual(len(te), 1)
            if n >= 3:
                union = sorted(np.concatenate([tr, va, te]).tolist())
                self.assertEqual(union, list(range(n)))
            again = bc.split_episodes(n, seed=3)
            for a, b in zip((tr, va, te), again):
                np.testing.assert_array_equal(a, b)


class StickDirTest(unittest.TestCase):
    def test_encoding(self):
        v = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1],
                      [0.6, 0.6], [0.1, 0.0]], np.float32)
        d, m = bc.stick_to_dir(v)
        self.assertEqual(d.tolist(), [0, 1, 3, 5, 7, 2, 1])
        self.assertEqual(m.tolist(), [0, 2, 2, 2, 2, 2, 1])
        back = bc.dir_to_xy(d, m)
        np.testing.assert_allclose(back[1], [1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(back[5], [0.7071, 0.7071], atol=1e-3)
        np.testing.assert_allclose(back[0], [0.0, 0.0])

    def test_targets_and_model(self):
        cfg = bc.BCConfig(window=4, horizon=2, stick_mode="dir")
        obs, acts = make_obs_actions(30, seed=3)
        acts["stick"] = np.tile(np.array([[1.0, 0.0]], np.float32), (30, 1))
        ep = episode_from_obs_actions(obs, acts, index=0)
        samples = bc.samples_for([ep], cfg)
        self.assertIn("stick", samples["targets"])
        self.assertIn("stick_mag", samples["targets"])
        self.assertTrue((samples["targets"]["stick"] == 1).all())
        net = bc.BCPolicyNet(cfg)
        out = net(torch.zeros(2, cfg.window, cfg.obs_dim),
                  torch.zeros(2, cfg.prev_dim))
        self.assertEqual(out["stick"].shape, (2, cfg.horizon, bc.STICK_DIRS))
        self.assertEqual(out["stick_mag"].shape, (2, cfg.horizon, bc.STICK_MAGS))

    def test_metrics_dir_mode(self):
        cfg = bc.BCConfig(window=4, horizon=2, stick_mode="dir")
        obs, acts = make_obs_actions(24, seed=4)
        ep = episode_from_obs_actions(obs, acts, index=0)
        samples = bc.samples_for([ep], cfg)
        pred = bc.copy_last_pred(samples, "dir")
        m = bc.compute_metrics(pred, bc.first_actions(samples),
                               prev=samples["prev"], mode="dir")
        self.assertEqual(m["stick_mode"], "dir")
        for key in ("stick_acc", "stick_mag_acc", "stick_mae", "stick_within",
                    "change_dir_acc"):
            self.assertIn(key, m)


def _untrained_policy(sample: bool):
    cfg = bc.BCConfig(window=4, horizon=2, hidden=16, cstick=False)
    torch.manual_seed(0)
    checkpoint = {
        "config": cfg.to_dict(),
        "state_dict": bc.BCPolicyNet(cfg).state_dict(),
        "norm": {"mean": np.zeros(bc.OBS_DIM, np.float32).tolist(),
                 "std": np.ones(bc.OBS_DIM, np.float32).tolist()},
        "thresholds": np.full(bc.N_BUTTONS, 0.5, np.float32).tolist(),
    }
    return bc.BCPolicy(checkpoint, sample=sample)


class ReturnWeightTest(unittest.TestCase):
    """Conservative return weighting (damage differential only)."""

    def test_rewards_match_gym_reward(self):
        from rl.env import gym_reward
        ep = bc.synthetic_episode(40, seed=2)
        cfg = gym_reward.RewardConfig(damage_scale=0.01, ko_bonus=0.0,
                                      stock_loss=0.0, win_bonus=0.0,
                                      loss_penalty=0.0, draw_reward=0.0,
                                      ego_slot=ep.player)
        names = bc.FIELDS
        prev = None
        expected = []
        for t in range(len(ep.raw)):
            cur = {"fight_on": 1,
                   "fighters": [{n: float(ep.raw[t, i, bc.FIELD_INDEX[n]])
                                 for n in names} for i in range(2)]}
            expected.append(gym_reward.compute_reward(prev, cur, cfg))
            prev = cur
        got = bc.decision_rewards(ep, 0.01)
        np.testing.assert_allclose(got, np.asarray(expected, np.float32),
                                   rtol=1e-5, atol=1e-6)

    def test_weights_are_bounded_and_favour_returns(self):
        rewards = np.array([0.0, 0.3, 0.0, 0.0, -0.2, 0.3], np.float32)
        w = bc.return_weights(rewards, gamma=0.99, beta=4.0, quantile=0.5,
                              w_min=0.1, w_max=10.0)
        self.assertEqual(w.shape, rewards.shape)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=5)
        self.assertGreaterEqual(float(w.min()), 0.1 - 1e-6)
        self.assertLessEqual(float(w.max()), 10.0 + 1e-6)
        self.assertGreater(float(w[1]), float(w[4]))   # +damage beats -damage

    def test_empty_weights(self):
        self.assertEqual(len(bc.return_weights(np.zeros(0, np.float32))), 0)

    def test_samples_carry_aligned_weights(self):
        cfg = bc.BCConfig(window=4, horizon=2)
        ep = bc.synthetic_episode(60, seed=1)
        plain = bc.samples_for([ep], cfg)
        weighted = bc.samples_for([ep], cfg, return_beta=5.0)
        self.assertNotIn("weights", plain)
        self.assertEqual(weighted["weights"].shape, (len(weighted["windows"]),))
        self.assertAlmostEqual(float(weighted["weights"].mean()), 1.0, places=4)

    def test_equal_weights_match_the_unweighted_loss(self):
        cfg = bc.BCConfig(window=4, horizon=2, hidden=8)
        net = bc.BCPolicyNet(cfg)
        obs = torch.randn(3, cfg.window, cfg.obs_dim)
        prev = torch.zeros(3, cfg.prev_dim)
        targets = {"buttons": torch.zeros(3, cfg.horizon, bc.N_BUTTONS),
                   "stick": torch.zeros(3, cfg.horizon, dtype=torch.long),
                   "stick_mag": torch.zeros(3, cfg.horizon, dtype=torch.long),
                   "cstick": torch.zeros(3, cfg.horizon, dtype=torch.long),
                   "trigger": torch.zeros(3, cfg.horizon)}
        pos = torch.ones(bc.N_BUTTONS)
        out = net(obs, prev)
        plain, _ = bc._loss(out, targets, pos, cfg)
        ones, _ = bc._loss(out, targets, pos, cfg, weights=torch.ones(3))
        self.assertAlmostEqual(float(plain.detach()), float(ones.detach()),
                               places=5)


class SamplingDecodeTest(unittest.TestCase):
    """--sample: stochastic decoding must still produce valid, varied actions."""

    def test_sampled_actions_are_binary_and_vary(self):
        pol = _untrained_policy(sample=True)
        obs = np.zeros(bc.OBS_DIM, np.float32)
        actions = [pol.act(obs) for _ in range(40)]
        buttons = np.stack([a["buttons"] for a in actions])
        self.assertTrue(np.isin(buttons, (0.0, 1.0)).all())
        sticks = np.stack([a["stick"] for a in actions])
        self.assertGreater(len({tuple(np.round(v, 2)) for v in sticks}), 1)

    def test_argmax_stays_deterministic(self):
        # two fresh policies see the same empty history, so they must agree
        obs = np.zeros(bc.OBS_DIM, np.float32)
        a1 = _untrained_policy(sample=False).act(obs)
        a2 = _untrained_policy(sample=False).act(obs)
        np.testing.assert_allclose(a1["buttons"], a2["buttons"])
        np.testing.assert_allclose(a1["stick"], a2["stick"])


class EgoSwapTest(unittest.TestCase):
    def setUp(self):
        obs, acts = make_obs_actions(24, seed=9)
        self.ep = episode_from_obs_actions(obs, acts, index=3)
        # give the two players clearly different actions
        self.ep.buttons_raw[:, 1] = np.bitwise_xor(self.ep.buttons_raw[:, 0], 0x5)
        self.ep.stick[:, 1] = -self.ep.stick[:, 0]
        self.ep = bc.EpisodeData(self.ep.index, self.ep.path, self.ep.meta,
                                 self.ep.raw, self.ep.buttons_raw,
                                 self.ep.stick, self.ep.cstick, self.ep.trigger,
                                 player=0, obs=self.ep.obs)

    def test_blocks_swap_and_actions_follow(self):
        sw = bc.ego_swap_episode(self.ep)
        n = bc.OBS_PER_FIGHTER
        np.testing.assert_allclose(sw.obs[:, :n], self.ep.obs[:, n:])
        np.testing.assert_allclose(sw.obs[:, n:], self.ep.obs[:, :n])
        other = self.ep.with_player(1)
        np.testing.assert_allclose(sw.acts["buttons"], other.acts["buttons"])
        np.testing.assert_allclose(sw.acts["stick"], self.ep.stick[:, 1])
        np.testing.assert_allclose(sw.acts["cstick"], self.ep.cstick[:, 1])
        self.assertEqual(sw.player, 1)

    def test_swap_is_an_involution(self):
        twice = bc.ego_swap_episode(bc.ego_swap_episode(self.ep))
        np.testing.assert_allclose(twice.obs, self.ep.obs)
        np.testing.assert_allclose(twice.acts["buttons"], self.ep.acts["buttons"])

    def test_samples_double(self):
        cfg = bc.BCConfig(window=4, horizon=2)
        base = bc.samples_for([self.ep], cfg, mirror=False)
        swapped = bc.samples_for([self.ep], cfg, mirror=False, ego_swap=True)
        self.assertEqual(tmp_len(swapped), 2 * tmp_len(base))

    def test_composes_with_mirror(self):
        cfg = bc.BCConfig(window=4, horizon=2)
        one = bc.samples_for([self.ep], cfg, mirror=False, ego_swap=True)
        two = bc.samples_for([self.ep], cfg, mirror=True, ego_swap=True)
        self.assertEqual(tmp_len(two), 2 * tmp_len(one))


def tmp_len(samples) -> int:
    return int(samples["windows"].shape[0])


class DecisionMetricTest(unittest.TestCase):
    def setUp(self):
        # hold-heavy actions: one change every 6 decisions
        n = 60
        buttons = np.zeros((n, bc.N_BUTTONS), np.float32)
        stick = np.zeros((n, 2), np.float32)
        for i in range(n):
            if (i // 6) % 2 == 1:
                buttons[i, 0] = 1.0            # A held for 6 decisions
            if (i // 12) % 2 == 1:
                stick[i, 0] = 1.0 if (i // 12) % 4 == 1 else -1.0
        acts = {"buttons": buttons, "stick": stick,
                "cstick": np.zeros((n, 2), np.float32),
                "trigger": np.zeros(n, np.float32)}
        obs = np.zeros((n, bc.OBS_DIM), np.float32)
        obs[:, 0] = np.arange(n, dtype=np.float32)
        self.ep = episode_from_obs_actions(obs, acts, index=0)
        self.cfg = bc.BCConfig(window=4, horizon=2)
        self.samples = bc.samples_for([self.ep], self.cfg)

    def test_transition_mask_matches_manual(self):
        mask = bc.transition_mask(self.samples)
        prev = self.samples["prev"]
        tb = self.samples["targets"]["buttons"][:, 0]
        pbtn = prev[:, :12] >= 0.5
        pstick = bc.stick_to_bin(prev[:, 12:14])
        tb_bins = self.samples["targets"]["stick"][:, 0]
        manual = (tb != pbtn).any(axis=1) | (tb_bins != pstick)
        np.testing.assert_array_equal(mask, manual)
        self.assertGreater(int(mask.sum()), 0)
        self.assertLess(int(mask.sum()), len(mask))
        self.assertLess(float(mask.mean()), 0.5)   # hold-dominated

    def test_change_metrics_present(self):
        target = bc.first_actions(self.samples)
        pred = bc.copy_last_pred(self.samples)
        m = bc.compute_metrics(pred, target, prev=self.samples["prev"])
        for key in ("change_frames", "change_rate", "change_mean_f1",
                    "change_mean_f1_active", "change_per_button_f1"):
            self.assertIn(key, m)
        self.assertGreater(m["change_frames"], 0)
        self.assertTrue(0.0 <= m["change_mean_f1"] <= 1.0)
        b = bc.copy_last_metrics(self.samples)
        self.assertIn("mean_f1", b)
        self.assertIn("change_mean_f1", b)


class MirrorTest(unittest.TestCase):
    def setUp(self):
        obs, acts = make_obs_actions(20, seed=5)
        self.ep = episode_from_obs_actions(obs, acts, index=0)
        self.mir = bc.mirror_episode(self.ep)

    def test_observation_fields(self):
        o, m = self.ep.obs, self.mir.obs
        for base in (0, bc.OBS_PER_FIGHTER):
            for idx in (0, 2, 5, 46):   # x, vx, facing, dx
                np.testing.assert_allclose(m[:, base + idx], -o[:, base + idx])
            for idx in (1, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 47, 48, 49):
                np.testing.assert_allclose(m[:, base + idx], o[:, base + idx])
        # action_state one-hot unchanged
        np.testing.assert_allclose(m[:, 14:46], o[:, 14:46])

    def test_action_fields(self):
        a, m = self.ep.acts, self.mir.acts
        np.testing.assert_allclose(m["stick"][:, 0], -a["stick"][:, 0])
        np.testing.assert_allclose(m["stick"][:, 1], a["stick"][:, 1])
        np.testing.assert_allclose(m["cstick"][:, 0], -a["cstick"][:, 0])
        np.testing.assert_allclose(m["trigger"], a["trigger"])
        np.testing.assert_allclose(m["buttons"][:, bc.BUTTON_INDEX["DPAD_L"]],
                                   a["buttons"][:, bc.BUTTON_INDEX["DPAD_R"]])
        np.testing.assert_allclose(m["buttons"][:, bc.BUTTON_INDEX["DPAD_R"]],
                                   a["buttons"][:, bc.BUTTON_INDEX["DPAD_L"]])
        keep = [i for i in range(bc.N_BUTTONS)
                if i not in (bc.BUTTON_INDEX["DPAD_L"],
                             bc.BUTTON_INDEX["DPAD_R"])]
        np.testing.assert_allclose(m["buttons"][:, keep], a["buttons"][:, keep])

    def test_involution(self):
        twice = bc.mirror_episode(self.mir)
        np.testing.assert_allclose(twice.obs, self.ep.obs)
        for key in ("buttons", "stick", "cstick", "trigger"):
            np.testing.assert_allclose(twice.acts[key], self.ep.acts[key])

    def test_mirrored_samples_are_aligned(self):
        cfg = bc.BCConfig(window=8, horizon=4)
        base = bc.samples_for([self.ep], cfg, mirror=False)
        mirrored = bc.samples_for([self.mir], cfg, mirror=False)
        np.testing.assert_allclose(mirrored["windows"][:, :, 0],
                                   -base["windows"][:, :, 0])
        np.testing.assert_allclose(mirrored["prev"][:, 12],
                                   -base["prev"][:, 12])
        np.testing.assert_array_equal(
            mirrored["targets"]["stick"], bc.window_samples(
                self.mir.obs, self.mir.acts, 8, 4)["targets"]["stick"])


class ShieldFromStatusTest(unittest.TestCase):
    """CPU guard masks carry 0x8004 (X); the status kind is the real shield."""

    def _episode(self, shield_from_status: bool) -> bc.EpisodeData:
        ep = bc.synthetic_episode(12, seed=9)
        raw = ep.raw.copy()
        buttons_raw = ep.buttons_raw.copy()
        raw[:, 0, bc.FIELD_INDEX["action_state"]] = np.tile(
            np.asarray(bc.GUARD_STATES, np.float32), 4)
        raw[:, 1, bc.FIELD_INDEX["action_state"]] = 0.0
        raw[:, 0, bc.FIELD_INDEX["buttons"]] = np.float32(0x8004)
        buttons_raw[:, 0] = 0x8004          # CPU guard frame: X bit + 0x8000
        buttons_raw[:, 1] = 0
        return bc.EpisodeData(ep.index, ep.path, ep.meta, raw, buttons_raw,
                              ep.stick, ep.cstick, ep.trigger, player=0,
                              shield_from_status=shield_from_status)

    def test_guard_frames_become_zl_and_clear_x(self):
        ep = self._episode(True)
        x = bc.BUTTON_INDEX["X"]
        zl = bc.BUTTON_INDEX["ZL"]
        for i in range(len(ep.raw)):
            self.assertEqual(float(ep.acts["buttons"][i, zl]), 1.0, i)
            self.assertEqual(float(ep.acts["buttons"][i, x]), 0.0, i)
        # the raw mask and the raw slot keep the CPU evidence unchanged
        np.testing.assert_array_equal(ep.buttons_raw[:, 0], 0x8004)
        np.testing.assert_array_equal(
            ep.raw[:, 0, bc.FIELD_INDEX["buttons"]], np.float32(0x8004))
        # target shape is unchanged
        self.assertEqual(ep.acts["buttons"].shape[1], bc.N_BUTTONS)

    def test_non_guard_frames_unchanged(self):
        ep = bc.synthetic_episode(6, seed=9)
        raw = ep.raw.copy()
        buttons_raw = ep.buttons_raw.copy()
        raw[:, 0, bc.FIELD_INDEX["action_state"]] = 0.0
        buttons_raw[:, 0] = 0x8004
        fixed = bc.EpisodeData(ep.index, ep.path, ep.meta, raw, buttons_raw,
                               ep.stick, ep.cstick, ep.trigger, player=0,
                               shield_from_status=True)
        x = bc.BUTTON_INDEX["X"]
        zl = bc.BUTTON_INDEX["ZL"]
        for i in range(len(raw)):
            self.assertEqual(float(fixed.acts["buttons"][i, x]), 1.0, i)
            self.assertEqual(float(fixed.acts["buttons"][i, zl]), 0.0, i)

    def test_flag_off_keeps_raw_mask(self):
        ep = self._episode(False)
        x = bc.BUTTON_INDEX["X"]
        zl = bc.BUTTON_INDEX["ZL"]
        for i in range(len(ep.raw)):
            self.assertEqual(float(ep.acts["buttons"][i, x]), 1.0, i)
            self.assertEqual(float(ep.acts["buttons"][i, zl]), 0.0, i)

    def test_label_count_and_resample(self):
        ep = self._episode(True)
        stats = bc.shield_label_stats([bc.resample_episode(ep)])
        # 12 decisions, every 5th kept: states 8, 9, 10
        self.assertEqual(stats["frames"], 3)
        self.assertEqual(stats["by_state"], {"8": 1, "9": 1, "10": 1})
        self.assertEqual(stats["spurious_x_cleared"], 3)
        merged = bc.merge_shield_stats([stats, stats])
        self.assertEqual(merged["frames"], 6)
        self.assertEqual(merged["spurious_x_cleared"], 6)


class NormalizationTest(unittest.TestCase):
    def test_stats_and_normalize(self):
        rng = np.random.default_rng(0)
        windows = rng.normal(5.0, 3.0, size=(50, 8, bc.OBS_DIM)).astype(np.float32)
        windows[:, :, 0] = 7.0                     # constant feature
        mean, std = bc.norm_stats(windows)
        self.assertEqual(std[0], 1.0)
        z = bc.normalize(windows, mean, std)
        self.assertAlmostEqual(float(z[:, :, 1].mean()), 0.0, places=5)
        self.assertAlmostEqual(float(z[:, :, 1].std()), 1.0, places=4)
        self.assertAlmostEqual(float(z[:, :, 0].mean()), 0.0, places=6)

    def test_dataset_normalizes_windows(self):
        obs, acts = make_obs_actions(30)
        cfg = bc.BCConfig(window=8, horizon=4)
        samples = bc.window_samples(obs, acts, 8, 4)
        mean, std = bc.norm_stats(samples["windows"])
        ds = bc.WindowDataset(samples, mean, std)
        self.assertIs(ds.windows, samples["windows"])   # no second copy
        w, p, t, wt = ds[0]
        self.assertEqual(float(wt), 1.0)               # unweighted -> 1
        expected = (samples["windows"][0] - mean) / std
        np.testing.assert_allclose(w.numpy(), expected, rtol=1e-5, atol=1e-5)
        self.assertEqual(tuple(p.shape), (15,))
        self.assertEqual(tuple(t["buttons"].shape), (4, 12))


class GymObsCompatibilityTest(unittest.TestCase):
    """The BC dataset observation must equal gym_obs.PolicyObservation."""

    def test_matches_agent_c_profile(self):
        try:
            from rl.env import gym_obs
        except Exception as exc:                       # pragma: no cover
            self.skipTest(f"gym_obs not importable: {exc}")
        ep = bc.synthetic_episode(12, seed=4)
        raw = ep.raw.copy()
        states = [0, 4, 6, 30, 43, 74, 263, 470, 999, 31, -1, 35]
        raw[:, 0, bc.FIELD_INDEX["action_state"]] = states
        raw[:, 1, bc.FIELD_INDEX["action_state"]] = states[::-1]
        raw[:, 0, bc.FIELD_INDEX["raw_flags"]] = 1
        raw[:, 1, bc.FIELD_INDEX["raw_flags"]] = 0
        raw[5, 1, :] = 0.0                             # empty slot
        dicts = [
            [{name: float(raw[i, k, j]) for j, name in enumerate(bc.FIELDS)}
             for k in range(2)]
            for i in range(len(raw))
        ]
        for normalize in (False, True):
            expected = np.stack([
                gym_obs.PolicyObservation(normalize=normalize).build(d)
                for d in dicts])
            got = bc.obs_from_raw(raw, normalize=normalize)
            np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5,
                                       err_msg=f"normalize={normalize}")
        # an unreadable slot is all zeros in both builders
        np.testing.assert_allclose(
            bc.obs_from_raw(raw[5:6])[0, bc.OBS_PER_FIGHTER:], 0.0)
        np.testing.assert_allclose(
            gym_obs.PolicyObservation().build(dicts[5])[gym_obs.PER_FIGHTER_SIZE:],
            0.0)

    def test_curated_onehot_mapping(self):
        # C's 32nd slot is OTHER for unknown/out-of-range states.
        raw = np.zeros((1, 2, len(bc.FIELDS)), np.float32)
        raw[0, 0, bc.FIELD_INDEX["x"]] = 1.0
        for state, index in ((0, 0), (4, 3), (35, 21), (999, 31), (-5, 31)):
            raw[0, 0, bc.FIELD_INDEX["action_state"]] = state
            obs = bc.obs_from_raw(raw)[0]
            self.assertEqual(int(np.argmax(obs[14:46])), index, state)
            self.assertAlmostEqual(float(obs[14:46].sum()), 1.0)


# ---------------------------------------------------------------------------
# action/observation helpers
# ---------------------------------------------------------------------------
class ActionHelperTest(unittest.TestCase):
    def test_button_roundtrip(self):
        vec = np.zeros((4, bc.N_BUTTONS), np.float32)
        vec[1, 0] = 1
        vec[2, 11] = 1
        vec[3, [0, 3, 7]] = 1
        mask = bc.encode_buttons(vec)
        np.testing.assert_array_equal(bc.decode_buttons(mask), vec)
        self.assertEqual(int(mask[1]), bc.BUTTON_BITS[0])
        self.assertEqual(int(mask[2]), bc.BUTTON_BITS[11])
        # the raw ControlModule positions are not 1<<i (B's evidence)
        self.assertEqual(bc.BUTTON_BITS[bc.BUTTON_INDEX["R"]], 0x0400)
        self.assertEqual(bc.BUTTON_BITS[bc.BUTTON_INDEX["MINUS"]], 0x0010)

    def test_stick_bins(self):
        xy = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0],
                       [-0.3, 0.4]], np.float32)
        bins = bc.stick_to_bin(xy)
        centers = bc.bin_centers()
        np.testing.assert_allclose(centers[bins[1] // 9], 0.0, atol=1e-6)
        np.testing.assert_allclose(centers[bins[0] // 9], -1.0)
        np.testing.assert_allclose(centers[bins[2] // 9], 1.0)
        back = bc.bin_to_xy(bins)
        self.assertLess(np.max(np.abs(back - xy)), 0.13)

    def test_calibration_matches_rate(self):
        rng = np.random.default_rng(0)
        probs = rng.uniform(size=(500, bc.N_BUTTONS)).astype(np.float32)
        targets = (rng.uniform(size=(500, bc.N_BUTTONS)) > 0.8).astype(np.float32)
        th = bc.calibrate_thresholds(probs, targets)
        pred = (probs >= th).astype(np.float32)
        np.testing.assert_allclose(pred.mean(axis=0), targets.mean(axis=0),
                                   atol=0.02)


# ---------------------------------------------------------------------------
# overfit / checkpoint
# ---------------------------------------------------------------------------
class OverfitTest(unittest.TestCase):
    def test_tiny_batch_overfits(self):
        _, metrics = bc.train_overfit(steps=150, seed=0, log=lambda *a: None)
        self.assertLess(metrics["train_loss"], 0.05)
        self.assertGreater(metrics["mean_f1_active"], 0.95)
        self.assertGreater(metrics["stick_acc"], 0.9)
        self.assertLess(metrics["mean_action_rate_divergence"], 0.05)


class CheckpointRoundTripTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data = os.path.join(cls.tmp.name, "data")
        os.makedirs(cls.data)
        for i in range(3):
            ep = bc.synthetic_episode(90, seed=i, index=i)
            write_episode(os.path.join(cls.data, f"episode_{i:04d}.npz"), ep)
        cls.out = os.path.join(cls.tmp.name, "bc.pt")
        cls.cfg = bc.BCConfig(hidden=32, cstick=False, frame_skip=5)
        cls.ckpt, cls.metrics = bc.train(
            cls.data, cls.out, cfg=cls.cfg, epochs=2, batch=8, seed=0,
            mirror=False, log=lambda *a: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_checkpoint_contents(self):
        self.assertTrue(os.path.exists(self.out))
        ck = self.ckpt
        self.assertEqual(ck["format"], "ssbu_bc_v1")
        self.assertEqual(len(ck["norm"]["mean"]), bc.OBS_DIM)
        self.assertEqual(len(ck["norm"]["std"]), bc.OBS_DIM)
        self.assertEqual(len(ck["thresholds"]), bc.N_BUTTONS)
        self.assertEqual(ck["config"]["window"], 8)
        self.assertEqual(ck["config"]["horizon"], 4)
        self.assertEqual(ck["config"]["frame_skip"], 5)
        self.assertFalse(ck["config"]["obs_normalize"])
        self.assertTrue(ck["config"]["shield_from_status"])
        self.assertTrue(ck["shield_from_status"])
        self.assertEqual(ck["shield_labels"]["total"]["frames"], 0)
        self.assertIn("alignment", ck)
        self.assertTrue(ck["alignment"]["ok"])
        self.assertIn("test", ck["metrics"])
        self.assertIn("per_button_f1", self.metrics["test"])
        self.assertIn("stick_acc", self.metrics["test"])
        self.assertIn("mean_action_rate_divergence", self.metrics["test"])

    def test_policy_roundtrip_and_determinism(self):
        p1 = bc.load_policy(self.out)
        p2 = bc.BCPolicy(bc.load_checkpoint(self.out))
        obs = np.linspace(-1, 1, bc.OBS_DIM, dtype=np.float32)
        a1 = p1.act(obs)
        a2 = p2.act(obs)
        np.testing.assert_allclose(a1["buttons_prob"], a2["buttons_prob"])
        np.testing.assert_allclose(a1["stick"], a2["stick"])
        p1.reset()
        b1 = p1.act(obs)
        np.testing.assert_allclose(a1["buttons_prob"], b1["buttons_prob"])
        self.assertEqual(a1["buttons"].shape, (bc.N_BUTTONS,))
        self.assertEqual(a1["stick"].shape, (2,))

    def test_temporal_ensemble_executes_two(self):
        policy = bc.load_policy(self.out)
        ens = bc.TemporalEnsemble(policy, execute=2, blend=0.5)
        obs = np.zeros(bc.OBS_DIM, np.float32)
        actions = [ens.act(obs) for _ in range(6)]
        self.assertEqual(len(actions), 6)
        self.assertEqual(policy.cfg.horizon, 4)
        self.assertEqual(ens.steps, 6)
        # gym_action.decode_action accepts [buttons(12), stick(2), cstick(2)]
        arr = bc.action_to_env(actions[0], "array")
        self.assertEqual(arr.shape, (16,))
        d = bc.action_to_env(actions[0], "dict")
        self.assertEqual(set(d), {"buttons", "stick", "cstick", "trigger"})

    def test_warm_start_keeps_the_prior_weights(self):
        """--init-from must continue from the checkpoint, not random init."""
        out = os.path.join(self.tmp.name, "warm.pt")
        prior = torch.load(self.out, map_location="cpu", weights_only=False)
        ckpt, metrics = bc.train(
            self.data, out, cfg=self.cfg, epochs=1, batch=8, seed=0,
            lr=0.0, mirror=False, init_from=self.out, log=lambda *a: None)
        for key, value in prior["state_dict"].items():
            torch.testing.assert_close(ckpt["state_dict"][key], value)
        self.assertEqual(ckpt["training"]["init_from"], os.path.abspath(self.out))
        self.assertGreater(metrics["epochs_run"], 0)

    def test_warm_start_rejects_a_different_architecture(self):
        other = replace(self.cfg, hidden=16)
        with self.assertRaises(ValueError):
            bc.train(self.data, os.path.join(self.tmp.name, "bad.pt"),
                     cfg=other, epochs=1, batch=8, seed=0, mirror=False,
                     init_from=self.out, log=lambda *a: None)

    def test_early_stopping_restores_best_weights(self):
        """A rising validation curve must stop early and save the best epoch."""
        cfg = bc.BCConfig(hidden=8, cstick=False)
        out = os.path.join(self.tmp.name, "early.pt")
        vals = [1.0, 0.5, 0.8, 1.1, 1.4]
        calls = {"i": 0}
        best_state = {}

        def fake_eval_loss(model, *args, **kwargs):
            i = calls["i"]
            calls["i"] += 1
            value = vals[min(i, len(vals) - 1)]
            if i == 0 or value < min(vals[:i]):
                best_state["state"] = copy.deepcopy(model.state_dict())
            return float(value)

        with mock.patch.object(bc, "_eval_loss", side_effect=fake_eval_loss):
            ckpt, metrics = bc.train(
                self.data, out, cfg=cfg, epochs=8, batch=8, seed=0,
                mirror=False, patience=2, min_epochs=1, dropout=0.0,
                log=lambda *a: None)

        self.assertEqual(calls["i"], 4)          # stopped after 2 bad epochs
        self.assertEqual(metrics["best_epoch"], 2)
        self.assertEqual(metrics["epochs_run"], 4)
        self.assertTrue(metrics["stopped_early"])
        self.assertAlmostEqual(metrics["best_val_loss"], 0.5)
        self.assertEqual(len(metrics["history"]), 4)
        self.assertEqual(ckpt["best_epoch"], 2)
        self.assertEqual(ckpt["epochs_run"], 4)

        saved = torch.load(out, map_location="cpu", weights_only=False)
        self.assertEqual(saved["best_epoch"], 2)
        self.assertTrue(saved["stopped_early"])
        for key, value in best_state["state"].items():
            torch.testing.assert_close(saved["state_dict"][key], value)

    def test_cstick_config_roundtrip_and_eval(self):
        cfg = bc.BCConfig(hidden=8, cstick=True, trigger=True, window=6,
                          horizon=3, frame_skip=5, obs_normalize=False,
                          dropout=0.2)
        torch.manual_seed(1)
        net = bc.BCPolicyNet(cfg)
        ckpt = {
            "format": "ssbu_bc_v1",
            "config": cfg.to_dict(),
            "state_dict": net.state_dict(),
            "norm": {"mean": [0.0] * bc.OBS_DIM, "std": [1.0] * bc.OBS_DIM},
            "thresholds": [0.5] * bc.N_BUTTONS,
        }
        path = os.path.join(self.tmp.name, "cstick.pt")
        torch.save(ckpt, path)

        policy = bc.load_policy(path)
        self.assertTrue(policy.cfg.cstick)
        self.assertTrue(policy.cfg.trigger)
        self.assertEqual(policy.cfg.window, 6)
        self.assertEqual(policy.cfg.horizon, 3)
        self.assertEqual(policy.cfg.frame_skip, 5)
        self.assertFalse(policy.cfg.obs_normalize)
        self.assertTrue(policy.cfg.shield_from_status)
        self.assertAlmostEqual(policy.cfg.dropout, 0.2)

        obs = np.zeros(bc.OBS_DIM, np.float32)
        a1 = policy.act(obs)
        policy.reset()
        a2 = policy.act(obs)
        np.testing.assert_allclose(a1["buttons_prob"], a2["buttons_prob"])
        plan = policy.plan()
        self.assertEqual(len(plan), 3)
        self.assertEqual(plan[0]["cstick"].shape, (2,))
        self.assertEqual(bc.action_to_env(a1, "array").shape, (16,))

        env = ge.MockEnv(max_steps=4, seed=0)
        result = ge.run_episodes(
            env, bc.TemporalEnsemble(policy, execute=2), episodes=1,
            max_steps=4, seed=0, action_format="array", mock=True,
            log=lambda *a: None)
        self.assertEqual(result["results"][0]["length"], 4)
        self.assertIn("cstick_mean", result["policy_stats"])

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = ge.main(["--model", path, "--mock", "--episodes", "1",
                          "--max-steps", "2", "--cstick", "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(out.getvalue())
        self.assertTrue(data["config"]["cstick"])
        self.assertTrue(data["config"]["trigger"])
        self.assertEqual(data["config"]["window"], 6)
        self.assertEqual(data["config"]["horizon"], 3)
        self.assertEqual(data["config"]["frame_skip"], 5)
        self.assertTrue(data["config"]["shield_from_status"])

    def test_mock_closed_loop(self):
        policy = bc.load_policy(self.out)
        ens = bc.TemporalEnsemble(policy, execute=2, blend=0.5)
        env = ge.MockEnv(max_steps=20, seed=0)
        result = ge.run_episodes(env, ens, episodes=2, max_steps=20, seed=0,
                                 action_format="array", mock=True,
                                 log=lambda *a: None)
        self.assertEqual(result["episodes"], 2)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(all(r["length"] <= 20 for r in result["results"]))
        self.assertIn("A", result["policy_stats"]["button_rate"])
        self.assertEqual(result["wins"] + result["losses"] +
                         result["unknown"], 2)


class RealEnvAdapterTest(unittest.TestCase):
    """Closed loop against Agent C's real SSBUGymEnv + ShmFixture (no game)."""

    def _policy(self):
        cfg = bc.BCConfig(hidden=8, cstick=False)
        torch.manual_seed(0)
        net = bc.BCPolicyNet(cfg)
        ckpt = {
            "format": "ssbu_bc_v1",
            "config": cfg.to_dict(),
            "state_dict": net.state_dict(),
            "norm": {"mean": [0.0] * bc.OBS_DIM, "std": [1.0] * bc.OBS_DIM},
            "thresholds": [0.5] * bc.N_BUTTONS,
        }
        return bc.BCPolicy(ckpt)

    def test_closed_loop_with_shm_fixture(self):
        try:
            from rl.env.gym_env import (SSBUGymEnv, ShmFixture, NoopInputAdapter)
        except Exception as exc:                       # pragma: no cover
            self.skipTest(f"gym_env not importable: {exc}")
        policy = self._policy()
        ensemble = bc.TemporalEnsemble(policy, execute=2)
        fixture = ShmFixture()
        fixture.start_publisher(hz=500)
        env = SSBUGymEnv(shm_path=fixture.path,
                         input_adapter=NoopInputAdapter(),
                         reset_fn=lambda: None, max_steps=6)
        try:
            action = policy.act(np.zeros(bc.OBS_DIM, np.float32))
            native = ge.format_action(action, "auto", env)
            self.assertIsInstance(native, dict)
            self.assertIn("buttons", native)
            result = ge.run_episodes(env, ensemble, episodes=1, max_steps=6,
                                     seed=0, action_format="auto", mock=False,
                                     log=lambda *a: None)
            self.assertEqual(result["episodes"], 1)
            self.assertEqual(result["results"][0]["length"], 6)
            # C's info uses `stocks`/`damages` lists; the adapter reads them.
            self.assertEqual(result["results"][0]["p1_stock"], 1.0)
            self.assertEqual(result["results"][0]["p2_stock"], 1.0)
            self.assertIn("A", result["policy_stats"]["button_rate"])
        finally:
            env.close()
            fixture.close()


class EvalInfoTest(unittest.TestCase):
    def test_real_env_info_shape(self):
        info = {"stocks": [1, 0], "damages": [31.0, 120.0],
                "fight_on": 0, "end_reason": "fight_off"}
        r = ge._result_from_info(info)
        self.assertEqual(r["winner"], 0)
        self.assertEqual(r["p1_stock"], 1.0)
        self.assertEqual(r["p2_damage"], 120.0)

        r = ge._result_from_info({"stocks": [0, 1], "damages": [150.0, 20.0]})
        self.assertEqual(r["winner"], 1)

        r = ge._result_from_info({"winner": "p1", "stocks": [1, 1]})
        self.assertEqual(r["winner"], 0)
        r = ge._result_from_info({"stocks": [1, 1]})
        self.assertIsNone(r["winner"])


class FakeSnapshot:
    def __init__(self, fight_on: bool):
        self.fight_on = fight_on


class FakePad:
    def __init__(self):
        self.calls = []

    def press(self, key):
        self.calls.append(("press", key))

    def release_all(self):
        self.calls.append(("release", None))

    def set_stick(self, x, y):
        self.calls.append(("stick", (x, y)))


class FakeEnv:
    """Just enough of SSBUGymEnv for the menu walk: a reader."""

    def __init__(self, fights_after: int | None):
        self.reads = 0
        self.fights_after = fights_after
        self.reader = self

    def read_latest(self, timeout=None):
        self.reads += 1
        if self.fights_after is None:
            return FakeSnapshot(False)
        return FakeSnapshot(self.reads > self.fights_after)


class MenuWalkTest(unittest.TestCase):
    """--start-menus: boot -> fight with the pad, no presses once it is fighting."""

    def test_stops_as_soon_as_the_fight_is_on(self):
        pad, env = FakePad(), FakeEnv(fights_after=0)
        with mock.patch.object(ge.time, "sleep"):
            self.assertTrue(ge.drive_menus(env, pad, 5.0, log=lambda *a: None))
        self.assertEqual(pad.calls, [])          # nothing to do: already fighting

    def test_walks_the_macro_when_the_game_is_in_the_menus(self):
        # the fight comes on right after the whole macro: the roster flick runs
        pad, env = FakePad(), FakeEnv(fights_after=11)
        with mock.patch.object(ge.time, "sleep"):
            self.assertTrue(ge.drive_menus(env, pad, 5.0, log=lambda *a: None))
        keys = [c[1] for c in pad.calls if c[0] == "press"]
        self.assertIn("A", keys)
        sticks = [c[1] for c in pad.calls if c[0] == "stick"]
        self.assertIn((0.0, 1.0), sticks)        # up into the roster
        # the walk stops with the macro (at most one retry press after it)
        self.assertLessEqual(len(keys), sum(
            1 for step in ge.MENU_MACRO
            if step[0] not in ("up", "down", "left", "right")) + 2)

    def test_repeats_the_macro_when_a_pass_misses_the_fight(self):
        # a boot slower than the first pass: the second pass has to repeat the flicks
        pad, env = FakePad(), FakeEnv(fights_after=20)
        logs = []
        with mock.patch.object(ge.time, "sleep"):
            self.assertTrue(ge.drive_menus(env, pad, 60.0, log=logs.append))
        self.assertGreater(sum(1 for c in pad.calls if c == ("stick", (0.0, 1.0))), 1)
        self.assertTrue(any("retrying" in line for line in logs))

    def test_gives_up_without_a_fight(self):
        pad, env = FakePad(), FakeEnv(fights_after=None)
        with mock.patch.object(ge.time, "sleep"), \
                mock.patch.object(ge.time, "monotonic", side_effect=[0, 0, 1e9]):
            self.assertFalse(ge.drive_menus(env, pad, 5.0, log=lambda *a: None))


# ---------------------------------------------------------------------------
# recorder
# ---------------------------------------------------------------------------
def pack_v7_block(seq=2, fight=1, stage=0, frame=10, buttons=(1, 0),
                  stick=((0.5, -0.25), (0.0, 0.0))) -> bytes:
    header = rec._HEADER_V7.pack(
        b"SSBU", 7, 2, seq, seq * 16_666_667, 60.0, os.getpid(), fight, 1,
        -240.0, 240.0, 180.0, -140.0, stage, frame, 0, 0, 0, 0)
    slots = []
    for i in range(2):
        vals = [0] * 32
        vals[bc.FIELD_INDEX["entry_id"]] = i
        vals[bc.FIELD_INDEX["kind"]] = i
        vals[bc.FIELD_INDEX["stock"]] = 3
        vals[bc.FIELD_INDEX["damage"]] = 10.0 * (i + 1)
        vals[bc.FIELD_INDEX["x"]] = 10.0 - 20.0 * i
        vals[bc.FIELD_INDEX["y"]] = 5.0
        vals[bc.FIELD_INDEX["facing"]] = 1.0 if i == 0 else -1.0
        vals[bc.FIELD_INDEX["action_state"]] = 0
        vals[bc.FIELD_INDEX["on_ground"]] = 1
        vals[bc.FIELD_INDEX["buttons"]] = buttons[i]
        vals[bc.FIELD_INDEX["stick_x"]] = stick[i][0]
        vals[bc.FIELD_INDEX["stick_y"]] = stick[i][1]
        vals[bc.FIELD_INDEX["trigger"]] = 0.5 * i
        vals[bc.FIELD_INDEX["cstick_x"]] = -0.75 * i
        vals[bc.FIELD_INDEX["raw_flags"]] = 0xFFFFFFFF
        slots.append(rec._FIGHTER_V7.pack(*vals))
    return header + b"".join(slots)


def pack_v6_block(seq=2, fight=1) -> bytes:
    header = rec._HEADER_V6.pack(
        b"SSBU", 6, 2, seq, seq * 16_666_667, 60.0, os.getpid(), fight, 1,
        -240.0, 240.0, 180.0, -140.0, 0, 0)
    slots = []
    for i in range(2):
        vals = [0] * 20
        vals[bc.FIELD_INDEX["entry_id"]] = i
        vals[bc.FIELD_INDEX["kind"]] = i
        vals[bc.FIELD_INDEX["stock"]] = 1
        vals[bc.FIELD_INDEX["x"]] = 7.0
        vals[18] = 0xFFFF              # raw_flags in v6
        slots.append(rec._FIGHTER_V6.pack(*vals))
    return header + b"".join(slots)


class RecorderTest(unittest.TestCase):
    def test_v7_reader_and_layout(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            path = fh.name
        try:
            with open(path, "wb") as fh:
                fh.write(pack_v7_block())
            reader = rec.ShmReader(path)
            self.assertEqual(reader.version, 7)
            self.assertEqual(reader.block_size, 336)
            frame = reader.read()
            self.assertIsNotNone(frame)
            self.assertEqual(frame["seq"], 2)
            self.assertEqual(frame["fight_on"], 1)
            self.assertEqual(frame["stage_id"], 0)
            self.assertEqual(frame["game_frame"], 10)
            self.assertEqual(frame["fighters"].shape, (2, 32))
            self.assertEqual(int(frame["buttons_raw"][0]), 1)
            self.assertAlmostEqual(float(frame["stick"][0, 0]), 0.5, places=5)
            self.assertAlmostEqual(float(frame["stick"][0, 1]), -0.25, places=5)
            self.assertAlmostEqual(float(frame["trigger"][0]), 0.0, places=5)
            self.assertAlmostEqual(float(frame["cstick"][1, 0]), -0.75, places=5)
            reader.close()
        finally:
            os.unlink(path)

    def test_v6_fallback_reader(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            path = fh.name
        try:
            with open(path, "wb") as fh:
                fh.write(pack_v6_block())
            reader = rec.ShmReader(path)
            self.assertEqual(reader.version, 6)
            frame = reader.read()
            self.assertIsNotNone(frame)
            self.assertEqual(frame["fighters"].shape, (2, 32))
            self.assertAlmostEqual(float(frame["fighters"][0, bc.FIELD_INDEX["x"]]),
                                   7.0)
            self.assertEqual(int(frame["buttons_raw"][0]), 0)
            reader.close()
        finally:
            os.unlink(path)

    def test_torn_read_is_detected(self):
        block = bytearray(pack_v7_block())
        struct.pack_into("<I", block, rec.SEQ_OFFSET, 3)   # odd seq
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as fh:
            path = fh.name
        try:
            with open(path, "wb") as fh:
                fh.write(block)
            reader = rec.ShmReader(path)
            self.assertIsNone(reader.read(retries=2))
            reader.close()
        finally:
            os.unlink(path)

    def test_episode_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = rec.Episode(4, {"index": 4, "p1_char": "mario"})
            for k in range(5):
                ep.append(fake_frame(2 + 2 * k, fight=1))
            path = rec.episode_path(tmp, 4)
            ep.save(path)
            loaded = rec.Episode.load(path)
            self.assertEqual(len(loaded), 5)
            np.testing.assert_allclose(loaded.raw[0], ep.raw[0])
            np.testing.assert_array_equal(loaded.buttons[0], ep.buttons[0])
            self.assertEqual(int(loaded.seq[-1]), 10)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(int(data["frame_skip"]), 5)
                self.assertEqual(data["raw_fighters"].shape, (5, 2, 32))
                self.assertEqual(data["stick"].shape, (5, 2, 2))
                self.assertEqual(data["buttons"].shape, (5, 2, 12))
                # A is bit 0 in the raw mask and index 0 in canonical order
                self.assertEqual(int(data["buttons"][0, 0, 0]), 1)
                meta = json.loads(str(data["meta_json"]))
                self.assertEqual(meta["p1_char"], "mario")
                self.assertEqual(json.loads(str(data["meta"])), meta)

    def test_recorder_episode_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = rec.RecorderCore(tmp, episodes=2, log=lambda *a: None)
            self.assertEqual(core.feed(fake_frame(2, 1)), "start")
            self.assertEqual(core.feed(fake_frame(4, 1)), "append")
            self.assertEqual(core.feed(fake_frame(8, 1)), "append")  # seq gap
            self.assertEqual(core.feed(fake_frame(10, 0)), "stop")
            self.assertEqual(len(core.done), 1)
            manifest = rec.read_manifest(tmp)
            self.assertEqual(len(manifest["episodes"]), 1)
            entry = manifest["episodes"][0]
            self.assertEqual(entry["frames"], 3)
            self.assertEqual(entry["seq_gaps"], 1)
            self.assertEqual(entry["meta"]["p1_char"], "mario")
            self.assertEqual(entry["meta"]["p2_char"], "donkey_kong")
            self.assertEqual(entry["result"]["stocks"], [1, 1])

            core.feed(fake_frame(12, 1))
            core.feed(fake_frame(14, 1))
            core.feed(fake_frame(16, 0))
            self.assertEqual(len(core.done), 2)
            self.assertTrue(core.finished)
            self.assertTrue(os.path.exists(rec.episode_path(tmp, 0)))
            self.assertTrue(os.path.exists(rec.episode_path(tmp, 1)))
            manifest = rec.read_manifest(tmp)
            self.assertEqual([e["index"] for e in manifest["episodes"]], [0, 1])

            # a new core skips the existing episodes and sees the target met
            again = rec.RecorderCore(tmp, episodes=2, log=lambda *a: None)
            self.assertTrue(again.finished)
            self.assertEqual(len(again.done), 2)

    def test_partial_resume_and_finalize(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = rec.RecorderCore(tmp, episodes=1, log=lambda *a: None)
            core.feed(fake_frame(2, 1))
            core.stop()
            self.assertTrue(os.path.exists(rec.partial_path(tmp, 0)))
            self.assertFalse(os.path.exists(rec.episode_path(tmp, 0)))

            resumed = rec.RecorderCore(tmp, episodes=1, log=lambda *a: None)
            self.assertEqual(resumed.feed(fake_frame(4, 1)), "resume")
            self.assertEqual(len(resumed.ep), 2)
            self.assertEqual(resumed.feed(fake_frame(6, 0)), "stop")
            self.assertTrue(os.path.exists(rec.episode_path(tmp, 0)))
            self.assertFalse(os.path.exists(rec.partial_path(tmp, 0)))
            final = rec.Episode.load(rec.episode_path(tmp, 0))
            self.assertEqual(len(final), 2)

    def test_stale_partial_is_finalised(self):
        with tempfile.TemporaryDirectory() as tmp:
            core = rec.RecorderCore(tmp, episodes=2, log=lambda *a: None)
            core.feed(fake_frame(2, 1))
            core.stop()
            # A new fight starts much later: the partial is a complete episode.
            resumed = rec.RecorderCore(tmp, episodes=2, resume_window_s=0.1, min_frames=1,
                                       log=lambda *a: None)
            resumed.feed(fake_frame(10_000, 1))
            self.assertEqual(len(resumed.done), 1)
            self.assertIsNotNone(resumed.ep)
            self.assertEqual(resumed.ep.index, 1)

    def test_manifest_recovered_from_existing_finals(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep = rec.Episode(0, {"index": 0, "p1_char": "mario",
                                 "p2_char": "donkey_kong",
                                 "stage": "final_destination"})
            for k in range(3):
                ep.append(fake_frame(2 + 2 * k, fight=1))
            ep.save(rec.episode_path(tmp, 0))
            self.assertFalse(os.path.exists(os.path.join(tmp, "manifest.json")))
            core = rec.RecorderCore(tmp, episodes=2, log=lambda *a: None)
            self.assertEqual(len(core.done), 1)
            manifest = rec.read_manifest(tmp)
            self.assertEqual(len(manifest["episodes"]), 1)
            self.assertTrue(manifest["episodes"][0]["recovered"])
            self.assertEqual(manifest["episodes"][0]["frames"], 3)

    def test_meta_char_names(self):
        self.assertEqual(rec.char_name(0), "mario")
        self.assertEqual(rec.char_name(1), "donkey_kong")
        self.assertEqual(rec.char_name(9), "kind_9")

    def test_layout_matches_agent_b_schema(self):
        """The duplicated v7 layout must not drift from B's schema.py."""
        try:
            from rl.memory import schema
        except Exception as exc:                       # pragma: no cover
            self.skipTest(f"schema not importable: {exc}")
        self.assertEqual(schema.VERSION, 7)
        self.assertEqual(schema.HEADER_SIZE, rec.HEADER_SIZE_V7)
        self.assertEqual(schema.FIGHTER_SIZE, rec.FIGHTER_SIZE_V7)
        self.assertEqual(schema.BLOCK_SIZE, rec.BLOCK_SIZE_V7)
        self.assertEqual(tuple(schema.FIELD_NAMES), rec.FIELDS)
        self.assertEqual(schema.SEQ_OFFSET, rec.SEQ_OFFSET)
        self.assertEqual(schema.FIGHT_OFFSET, rec.FIGHT_OFFSET)
        self.assertEqual(schema.MATCH_STATE_OFFSET, rec.MATCH_STATE_OFFSET)
        self.assertEqual(schema.STAGE_ID_OFFSET, rec.STAGE_OFFSET)
        self.assertEqual(schema.GAME_FRAME_OFFSET, rec.GAME_FRAME_OFFSET)
        self.assertEqual(tuple(schema.FIELD_NAMES), bc.FIELDS)
        self.assertEqual(tuple(schema.BUTTON_ORDER), bc.BUTTONS)
        self.assertEqual(tuple(schema.BUTTON_BITS[n] for n in schema.BUTTON_ORDER),
                         tuple(bc.BUTTON_BITS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
