#!/usr/bin/env python3
"""Unit tests for the PPO fine-tuner (no game, no live reader)."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rl.bc import gym_bc as bc          # noqa: E402
from rl.rl import ppo                  # noqa: E402

torch.set_num_threads(1)


def tiny_checkpoint(path: str) -> dict:
    cfg = bc.BCConfig(hidden=8, window=4, horizon=4, action_mode="vocab",
                      vocab_size=8, stick_mode="dir", dropout=0.0)
    net = bc.BCPolicyNet(cfg)
    keys = np.array([bc.action_keys({
        "buttons": np.zeros((1, bc.N_BUTTONS), np.float32),
        "stick": np.ones((1, 2), np.float32),
        "cstick": np.zeros((1, 2), np.float32),
        "trigger": np.zeros(1, np.float32),
    })[0]], np.int64)
    ckpt = {
        "format": "ssbu_bc_v1",
        "config": cfg.to_dict(),
        "state_dict": net.state_dict(),
        "norm": {"mean": [0.0] * bc.OBS_DIM, "std": [1.0] * bc.OBS_DIM},
        "thresholds": [0.5] * bc.N_BUTTONS,
        "vocab": bc.ActionVocab(keys=keys).to_dict(),
    }
    torch.save(ckpt, path)
    return ckpt


class ValueHeadTest(unittest.TestCase):
    def test_value_head_is_optional(self):
        cfg = bc.BCConfig(hidden=8, value_head=True)
        net = bc.BCPolicyNet(cfg)
        obs = torch.zeros(3, cfg.window, cfg.obs_dim)
        prev = torch.zeros(3, cfg.prev_dim)
        out = net(obs, prev)
        self.assertEqual(tuple(out["value"].shape), (3,))
        self.assertTrue(torch.isfinite(out["value"]).all())
        plain = bc.BCPolicyNet(replace(cfg, value_head=False))(obs, prev)
        self.assertNotIn("value", plain)


class TokenDecodeTest(unittest.TestCase):
    def test_repeat_replays_the_previous_action(self):
        cfg = bc.BCConfig(action_mode="vocab", stick_mode="dir",
                          window=4, horizon=4)
        action = {
            "buttons": np.zeros((1, bc.N_BUTTONS), np.float32),
            "stick": np.zeros((1, 2), np.float32),
            "cstick": np.zeros((1, 2), np.float32),
            "trigger": np.zeros(1, np.float32),
        }
        action["stick"][0] = [1.0, 0.0]          # E
        action["buttons"][0, bc.BUTTON_INDEX["A"]] = 1.0
        key = int(bc.action_keys(action)[0])
        vocab = bc.ActionVocab(keys=np.array([key], np.int64))
        decoded = ppo.token_action(1, np.zeros(cfg.prev_dim, np.float32),
                                   vocab, cfg)
        self.assertEqual(float(decoded["buttons"][bc.BUTTON_INDEX["A"]]), 1.0)
        np.testing.assert_allclose(decoded["stick"], [1.0, 0.0])
        prev = bc.action_vector(decoded, cfg)
        repeat = ppo.token_action(bc.REPEAT_TOKEN, prev, vocab, cfg)
        np.testing.assert_array_equal(repeat["buttons"], decoded["buttons"])
        np.testing.assert_allclose(repeat["stick"], decoded["stick"])


class TrainingStatsTest(unittest.TestCase):
    def test_stats_file_tracks_episodes_and_iteration(self):
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "stats.json")
            stats = ppo.TrainingStats(path, cpu_level=3, iterations=20)
            stats.record_episode({"p1_damage": 40.0, "p2_damage": 50.0,
                                  "winner": None})
            stats.record_episode({"p1_damage": 30.0, "p2_damage": 10.0,
                                  "winner": 1})
            stats.record_iteration(2, {"policy_loss": -0.1,
                                       "value_loss": 0.01,
                                       "entropy": 1.2,
                                       "kl_bc": 0.03})
            stats.finish()
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            self.assertEqual(data["cpu_level"], 3)
            self.assertEqual(data["episodes"], 2)
            self.assertEqual(data["positive"], 1)
            self.assertEqual(data["losses"], 1)
            self.assertAlmostEqual(data["mean_dealt"], 30.0)
            self.assertAlmostEqual(data["mean_diff"], -5.0)
            self.assertAlmostEqual(data["positive_rate"], 0.5)
            self.assertEqual(data["iteration"], 2)
            self.assertFalse(data["active"])


class CheckpointNamingTest(unittest.TestCase):
    def test_per_iteration_path_is_stable(self):
        self.assertEqual(
            ppo.iteration_checkpoint_path("data/models/rl.pt", 7),
            "data/models/rl_iter007.pt")
        self.assertEqual(
            ppo.iteration_checkpoint_path("data/models/rl", 12),
            "data/models/rl_iter012.pt")


class EpisodeRowTest(unittest.TestCase):
    def test_peak_damage_is_reported_when_final_frame_reset_to_zero(self):
        row = ppo._episode_row({"damages": [0.0, 0.0]}, 900, 1.0,
                               False, True,
                               peak={"p1_damage": 42.0, "p2_damage": 17.0})
        self.assertEqual(row["p1_damage"], 42.0)
        self.assertEqual(row["p2_damage"], 17.0)
        plain = ppo._episode_row({"damages": [0.0, 0.0]}, 900, 1.0,
                                 False, True)
        self.assertEqual(plain["p1_damage"], 0.0)


class GAETest(unittest.TestCase):
    def test_matches_manual_returns(self):
        rewards = np.array([1.0, 1.0], np.float32)
        values = np.array([0.5, 0.5], np.float32)
        next_values = np.array([0.5, 0.0], np.float32)
        terminals = np.array([False, True])
        adv, ret = ppo.compute_gae(rewards, values, next_values, terminals,
                                   np.zeros(2, bool), gamma=0.5, lam=1.0)
        np.testing.assert_allclose(adv, [1.0, 0.5], rtol=1e-6)
        np.testing.assert_allclose(ret, [1.5, 1.0], rtol=1e-6)

    def test_truncation_bootstraps_from_the_next_value(self):
        rewards = np.array([1.0, 1.0], np.float32)
        values = np.array([0.5, 0.5], np.float32)
        next_values = np.array([0.5, 2.0], np.float32)
        terminals = np.array([False, False])
        truncateds = np.array([False, True])
        adv, ret = ppo.compute_gae(rewards, values, next_values, terminals,
                                   truncateds, gamma=0.5, lam=1.0)
        # t=1: 1 + .5*2 - .5 = 1.5; t=0 chains the same-episode return
        np.testing.assert_allclose(adv, [1.5, 1.5], rtol=1e-6)
        np.testing.assert_allclose(ret, [2.0, 2.0], rtol=1e-6)

    def test_new_episode_return_does_not_leak_backwards(self):
        # t=1 is the truncated end of episode A; t=2 is the first step of B.
        rewards = np.array([1.0, 1.0, 10.0], np.float32)
        values = np.array([0.5, 0.5, 0.0], np.float32)
        next_values = np.array([0.5, 2.0, 0.0], np.float32)
        terminals = np.array([False, False, False])
        truncateds = np.array([False, True, False])
        adv, _ret = ppo.compute_gae(rewards, values, next_values, terminals,
                                    truncateds, gamma=0.5, lam=1.0)
        np.testing.assert_allclose(adv, [1.5, 1.5, 10.0], rtol=1e-6)


class PPOUpdateTest(unittest.TestCase):
    def test_update_and_mock_rollout_are_finite(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bc.pt")
            tiny_checkpoint(path)
            model, cfg, vocab, mean, std, _ = ppo.load_bc_model(
                path, "cpu", dropout=0.0)
            self.assertIsNotNone(model.value_head)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            n = 8
            batch = {
                "windows": np.zeros((n, cfg.window, cfg.obs_dim), np.float32),
                "prevs": np.zeros((n, cfg.prev_dim), np.float32),
                "actions": np.zeros(n, np.int64),
                "logps": np.zeros(n, np.float32),
                "values": np.zeros(n, np.float32),
                "advantages": np.ones(n, np.float32),
                "returns": np.ones(n, np.float32),
            }
            stats = ppo.ppo_update(model, optimizer, batch,
                                   ppo.PPOConfig(epochs=1, minibatch=4,
                                                 device="cpu"),
                                   mean, std, "cpu")
            for value in stats.values():
                self.assertTrue(np.isfinite(value))
            kl_batch = dict(batch)
            kl_batch["ref_logits"] = np.zeros((n, cfg.vocab_size),
                                              np.float32)
            kl_stats = ppo.ppo_update(
                model, optimizer, kl_batch,
                ppo.PPOConfig(epochs=1, minibatch=4, device="cpu",
                              kl_coef=0.1),
                mean, std, "cpu")
            self.assertTrue(np.isfinite(kl_stats["kl_bc"]))
            self.assertGreaterEqual(kl_stats["kl_bc"], 0.0)
            state = ppo.PolicyState(cfg)
            env = ppo.MockCombatEnv(max_steps=6, seed=0)
            rollout, episodes = ppo.collect_rollout(
                env, model, state, cfg, vocab, mean, std, "cpu",
                steps=6, seed=0)
            self.assertEqual(rollout["actions"].shape, (6,))
            self.assertEqual(rollout["windows"].shape,
                             (6, cfg.window, cfg.obs_dim))
            self.assertTrue(np.isfinite(rollout["rewards"]).all())
            self.assertGreaterEqual(len(episodes), 1)


if __name__ == "__main__":
    unittest.main()
