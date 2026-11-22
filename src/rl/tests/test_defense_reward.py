"""Shield-block / dodge reward and recovery from a failed step."""
from __future__ import annotations

import unittest

import numpy as np

from rl.env.gym_defense import DefenseRewardEnv
from rl.memory import status_kinds as sk


def info(ego_state, ego_dmg=0.0, opp_state=0, dx=20.0, dy=0.0):
    return {"raw_fighters": [
        {"action_state": ego_state, "damage": ego_dmg, "x": 0.0, "y": 0.0},
        {"action_state": opp_state, "damage": 0.0, "x": dx, "y": dy}]}


class _Seq:
    """Replays a list of infos; reward 0."""
    ego_slot = 0

    def __init__(self, infos):
        self.infos = infos
        self.i = 0

    def reset(self, seed=None):
        self.i = 0
        return np.zeros(2, np.float32), self.infos[0]

    def step(self, action):
        self.i = min(self.i + 1, len(self.infos) - 1)
        return np.zeros(2, np.float32), 0.0, False, False, self.infos[self.i]


def run(infos, **kw):
    env = DefenseRewardEnv(_Seq(infos), **kw)
    env.reset()
    return [env.step(None)[1] for _ in range(len(infos) - 1)], env


class BlockTest(unittest.TestCase):
    def test_block_pays_once_when_the_shield_is_hit_unhurt(self):
        r, env = run([info(sk.GUARD), info(sk.GUARD), info(sk.GUARD_DAMAGE),
                      info(sk.GUARD_DAMAGE), info(sk.GUARD)], block_bonus=0.2,
                     dodge_bonus=0.0)
        self.assertEqual(r, [0.0, 0.2, 0.0, 0.0])
        self.assertEqual(env.blocks, 1)

    def test_no_bonus_if_damage_rose(self):
        r, _ = run([info(sk.GUARD, 10), info(sk.GUARD_DAMAGE, 18)],
                   block_bonus=0.2)
        self.assertEqual(r, [0.0])

    def test_guard_damage_without_guard_before_it_pays_nothing(self):
        r, _ = run([info(0), info(sk.GUARD_DAMAGE)], block_bonus=0.2)
        self.assertEqual(r, [0.0])


class DodgeTest(unittest.TestCase):
    def test_dodging_a_real_attack_pays_when_the_dodge_ends(self):
        r, env = run([info(0), info(31, opp_state=54), info(31, opp_state=54),
                      info(0)], dodge_bonus=0.15, block_bonus=0.0)
        self.assertEqual(r, [0.0, 0.0, 0.15])
        self.assertEqual(env.dodges, 1)

    def test_dodging_nothing_pays_nothing(self):
        r, _ = run([info(0), info(34, opp_state=0), info(34), info(0)],
                   dodge_bonus=0.15)
        self.assertEqual(sum(r), 0.0)

    def test_attack_out_of_range_pays_nothing(self):
        r, _ = run([info(0), info(32, opp_state=54, dx=200.0), info(0)],
                   dodge_bonus=0.15)
        self.assertEqual(sum(r), 0.0)

    def test_getting_hit_during_the_dodge_pays_nothing(self):
        r, _ = run([info(0, 0), info(31, 0, opp_state=54),
                    info(31, 12, opp_state=54), info(71, 12, opp_state=54)],
                   dodge_bonus=0.15)
        self.assertEqual(sum(r), 0.0)

    def test_reset_clears_the_tracking(self):
        env = DefenseRewardEnv(
            _Seq([info(0), info(31, opp_state=54), info(31, opp_state=54)]),
            dodge_bonus=0.15)
        env.reset()
        env.step(None)
        env.step(None)                                # mid-dodge, threat seen
        self.assertTrue(env._dodge_threat)
        env.reset()                                   # a new episode
        self.assertFalse(env._dodge_threat)
        self.assertIsNone(env._dodge_damage)


class StepRecoveryTest(unittest.TestCase):
    def test_a_failed_step_is_recovered_and_the_rollout_completes(self):
        import torch
        from rl.bc import gym_bc as bc
        from rl.rl import ppo

        class Flaky(ppo.MockCombatEnv):
            calls = 0

            def step(self, action):
                Flaky.calls += 1
                if Flaky.calls == 3:
                    raise TimeoutError("no snapshot tagged game frame")
                return super().step(action)

        cfg = bc.BCConfig(hidden=16, window=4, horizon=2, action_mode="vocab",
                          vocab_size=9, stick_mode="dir", value_head=True)
        model = bc.BCPolicyNet(cfg)
        vocab = bc.ActionVocab(keys=np.arange(9, dtype=np.int64),
                               observed=9) if hasattr(bc, "ActionVocab") else None
        mean = torch.zeros(cfg.obs_dim)
        std = torch.ones(cfg.obs_dim)
        state = ppo.PolicyState(cfg)
        seen = []
        env = Flaky(max_steps=50, seed=0)
        batch, _eps = ppo.collect_rollout(
            env, model, state, cfg, vocab, mean, std, "cpu", steps=8, seed=0,
            on_step_error=lambda exc: seen.append(type(exc).__name__))
        self.assertEqual(seen, ["TimeoutError"])
        self.assertEqual(len(batch["rewards"]), 8)
        self.assertTrue(bool(batch["truncateds"][1]))    # chain cut before the gap

    def test_without_a_handler_the_error_propagates(self):
        import torch
        from rl.bc import gym_bc as bc
        from rl.rl import ppo

        class Dead(ppo.MockCombatEnv):
            def step(self, action):
                raise TimeoutError("x")

        cfg = bc.BCConfig(hidden=16, window=4, horizon=2, action_mode="vocab",
                          vocab_size=9, stick_mode="dir", value_head=True)
        model = bc.BCPolicyNet(cfg)
        vocab = bc.ActionVocab(keys=np.arange(9, dtype=np.int64), observed=9)
        with self.assertRaises(TimeoutError):
            ppo.collect_rollout(Dead(max_steps=5, seed=0), model,
                                ppo.PolicyState(cfg), cfg, vocab,
                                torch.zeros(cfg.obs_dim), torch.ones(cfg.obs_dim),
                                "cpu", steps=4, seed=0)


if __name__ == "__main__":
    unittest.main()
