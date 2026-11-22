"""Memory features (Up Special used) and widening a trained model for them."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.bc import gym_bc as bc
from rl.env import gym_memory
from rl.memory import status_kinds as sk
from rl.scripts.add_obs_features import widen_checkpoint


def fighter(state, on_ground=0, rebirth=0):
    return {"action_state": state, "on_ground": on_ground,
            "is_on_rebirth": rebirth}


class UpbRuleTest(unittest.TestCase):
    def test_set_by_up_special_and_helpless_fall(self):
        self.assertTrue(gym_memory.upb_update(False, fighter(sk.SPECIAL_HI)))
        self.assertTrue(gym_memory.upb_update(False, fighter(sk.HELPLESS_FALL)))

    def test_kept_through_a_fall_and_through_being_hit(self):
        for state in (14, 11, 71, 72, 73, 54, 477):          # fall, jump, hit, aerial
            self.assertTrue(gym_memory.upb_update(True, fighter(state)), state)

    def test_restored_by_ground_ledge_ko_and_platform(self):
        self.assertFalse(gym_memory.upb_update(True, fighter(0, on_ground=1)))
        for state in (117, 118, 119, 124, sk.KO, sk.DEAD_ENTRY, sk.REBIRTH_PLATFORM):
            self.assertFalse(gym_memory.upb_update(True, fighter(state)), state)
        self.assertFalse(gym_memory.upb_update(True, fighter(14, rebirth=1)))

    def test_other_specials_do_not_set_it(self):
        for state in (476, 477, 479):
            self.assertFalse(gym_memory.upb_update(False, fighter(state)))

    def test_garbage_input_keeps_the_value(self):
        self.assertTrue(gym_memory.upb_update(True, {"action_state": "x"}))


class _Env:
    ego_slot = 0

    def __init__(self, states):
        self.states = list(states)
        self.i = 0

    def _info(self):
        f = self.states[min(self.i, len(self.states) - 1)]
        return {"raw_fighters": [f, fighter(0, on_ground=1)]}

    def reset(self, seed=None):
        self.i = 0
        return np.arange(4, dtype=np.float32), self._info()

    def step(self, action):
        self.i += 1
        return np.arange(4, dtype=np.float32), 0.0, False, False, self._info()


class ExtraObsEnvTest(unittest.TestCase):
    def test_appends_the_flag_and_tracks_a_recovery(self):
        env = gym_memory.ExtraObsEnv(_Env([
            fighter(0, on_ground=1), fighter(14), fighter(sk.SPECIAL_HI),
            fighter(sk.HELPLESS_FALL), fighter(71), fighter(119),
            fighter(14)]), ["ego_upb_used"])
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (5,))
        flags = [float(obs[-1])]
        for _ in range(6):
            obs, *_ = env.step(None)
            flags.append(float(obs[-1]))
        #            ground fall  upb  helpless hit  ledge fall
        self.assertEqual(flags, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertTrue(np.array_equal(obs[:4], np.arange(4)))

    def test_reset_clears_the_flag(self):
        env = gym_memory.ExtraObsEnv(
            _Env([fighter(0, on_ground=1), fighter(sk.SPECIAL_HI)]),
            ["ego_upb_used"])
        env.reset()
        obs, *_ = env.step(None)
        self.assertEqual(float(obs[-1]), 1.0)
        obs, _ = env.reset()
        self.assertEqual(float(obs[-1]), 0.0)

    def test_unknown_feature_is_rejected(self):
        with self.assertRaises(ValueError):
            gym_memory.ExtraObsEnv(_Env([fighter(0)]), ["nope"])

    def test_mirror_keeps_extra_features(self):
        from rl.rl import mirror
        obs = np.zeros(bc.OBS_DIM + 1, np.float32)
        obs[0], obs[-1] = 5.0, 1.0
        out = mirror.mirror_observation(obs)
        self.assertEqual(out.shape, obs.shape)
        self.assertEqual(float(out[0]), -5.0)
        self.assertEqual(float(out[-1]), 1.0)


def _tiny_checkpoint():
    cfg = bc.BCConfig(hidden=16, window=4, horizon=2, action_mode="vocab",
                      vocab_size=9, stick_mode="dir", value_head=True,
                      aux_weight=0.5)
    model = bc.BCPolicyNet(cfg)
    return model, {
        "config": cfg.to_dict(), "state_dict": model.state_dict(),
        "norm": {"mean": [0.0] * cfg.obs_dim, "std": [1.0] * cfg.obs_dim},
        "meta": {},
    }


class WidenTest(unittest.TestCase):
    def test_widened_model_is_identical_for_any_value_of_the_new_input(self):
        model, ck = _tiny_checkpoint()
        out = widen_checkpoint(ck, ["ego_upb_used"])
        cfg0 = bc.BCConfig(**{k: v for k, v in ck["config"].items()
                              if k in bc.BCConfig.__dataclass_fields__})
        cfg1 = bc.BCConfig(**{k: v for k, v in out["config"].items()
                              if k in bc.BCConfig.__dataclass_fields__})
        self.assertEqual(cfg1.obs_dim, cfg0.obs_dim + 1)
        self.assertEqual(cfg1.extra_obs, ["ego_upb_used"])
        wide = bc.BCPolicyNet(cfg1)
        wide.load_state_dict(out["state_dict"])
        model.eval(); wide.eval()
        torch.manual_seed(0)
        w = torch.randn(5, cfg0.window, cfg0.obs_dim)
        p = torch.randn(5, cfg0.prev_dim)
        with torch.no_grad():
            ref = model(w, p)
            for extra in (0.0, 1.0, -4.0):
                got = wide(torch.cat([w, torch.full((5, cfg0.window, 1), extra)], -1), p)
                for k, v in ref.items():
                    if torch.is_tensor(v):
                        self.assertEqual(float((v - got[k]).abs().max()), 0.0, k)
        self.assertEqual(len(out["norm"]["mean"]), cfg1.obs_dim)
        self.assertEqual(out["norm"]["std"][-1], 1.0)

    def test_new_weights_receive_gradient(self):
        model, ck = _tiny_checkpoint()
        out = widen_checkpoint(ck, ["ego_upb_used"])
        cfg1 = bc.BCConfig(**{k: v for k, v in out["config"].items()
                              if k in bc.BCConfig.__dataclass_fields__})
        wide = bc.BCPolicyNet(cfg1)
        wide.load_state_dict(out["state_dict"])
        w = torch.randn(3, cfg1.window, cfg1.obs_dim)
        w[..., -1] = 1.0
        loss = wide(w, torch.randn(3, cfg1.prev_dim))["action"].pow(2).sum()
        loss.backward()
        g = wide.encoder.weight_ih_l0.grad[:, -1]
        self.assertGreater(float(g.abs().sum()), 0.0)      # it can learn

    def test_cannot_add_the_same_feature_twice(self):
        _m, ck = _tiny_checkpoint()
        out = widen_checkpoint(ck, ["ego_upb_used"])
        with self.assertRaises(ValueError):
            widen_checkpoint(out, ["ego_upb_used"])

    def test_batch_logits_lets_an_old_reference_look_at_the_leading_columns(self):
        from rl.rl import ppo
        model, ck = _tiny_checkpoint()
        model.eval()
        n, window = 6, model.cfg.window
        rng = np.random.default_rng(1)
        base = rng.normal(size=(n, window, model.cfg.obs_dim)).astype(np.float32)
        wide = np.concatenate([base, np.ones((n, window, 1), np.float32)], -1)
        prevs = rng.normal(size=(n, model.cfg.prev_dim)).astype(np.float32)
        mean = torch.zeros(model.cfg.obs_dim + 1)
        std = torch.ones(model.cfg.obs_dim + 1)
        a = ppo.batch_logits(model, base, prevs, mean[:-1], std[:-1], "cpu")
        b = ppo.batch_logits(model, wide, prevs, mean, std, "cpu",
                             obs_dim=model.cfg.obs_dim)
        np.testing.assert_allclose(a, b, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
