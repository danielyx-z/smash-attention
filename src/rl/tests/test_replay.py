"""Recent-batch replay and the safer PPO update."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from rl.bc import gym_bc as bc
from rl.rl import ppo


def _setup(n=64):
    cfg = bc.BCConfig(hidden=16, window=4, horizon=2, action_mode="vocab",
                      vocab_size=9, stick_mode="dir", value_head=True)
    model = bc.BCPolicyNet(cfg)
    d = cfg.obs_dim
    rng = np.random.default_rng(0)

    def batch(seed):
        r = np.random.default_rng(seed)
        return {
            "windows": r.normal(size=(n, cfg.window, d)).astype(np.float32),
            "prevs": np.zeros((n, cfg.prev_dim), np.float32),
            "actions": r.integers(1, 9, n).astype(np.int64),
            "logps": np.full(n, -2.0, np.float32),
            "values": np.zeros(n, np.float32),
            "rewards": r.normal(size=n).astype(np.float32),
            "next_windows": r.normal(size=(n, cfg.window, d)).astype(np.float32),
            "next_prevs": np.zeros((n, cfg.prev_dim), np.float32),
            "terminals": np.zeros(n, bool), "truncateds": np.zeros(n, bool),
            "offstage": np.zeros(n, bool),
            "advantages": r.normal(size=n).astype(np.float32),
            "returns": r.normal(size=n).astype(np.float32),
        }
    return cfg, model, d, batch, torch.zeros(d), torch.ones(d)


class ReplayTest(unittest.TestCase):
    def test_merge_concatenates_and_marks_fresh(self):
        cfg, model, _d, batch, mean, std = _setup()
        pc = ppo.PPOConfig(epochs=1, minibatch=32, device="cpu")
        cur, old1, old2 = batch(1), batch(2), batch(3)
        m = ppo.merge_replay(cur, [old1, old2], model, pc, mean, std, "cpu")
        self.assertEqual(len(m["actions"]), 64 * 3)
        self.assertEqual(int(m["fresh"].sum()), 64)
        self.assertTrue(m["fresh"][:64].all() and not m["fresh"][64:].any())
        # the stored (stale) advantages are replaced, normalised per batch
        self.assertNotEqual(float(m["advantages"][64]), float(old1["advantages"][0]))
        self.assertAlmostEqual(float(m["advantages"][64:128].mean()), 0.0, places=4)
        np.testing.assert_array_equal(m["logps"][64:128], old1["logps"])

    def test_update_runs_on_merged_data_and_reports_diagnostics(self):
        cfg, model, _d, batch, mean, std = _setup()
        pc = ppo.PPOConfig(epochs=2, minibatch=32, device="cpu", lr=1e-3,
                           dual_clip=3.0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        m = ppo.merge_replay(batch(1), [batch(2)], model, pc, mean, std, "cpu")
        out = ppo.ppo_update(model, opt, m, pc, mean, std, "cpu")
        for k in ("clipfrac", "approx_kl", "epochs_run", "policy_loss"):
            self.assertIn(k, out)
        self.assertTrue(np.isfinite(out["policy_loss"]))
        self.assertEqual(out["epochs_run"], 2.0)

    def test_target_kl_stops_the_epochs_early(self):
        cfg, model, _d, batch, mean, std = _setup()
        pc = ppo.PPOConfig(epochs=6, minibatch=32, device="cpu", lr=1e-3,
                           target_kl=1e-9)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        b = batch(1)
        b["fresh"] = np.ones(64, bool)
        b["logps"] = np.full(64, -0.1, np.float32)       # far from the policy
        out = ppo.ppo_update(model, opt, b, pc, mean, std, "cpu")
        self.assertLess(out["epochs_run"], 6.0)

    def test_dual_clip_bounds_the_loss_for_negative_advantages(self):
        cfg, model, _d, batch, mean, std = _setup()
        pc = ppo.PPOConfig(epochs=1, minibatch=64, device="cpu", dual_clip=3.0)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0 + 1e-12)
        b = batch(1)
        b["advantages"] = np.full(64, -1.0, np.float32)
        b["logps"] = np.full(64, -40.0, np.float32)      # ratio is enormous
        out = ppo.ppo_update(model, opt, b, pc, mean, std, "cpu")
        self.assertLess(abs(out["policy_loss"]), 3.0 + 1e-3)


if __name__ == "__main__":
    unittest.main()


class CpuRolloutTest(unittest.TestCase):
    def test_cpu_copy_gives_the_same_actions_after_a_sync(self):
        import copy
        cfg, model, d, _batch, mean, std = _setup()
        vocab = bc.ActionVocab(keys=np.arange(9, dtype=np.int64), observed=9)
        play = copy.deepcopy(model).eval()
        with torch.no_grad():                      # the learner moves on
            for p in model.parameters():
                p.add_(0.1)
        play.load_state_dict({k: v.detach().cpu()
                              for k, v in model.state_dict().items()})
        st = ppo.PolicyState(cfg)
        for _ in range(cfg.window):
            st.observe(np.random.default_rng(1).normal(size=d))
        w = torch.from_numpy(st.window())[None]
        pv = torch.from_numpy(st.prev)[None]
        model.eval()
        with torch.no_grad():
            a = model(w, pv)["action"]
            b = play(w, pv)["action"]
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
