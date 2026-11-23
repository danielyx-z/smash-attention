"""Teleport drill: the runtime drops P1 offstage with damage, no run-off."""
from __future__ import annotations

import unittest

import numpy as np

from rl.env.gym_drill import DrillSetupError, RecoveryDrillEnv


def fighter(y, ground=0, state=14, dmg=0.0, x=100.0):
    return {"x": x, "y": y, "on_ground": ground, "action_state": state,
            "damage": dmg, "is_on_rebirth": 0}


class FakeBase:
    """Records the drop request and answers like the runtime would."""
    ego_slot = 0
    max_steps = 900

    def __init__(self, land=True):
        self.drill_drop = None
        self.drops = []
        self.steps = 0
        self.land = land

    @property
    def base_env(self):
        return self

    def _info(self, f):
        return {"raw_fighters": [f, fighter(0, 1, 0)]}

    def reset(self, seed=None):
        drop, self.drill_drop = self.drill_drop, None
        self.drops.append(drop)
        self.steps = 0
        if drop is None:
            return np.zeros(2, np.float32), self._info(fighter(0, 1, 0))
        y = drop["y"] if self.land else 5.0
        return np.zeros(2, np.float32), self._info(
            fighter(y, 0, 16 if drop["helpless"] else 14, drop["damage"],
                    drop["x"]))

    def step(self, action):
        self.steps += 1
        back = self.steps >= 5
        f = fighter(0, 1, 0) if back else fighter(-50, 0, 14)
        return np.zeros(2, np.float32), 0.0, False, False, self._info(f)


class TeleportDrillTest(unittest.TestCase):
    def _env(self, **kw):
        base = FakeBase(kw.pop("land", True))
        return base, RecoveryDrillEnv(base, teleport=True, drill_prob=1.0,
                                      handoff_y_range=(-90.0, -20.0), **kw)

    def test_drop_is_requested_with_damage_offstage_and_no_setup_steps(self):
        base, env = self._env(damage_range=(0.0, 100.0), helpless_prob=0.0)
        _obs, info = env.reset(seed=3)
        drop = base.drops[0]
        self.assertTrue(95.0 <= abs(drop["x"]) <= 140.0)
        self.assertTrue(-90.0 <= drop["y"] <= -20.0)
        self.assertTrue(0.0 <= drop["damage"] <= 100.0)
        self.assertEqual(base.steps, 0)                  # nothing scripted
        self.assertEqual(env.setup_observations, [])

    def test_drops_vary_across_seeds_and_are_reproducible(self):
        base, env = self._env()
        for s in range(40):
            env.reset(seed=s)
        dmg = {round(d["damage"]) for d in base.drops}
        self.assertGreater(len(dmg), 20)
        self.assertTrue(any(d["helpless"] for d in base.drops))
        self.assertTrue(any(not d["helpless"] for d in base.drops))
        base2, env2 = self._env()
        for s in range(40):
            env2.reset(seed=s)
        self.assertEqual(base.drops, base2.drops)

    def test_jump_spent_takes_one_step_with_jump_held(self):
        base, env = self._env(helpless_prob=0.0, jump_spent_prob=1.0)
        seen = []
        orig = base.step
        base.step = lambda a: (seen.append(a), orig(a))[1]
        env.reset(seed=1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(float(seen[0]["buttons"][2]), 1.0)      # X
        self.assertEqual(len(env.setup_observations), 1)

    def test_helpless_drops_stay_in_the_winnable_zone(self):
        base, env = self._env(helpless_prob=1.0)
        for s in range(30):
            env.reset(seed=s)
        for d in base.drops:
            self.assertTrue(d["helpless"])
            self.assertTrue(85.0 <= abs(d["x"]) <= 100.0)
            self.assertTrue(-35.0 <= d["y"] <= -15.0)

    def test_helpless_start_never_spends_a_jump(self):
        base, env = self._env(helpless_prob=1.0, jump_spent_prob=1.0)
        env.reset(seed=1)
        self.assertEqual(base.steps, 0)

    def test_a_drop_that_stays_on_stage_is_a_setup_error(self):
        _base, env = self._env(land=False)
        env.attempts = 1
        with self.assertRaises(DrillSetupError):
            env._reset_teleport(1)

    def test_drill_ends_on_recovery_and_respects_the_step_cap(self):
        base, env = self._env(drill_max_steps=3)
        env.reset(seed=1)
        out = [env.step({})[2:4] for _ in range(3)]
        self.assertEqual(out[-1], (False, True))        # truncated at the cap


if __name__ == "__main__":
    unittest.main()
