#!/usr/bin/env python3
"""The status ids are verified against the live game, and used consistently.

``status_ids_verified.json`` is written by ``rl.scripts.status_scenarios``
(scripted pad input, ids recorded per decision, behaviour-based checks).  This
test needs no game: it asserts that suite's checks all passed, that every id
the code treats as a family was actually observed, and that the reader, the
observation, the BC fallback tables and the reward all agree.
"""

from __future__ import annotations

import json
import os
import unittest

from rl.bc import gym_bc as bc
from rl.env import gym_obs
from rl.memory import gamestate
from rl.memory import status_kinds as sk

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFIED = os.path.join(HERE, "..", "memory", "status_ids_verified.json")


class VerifiedTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(VERIFIED, encoding="utf-8") as fh:
            cls.data = json.load(fh)
        cls.observed = {int(k) for k in cls.data["ids"]}

    def test_every_live_check_passed(self):
        failed = [c["name"] for c in self.data["checks"] if not c["ok"]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(self.data["checks"]), 25)

    def test_the_ids_the_code_relies_on_were_observed_live(self):
        must = {sk.WAIT, sk.JUMP_SQUAT, 11, 12, 14, sk.HELPLESS_FALL,
                sk.GUARD_ON, sk.GUARD, sk.GUARD_OFF, sk.ATTACK_AIR,
                sk.LEDGE_CATCH, sk.LEDGE_HANG, 120, 122, 123, 124,
                sk.REBIRTH_PLATFORM, sk.KO, *sk.SPECIAL_KINDS}
        self.assertEqual(sorted(must - self.observed), [])

    def test_ledge_hang_point_matches_the_measurement(self):
        self.assertEqual(list(self.data["ledge"]), [sk.LEDGE_X, sk.LEDGE_Y])


class ConstantsAreConsistent(unittest.TestCase):
    def test_families_do_not_overlap(self):
        groups = {"ledge": sk.LEDGE_KINDS, "helpless": sk.HELPLESS_KINDS,
                  "special": sk.SPECIAL_KINDS, "guard": sk.GUARD_KINDS,
                  "ko": sk.KO_KINDS, "rebirth": sk.REBIRTH_KINDS,
                  "damage": sk.DAMAGE_KINDS}
        names = list(groups)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                self.assertFalse(groups[a] & groups[b], (a, b))
        # the old wrong beliefs stay wrong
        self.assertNotIn(sk.JUMP_SQUAT, sk.GUARD_KINDS)
        self.assertNotIn(sk.ATTACK_AIR, sk.HITSTUN_KINDS)
        self.assertFalse(sk.SPECIAL_KINDS & sk.HITSTUN_KINDS)
        self.assertFalse({505, 506, 507, 508, 509, 510, 511} & sk.HITSTUN_KINDS)
        self.assertNotIn(73, sk.LEDGE_KINDS)       # 73 is a damage state
        self.assertIn(73, sk.DAMAGE_KINDS)
        self.assertNotIn(20, sk.HELPLESS_KINDS)

    def test_reader_uses_the_shared_ids(self):
        self.assertEqual(gamestate.DAMAGE_STATUS_KINDS, sk.DAMAGE_KINDS)
        self.assertEqual(gamestate.GUARD_STATUS_KINDS, sk.GUARD_HELD_KINDS)

    def test_bc_fallback_tables_match_gym_obs(self):
        self.assertEqual(bc._LOCAL_CURATED_STATES,
                         tuple(s for _n, s in gym_obs.CURATED_ACTION_STATES))
        self.assertEqual(
            tuple((n, tuple(ids)) for n, ids in bc._LOCAL_FAMILIES),
            tuple((n, tuple(ids)) for n, ids in gym_obs.FAMILY_FLAGS))
        self.assertEqual(bc.GUARD_STATES, tuple(sorted(sk.GUARD_HELD_KINDS)))

    def test_named_one_hot_slots_use_the_verified_ids(self):
        named = {n: s for n, s in gym_obs.CURATED_ACTION_STATES}
        self.assertEqual(named["JUMP_SQUAT"], sk.JUMP_SQUAT)
        self.assertEqual(named["GUARD"], sk.GUARD)
        self.assertEqual(named["HELPLESS_FALL"], sk.HELPLESS_FALL)
        self.assertEqual(named["LEDGE_HANG"], sk.LEDGE_HANG)
        self.assertEqual(named["KO"], sk.KO)
        self.assertEqual(named["SPECIAL_HI"], sk.SPECIAL_HI)
        self.assertEqual(named["ATTACK_AIR"], sk.ATTACK_AIR)
        self.assertEqual(len(set(named.values())), 31)


if __name__ == "__main__":
    unittest.main()
