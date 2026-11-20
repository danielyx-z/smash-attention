"""Force a way off the ledge instead of hanging there forever.

A fighter holding the ledge (status 118/119) is safe, so a policy that has no
reason to move can sit there for the whole episode.  After ``max_hang``
consecutive decisions on the ledge this wrapper replaces the policy's action
with a getup (stick up) and charges ``penalty`` per forced decision, so the
policy learns to act before the timeout.  Training only: it sits *under* the
drill and mirror wrappers, so the override is in the game's frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.memory import status_kinds

HANG_STATES = frozenset({status_kinds.LEDGE_CATCH, status_kinds.LEDGE_HANG})


def _ego_state(info: dict, slot: int = 0) -> int:
    f = (info.get("raw_fighters") or [])
    if len(f) <= slot:
        return -1
    try:
        return int(f[slot].get("action_state", -1))
    except (TypeError, ValueError):
        return -1


def getup_action() -> dict:
    return {"buttons": np.zeros(12, np.float32),
            "stick": np.array([0.0, 1.0], np.float32),
            "cstick": np.zeros(2, np.float32), "trigger": 0.0}


class LedgeTimeoutEnv:
    def __init__(self, env: Any, max_hang: int = 45, penalty: float = 0.02,
                 log=None):
        self.env = env
        self.max_hang = max(1, int(max_hang))
        self.penalty = float(penalty)
        self.log = log or (lambda *_a, **_k: None)
        self._hang = 0
        self.forced = 0

    def __getattr__(self, name):          # delegate everything else
        return getattr(self.env, name)

    @property
    def base_env(self):
        return getattr(self.env, "base_env", self.env)

    def reset(self, *args, **kwargs):
        self._hang = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        forced = self._hang >= self.max_hang
        if forced:
            action = getup_action()
            self.forced += 1
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        slot = int(getattr(self.env, "ego_slot", 0) or 0)
        if _ego_state(info, slot) in HANG_STATES:
            self._hang += 1
        elif not forced or _ego_state(info, slot) not in HANG_STATES:
            self._hang = 0
        if forced:
            reward = float(reward) - self.penalty
            info = dict(info, ledge_forced=True)
        return obs, reward, terminated, truncated, info
