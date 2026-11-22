"""Extra observation features the game state does not carry.

The base observation is one frame of state.  Some facts need memory: whether
the Up Special was already used since the last safe state decides whether a
recovery can still be saved (Up Special ends in a helpless fall), and 58-70 %
of the agent's deaths were exactly that.  ``ExtraObsEnv`` appends such
features to the observation vector; a checkpoint lists them in
``config.extra_obs`` so the trainer and the evaluator wrap the env the same
way.  ``scripts/add_obs_features.py`` widens an existing model for them
without retraining (zero-initialised input weights: identical behaviour at
first, then the new input is learned).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from rl.memory import status_kinds as sk

EXTRA_OBS_NAMES = ("ego_upb_used",)


def upb_update(used: bool, fighter: dict) -> bool:
    """Next value of "Up Special used since the last safe state".

    Set by Up Special (478) or a helpless fall (16, which only follows one);
    cleared by anything that restores it: touching the ground, the ledge, a KO
    or the revival platform.  Being hit does not restore it.
    """
    try:
        status = int(fighter.get("action_state", -1))
        on_ground = int(fighter.get("on_ground", 0)) == 1
        rebirth = int(fighter.get("is_on_rebirth", 0)) == 1
    except (TypeError, ValueError):
        return used
    if status == sk.SPECIAL_HI or status == sk.HELPLESS_FALL:
        return True
    if (on_ground or rebirth or status in sk.LEDGE_KINDS
            or status in sk.KO_KINDS or status in sk.REBIRTH_KINDS):
        return False
    return used


class ExtraObsEnv:
    """Append memory features to the observation of the wrapped env."""

    def __init__(self, env: Any, names: Sequence[str], ego_slot: int | None = None):
        unknown = [n for n in names if n not in EXTRA_OBS_NAMES]
        if unknown:
            raise ValueError(f"unknown extra observation(s) {unknown}; "
                             f"known: {EXTRA_OBS_NAMES}")
        self.env = env
        self.names = tuple(names)
        self._ego = ego_slot
        self.upb_used = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def base_env(self):
        return getattr(self.env, "base_env", self.env)

    @property
    def n_extra(self) -> int:
        return len(self.names)

    def _ego_slot(self) -> int:
        return int(self._ego if self._ego is not None
                   else getattr(self.env, "ego_slot", 0) or 0)

    def _features(self) -> np.ndarray:
        values = []
        for name in self.names:
            if name == "ego_upb_used":
                values.append(1.0 if self.upb_used else 0.0)
        return np.asarray(values, np.float32)

    def _track(self, info: dict) -> None:
        fighters = info.get("raw_fighters") or []
        slot = self._ego_slot()
        if len(fighters) > slot:
            self.upb_used = upb_update(self.upb_used, fighters[slot])

    def _append(self, obs) -> np.ndarray:
        return np.concatenate([np.asarray(obs, np.float32).reshape(-1),
                               self._features()])

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        obs, info = result if isinstance(result, tuple) else (result, {})
        self.upb_used = False
        self._track(info)
        return self._append(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._track(info)
        return self._append(obs), reward, terminated, truncated, info
