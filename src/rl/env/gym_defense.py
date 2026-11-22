"""Reward for defending: blocking with the shield and dodging attacks.

* **block**: the ego enters the shield-hit reaction (status 30) from a guard
  state without its damage rising -> ``block_bonus``.
* **dodge**: the ego ends a spot dodge / roll / air dodge (statuses 31-34)
  during which the opponent had an attack status within ``range_x`` x
  ``range_y`` and the ego took no damage meanwhile -> ``dodge_bonus``.

Both need an actual attack to defend against, so spamming shield or dodge with
nobody attacking pays nothing.  The bonuses are small next to a hit (damage
reward is 0.01 per percent): a 10-15 % hit costs 0.1-0.15.
"""

from __future__ import annotations

from typing import Any

from rl.memory import status_kinds as sk


def _f(fighter: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(fighter.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _state(fighter: dict) -> int:
    try:
        return int(fighter.get("action_state", -1))
    except (TypeError, ValueError):
        return -1


class DefenseRewardEnv:
    def __init__(self, env: Any, block_bonus: float = 0.15,
                 dodge_bonus: float = 0.15, range_x: float = 45.0,
                 range_y: float = 45.0, ego_slot: int | None = None):
        self.env = env
        self.block_bonus = float(block_bonus)
        self.dodge_bonus = float(dodge_bonus)
        self.range_x, self.range_y = float(range_x), float(range_y)
        self._ego = ego_slot
        self.blocks = 0
        self.dodges = 0
        self._reset_tracking()

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def base_env(self):
        return getattr(self.env, "base_env", self.env)

    @property
    def defense_counts(self) -> dict:
        return {"blocks": self.blocks, "dodges": self.dodges}

    def _reset_tracking(self) -> None:
        self._prev_state = -1
        self._prev_damage = None
        self._dodge_threat = False
        self._dodge_damage = None

    def _slot(self) -> int:
        return int(self._ego if self._ego is not None
                   else getattr(self.env, "ego_slot", 0) or 0)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        info = result[1] if isinstance(result, tuple) and len(result) > 1 else {}
        self._reset_tracking()
        self._observe(info)
        return result

    def _threatened(self, ego: dict, opp: dict) -> bool:
        return (_state(opp) in sk.ATTACK_KINDS
                and abs(_f(opp, "x") - _f(ego, "x")) <= self.range_x
                and abs(_f(opp, "y") - _f(ego, "y")) <= self.range_y)

    def _observe(self, info: dict) -> float:
        """Update the trackers from one frame; returns the bonus to add."""
        fighters = info.get("raw_fighters") or []
        slot = self._slot()
        if len(fighters) < 2:
            return 0.0
        ego, opp = fighters[slot], fighters[1 - slot]
        state, damage = _state(ego), _f(ego, "damage")
        bonus = 0.0
        # shield block: entered the shield-hit reaction, no damage taken
        if (state == sk.GUARD_DAMAGE and self._prev_state != sk.GUARD_DAMAGE
                and self._prev_state in sk.GUARD_KINDS
                and (self._prev_damage is None
                     or damage <= self._prev_damage + 1e-6)):
            bonus += self.block_bonus
            self.blocks += 1
        # dodge: an attack was in range during the dodge and nothing hit
        if state in sk.DODGE_KINDS:
            if self._prev_state not in sk.DODGE_KINDS:
                self._dodge_threat = False
                self._dodge_damage = damage
            if self._threatened(ego, opp):
                self._dodge_threat = True
        elif self._prev_state in sk.DODGE_KINDS:
            if (self._dodge_threat and self._dodge_damage is not None
                    and damage <= self._dodge_damage + 1e-6
                    and state not in sk.DAMAGE_KINDS):
                bonus += self.dodge_bonus
                self.dodges += 1
            self._dodge_threat = False
        self._prev_state, self._prev_damage = state, damage
        return bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        bonus = self._observe(info)
        if bonus:
            info = dict(info, defense_bonus=bonus)
        return obs, float(reward) + bonus, terminated, truncated, info
