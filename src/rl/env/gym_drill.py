#!/usr/bin/env python3
"""Recovery drill wrapper for the live SSBU gym.

The policy spends almost all of its time on stage, so PPO sees very few
recoverable ledge states.  This wrapper manufactures them with the existing
virtual pad:

1. reset the match;
2. run player 1 toward the nearest ledge until the fighter is airborne below
   the stage (the setup uses only the stick, so both the midair jump and Up
   Special stay available);
3. hand control to the policy mid-fall;
4. end the episode on recovery (back on the ground) or death (respawn).

The setup steps are not part of the policy's training data.  The observations
that led into the fall are exposed as ``setup_observations`` so the policy's
history window starts with the approach, and ``setup_action`` lets PPO seed
its previous-action vector.

This is a training-only wrapper.  It should not be used for tournament
evaluation.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.memory import status_kinds

EGO_SLOT = 0


def _fighters(info: dict) -> list[dict]:
    return list(info.get("raw_fighters") or [])


def _ego(info: dict, slot: int = EGO_SLOT) -> dict:
    fighters = _fighters(info)
    if len(fighters) <= int(slot):
        return {}
    return fighters[int(slot)]


def _num(fighter: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(fighter.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def _flag(fighter: dict, key: str) -> bool:
    try:
        return int(fighter.get(key, 0) or 0) == 1
    except (TypeError, ValueError):
        return False


def on_ledge(fighter: dict) -> bool:
    """Catching, hanging on, or climbing out of the ledge (safe)."""
    return status_kinds.is_ledge(fighter.get("action_state", -1))


def is_offstage(info: dict, slot: int = EGO_SLOT) -> bool:
    """Airborne below the stage, not on the revival platform or the ledge."""
    fighter = _ego(info, slot)
    if not fighter or _flag(fighter, "is_on_rebirth") or on_ledge(fighter):
        return False
    if _flag(fighter, "on_ground"):
        return False
    return _num(fighter, "y") < -10.0


def _neutral(stick_x: float, stick_y: float = 0.0) -> dict:
    return {
        "buttons": np.zeros(12, np.float32),
        "stick": np.array([float(stick_x), float(stick_y)], np.float32),
        "cstick": np.zeros(2, np.float32),
        "trigger": 0.0,
    }


def _split_reset(result):
    if isinstance(result, tuple):
        obs = result[0]
        info = result[1] if len(result) > 1 and isinstance(result[1], dict) \
            else {}
        return obs, info
    return result, {}


def _split_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
    elif len(result) == 4:
        obs, reward, done, info = result
        terminated, truncated = bool(done), False
    else:
        raise TypeError(f"env.step() returned {len(result)} values")
    return obs, float(reward), bool(terminated), bool(truncated), info or {}


class RecoveryDrillEnv:
    """Run P1 off the ledge, then let the policy try to recover.

    Parameters
    ----------
    env:
        The live ``SSBUGymEnv`` (or any env with the same 5-tuple API).
    setup_steps:
        Maximum decisions spent running toward the ledge before giving up.
    settle_steps:
        Consecutive on-ground decisions required to count as a recovery.
    recovery_bonus:
        Reward added once when the fighter returns to the ground.
    attempts:
        Reset attempts if the setup dies or never leaves the stage.
    """

    def __init__(self, env: Any, setup_steps: int = 180,
                 settle_steps: int = 2, recovery_bonus: float = 0.5,
                 attempts: int = 3, settle_reset_steps: int = 90,
                 drill_prob: float = 1.0, fall_delay_max: int = 24,
                 log=None):
        self.env = env
        self.setup_steps = max(1, int(setup_steps))
        self.settle_steps = max(1, int(settle_steps))
        self.recovery_bonus = float(recovery_bonus)
        self.attempts = max(1, int(attempts))
        self.settle_reset_steps = max(0, int(settle_reset_steps))
        self.drill_prob = min(1.0, max(0.0, float(drill_prob)))
        self.fall_delay_max = max(0, int(fall_delay_max))
        self.log = log or (lambda *_a, **_k: None)
        self._drill = True
        self.setup_observations: list[np.ndarray] = []
        self.setup_action: dict | None = None
        self._was_offstage = False
        self._prev_rebirth = False
        self._ground_steps = 0
        self._steps = 0

    # -- Gym plumbing -----------------------------------------------------
    @property
    def base_env(self):
        return self.env

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def metadata(self):
        return getattr(self.env, "metadata", {})

    @property
    def max_steps(self):
        return getattr(self.env, "max_steps", 1800)

    @property
    def ego_slot(self):
        return getattr(self.env, "ego_slot", EGO_SLOT)

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def render(self, *args, **kwargs):
        render = getattr(self.env, "render", None)
        if callable(render):
            return render(*args, **kwargs)
        return None

    # -- reset / step -----------------------------------------------------
    def _choose_drill(self, seed: int | None) -> bool:
        if self.drill_prob <= 0.0:
            return False
        if self.drill_prob >= 1.0:
            return True
        if seed is not None:
            return bool(np.random.default_rng(int(seed)).random()
                        < self.drill_prob)
        return bool(np.random.default_rng().random() < self.drill_prob)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._drill = self._choose_drill(seed)
        if not self._drill:
            result = self.env.reset(seed=seed)
            obs, info = _split_reset(result)
            self.setup_observations = []
            self.setup_action = None
            self._was_offstage = False
            self._prev_rebirth = _flag(
                _ego(info, self.ego_slot), "is_on_rebirth")
            self._ground_steps = 0
            self._steps = 0
            self.log("[drill] normal episode")
            return obs, info
        last_error: Exception | None = None
        for _attempt in range(self.attempts):
            try:
                return self._reset_once(seed)
            except RuntimeError as exc:
                last_error = exc
        raise RuntimeError(
            f"recovery drill setup failed after {self.attempts} attempts "
            f"({last_error})")

    def _reset_once(self, seed: int | None):
        result = self.env.reset(seed=seed)
        obs, info = _split_reset(result)
        fighter = _ego(info, self.ego_slot)
        if not fighter:
            raise RuntimeError("recovery drill: reset info has no ego fighter")
        # The fast rematch can hand back a fighter that is still on the
        # revival platform.  Wait for a normal on-stage state before running
        # off the ledge; otherwise the setup immediately breaks on rebirth.
        observations: list[np.ndarray] = []
        for _ in range(self.settle_reset_steps):
            fighter = _ego(info, self.ego_slot)
            if (not _flag(fighter, "is_on_rebirth")
                    and _flag(fighter, "on_ground")):
                break
            result = self.env.step(_neutral(0.0))
            obs, _reward, terminated, truncated, info = _split_step(result)
            observations.append(np.asarray(obs, np.float32))
            if terminated or truncated:
                break
        fighter = _ego(info, self.ego_slot)
        x = _num(fighter, "x")
        direction = -1.0 if x <= 0.0 else 1.0
        action = _neutral(direction)
        # Setup state machine.  Running off the edge often ends with the ego
        # *catching the ledge*, which is safe (not offstage), so the setup
        # must drop from it (stick down) instead of waiting forever.  Once
        # falling offstage it falls a random few more decisions so the policy
        # starts from a range of depths.
        rng = np.random.default_rng(None if seed is None else int(seed) + 7919)
        delay = int(rng.integers(0, self.fall_delay_max + 1))
        offstage = False
        drops = 0
        for _ in range(self.setup_steps):
            fighter = _ego(info, self.ego_slot)
            if on_ledge(fighter):
                action = _neutral(0.0, -1.0)           # drop from the ledge
                drops += 1
            elif is_offstage(info, self.ego_slot):
                if delay <= 0:
                    offstage = True
                    break
                delay -= 1
                action = _neutral(0.0)                 # fall straight
            else:
                action = _neutral(direction)           # run toward the edge
            result = self.env.step(action)
            obs, _reward, terminated, truncated, info = _split_step(result)
            observations.append(np.asarray(obs, np.float32))
            fighter = _ego(info, self.ego_slot)
            if _flag(fighter, "is_on_rebirth") or terminated or truncated:
                break
        if not offstage and is_offstage(info, self.ego_slot):
            offstage = True
        if not offstage:
            raise RuntimeError(
                "recovery drill: P1 did not leave the stage during setup")
        self.setup_observations = observations[:-1]
        self.setup_action = action
        self._was_offstage = True
        self._prev_rebirth = _flag(_ego(info, self.ego_slot),
                                   "is_on_rebirth")
        self._ground_steps = 0
        self._steps = 0
        self.log(f"[drill] setup {len(observations)} steps, direction "
                 f"{direction:+.0f}, ledge drops {drops}, offstage P1 at "
                 f"x={_num(_ego(info, self.ego_slot), 'x'):+.1f} "
                 f"y={_num(_ego(info, self.ego_slot), 'y'):+.1f}")
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = _split_step(result)
        if not self._drill:
            return obs, reward, terminated, truncated, info
        fighter = _ego(info, self.ego_slot)
        rebirth = _flag(fighter, "is_on_rebirth")
        if rebirth and not self._prev_rebirth:
            # The environment already paid the stock/death reward.
            terminated = True
        elif (self._was_offstage and not rebirth
              and (_flag(fighter, "on_ground") or on_ledge(fighter))):
            self._ground_steps += 1
            if self._ground_steps >= self.settle_steps:
                reward += self.recovery_bonus
                terminated = True
        else:
            self._ground_steps = 0
        self._prev_rebirth = rebirth
        self._steps += 1
        if not terminated and not truncated and self._steps >= self.max_steps:
            truncated = True
        return obs, reward, bool(terminated), bool(truncated), info
