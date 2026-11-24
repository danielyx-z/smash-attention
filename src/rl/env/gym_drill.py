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


class DrillSetupError(RuntimeError):
    """The scripted run-off failed; the game itself is fine."""


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
                 drill_prob: float = 1.0, handoff_y_range: tuple = (-60.0, -10.0),
                 log=None, teleport: bool = False,
                 damage_range: tuple = (0.0, 100.0), helpless_prob: float = 0.3,
                 jump_spent_prob: float = 0.3, x_range: tuple = (95.0, 140.0),
                 drill_max_steps: int = 0,
                 helpless_x_range: tuple = (85.0, 100.0),
                 helpless_y_range: tuple = (-35.0, -15.0),
                 launch_prob: float = 0.0,
                 launch_speed: tuple = (0.8, 2.5),
                 launch_vy: tuple = (-0.5, 1.5),
                 launch_x_range: tuple = (90.0, 110.0),
                 launch_y_range: tuple = (-45.0, -10.0),
                 launch_hitstun_prob: float = 0.8,
                 launch_hitstun_steps: tuple = (2, 6)):
        self.env = env
        self.setup_steps = max(1, int(setup_steps))
        self.settle_steps = max(1, int(settle_steps))
        self.recovery_bonus = float(recovery_bonus)
        self.attempts = max(1, int(attempts))
        self.settle_reset_steps = max(0, int(settle_reset_steps))
        self.drill_prob = min(1.0, max(0.0, float(drill_prob)))
        self.max_fallbacks = 2          # consecutive setup failures = dead pad
        self._fallbacks = 0
        lo, hi = sorted(float(v) for v in handoff_y_range)
        self.handoff_y_range = (lo, hi)
        self.log = log or (lambda *_a, **_k: None)
        # teleport: the runtime drops P1 airborne offstage with some damage
        # (no scripted run-off, no settle wait); helpless = Up-B already spent,
        # jump_spent = the midair jump already used.
        self.teleport = bool(teleport)
        self.damage_range = tuple(float(v) for v in damage_range)
        self.helpless_prob = float(helpless_prob)
        self.jump_spent_prob = float(jump_spent_prob)
        self.x_range = tuple(float(v) for v in x_range)
        # A helpless fighter (Up Special spent) can only drift: from far out or
        # deep it is lost whatever the policy does (live data: 92 % of helpless
        # drops died, 6 % grabbed the ledge), so those drops teach nothing but
        # noise.  Keep them to a zone close to the ledge where a ledge grab is
        # possible.
        # Launch drops: how a stock is really lost.  The fighter starts nearer
        # the stage and is thrown outward at ``launch_speed`` (game units per
        # frame); with ``launch_hitstun_prob`` the first few decisions are
        # neutral input (the policy does not act yet), standing in for hitstun.
        # Verified live: the x speed is in the fighter's facing frame (it faces
        # the stage centre after the reset), so outward is negative.  A real
        # damage status (72) was tried and dropped: it keeps the motion of the
        # fighter's previous hit, so 4 of 16 drops flew inward at random.
        self.launch_prob = float(launch_prob)
        self.launch_speed = tuple(float(v) for v in launch_speed)
        self.launch_vy = tuple(float(v) for v in launch_vy)
        self.launch_x_range = tuple(float(v) for v in launch_x_range)
        self.launch_y_range = tuple(float(v) for v in launch_y_range)
        self.launch_hitstun_prob = float(launch_hitstun_prob)
        self.launch_hitstun_steps = (int(launch_hitstun_steps[0]),
                                     int(launch_hitstun_steps[1]))
        self.helpless_x_range = tuple(float(v) for v in helpless_x_range)
        self.helpless_y_range = tuple(float(v) for v in helpless_y_range)
        self.drill_max_steps = max(0, int(drill_max_steps))   # 0 = env max
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
        return getattr(self.env, "base_env", self.env)

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
                out = self._reset_once(seed)
                self._fallbacks = 0
                return out
            except DrillSetupError as exc:
                last_error = exc
        self._fallbacks += 1
        if self._fallbacks >= self.max_fallbacks:
            # A single failure is a quirk; several in a row mean P1 no longer
            # answers the pad (observed: 0 damage dealt for 25 iterations).
            # Raise a plain error so the trainer recovers the pad / game.
            self._fallbacks = 0
            raise RuntimeError(
                f"P1 unresponsive: {self.max_fallbacks} drill setups in a row "
                f"failed ({last_error})")
        # The setup is only a scripted run-off.  A failure is not a game
        # wedge, so play a normal episode instead of escalating to a restart.
        self.log(f"[drill] setup failed {self.attempts}x ({last_error}); "
                 "playing a normal episode instead")
        self._drill = False
        result = self.env.reset(seed=seed)
        obs, info = _split_reset(result)
        self.setup_observations = []
        self.setup_action = None
        self._was_offstage = False
        self._prev_rebirth = _flag(_ego(info, self.ego_slot), "is_on_rebirth")
        self._ground_steps = 0
        self._steps = 0
        return obs, info

    def _reset_teleport(self, seed: int | None):
        """Drop P1 offstage in one reset: no run-off, no settle wait."""
        rng = np.random.default_rng(None if seed is None else int(seed) + 7919)
        side = -1.0 if rng.random() < 0.5 else 1.0
        damage_u = rng.random()
        helpless = bool(rng.random() < self.helpless_prob)
        xr = self.helpless_x_range if helpless else self.x_range
        yr = self.helpless_y_range if helpless else self.handoff_y_range
        x = side * float(rng.uniform(*xr))
        y = float(rng.uniform(*yr))
        damage = self.damage_range[0] + damage_u * (self.damage_range[1]
                                                    - self.damage_range[0])
        spend_jump = (not helpless) and bool(rng.random() < self.jump_spent_prob)
        launch = (not helpless) and bool(rng.random() < self.launch_prob)
        hitstun_steps = 0
        drop = {"x": x, "y": y, "damage": damage, "helpless": helpless}
        if launch:
            speed = float(rng.uniform(*self.launch_speed))
            # Right side only: a launched drop on the left side lands at a
            # wrong, positive x in the live game (cause not found; plain
            # drops and right-side launches are fine).  The mirror wrapper
            # shows the policy both sides anyway.
            drop["x"] = float(rng.uniform(*self.launch_x_range))
            drop["y"] = float(rng.uniform(*self.launch_y_range))
            drop["vx"] = -speed                       # outward (facing frame)
            drop["vy"] = float(rng.uniform(*self.launch_vy))
            hitstun = bool(rng.random() < self.launch_hitstun_prob)
            hitstun_steps = (int(rng.integers(self.launch_hitstun_steps[0],
                                              self.launch_hitstun_steps[1] + 1))
                             if hitstun else 0)
            if hitstun:
                spend_jump = False                    # cannot act yet
        base = self.base_env
        base.drill_drop = drop
        result = self.env.reset(seed=seed)
        obs, info = _split_reset(result)
        fighter = _ego(info, self.ego_slot)
        if not fighter or not is_offstage(info, self.ego_slot):
            raise DrillSetupError(
                "recovery drill: the drop did not leave P1 offstage "
                f"(y={_num(fighter, 'y'):+.1f})")
        observations: list[np.ndarray] = []
        action = _neutral(0.0)
        for _ in range(hitstun_steps):               # "hitstun": no input yet
            result = self.env.step(action)
            obs2, _r, terminated, truncated, info = _split_step(result)
            observations.append(np.asarray(obs, np.float32))
            obs = obs2
            if terminated or truncated or _flag(_ego(info, self.ego_slot),
                                                 "is_on_rebirth"):
                raise DrillSetupError("recovery drill: ended during the launch")
        if spend_jump:
            # one tap of jump toward the stage uses the midair jump
            action = _neutral(-side)
            action["buttons"][2] = 1.0                 # X = jump
            result = self.env.step(action)
            obs2, _r, terminated, truncated, info = _split_step(result)
            observations.append(np.asarray(obs, np.float32))
            obs = obs2
            if terminated or truncated:
                raise DrillSetupError("recovery drill: ended during the drop")
        self.setup_observations = observations
        self.setup_action = action
        self._was_offstage = True
        self._prev_rebirth = _flag(_ego(info, self.ego_slot), "is_on_rebirth")
        self._ground_steps = 0
        self._steps = 0
        f = _ego(info, self.ego_slot)
        self.log(f"[drill] drop x={_num(f, 'x'):+.1f} y={_num(f, 'y'):+.1f} "
                 f"damage {damage:.0f} helpless {int(helpless)} "
                 f"jump_spent {int(spend_jump)}"
                 + (f" launch {-drop['vx']:.1f} hitstun_steps {hitstun_steps}"
                    if launch else ""))
        return obs, info

    def _reset_once(self, seed: int | None):
        if self.teleport:
            return self._reset_teleport(seed)
        result = self.env.reset(seed=seed)
        obs, info = _split_reset(result)
        fighter = _ego(info, self.ego_slot)
        if not fighter:
            raise DrillSetupError("recovery drill: reset info has no ego fighter")
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
        # the policy gets control once it has fallen to this depth
        target_y = float(rng.uniform(*self.handoff_y_range))
        offstage = False
        drops = 0
        for _ in range(self.setup_steps):
            fighter = _ego(info, self.ego_slot)
            if on_ledge(fighter):
                action = _neutral(0.0, -1.0)           # drop from the ledge
                drops += 1
            elif is_offstage(info, self.ego_slot):
                if _num(fighter, "y") <= target_y:
                    offstage = True
                    break
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
            raise DrillSetupError(
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
        limit = self.drill_max_steps or self.max_steps
        if not terminated and not truncated and self._steps >= limit:
            truncated = True
        return obs, reward, bool(terminated), bool(truncated), info
