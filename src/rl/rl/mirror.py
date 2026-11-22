#!/usr/bin/env python3
"""Left/right mirror wrapper for live PPO training.

The game is symmetric on the stages used for training, so an episode can be
presented mirrored: the policy sees a canonical state and emits a canonical
action, and this wrapper flips both before they reach the game.

This is done at the environment boundary, not on the action token.  The policy
head is a vocabulary of absolute actions; mirroring a token is not guaranteed
to be a valid token, so token-level augmentation is avoided.

Mirroring is per episode, never per frame.  Flipping whenever the fighter
crosses the stage centre would make the frame of reference jump and turn a
smooth trajectory into a non-stationary one.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl.bc import gym_bc as bc


def mirror_action(action: Any) -> Any:
    """Mirror one canonical action (dict or flat array) left <-> right."""
    if isinstance(action, dict):
        out = dict(action)
        stick = np.asarray(action["stick"], np.float32).copy()
        stick[0] = -stick[0]
        out["stick"] = stick
        if "cstick" in action:
            cstick = np.asarray(action["cstick"], np.float32).copy()
            cstick[0] = -cstick[0]
            out["cstick"] = cstick
        if "buttons" in action:
            buttons = np.asarray(action["buttons"], np.float32).copy()
            left = bc.BUTTON_INDEX["DPAD_L"]
            right = bc.BUTTON_INDEX["DPAD_R"]
            buttons[left], buttons[right] = buttons[right], buttons[left]
            out["buttons"] = buttons
        if "button_names" in action and isinstance(action["button_names"],
                                                   tuple):
            names = list(action["button_names"])
            names = [{"DPAD_L": "DPAD_R", "DPAD_R": "DPAD_L"}.get(n, n)
                     for n in names]
            out["button_names"] = tuple(names)
        return out
    arr = np.asarray(action, np.float32).copy()
    if arr.size < bc.N_BUTTONS + 2:
        raise ValueError(f"action needs at least {bc.N_BUTTONS + 2} values")
    left = bc.BUTTON_INDEX["DPAD_L"]
    right = bc.BUTTON_INDEX["DPAD_R"]
    arr[left], arr[right] = arr[right], arr[left]
    arr[bc.N_BUTTONS] = -arr[bc.N_BUTTONS]          # stick x
    cstick = bc.N_BUTTONS + 2
    if arr.size > cstick:
        arr[cstick] = -arr[cstick]                  # cstick x
    return arr


def mirror_observation(obs: Any) -> np.ndarray:
    """Mirror the policy observation vector (OBS_DIM)."""
    arr = np.asarray(obs, np.float32).reshape(-1)
    if arr.size < bc.OBS_DIM:
        raise ValueError(
            f"mirror wrapper expects the {bc.OBS_DIM}-dim policy "
            f"observation (plus optional extra features), got {arr.size}")
    # extra memory features (e.g. Up Special used) are left/right symmetric
    return bc.mirror_obs(arr)


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


class MirrorWrapper:
    """Mirror a whole episode with probability ``mirror_prob``.

    The wrapper is deliberately outside any recovery-drill wrapper so that the
    drill's setup observations and previous action are mirrored too:

        MirrorWrapper(RecoveryDrillEnv(SSBUGymEnv(...)))
    """

    def __init__(self, env: Any, mirror_prob: float = 0.5, log=None):
        self.env = env
        self.mirror_prob = min(1.0, max(0.0, float(mirror_prob)))
        self.log = log or (lambda *_a, **_k: None)
        self._rng = np.random.default_rng(0)
        self._mirror = False
        self.setup_observations: list[np.ndarray] = []
        self.setup_action: dict | None = None

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
        return getattr(self.env, "ego_slot", 0)

    def close(self):
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def render(self, *args, **kwargs):
        render = getattr(self.env, "render", None)
        if callable(render):
            return render(*args, **kwargs)
        return None

    # -- episode frame ----------------------------------------------------
    def _choose(self, seed: int | None) -> bool:
        if self.mirror_prob <= 0.0:
            return False
        if self.mirror_prob >= 1.0:
            return True
        if seed is not None:
            return bool(np.random.default_rng(int(seed)).random()
                        < self.mirror_prob)
        return bool(self._rng.random() < self.mirror_prob)

    def _mirror_setup(self):
        if not self._mirror:
            self.setup_observations = list(
                getattr(self.env, "setup_observations", None) or [])
            self.setup_action = getattr(self.env, "setup_action", None)
            return
        self.setup_observations = [
            mirror_observation(o)
            for o in (getattr(self.env, "setup_observations", None) or [])]
        action = getattr(self.env, "setup_action", None)
        self.setup_action = mirror_action(action) if action is not None else None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._mirror = self._choose(seed)
        if options is None:
            result = self.env.reset(seed=seed)
        else:
            result = self.env.reset(seed=seed, options=options)
        obs, info = _split_reset(result)
        self._mirror_setup()
        if self._mirror:
            obs = mirror_observation(obs)
        self.log(f"[mirror] episode {'MIRRORED' if self._mirror else 'normal'}")
        return obs, info

    def step(self, action):
        base_action = mirror_action(action) if self._mirror else action
        result = self.env.step(base_action)
        obs, reward, terminated, truncated, info = _split_step(result)
        if self._mirror:
            obs = mirror_observation(obs)
        return obs, reward, terminated, truncated, info
