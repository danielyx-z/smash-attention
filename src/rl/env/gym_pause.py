"""Resume a paused match instead of training on frozen frames.

Found by audit: when the controller disconnects/reconnects (or anything else
pauses the match) the simulation stops while the window keeps rendering, the
reader keeps publishing, resets keep being acknowledged, and every observation
is a perfectly valid *identical* frame.  The agent then collects hundreds of
zero-reward steps (two 4x training runs lost 25-35 iterations this way).
Pressing PLUS resumes the match.

The guard watches a signature of both fighters (status, status frame,
position, damage).  A running match changes it at least every few frames, even
idle (the status frame advances); hitlag is the longest legitimate freeze
(about 30 game frames), so ``freeze_steps`` is set well above that.  On a
freeze it presses PLUS and checks that the signature moves again.  PLUS is a
toggle, so a false positive is undone by the next attempt.  If the match
cannot be resumed the episode is truncated and flagged, so the trainer's
reset/recovery path takes over.
"""

from __future__ import annotations

from typing import Any

import numpy as np

PLUS_INDEX = 8          # canonical button order: ... ZR, PLUS, MINUS ...


def _signature(info: dict) -> tuple | None:
    fighters = info.get("raw_fighters") or []
    if len(fighters) < 2:
        return None
    sig = []
    for f in fighters[:2]:
        sig.append((int(f.get("action_state", -1) or -1),
                    round(float(f.get("action_frame", 0.0) or 0.0), 2),
                    round(float(f.get("x", 0.0) or 0.0), 3),
                    round(float(f.get("y", 0.0) or 0.0), 3),
                    round(float(f.get("damage", 0.0) or 0.0), 2)))
    return tuple(sig)


def _press_plus() -> dict:
    buttons = np.zeros(12, np.float32)
    buttons[PLUS_INDEX] = 1.0
    return {"buttons": buttons, "stick": np.zeros(2, np.float32),
            "cstick": np.zeros(2, np.float32), "trigger": 0.0}


def _neutral() -> dict:
    return {"buttons": np.zeros(12, np.float32),
            "stick": np.zeros(2, np.float32),
            "cstick": np.zeros(2, np.float32), "trigger": 0.0}


class PauseGuardEnv:
    def __init__(self, env: Any, freeze_steps: int = 40, attempts: int = 3,
                 log=None):
        self.env = env
        self.freeze_steps = max(2, int(freeze_steps))
        self.attempts = max(1, int(attempts))
        self.log = log or (lambda *_a, **_k: None)
        self._last: tuple | None = None
        self._frozen = 0
        self.resumed = 0          # times a pause was undone
        self.wedged = 0           # times it could not be undone

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def base_env(self):
        return getattr(self.env, "base_env", self.env)

    def reset(self, *args, **kwargs):
        self._last, self._frozen = None, 0
        return self.env.reset(*args, **kwargs)

    # -- helpers ----------------------------------------------------------
    def _track(self, info: dict) -> bool:
        """Update the freeze counter; True when the match is frozen."""
        if int(info.get("fight_on", 1) or 0) != 1:
            self._last, self._frozen = None, 0
            return False
        sig = _signature(info)
        if sig is None:
            return False
        if sig == self._last:
            self._frozen += 1
        else:
            self._frozen = 0
        self._last = sig
        return self._frozen >= self.freeze_steps

    def _try_resume(self, result):
        """Press PLUS until the signature moves; returns (result, ok)."""
        for attempt in range(1, self.attempts + 1):
            ref = _signature(result[4])
            moved = False
            for press in (_press_plus(), _press_plus(), _neutral(),
                          _neutral(), _neutral(), _neutral()):
                result = self.env.step(press)
                if _signature(result[4]) != ref:
                    moved = True
                    break
                if result[2] or result[3]:
                    return result, True            # episode ended anyway
            if moved:
                self.resumed += 1
                self.log(f"[guard] match was paused (frozen "
                         f"{self.freeze_steps} steps); PLUS resumed it "
                         f"(attempt {attempt})")
                self._last, self._frozen = _signature(result[4]), 0
                return result, True
        return result, False

    # -- step -------------------------------------------------------------
    def step(self, action):
        result = self.env.step(action)
        obs, reward, terminated, truncated, info = result
        if terminated or truncated or not self._track(info):
            return result
        result, ok = self._try_resume(result)
        obs, reward, terminated, truncated, info = result
        if not ok:
            self.wedged += 1
            self.log("[guard] match frozen and PLUS did not resume it; "
                     "truncating the episode")
            self._last, self._frozen = None, 0
            info = dict(info, frozen_wedge=True)
            return obs, reward, terminated, True, info
        return obs, reward, terminated, truncated, dict(info, pause_resumed=True)
