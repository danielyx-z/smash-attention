#!/usr/bin/env python3
"""Audit: is the observation right after reset() the *new* state?

Moves P1 far from its spawn and gets damaged/KO'd where possible, resets,
then prints the first few observations after each reset.  Stale data would
show the old position/damage; a correct reset shows spawn x, damage 0.
"""
from __future__ import annotations

import os

import numpy as np

from rl.bc import gym_eval as ge
from rl.env.gym_env import SSBUGymEnv
from rl.input.gym_input import VirtualPad
from rl.input.gym_reset import ResetClient


def main():
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    pad = VirtualPad().create()
    try:
        env = SSBUGymEnv(frame_skip=2, max_steps=10**6, reward_mode="none",
                         stock_termination=False, require_positive_stock=False,
                         input_adapter=pad,
                         reset_fn=ResetClient(timeout=20.0).reset)
        ge.wait_for_pad("~/smash.log", 15.0, log=log)
        speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
        if not ge.drive_menus(env, pad, 120.0, log=log, speed=speed):
            log("no fight")
            return 2
        env.reset()
        for trial in range(8):
            direction = 1.0 if trial % 2 == 0 else -1.0
            a = {"buttons": np.zeros(12, np.float32),
                 "stick": np.array([direction, 0.0], np.float32),
                 "cstick": np.zeros(2, np.float32), "trigger": 0.0}
            for _ in range(90):
                _o, _r, term, trunc, info = env.step(a)
            pre = info["raw_fighters"][0]
            _o, info = env.reset()
            post = info["raw_fighters"][0]
            first = []
            idle = {"buttons": np.zeros(12, np.float32),
                    "stick": np.zeros(2, np.float32),
                    "cstick": np.zeros(2, np.float32), "trigger": 0.0}
            for _ in range(4):
                _o, _r, term, trunc, info = env.step(idle)
                first.append(info["raw_fighters"][0])
            log(f"trial {trial}: before reset x={pre['x']:7.1f} st={int(pre['action_state']):>3} dmg={pre['damage']:5.1f}"
                f" | reset obs x={post['x']:7.1f} y={post['y']:6.1f} st={int(post['action_state']):>3}"
                f" dmg={post['damage']:4.1f} | next steps x="
                + ",".join(f"{f['x']:.1f}" for f in first)
                + " st=" + ",".join(str(int(f['action_state'])) for f in first))
        env.close()
    finally:
        pad.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
