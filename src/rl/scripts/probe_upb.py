#!/usr/bin/env python3
"""Probe the live build's action-state ids during a recovery.

Runs P1 off the nearest ledge with the pad, then tries (a) midair jump, then
Up+B and (b) Up+B with no jump, logging action_state, jumps_left,
is_actionable, vy, hitstun every decision.  Prints the state sequence so the
helpless / up-special ids can be read off, instead of trusting the schema's
name table (which does not match this build).

    src/rl/scripts/live_rl.sh-style game must already be running with the pad
    free:  python -m rl.scripts.probe_upb --trials 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

from rl.bc import gym_eval as ge
from rl.env.gym_env import SSBUGymEnv
from rl.input.gym_input import VirtualPad
from rl.input.gym_reset import ResetClient

BTN = ["A", "B", "X", "Y", "L", "R", "ZL", "ZR", "PLUS", "MINUS",
       "DPAD_L", "DPAD_R"]


def act(buttons=(), sx=0.0, sy=0.0):
    b = np.zeros(12, np.float32)
    for name in buttons:
        b[BTN.index(name)] = 1.0
    return {"buttons": b, "stick": np.array([sx, sy], np.float32),
            "cstick": np.zeros(2, np.float32), "trigger": 0.0}


def ego(info):
    f = (info.get("raw_fighters") or [{}])[0]
    return {k: f.get(k, 0) for k in
            ("x", "y", "vx", "vy", "action_state", "action_frame",
             "jumps_left", "on_ground", "is_actionable", "hitstun",
             "is_on_rebirth", "damage")}


def run_trial(env, mode: str, log):
    obs, info = env.reset()
    # settle on stage
    for _ in range(60):
        obs, _r, term, trunc, info = env.step(act())
        f = ego(info)
        if f["on_ground"] and not f["is_on_rebirth"]:
            break
    direction = -1.0 if f["x"] <= 0 else 1.0
    rows = []
    phase = "run"
    for _ in range(200):
        obs, _r, term, trunc, info = env.step(act(sx=direction))
        f = ego(info)
        if f["y"] < -10 and not f["on_ground"]:
            break
    else:
        log(f"[{mode}] never left the stage")
        return rows
    toward = -direction
    # fall straight (stick neutral) until deep enough that a recovery is real
    for _ in range(120):
        obs, _r, term, trunc, info = env.step(act())
        f = ego(info)
        if f["y"] <= -50 or f["on_ground"] or f["is_on_rebirth"]:
            break
    away = 0.0 if mode != "upb_away" else direction
    steer = toward if mode != "upb_away" else away
    for step in range(220):
        if mode in ("jump_upb", "upb_away"):
            if step == 0:
                action, phase = act(["X"], sx=steer), "jump"
            elif step < 18:
                action, phase = act(sx=steer), "wait"
            elif step < 22:
                action, phase = act(["B"], sx=steer * 0.3, sy=1.0), "upB"
            else:
                action, phase = act(sx=steer), "after"
        else:  # upb_only
            if step < 4:
                action, phase = act(["B"], sx=toward * 0.3, sy=1.0), "upB"
            else:
                action, phase = act(sx=toward), "after"
        obs, _r, term, trunc, info = env.step(action)
        f = ego(info)
        rows.append((step, phase, f))
        if f["is_on_rebirth"] or f["on_ground"]:
            break
    return rows


def summarize(mode, rows, log):
    log(f"=== {mode}: {len(rows)} steps")
    runs = []
    for step, phase, f in rows:
        st = int(f["action_state"])
        if runs and runs[-1][0] == st:
            runs[-1][2] = step
            runs[-1][4] = f
        else:
            runs.append([st, step, step, phase, f])
    for st, a, b, phase, f in runs:
        log(f"  steps {a:3d}-{b:3d} {phase:5s} state={st:4d} "
            f"jumps={int(f['jumps_left'])} act={int(f['is_actionable'])} "
            f"hit={f['hitstun']:.0f} vy={f['vy']:+.1f} "
            f"x={f['x']:+.0f} y={f['y']:+.0f}")
    end = rows[-1][2] if rows else {}
    log(f"  end: on_ground={end.get('on_ground')} "
        f"rebirth={end.get('is_on_rebirth')} y={end.get('y')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--game-log", default="~/smash.log")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    pad = VirtualPad().create()
    try:
        env = SSBUGymEnv(frame_skip=2, max_steps=100000,
                         reward_mode="none", stock_termination=False,
                         require_positive_stock=False, input_adapter=pad,
                         reset_fn=ResetClient(timeout=20.0).reset)
        ge.wait_for_pad(args.game_log, 15.0, log=log)
        if not ge.drive_menus(env, pad, 120.0, log=log):
            log("no fight")
            return 2
        for t in range(args.trials):
            for mode in ("jump_upb", "upb_only", "upb_away"):
                rows = run_trial(env, mode, log)
                if rows:
                    summarize(f"{mode} #{t + 1}", rows, log)
        env.close()
    finally:
        pad.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
