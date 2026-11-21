#!/usr/bin/env python3
"""Capture raw fighter state, header and the built observation per decision.

Drives P1 with a mix of random and held inputs against the CPU so that the
trace contains runs, jumps, attacks, hits taken, ledge grabs and KOs, then
dumps everything for offline checks (see audit_check.py).

    python -m rl.scripts.audit_capture --seconds 240 --out /tmp/audit.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

from rl.bc import gym_eval as ge
from rl.env.gym_env import SSBUGymEnv
from rl.input.gym_input import VirtualPad
from rl.input.gym_reset import ResetClient

BTN = ["A", "B", "X", "Y", "L", "R", "ZL", "ZR", "PLUS", "MINUS",
       "DPAD_L", "DPAD_R"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--frame-skip", type=int, default=1)
    ap.add_argument("--out", default="/tmp/audit.pkl")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    rng = np.random.default_rng(args.seed)
    pad = VirtualPad().create()
    rows = []
    try:
        env = SSBUGymEnv(frame_skip=args.frame_skip, max_steps=10**7,
                         reward_mode="none", stock_termination=False,
                         require_positive_stock=False, input_adapter=pad,
                         reset_fn=ResetClient(timeout=20.0).reset)
        ge.wait_for_pad("~/smash.log", 15.0, log=log)
        speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
        if not ge.drive_menus(env, pad, 120.0, log=log, speed=speed):
            log("no fight")
            return 2
        obs, info = env.reset()
        hold = None
        hold_left = 0
        t_end = time.time() + args.seconds
        n = 0
        while time.time() < t_end:
            if hold_left <= 0:
                kind = rng.integers(0, 6)
                b = np.zeros(12, np.float32)
                sx = sy = 0.0
                if kind == 0:   # run one way
                    sx = rng.choice([-1.0, 1.0])
                elif kind == 1:  # jump / double jump
                    b[BTN.index("X")] = 1.0
                    sx = rng.uniform(-1, 1)
                elif kind == 2:  # attack
                    b[BTN.index("A")] = 1.0
                    sx = rng.choice([-1.0, 0.0, 1.0])
                    sy = rng.choice([-1.0, 0.0, 1.0])
                elif kind == 3:  # special
                    b[BTN.index("B")] = 1.0
                    sx = rng.choice([-1.0, 0.0, 1.0])
                    sy = rng.choice([-1.0, 0.0, 1.0])
                elif kind == 4:  # shield
                    b[BTN.index("ZL")] = 1.0
                else:            # idle/stick noise
                    sx, sy = rng.uniform(-1, 1, 2)
                hold = {"buttons": b,
                        "stick": np.array([sx, sy], np.float32),
                        "cstick": np.zeros(2, np.float32), "trigger": 0.0}
                hold_left = int(rng.integers(3, 40))
            hold_left -= 1
            obs, _r, term, trunc, info = env.step(hold)
            snap = env.last_snapshot if hasattr(env, "last_snapshot") else None
            rows.append({
                "t": time.time(), "obs": np.asarray(obs, np.float32).copy(),
                "fighters": [dict(f) for f in info.get("raw_fighters", [])],
                "info": {k: info[k] for k in ("game_frame", "fight_on",
                                              "match_state", "stage_id")
                         if k in info},
                "blast": {k: info.get(k) for k in info
                          if "blast" in str(k)},
                "action": {"stick": hold["stick"].copy(),
                           "buttons": hold["buttons"].copy()},
            })
            n += 1
            if term or trunc:
                try:
                    obs, info = env.reset()
                except Exception as exc:          # noqa: BLE001
                    log(f"fight ended ({exc}); walking the menus again")
                    rows.append({"marker": "fight_ended", "t": time.time(),
                                 "step": n, "info": dict(info)})
                    if not ge.drive_menus(env, pad, 120.0, log=log,
                                          speed=speed):
                        break
                    obs, info = env.reset()
        log(f"captured {n} steps")
        env.close()
    finally:
        pad.destroy()
    with open(args.out, "wb") as fh:
        pickle.dump(rows, fh)
    log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
