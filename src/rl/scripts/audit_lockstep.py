#!/usr/bin/env python3
"""Live audit of lockstep stepping (needs a game started with SSBU_GATE=1 and
SSBU_SPEED != 1).

A  exactness     every env.step advances exactly frame_skip game frames, and a
                 fighter's status frame advances by exactly that much
B  determinism   the same scripted actions from two resets give identical state
                 traces (both fighters, every field of interest)
C  stall         a multi-second stall between decisions changes nothing
D  latency       an input applied at step t shows up in the result of step t

    python -m rl.scripts.audit_lockstep --steps 240
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
KEYS = ("action_state", "action_frame", "x", "y", "vx", "vy", "damage",
        "facing", "jumps_left", "on_ground", "hitstun")


def act(buttons=(), sx=0.0, sy=0.0):
    b = np.zeros(12, np.float32)
    for n in buttons:
        b[BTN.index(n)] = 1.0
    return {"buttons": b, "stick": np.array([sx, sy], np.float32),
            "cstick": np.zeros(2, np.float32), "trigger": 0.0}


def script(n, seed=7):
    """A fixed, varied action sequence (runs, jumps, attacks, specials, shield)."""
    rng = np.random.default_rng(seed)
    out, left, cur = [], 0, act()
    for _ in range(n):
        if left <= 0:
            k = int(rng.integers(0, 6))
            sx = float(rng.choice([-1.0, 0.0, 1.0]))
            sy = float(rng.choice([-1.0, 0.0, 1.0]))
            cur = [act(sx=sx), act(("X",), sx), act(("A",), sx, sy),
                   act(("B",), sx, sy), act(("ZL",)), act()][k]
            left = int(rng.integers(2, 14))
        left -= 1
        out.append(cur)
    return out


VELOCITY_LIMIT = 3000.0      # same rule as the observation builders


def row(info):
    """State compared across runs.  The raw velocity readings are left out:
    on the first frame of a new status they hold uninitialised values (seen:
    4969.99 vs 4934.58, 2.48 vs 1.07 with identical positions).  Positions
    integrate velocity, so a real difference shows up in x/y."""
    r = []
    for f in info["raw_fighters"][:2]:
        vals = []
        for k in KEYS:
            v = float(f.get(k, 0) or 0)
            if k in ("vx", "vy"):
                v = 0.0
            vals.append(round(v, 4))
        r.append(tuple(vals))
    return tuple(r)


def quiet_prefix(*traces, gap=30.0):
    """Steps before P1 and the CPU first interact in any trace.

    The CPU AI keeps internal state across rematches, so its play differs
    between episodes; P1's own trace depends only on the pad and the physics
    until the two fighters touch, which is the part that can be compared.
    """
    cut = min(len(t) for t in traces)
    for t in traces:
        for i, (p1, p2) in enumerate(t):
            near = abs(p1[2] - p2[2]) < gap and abs(p1[3] - p2[3]) < 40
            if near or p1[6] != 0.0 or p1[10] != 0.0:
                cut = min(cut, i)
                break
    return cut


def run(env, actions, stall_every=0, log=None):
    _o, info = env.reset(seed=1)
    if log:
        log(f"   reset: parked frame {info['game_frame']}, "
            f"+{info.get('reset_frames_advanced')} frames until live")
    trace, frames = [row(info)], [info["game_frame"]]
    for i, a in enumerate(actions):
        if stall_every and i % stall_every == stall_every - 1:
            time.sleep(1.5)                     # the trainer is "busy"
        _o, _r, term, trunc, info = env.step(a)
        trace.append(row(info))
        frames.append(info["game_frame"])
        if term or trunc:
            break
    return trace, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--frame-skip", type=int, default=2)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    pad = VirtualPad().create()
    ok_all = True
    try:
        env = SSBUGymEnv(frame_skip=args.frame_skip, max_steps=10**6,
                         reward_mode="none", stock_termination=False,
                         require_positive_stock=False, input_adapter=pad,
                         reset_fn=ResetClient(timeout=20.0).reset)
        ge.wait_for_pad("~/smash.log", 15.0, log=log)
        speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
        if not ge.drive_menus(env, pad, 120.0, log=log, speed=speed):
            log("no fight")
            return 2
        env.enable_lockstep(seed=12345)
        actions = script(args.steps)

        # A: exactness + status-frame advance
        t0 = time.time()
        trace, frames = run(env, actions, log=log)
        wall = time.time() - t0
        d = np.diff(np.array(frames, np.int64))
        okA = bool(np.all(d == args.frame_skip))
        log(f"A exactness: {len(d)} steps, frame deltas {sorted(set(d.tolist()))} "
            f"(want only {args.frame_skip}) -> {'PASS' if okA else 'FAIL'}; "
            f"{len(d) / wall:.0f} decisions/s wall, "
            f"{len(d) * args.frame_skip / wall:.0f} game frames/s")
        # WAIT (status 0) plays at rate 1.0, so its status frame must advance
        # by exactly frame_skip per step (other animations have their own rates)
        wait_adv = [round(b[s][1] - a[s][1], 3)
                    for a, b in zip(trace, trace[1:]) for s in (0, 1)
                    if a[s][0] == 0 and b[s][0] == 0 and b[s][1] > a[s][1]]
        bad = sum(1 for v in wait_adv if abs(v - args.frame_skip) > 0.01)
        log(f"A WAIT status-frame advance: {len(wait_adv)} steps, "
            f"{bad} not equal to {args.frame_skip} -> "
            f"{'PASS' if bad == 0 else 'FAIL'}")
        okA = okA and bad == 0
        ok_all &= okA

        # B: determinism
        t1, f1 = run(env, actions, log=log)
        t2, f2 = run(env, actions, log=log)
        n = quiet_prefix(t1, t2)
        p1a = [r[0] for r in t1[:n]]
        p1b = [r[0] for r in t2[:n]]
        div = next((i for i in range(n) if p1a[i] != p1b[i]), None)
        okB = div is None and n >= 40
        if okB:
            log(f"B determinism: P1's trace is identical over the {n - 1} "
                "steps before the CPU interacts (all fields) -> PASS")
        else:
            log(f"B determinism: FIRST DIVERGENCE at step {div} of {n - 1}"
                if div is not None else
                f"B determinism: only {n - 1} comparable steps (need 40)")
            if div is not None:
                log(f"   P1 run1 {dict(zip(KEYS, p1a[div]))}")
                log(f"   P1 run2 {dict(zip(KEYS, p1b[div]))}")
        ok_all &= okB

        # C: stall
        t3, f3 = run(env, actions, stall_every=40)
        n3 = quiet_prefix(t1, t3)
        div = next((i for i in range(n3) if t1[i][0] != t3[i][0]), None)
        okC = div is None and n3 >= 40
        log(f"C stall (1.5 s every 40 steps): P1 identical to the unstalled "
            f"run over {n3 - 1} steps -> {'PASS' if okC else f'FAIL (diverges at {div})'}")
        if div is not None:
            log(f"   unstalled {dict(zip(KEYS, t1[div][0]))}")
            log(f"   stalled   {dict(zip(KEYS, t3[div][0]))}")
        ok_all &= okC

        # D: latency - jump (X) must put the ego in jump-squat on this step
        _o, info = env.reset(seed=1)
        for _ in range(20):
            _o, _r, *_x, info = env.step(act())
        before = info["raw_fighters"][0]
        _o, _r, _t, _u, info = env.step(act(("X",)))
        after = info["raw_fighters"][0]
        moved = (int(after["action_state"]) != int(before["action_state"]))
        log(f"D latency: status {int(before['action_state'])} -> "
            f"{int(after['action_state'])} on the step the jump was pressed "
            f"-> {'PASS' if moved else 'FAIL (input arrives a step late)'}")
        ok_all &= moved
        env.close()
    finally:
        pad.destroy()
    log("ALL PASS" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
