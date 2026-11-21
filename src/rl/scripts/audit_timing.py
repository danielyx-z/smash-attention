#!/usr/bin/env python3
"""Audit: is every observation tied to exactly one game frame?

The reader publishes the state block on its own wall-clock timer and numbers
its samples with a private counter (header ``game_frame``).  The runtime
publishes the game's real simulation frame in the control block.  This script
samples both at a high rate while a fight is running and reports, per reader
publish, how many *real* game frames passed since the previous publish:

    0   duplicate  (the agent saw the same game frame twice)
    1   ideal
    2+  skipped    (the agent missed game frames)

and the same for the decision step the env actually takes (frame_skip reads).

    python -m rl.scripts.audit_timing --seconds 12 --out /tmp/timing.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

import numpy as np

from rl.bc import gym_eval as ge
from rl.env.gym_env import SSBUGymEnv
from rl.input.control_block import ControlBlock
from rl.input.gym_input import VirtualPad
from rl.input.gym_reset import ResetClient


def sample(env, ctl, seconds):
    rows = []
    t_end = time.perf_counter() + seconds
    last_seq = None
    while time.perf_counter() < t_end:
        snap = env.reader.read_latest(timeout=1.0)
        if snap.seq == last_seq:
            continue
        last_seq = snap.seq
        c = ctl.read()
        f0 = (snap.fighters or [{}])[0]
        rows.append((time.perf_counter(), snap.seq, snap.game_frame,
                     c.game_frame, float(f0.get("x", 0.0)),
                     float(f0.get("action_frame", 0.0) or 0.0)))
    return rows


def hist(deltas):
    c = collections.Counter(int(d) for d in deltas)
    n = max(1, len(deltas))
    return {k: round(v / n, 4) for k, v in sorted(c.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--frame-skip", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    pad = VirtualPad().create()
    try:
        env = SSBUGymEnv(frame_skip=args.frame_skip, max_steps=100000,
                         reward_mode="none", stock_termination=False,
                         require_positive_stock=False, input_adapter=pad,
                         reset_fn=ResetClient(timeout=20.0).reset)
        ge.wait_for_pad(" ~/smash.log".strip(), 15.0, log=log)
        speed = float(os.environ.get("SSBU_SPEED", "1") or 1.0)
        if not ge.drive_menus(env, pad, 120.0, log=log, speed=speed):
            log("no fight")
            return 2
        ctl = ControlBlock(create=False)
        ctl.open() if hasattr(ctl, "open") else None
        env.reset()

        # 1. raw reader publishes vs the runtime's true frame
        rows = sample(env, ctl, args.seconds)
        gf = np.array([r[3] for r in rows], np.int64)
        rd = np.array([r[2] for r in rows], np.int64)
        d_true = np.diff(gf)
        d_reader = np.diff(rd)
        span_true = gf[-1] - gf[0]
        span_read = rd[-1] - rd[0]
        log(f"reader publishes: {len(rows)} in {args.seconds:.0f}s; "
            f"true game frames advanced {span_true}; reader counter {span_read}")
        log(f"  true game-frame delta per reader publish: {hist(d_true)}")
        log(f"  reader-counter delta per publish        : {hist(d_reader)}")
        log(f"  ctl.game_frame first/last: {gf[0]} {gf[-1]}  (zero => not live)")

        # 2. what a decision step really spans, with a random-ish action
        rng = np.random.default_rng(0)
        steps = []
        env.reset()
        for i in range(args.steps):
            before = ctl.read().game_frame
            a = {"buttons": np.zeros(12, np.float32),
                 "stick": rng.uniform(-1, 1, 2).astype(np.float32),
                 "cstick": np.zeros(2, np.float32), "trigger": 0.0}
            t0 = time.perf_counter()
            _o, _r, term, trunc, info = env.step(a)
            after = ctl.read().game_frame
            steps.append((int(after) - int(before),
                          time.perf_counter() - t0))
            if term or trunc:
                env.reset()
        d = [s[0] for s in steps]
        log(f"env.step (frame_skip={args.frame_skip}) spans true game frames: "
            f"{hist(d)}  mean {np.mean(d):.2f}")
        log(f"  wall time per step: mean {1000*np.mean([s[1] for s in steps]):.1f} ms")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump({"rows": rows[:4000], "steps": steps}, fh)
        env.close()
    finally:
        pad.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
