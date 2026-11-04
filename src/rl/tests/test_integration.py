#!/usr/bin/env python3
"""Coordinator integration smoke test for the SSBU stack.

Live modes must run under the game lock; the easiest way is:

    .venv/bin/python tools/mem/game_session.py run --owner coord --restart-game \
        --restart-reader --keep-game -- .venv/bin/python tools/mem/test_integration.py --all

The session manager sets SSBU_GAME_LOCK_HELD=1, so the live modes below do not
take the lock a second time.  Mock/unit tests are in test_gym_env / test_gym_bc.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import numpy as np


def _live_lock():
    from rl.utils.locks import game_lock
    return game_lock()


def wait_live(timeout: float) -> dict:
    """Wait for a live match with two readable fighters, then return a summary."""
    from rl.env.gym_env import RawSnapshotReader
    r = RawSnapshotReader()
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout:
        try:
            s = r.read_latest(timeout=1.0)
        except Exception:
            time.sleep(0.05)
            continue
        last = s
        if int(s.fight_on) == 1 and int(s.n_fighters) >= 2 and \
                all(int(f.get("raw_flags", 0)) for f in s.fighters):
            return {
                "seq": int(s.seq),
                "stage_id": int(s.stage_id),
                "fighters": [
                    [int(f.get("entry_id", 0)), int(f.get("kind", -1)),
                     int(f.get("is_cpu", 0))] for f in s.fighters],
            }
        time.sleep(0.05)
    raise TimeoutError(
        f"no live match within {timeout}s (last fight_on={getattr(last, 'fight_on', None)}, "
        f"n={getattr(last, 'n_fighters', None)})")


def probe() -> dict:
    from rl.env.gym_env import RawSnapshotReader
    r = RawSnapshotReader()
    snap = r.read_latest(timeout=3.0)
    out = {
        "version": int(snap.version),
        "seq": int(snap.seq),
        "fight_on": int(snap.fight_on),
        "n_fighters": int(snap.n_fighters),
        "match_state": int(snap.match_state),
        "stage_id": int(snap.stage_id),
        "fps": float(snap.fps),
        "game_frame": int(snap.game_frame),
        "flags": int(snap.flags),
        "fighters": [],
    }
    for f in snap.fighters:
        out["fighters"].append({
            "entry_id": int(f.get("entry_id", 0)),
            "kind": int(f.get("kind", -1)),
            "stock": int(f.get("stock", 0)),
            "damage": round(float(f.get("damage", 0.0)), 3),
            "x": round(float(f.get("x", 0.0)), 3),
            "y": round(float(f.get("y", 0.0)), 3),
            "action_state": int(f.get("action_state", 0)),
            "shield": int(f.get("shield", 0)),
            "is_cpu": int(f.get("is_cpu", 0)),
            "buttons": int(f.get("buttons", 0)),
            "stick": [round(float(f.get("stick_x", 0.0)), 3),
                      round(float(f.get("stick_y", 0.0)), 3)],
            "raw_flags": int(f.get("raw_flags", 0)),
        })
        for k in ("x", "y", "vx", "vy", "damage"):
            v = float(f.get(k, 0.0))
            if not math.isfinite(v):
                raise AssertionError(f"non-finite {k} in fighter slot")
    if out["version"] != 7:
        raise AssertionError(f"expected v7 shm, got version {out['version']}")
    return out


def live_reset(timeout: float) -> dict:
    from rl.input.gym_reset import ResetClient
    with ResetClient(timeout=timeout) as client:
        res = client.reset(timeout=timeout)
    return {
        "seq": res.seq,
        "ack_seq": res.ack_seq,
        "state": res.state,
        "stable_frames": res.stable_frames,
        "elapsed": round(res.elapsed, 3),
    }


def env_steps(n: int, reset_first: bool, timeout: float) -> dict:
    from rl.env.gym_env import SSBUGymEnv, NoopInputAdapter
    from rl.env.gym_action import neutral_action
    from rl.env.gym_obs import OBS_SIZE

    reset_fn = None
    if reset_first:
        from rl.input.gym_reset import ResetClient
        client = ResetClient(timeout=timeout)
        reset_fn = client.reset
    else:
        reset_fn = lambda: None          # wait for an already-running match

    env = SSBUGymEnv(input_adapter=NoopInputAdapter(), reset_fn=reset_fn,
                     frame_skip=5, reset_timeout=timeout,
                     require_positive_stock=False)
    try:
        obs, info = env.reset()
        assert obs.shape == (OBS_SIZE,), obs.shape
        assert np.all(np.isfinite(obs)), "non-finite observation"
        rewards = []
        for _ in range(n):
            obs, rew, term, trunc, info = env.step(neutral_action())
            rewards.append(float(rew))
            assert np.all(np.isfinite(obs)), "non-finite observation in step"
            if term or trunc:
                break
        return {
            "obs_shape": list(obs.shape),
            "steps": int(info.get("step", n)),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "end_reason": info.get("end_reason"),
            "last_seq": int(info.get("seq", 0)),
            "frames_advanced": int(info.get("frames_advanced", 0)),
            "reward_sum": round(sum(rewards), 4),
            "stock_usable": info.get("stock_usable"),
            "fight_on": info.get("fight_on"),
            "damages": info.get("damages"),
            "stocks": info.get("stocks"),
            "p1_damage": (info.get("damages") or [None, None])[0],
            "p2_damage": (info.get("damages") or [None, None])[1],
        }
    finally:
        env.close()
        if reset_first:
            client.close()


def record(n: int, out: str, reset: bool) -> dict:
    cmd = [sys.executable, os.path.join(HERE, "gym_record.py"),
           "--episodes", str(n), "--out", out, "--hz", "60", "--no-lock"]
    if not reset:
        cmd.append("--no-reset")
    t0 = time.monotonic()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return {
        "rc": p.returncode,
        "elapsed": round(time.monotonic() - t0, 2),
        "stdout_tail": p.stdout.splitlines()[-8:],
        "stderr_tail": p.stderr.splitlines()[-8:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--wait-live", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--env-steps", type=int, default=0)
    ap.add_argument("--reset-first", action="store_true")
    ap.add_argument("--record", type=int, default=0)
    ap.add_argument("--record-out", default=os.path.join(HERE, "test_artifacts", "integration"))
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    need_live = (args.probe or args.wait_live or args.reset or args.env_steps
                 or args.record or args.all)
    if not need_live:
        ap.error("choose --probe/--wait-live/--reset/--env-steps/--record/--all")
    results = {}
    with _live_lock():
        if args.wait_live or args.all:
            results["wait_live"] = wait_live(args.timeout)
        if args.probe or args.all:
            results["probe"] = probe()
        if args.reset or args.all:
            results["reset"] = live_reset(args.timeout)
            results["probe_after_reset"] = probe()
        if args.env_steps or args.all:
            results["env"] = env_steps(args.env_steps or 20, args.reset_first or args.all,
                                       args.timeout)
        if args.record or args.all:
            results["record"] = record(args.record or 1, args.record_out, reset=True)

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
