#!/usr/bin/env python3
"""Wait until the state block reports a live match, then exit 0.

Use inside a game_session command before an eval/recorder step that expects the
game to already be in a match (the R2 reset refuses when no fight is live).

    .venv/bin/python tools/mem/wait_live.py [timeout_s]
"""

from __future__ import annotations

import sys
import time

import os

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rl.env.gym_env import RawSnapshotReader  # noqa: E402


def main() -> int:
    timeout = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    r = RawSnapshotReader()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = r.read_latest(timeout=0.5)
        except Exception:
            time.sleep(0.05)
            continue
        if s.fight_on and s.n_fighters >= 2:
            print("WAIT_LIVE OK", s.seq, s.stage_id,
                  [(f.get("entry_id"), f.get("kind"), f.get("is_cpu"))
                   for f in s.fighters], flush=True)
            return 0
        time.sleep(0.05)
    print("WAIT_LIVE TIMEOUT", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
