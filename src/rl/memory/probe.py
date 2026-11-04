#!/usr/bin/env python3
"""One-shot CLI probe: print the live fighter state, one frame at a time.

    python3 tools/mem/probe.py                  # one frame, human readable
    python3 tools/mem/probe.py --frames 10 --interval 0.1
    python3 tools/mem/probe.py --json           # machine readable
    python3 tools/mem/probe.py --raw            # raw bytes per fighter

This reads /proc/<pid>/mem directly; it does not use shared memory.  Use it to
cross-check what reader.py publishes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from rl.memory import schema
from rl.memory.gamestate import GameState
from rl.memory.memapi import check_access, find_game_pid


def fmt_row(row: dict) -> str:
    mask = int(row.get("buttons", 0) or 0)
    names = ",".join(schema.buttons_pressed(mask)) or "-"
    return (
        f"P{row['entry_id'] + 1} "
        f"kind={row.get('kind', -1):<2d} "
        f"cpu={row.get('is_cpu', -1)} "
        f"stock={row.get('stock', -1):<2d} "
        f"dmg={row.get('damage', float('nan')):6.2f} "
        f"x={row.get('x', float('nan')):8.2f} "
        f"y={row.get('y', float('nan')):8.2f} "
        f"vx={row.get('vx', 0.0):8.2f} "
        f"vy={row.get('vy', 0.0):8.2f} "
        f"act={row.get('action_state', -1):4d} "
        f"frame={row.get('action_frame', float('nan')):6.2f} "
        f"face={row.get('facing', float('nan')):+.0f} "
        f"ground={row.get('on_ground', -1)} "
        f"actable={row.get('is_actionable', -1)} "
        f"jumps={row.get('jumps_left', -1)} "
        f"blast={row.get('dist_to_blast', float('nan')):6.2f} "
        f"dx={row.get('dx', float('nan')):7.2f} "
        f"dy={row.get('dy', float('nan')):7.2f} "
        f"rel={row.get('facing_rel', 0):+d} "
        f"rebirth={row.get('is_on_rebirth', -1)} "
        f"cancel={row.get('is_enable_cancel', -1)} "
        f"shield={row.get('shield', -1)} "
        f"shsize={row.get('shield_size', 0.0):.2f} "
        f"hitstun={row.get('hitstun', -1)} "
        f"btn=0x{mask:08x}[{names}] "
        f"stick=({row.get('stick_x', 0.0):+.2f},{row.get('stick_y', 0.0):+.2f}) "
        f"cstick=({row.get('cstick_x', 0.0):+.2f},{row.get('cstick_y', 0.0):+.2f}) "
        f"trig={row.get('trigger', 0.0):.3f} "
        f"flags=0x{int(row.get('raw_flags', 0)):08x}"
    )


def hex_row(row: dict) -> str:
    body = schema.pack_fighter(row, row.get("raw_flags", 0))
    return " ".join(f"{b:02x}" for b in body)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, help="game pid (default: first ssbu_native)")
    ap.add_argument("--frames", type=int, default=1, help="how many frames to print")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between frames")
    ap.add_argument("--json", action="store_true", help="print JSON lines")
    ap.add_argument("--raw", action="store_true", help="also print the shared-memory row")
    args = ap.parse_args()

    pid = args.pid or find_game_pid()
    if not pid:
        print("error: no ssbu_native process found", file=sys.stderr)
        return 2

    ok, hint = check_access(pid)
    if not ok:
        print(f"error: cannot read pid {pid}: {hint}", file=sys.stderr)
        return 2

    with GameState(pid) as gs:
        for frame in range(args.frames):
            fight, rows, ms = gs.read_fighters()
            ms_name = {0: "MENU", 1: "IN_MATCH", 2: "RESULTS"}.get(ms, str(ms))

            if args.json:
                print(json.dumps({"pid": pid, "fight_on": fight, "match_state": ms,
                                  "match_state_name": ms_name,
                                  "stage_id": gs.last_stage_id,
                                  "blast": gs.last_bounds,
                                  "control_layout": repr(gs.control_layout) if gs.control_layout else None,
                                  "slots": rows}))
            else:
                if frame == 0:
                    print(gs.describe())
                print(f"frame {frame}: fight_on={fight}  match={ms_name}  "
                      f"stage_id={gs.last_stage_id}  slots={len(rows)}")
                for slot, row in enumerate(rows):
                    if row is None:
                        print(f"   slot {slot}: (empty)")
                        continue
                    print("   " + fmt_row(row))
                    if args.raw:
                        print("      " + hex_row(row))
            if frame + 1 < args.frames:
                time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())