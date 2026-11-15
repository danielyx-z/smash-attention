#!/usr/bin/env python3
"""Player 2's pad: a uinput virtual Xbox 360 controller.

The recomp's pad layer (nnpad_sdl.c) gives a pad that connects while the
keyboard holds player 1 the first free slot -- player 2 -- and tells the game
so ("connected ... -> player 2").  The game then takes it at the character
select on its first press, exactly like a pad plugged into the console.  That
is the reliable way to drive player 2: a real HID device, not a synthesized
XInput state.

    python -m rl.scripts.p2_pad            # pad up, joins player 2 in the menus
    python -m rl.scripts.p2_pad --fifo /tmp/opencode/p2fifo

While the game is in a menu it presses L + R (the game's own join gesture) once
every ``--period`` seconds; during a fight it leaves the pad to the policy.  A
FIFO, if given, also takes manual commands: ``a``, ``b``, ``start``,
``stick X Y``, ``neutral``.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time

from rl.input.gym_input import VirtualPad

JOIN = {"buttons": ["L", "R"]}
NEUTRAL = {}
BUTTONS = {
    "a": ["A"], "b": ["B"], "x": ["X"], "y": ["Y"],
    "start": ["PLUS"], "back": ["MINUS"],
    "up": ["UP"], "down": ["DOWN"], "left": ["LEFT"], "right": ["RIGHT"],
}


def _wait_for_game(timeout: float = 180.0) -> bool:
    """The pad must appear after the game's pad layer is up: the device takes the first free player
    slot, and the keyboard is player 1.  A pad created before the game starts can land on player 1."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            from rl.memory.memapi import find_game_pid
            find_game_pid()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def _fight_on(reader) -> bool:
    if reader is None:
        return False
    try:
        snap = reader.read_latest(timeout=0.05)
        return bool(snap is not None and snap.fight_on)
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", type=float, default=1.5, help="seconds between join presses in menus")
    ap.add_argument("--hold", type=float, default=0.25, help="how long the join press is held")
    ap.add_argument("--fifo", default=None, help="also read manual commands from this FIFO")
    ap.add_argument("--name", default=None, help="uinput device name")
    args = ap.parse_args(argv)

    lock_fd = os.open("/dev/shm/ssbu_p2pad.lock", os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("p2 pad: another copy already holds /dev/shm/ssbu_p2pad.lock", file=sys.stderr)
        return 1

    if not _wait_for_game(60.0):
        print("p2 pad: the game is not running yet; creating the pad anyway", file=sys.stderr)

    with open("/dev/shm/ssbu_p2pad.pid", "w") as fh:
        fh.write(str(os.getpid()))

    pad = VirtualPad(name=args.name) if args.name else VirtualPad()
    pad.create()
    pad.apply(NEUTRAL)

    reader = None
    try:
        from rl.env.gym_env import RawSnapshotReader
        reader = RawSnapshotReader()
    except Exception:
        pass

    fifo_fd = None
    if args.fifo:
        try:
            os.mkfifo(args.fifo)
        except FileExistsError:
            pass
        fifo_fd = os.open(args.fifo, os.O_RDONLY | os.O_NONBLOCK)

    print(f"p2 pad: {pad.event_path} (player 2)", flush=True)
    stick = (0.0, 0.0)
    manual_until = 0.0
    last_join = 0.0
    try:
        while True:
            now = time.monotonic()
            if fifo_fd is not None:
                try:
                    data = os.read(fifo_fd, 4096).decode("utf-8", "replace")
                except BlockingIOError:
                    data = ""
                for line in data.splitlines():
                    cmd = line.strip().lower().split()
                    if not cmd:
                        continue
                    if cmd[0] == "stick" and len(cmd) == 3:
                        stick = (float(cmd[1]), float(cmd[2]))
                        pad.apply({"stick": stick})
                        manual_until = now + 30.0
                    elif cmd[0] == "neutral":
                        stick = (0.0, 0.0)
                        pad.apply(NEUTRAL)
                        manual_until = now + 5.0
                    elif cmd[0] in BUTTONS:
                        pad.apply({"buttons": BUTTONS[cmd[0]], "stick": stick})
                        time.sleep(0.15)
                        pad.apply({"stick": stick})
                        manual_until = now + 5.0
                    elif cmd[0] == "join":
                        pad.apply(JOIN)
                        time.sleep(args.hold)
                        pad.apply(NEUTRAL)
                        manual_until = now + 5.0
            if now < manual_until:
                time.sleep(0.05)
                continue
            if _fight_on(reader):
                last_join = 0.0
                time.sleep(0.2)
                continue
            if now - last_join >= args.period:
                last_join = now
                pad.apply(JOIN)
                time.sleep(args.hold)
                pad.apply(NEUTRAL)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            pad.close()
        except Exception:
            pass
        if fifo_fd is not None:
            os.close(fifo_fd)
        os.close(lock_fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
