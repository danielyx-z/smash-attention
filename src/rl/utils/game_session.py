#!/usr/bin/env python3
"""Serialize every live test on the one game instance.

    python3 tools/mem/game_session.py run --owner B -- python3 tools/mem/test_b.py
    python3 tools/mem/game_session.py run --owner D -- sh -c '...'

Holds /tmp/ssbu_game.lock for the whole command.  Starts the game/reader only
if they are not already running; on exit it stops only what it started.
Do NOT call play.sh or pkill ssbu_native directly; use this.
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RLROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))   # ssbu-rl
REPO = os.environ.get("SSBU_RECOMP_DIR") or os.path.abspath(
    os.path.join(RLROOT, "..", "ssbu-recomp"))
sys.path.insert(0, HERE)

from rl.utils.locks import GAME_LOCK, FileLock  # noqa: E402

DISPLAY = os.environ.get("SSBU_SESSION_DISPLAY", ":0")
DEFAULT_MATCH = os.environ.get(
    "SSBU_SESSION_MATCH",
    "p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end,players=2")


def procs(pattern: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True, text=True).stdout
    hits = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, args = line.partition(" ")
        if pattern in args and "game_session.py" not in args:
            hits.append(int(pid))
    return hits


def game_pid() -> int | None:
    hits = procs("build-linux/work/ssbu_native")
    return hits[0] if hits else None


def reader_pid() -> int | None:
    hits = procs("rl.memory.reader")
    return hits[0] if hits else None


def wait_for(fn, timeout: float, interval: float = 0.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = fn()
        if v:
            return v
        time.sleep(interval)
    return None


def stop(pid: int, name: str) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(40):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--owner", required=True)
    r.add_argument("--match", default=DEFAULT_MATCH)
    r.add_argument("--no-launch", action="store_true", help="fail if the game is not running")
    r.add_argument("--no-reader", action="store_true", help="do not start the reader")
    r.add_argument("--restart-reader", action="store_true", help="stop and restart the reader")
    r.add_argument("--restart-game", action="store_true", help="stop and restart the game")
    r.add_argument("--keep-game", action="store_true",
                   help="leave the game and the reader it started running on exit")
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="extra SSBU_* environment for the game (repeatable)")
    r.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.cmd != "run":
        return 2
    cmd = args.command
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("missing command after --")

    env = dict(os.environ)
    env["DISPLAY"] = DISPLAY
    env["SSBU_SESSION_OWNER"] = args.owner
    log_path = f"/tmp/opencode/session_{args.owner}.log"
    os.makedirs("/tmp/opencode", exist_ok=True)
    log = open(log_path, "a", buffering=1)
    print(f"[{args.owner}] lock {GAME_LOCK}", file=log)

    with FileLock(GAME_LOCK, timeout=args.timeout):
        env["SSBU_GAME_LOCK_HELD"] = "1"
        started_game = started_reader = None

        if args.restart_reader:
            p = reader_pid()
            if p:
                stop(p, "reader")
        if args.restart_game:
            p = game_pid()
            if p:
                stop(p, "game")

        if game_pid() is None:
            if args.no_launch:
                print(f"[{args.owner}] no game and --no-launch", file=log)
                return 3
            print(f"[{args.owner}] starting game", file=log)
            play = os.path.join(REPO, "play.sh")
            play_env = ["--env", "SSBU_SKIP_MOVIES=1",
                        "--env", f"SSBU_KEYBOARD_PLAYER={os.environ.get('SSBU_KEYBOARD_PLAYER', '1')}",
                        "--env", "SSBU_RL=1",
                        "--env", f"SSBU_DEBUG_MATCH={args.match}"]
            for kv in args.env:
                play_env += ["--env", kv]
            started_game = subprocess.Popen(
                [play] + play_env,
                cwd=REPO, env=env, stdout=log, stderr=log,
                start_new_session=True)
            if not wait_for(game_pid, 120.0):
                print(f"[{args.owner}] game did not start", file=log)
                return 4

        if not args.no_reader and reader_pid() is None:
            print(f"[{args.owner}] starting reader", file=log)
            started_reader = subprocess.Popen(
                [sys.executable, "-m", "rl.memory.reader"],
                cwd=RLROOT, env=env, stdout=log, stderr=log,
                start_new_session=True)
            time.sleep(1.0)

        print(f"[{args.owner}] run: {shlex.join(cmd)}", file=log)
        try:
            rc = subprocess.run(cmd, cwd=RLROOT, env=env, stdout=log, stderr=log).returncode
        finally:
            if started_reader is not None and not args.keep_game:
                stop(started_reader.pid, "reader")
            if started_game is not None and not args.keep_game:
                p = game_pid()
                if p:
                    stop(p, "game")
        print(f"[{args.owner}] exit {rc}", file=log)
        return rc


if __name__ == "__main__":
    raise SystemExit(main())
