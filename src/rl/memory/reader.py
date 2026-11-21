#!/usr/bin/env python3
"""Reader daemon: sample the game at 60 fps and publish it to shared memory.

    python3 tools/mem/reader.py                 # attach to the running game
    python3 tools/mem/reader.py --spawn         # start the game as our child
    python3 tools/mem/reader.py --duration 10   # benchmark for 10 s, then stop

Why --spawn matters: with the default Yama setting (ptrace_scope=1) a process
may only read the memory of its own descendants.  If you start the game
yourself, reader.py needs 'sudo sysctl -w kernel.yama.ptrace_scope=0' or the
game must be launched by --spawn.  See README.md.

The block is written with a sequence counter (odd while writing, even when
complete), so readers never see a half-updated frame.
"""

from __future__ import annotations

import argparse
import os
import signal
import struct
import subprocess
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from rl.memory import schema
from rl.memory.gamestate import GameState
from rl.memory.memapi import check_access, find_game_pid, ptrace_scope

RLROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPO = os.environ.get("SSBU_RECOMP_DIR") or os.path.abspath(
    os.path.join(RLROOT, "..", "ssbu-recomp"))

_SEQ = struct.Struct("<I")
_TS = struct.Struct("<Q")
_FPS = struct.Struct("<f")
_FIGHT = struct.Struct("<I")
_NFIGHTERS = struct.Struct("<I")
_MSTATE = struct.Struct("<I")
_BLAST = struct.Struct("<4f")
_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")

TS_OFFSET = 16
FPS_OFFSET = 24
FIGHT_OFFSET = 32
NFIGHTERS_OFFSET = 8
MSTATE_OFFSET = 36
BLAST_OFFSET = schema.BLAST_OFFSET


def spawn_game(log_path: str | None = None, extra: list[str] | None = None) -> subprocess.Popen:
    """Start play.sh as a child so ptrace_scope=1 still allows reads."""
    play = os.path.join(REPO, "play.sh")
    if not os.path.exists(play):
        raise SystemExit(f"error: {play} not found")
    log = open(log_path, "ab") if log_path else subprocess.DEVNULL
    cmd = [play] + (extra or [])
    return subprocess.Popen(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=os.environ.copy())


def wait_for_game(timeout: float = 180.0) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pid = find_game_pid()
        if pid:
            return pid
        time.sleep(0.2)
    raise SystemExit("error: the game did not start within the timeout")


class Publisher:
    """Owns the shared-memory block and the odd/even publish protocol."""

    def __init__(self, path: str = schema.SHM_PATH):
        self.mm = schema.open_writer(path)
        self.seq = 0
        self.mm[0:schema.BLOCK_SIZE] = b"\0" * schema.BLOCK_SIZE
        self.mm[0:schema.HEADER_SIZE] = schema.pack_header(
            0, 0, 0, 0.0, os.getpid(), 0, 0)

    def publish(self, fight_on: bool, rows: list[dict | None], match_state: int,
                ts_ns: int, fps: float, bounds=None, stage_id: int | None = None,
                game_frame: int = 0, flags: int = 0) -> None:
        mm = self.mm
        self.seq = (self.seq + 1) | 1          # odd: body is being written
        mm[schema.SEQ_OFFSET:schema.SEQ_OFFSET + 4] = _SEQ.pack(self.seq)
        mm[TS_OFFSET:TS_OFFSET + 8] = _TS.pack(ts_ns)
        mm[FPS_OFFSET:FPS_OFFSET + 4] = _FPS.pack(fps)
        mm[FIGHT_OFFSET:FIGHT_OFFSET + 4] = _FIGHT.pack(1 if fight_on else 0)
        n_fighters = sum(1 for row in rows if row is not None)
        mm[NFIGHTERS_OFFSET:NFIGHTERS_OFFSET + 4] = _NFIGHTERS.pack(n_fighters)
        mm[MSTATE_OFFSET:MSTATE_OFFSET + 4] = _MSTATE.pack(match_state)
        mm[BLAST_OFFSET:BLAST_OFFSET + 16] = _BLAST.pack(*(bounds or (0.0, 0.0, 0.0, 0.0)))
        mm[schema.STAGE_ID_OFFSET:schema.STAGE_ID_OFFSET + 4] = _I32.pack(int(stage_id or 0))
        mm[schema.GAME_FRAME_OFFSET:schema.GAME_FRAME_OFFSET + 4] = _U32.pack(game_frame & 0xFFFFFFFF)
        mm[schema.FLAGS_OFFSET:schema.FLAGS_OFFSET + 4] = _U32.pack(flags & 0xFFFFFFFF)
        # `rows` is already in player order: slot 0 = entry 0 (P1), slot 1 =
        # entry 1 (P2).  A missing player is None -> an all-zero slot.
        for i in range(schema.MAX_FIGHTERS):
            off = schema.fighter_offset(i)
            row = rows[i] if i < len(rows) else None
            if row is None:
                mm[off:off + schema.FIGHTER_SIZE] = b"\0" * schema.FIGHTER_SIZE
            else:
                mm[off:off + schema.FIGHTER_SIZE] = schema.pack_fighter(
                    row, row.get("raw_flags", 0))
        self.seq += 1                          # even: frame is complete
        mm[schema.SEQ_OFFSET:schema.SEQ_OFFSET + 4] = _SEQ.pack(self.seq)

    def close(self) -> None:
        self.mm.close()


class FpsMeter:
    def __init__(self, window: int = 120):
        self.window = window
        self.stamps: list[float] = []

    def tick(self, now: float) -> float:
        self.stamps.append(now)
        if len(self.stamps) > self.window:
            del self.stamps[:len(self.stamps) - self.window]
        if len(self.stamps) < 2:
            return 0.0
        span = self.stamps[-1] - self.stamps[0]
        return (len(self.stamps) - 1) / span if span > 0 else 0.0


def gated_sample(gate, state: dict, read, publish) -> bool:
    """One lockstep poll.  Returns True when it handled the iteration.

    While the runtime parks the game at frame F (gate enabled), read the game
    memory once, check that the game was still parked at F afterwards (so the
    sample is one consistent frame) and publish it tagged F.  One publish per
    parked frame.  Returns False when the gate is off: the caller then runs
    the timer-driven path.
    """
    st = gate.read()
    state["published"] = False
    if st.epoch != state.get("epoch"):
        state["epoch"], state["last"] = st.epoch, None
    if not st.enable:
        state["last"] = None
        return False
    if not st.parked or st.blocked == state.get("last"):
        time.sleep(0.0002)
        return True
    tag = st.blocked
    data = read()
    st2 = gate.read()
    if not (st2.enable and st2.parked and st2.blocked == tag):
        return True                           # the game moved during the read
    publish(tag, data)
    state["last"] = tag
    state["published"] = True
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, help="game pid (default: first ssbu_native)")
    ap.add_argument("--spawn", action="store_true", help="start play.sh as our child")
    ap.add_argument("--spawn-log", default=os.path.expanduser("~/smash_reader.log"),
                    help="where --spawn sends the game output")
    ap.add_argument("--fps", type=float, default=60.0, help="target sample rate")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    ap.add_argument("--shm", default=schema.SHM_PATH, help="shared memory path")
    ap.add_argument("--elf", help="override main.elf path")
    args = ap.parse_args()

    child = None
    if args.spawn:
        print(f"spawning {os.path.join(REPO, 'play.sh')} (log: {args.spawn_log})", flush=True)
        child = spawn_game(args.spawn_log)

    pid = args.pid or wait_for_game()
    ok, hint = check_access(pid)
    if not ok:
        print(f"error: cannot read pid {pid}: {hint}", file=sys.stderr)
        if child:
            child.terminate()
        return 2

    print(f"reading pid {pid}", flush=True)
    gs = GameState(pid, args.elf)
    print(gs.describe(), flush=True)

    pub = Publisher(args.shm)
    meter = FpsMeter()

    stop = False

    def on_signal(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    from rl.input.gate import FrameGate, GateError
    gate = None
    gate_checked = 0.0
    gate_state: dict = {}       # lockstep bookkeeping (see gated_sample)

    period = 1.0 / args.fps
    start = time.perf_counter()
    next_at = start
    frames = 0
    t_end = start + args.duration if args.duration > 0 else None
    last_report = start
    last_fps = 0.0

    try:
        while not stop:
            now = time.perf_counter()
            if t_end and now >= t_end:
                break

            # lockstep: while the runtime parks the game at a frame boundary,
            # publish exactly one consistent sample per parked frame, tagged
            # with that frame number (header game_frame).  Otherwise the
            # timer-driven path below runs as before.
            if gate is None and now - gate_checked > 0.5:
                gate_checked = now
                try:
                    gate = FrameGate().open()
                except GateError:
                    gate = None
            if gate is not None:
                def _read():
                    return gs.read_fighters()

                def _publish(tag, data):
                    fight_, rows_, ms_ = data
                    hflags_ = 0
                    if gs.last_stage_id is not None:
                        hflags_ |= schema.HEADER_FLAG_STAGE_ID
                    if gs.control_layout is not None:
                        hflags_ |= schema.HEADER_FLAG_CONTROL_LAYOUT
                    pub.publish(fight_, rows_, ms_,
                                time.clock_gettime_ns(time.CLOCK_MONOTONIC),
                                last_fps, gs.last_bounds,
                                stage_id=gs.last_stage_id, game_frame=tag,
                                flags=hflags_)

                try:
                    handled = gated_sample(gate, gate_state, _read, _publish)
                except Exception as exc:      # noqa: BLE001
                    print(f"read error (retrying): {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(0.001)
                    continue
                if handled:
                    if gate_state.get("published"):
                        frames += 1
                        next_at = time.perf_counter()
                    continue
            try:
                fight, rows, ms = gs.read_fighters()
            except Exception as exc:          # a torn-down match must not kill the daemon
                print(f"read error (retrying): {type(exc).__name__}: {exc}", flush=True)
                time.sleep(period)
                continue
            ts_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
            last_fps = meter.tick(now)
            hflags = 0
            if gs.last_stage_id is not None:
                hflags |= schema.HEADER_FLAG_STAGE_ID
            if gs.control_layout is not None:
                hflags |= schema.HEADER_FLAG_CONTROL_LAYOUT
            pub.publish(fight, rows, ms, ts_ns, last_fps, gs.last_bounds,
                        stage_id=gs.last_stage_id, game_frame=frames,
                        flags=hflags)
            frames += 1

            if now - last_report >= 1.0:
                last_report = now
                ms_name = {0: "MENU", 1: "IN_MATCH", 2: "RESULTS"}.get(ms, str(ms))
                n = sum(1 for row in rows if row is not None)
                print(f"  {frames} frames, {last_fps:5.1f} fps, fight_on={fight}, "
                      f"match={ms_name}, fighters={n}, stage={gs.last_stage_id}", flush=True)

            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0.0015:
                time.sleep(delay - 0.001)
                delay = next_at - time.perf_counter()
            while delay > 0:
                delay = next_at - time.perf_counter()
            if next_at < time.perf_counter() - period * 5:
                next_at = time.perf_counter()   # resync after a long stall

            if not gs.mem.alive:
                print("game exited", flush=True)
                break
    finally:
        elapsed = time.perf_counter() - start
        pub.close()
        gs.close()
        if elapsed > 0:
            print(f"stopped: {frames} frames in {elapsed:.2f}s = {frames / elapsed:.2f} fps", flush=True)
        if child and child.poll() is None:
            child.terminate()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())