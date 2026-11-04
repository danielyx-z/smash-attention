#!/usr/bin/env python3
"""Reset client: ask the runtime for a match reset and wait for a ready match.

Two shared-memory blocks are involved:

  /dev/shm/ssbu_control.bin  reset_seq / ack_seq / state   (control_block.py)
  /dev/shm/ssbu_state.bin    reader frames, fight_on, seq (B's state bridge)

Protocol:

    seq = control.request_reset()          # reset_seq += 1, state = requested
    control.wait_for_done(seq)             # ack_seq == seq and state == done
    wait for fight_on == 1 on N fresh reader frames

"Fresh" means a new state-block ``seq`` value, so the stable-frame count is
not satisfied by re-reading the same frame.  The default is 3 frames (50 ms at
60 Hz), matching the contract's "stable spawn frames".

The runtime gating itself is owned by Agent A; this client only speaks the
documented protocol.  ``wait_for_done`` accepts a runtime that returns to
``idle`` after acking, which keeps the client compatible with a simpler
runtime implementation.

Mock mode (no game, no shared memory) is built in:

    with ResetClient(mock=True) as client:
        result = client.reset()

    python3 tools/mem/gym_reset.py --mock
    python3 tools/mem/gym_reset.py --timeout 5      # live, needs the runtime
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.input import control_block
from rl.input.control_block import (
    ControlBlock, ControlBlockError, ControlBlockSnapshot, ControlBlockTimeout,
    STATE_DONE, STATE_ERROR, STATE_IDLE, STATE_REQUESTED, STATE_RUNNING,
)

STATE_PATH = "/dev/shm/ssbu_state.bin"
STATE_MAGIC = b"SSBU"
STATE_SEQ_OFFSET = 12       # v6 and v7: magic, version, n_fighters, seq
STATE_FIGHT_ON_OFFSET = 32  # v6 and v7: ..., fps, reader_pid, fight_on
STATE_MATCH_OFFSET = 36
STATE_TS_OFFSET = 16

_HEADER_PEEK = 40
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


class ResetError(RuntimeError):
    """A reset could not be requested or did not become ready."""


class ResetTimeout(ResetError, TimeoutError):
    """The runtime or the state reader did not reach the expected state."""


@dataclass(frozen=True)
class StateSample:
    """The few state-block fields the reset client needs."""

    seq: int
    fight_on: int
    match_state: int
    ts_ns: int

    @property
    def ready(self) -> bool:
        return self.fight_on == 1


@dataclass(frozen=True)
class ResetResult:
    seq: int
    ack_seq: int
    state: int
    stable_frames: int
    frames_seen: int
    elapsed: float
    last: StateSample | None = None

    @property
    def ok(self) -> bool:
        return self.state != STATE_ERROR and self.stable_frames > 0


def read_state(path: str = STATE_PATH) -> StateSample | None:
    """Read the small header fields from the state block; None if unavailable.

    No import of schema/gamestate on purpose: the offsets used here are stable
    across the v6 -> v7 layouts, so the reset client keeps working while Agent B
    changes the rest of the block.
    """
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return None
    try:
        data = os.pread(fd, _HEADER_PEEK, 0)
    except OSError:
        return None
    finally:
        os.close(fd)
    if len(data) < _HEADER_PEEK or data[:4] != STATE_MAGIC:
        return None
    seq = _U32.unpack_from(data, STATE_SEQ_OFFSET)[0]
    fight_on = _U32.unpack_from(data, STATE_FIGHT_ON_OFFSET)[0]
    match_state = _U32.unpack_from(data, STATE_MATCH_OFFSET)[0]
    ts_ns = _U64.unpack_from(data, STATE_TS_OFFSET)[0]
    return StateSample(seq=seq, fight_on=fight_on, match_state=match_state,
                       ts_ns=ts_ns)


def wait_fight_on(reader=read_state, timeout: float = 5.0,
                  stable_frames: int = 3, poll: float = 0.002,
                  require_new: bool = True) -> tuple[StateSample, int, int]:
    """Wait for ``stable_frames`` fresh frames with fight_on == 1.

    Returns ``(last_sample, frames_seen, distinct_ready_frames)``.  ``reader``
    is any callable returning a StateSample or None.
    """
    if stable_frames < 1:
        raise ValueError("stable_frames must be >= 1")
    deadline = time.monotonic() + timeout
    last_seq: int | None = None
    ready_run = 0
    frames_seen = 0
    last: StateSample | None = None
    while time.monotonic() < deadline:
        sample = reader()
        if sample is None:
            time.sleep(poll)
            continue
        # only count a new seq once
        if sample.seq == last_seq:
            time.sleep(poll)
            continue
        last_seq = sample.seq
        frames_seen += 1
        last = sample
        if sample.ready:
            ready_run += 1
            if ready_run >= stable_frames:
                return sample, frames_seen, ready_run
        else:
            ready_run = 0
        time.sleep(poll)
    if last is None:
        raise ResetTimeout(
            f"no state frames from {getattr(reader, '__name__', 'reader')} "
            f"within {timeout:.2f}s (is reader.py running?)")
    raise ResetTimeout(
        f"fight_on == 1 for {ready_run}/{stable_frames} stable frames within "
        f"{timeout:.2f}s (last seq {last.seq}, fight_on {last.fight_on}, "
        f"match_state {last.match_state})")


class MockRuntime:
    """A fake runtime + state source, for unit tests and ``--mock``.

    It implements the same ``read``/``wait_for_ack``/``wait_for_done`` surface
    as control_block.ControlBlock and is also callable as a state reader.
    """

    def __init__(self, ack_delay: float = 0.02, fight_on_delay: float = 0.05,
                 fail: bool = False, stable_frames: int = 3,
                 start_fight_on: bool = False):
        self.ack_delay = ack_delay
        self.fight_on_delay = fight_on_delay
        self.fail = fail
        self.start_fight_on = start_fight_on
        self.reset_seq = 0
        self.ack_seq = 0
        self.state = STATE_IDLE
        self._requested_at: float | None = None
        self._seq = 0
        self._fight_on = 1 if start_fight_on else 0
        self._match_state = 0
        self._now = time.monotonic

    # control-block surface
    def read(self) -> ControlBlockSnapshot:
        if self._requested_at is not None and self.ack_seq != self.reset_seq:
            elapsed = self._now() - self._requested_at
            if elapsed >= self.ack_delay:
                if self.fail:
                    self.state = STATE_ERROR
                    self.ack_seq = self.reset_seq
                elif elapsed >= self.ack_delay:
                    self.state = STATE_DONE
                    self.ack_seq = self.reset_seq
                    self._fight_on = 0
                    self._match_state = 0
        return ControlBlockSnapshot(
            control_block.MAGIC, control_block.VERSION, self.reset_seq,
            self.ack_seq, self.state, 0)

    def request_reset(self) -> int:
        self.reset_seq = (self.reset_seq + 1) & 0xFFFFFFFF
        self.state = STATE_REQUESTED
        self._requested_at = self._now()
        return self.reset_seq

    def wait_for_ack(self, seq: int, timeout: float = 5.0,
                     poll: float = 0.002) -> ControlBlockSnapshot:
        deadline = self._now() + timeout
        while True:
            snap = self.read()
            if snap.failed:
                raise ControlBlockError("mock runtime reported an error")
            if snap.ack_seq == seq:
                return snap
            if self._now() >= deadline:
                raise ControlBlockTimeout("mock ack timeout")
            time.sleep(poll)

    def wait_for_done(self, seq: int, timeout: float = 5.0,
                      poll: float = 0.002) -> ControlBlockSnapshot:
        snap = self.wait_for_ack(seq, timeout, poll)
        deadline = self._now() + timeout
        while snap.state in (STATE_REQUESTED, STATE_RUNNING):
            if self._now() >= deadline:
                raise ControlBlockTimeout("mock done timeout")
            time.sleep(poll)
            snap = self.read()
        if snap.state == STATE_ERROR:
            raise ControlBlockError("mock runtime reported an error")
        return snap

    # state-reader surface
    def __call__(self) -> StateSample:
        self.read()
        if self._requested_at is not None and self.ack_seq == self.reset_seq:
            if self._now() - self._requested_at >= self.fight_on_delay and not self.fail:
                self._fight_on = 1
                self._match_state = 1
        self._seq += 1
        return StateSample(self._seq, self._fight_on, self._match_state,
                           int(time.time_ns()))


class ResetClient:
    """Request a reset and wait for a ready match.

    ``control`` is a ControlBlock (default) or anything with the same surface.
    ``state_reader`` defaults to :func:`read_state`; pass None explicitly to
    keep the client from waiting on the state block (useful for pure protocol
    tests).
    """

    def __init__(self, control=None, state_reader=read_state,
                 state_path: str = STATE_PATH, timeout: float = 5.0,
                 stable_frames: int = 3, poll: float = 0.002,
                 mock: bool = False, mock_ack_delay: float = 0.02,
                 mock_fight_on_delay: float = 0.05, mock_fail: bool = False):
        self.timeout = timeout
        self.stable_frames = stable_frames
        self.poll = poll
        self.state_path = state_path
        self._owns_control = False
        self.runtime = None
        if mock:
            self.runtime = MockRuntime(ack_delay=mock_ack_delay,
                                       fight_on_delay=mock_fight_on_delay,
                                       fail=mock_fail, stable_frames=stable_frames)
            self.control = self.runtime
            self.state_reader = self.runtime
        else:
            if control is None:
                self.control = ControlBlock(timeout=timeout, poll=poll)
                self._owns_control = True
            else:
                self.control = control
            self.state_reader = state_reader

    # -- lifecycle -----------------------------------------------------
    def open(self) -> "ResetClient":
        if self.runtime is None and isinstance(self.control, ControlBlock):
            self.control.open()
        return self

    def close(self) -> None:
        if self._owns_control and isinstance(self.control, ControlBlock):
            self.control.close()
            self._owns_control = False

    def __enter__(self) -> "ResetClient":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- reset -----------------------------------------------------------
    def request(self) -> int:
        """Increment reset_seq; returns the sequence to wait for."""
        if isinstance(self.control, ControlBlock):
            self.control.open()
        return self.control.request_reset()

    def wait(self, seq: int, timeout: float | None = None,
             stable_frames: int | None = None) -> ResetResult:
        """Wait for the ack/done and then for stable fight_on frames."""
        timeout = self.timeout if timeout is None else timeout
        stable_frames = self.stable_frames if stable_frames is None else stable_frames
        t0 = time.monotonic()
        deadline = t0 + timeout
        try:
            snap = self.control.wait_for_done(seq, timeout)
        except (ControlBlockError, ControlBlockTimeout) as e:
            raise ResetError(f"reset seq {seq} failed: {e}") from e
        if snap.state == STATE_ERROR:
            raise ResetError(f"reset seq {seq} reported state=error")
        remaining = max(0.0, deadline - time.monotonic())
        last = None
        frames_seen = 0
        ready_run = 0
        if self.state_reader is not None:
            try:
                last, frames_seen, ready_run = wait_fight_on(
                    self.state_reader, timeout=remaining,
                    stable_frames=stable_frames, poll=self.poll)
            except ResetTimeout as e:
                raise ResetTimeout(f"reset seq {seq} acked but not ready: {e}") from e
        else:
            # protocol-only client: treat the ack itself as success
            ready_run = stable_frames
        elapsed = time.monotonic() - t0
        if snap.state != STATE_DONE and snap.state != STATE_IDLE and \
                snap.state not in (STATE_REQUESTED, STATE_RUNNING):
            raise ResetError(f"reset seq {seq} ended in state {snap.state}")
        return ResetResult(seq=seq, ack_seq=snap.ack_seq, state=snap.state,
                           stable_frames=ready_run, frames_seen=frames_seen,
                           elapsed=elapsed, last=last)

    def reset(self, timeout: float | None = None,
              stable_frames: int | None = None) -> ResetResult:
        seq = self.request()
        return self.wait(seq, timeout, stable_frames)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SSBU reset client")
    ap.add_argument("--mock", action="store_true", help="use the in-process mock runtime")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--stable-frames", type=int, default=3)
    ap.add_argument("--control-path", default=control_block.SHM_PATH)
    ap.add_argument("--state-path", default=STATE_PATH)
    ap.add_argument("--no-state", action="store_true",
                    help="wait only for the ack (no fight_on check)")
    ap.add_argument("--mock-fail", action="store_true")
    args = ap.parse_args(argv)

    client: ResetClient
    if args.mock:
        client = ResetClient(mock=True, mock_fail=args.mock_fail,
                             timeout=args.timeout,
                             stable_frames=args.stable_frames)
    else:
        client = ResetClient(
            control=ControlBlock(args.control_path, timeout=args.timeout),
            state_reader=None if args.no_state else read_state,
            state_path=args.state_path, timeout=args.timeout,
            stable_frames=args.stable_frames)
    try:
        with client:
            result = client.reset()
        print(f"reset seq {result.seq}: ack {result.ack_seq}, state {result.state}, "
              f"{result.stable_frames} stable fight_on frames "
              f"({result.frames_seen} seen) in {result.elapsed:.3f}s")
        if result.last is not None:
            print(f"  last state seq {result.last.seq}, fight_on {result.last.fight_on}, "
                  f"match_state {result.last.match_state}")
        return 0 if result.ok else 1
    except (ResetError, ControlBlockError, ControlBlockTimeout) as e:
        print(f"gym_reset: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
