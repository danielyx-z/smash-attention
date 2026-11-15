#!/usr/bin/env python3
"""Record and compare per-frame ``game_frame`` / ``state_hash`` samples.

The runtime publishes a v2 control block (80 bytes) whose ``state_hash`` is a
hash of the *fighter simulation* for the presented frame.  If turbo
(``SSBU_SPEED``) is simulation-correct, then at the same seed and with the same
match and inputs, a 1x run and a turbo run must produce the same hash for every
corresponding (relative) game frame.  This tool records those samples and
aligns two runs to prove it.

The block is read directly with ``mmap`` + ``struct`` (no per-sample file
reads), so the tool works today even if the v2 ``ControlBlock`` client has not
landed yet; it deliberately does not import that client.  Layout used
(little-endian; offsets from the v2 contract):

    offset  type  field
    0       char4 magic        "SSBC"
    4       u32   version      >= 2 for the hashing layout
    20      u32   flags        bit 0 fight_on, bit 6 hash_valid
    24      u32   game_frame   game frame (player 1's pad poll, one per step)
    76      u32   match_start  game_frame at the ready-go (alignment base)
    28      u32   state_hash   hash of the fighter simulation
    32      u32   seed         seed the runtime was started with
    68      u32   speed_milli  SSBU_SPEED in thousandths (1000 = 1x,
                                0 = uncapped)

Usage
-----
Start the game twice from the *same* seed, match and recorded inputs -- once at
normal speed and once with turbo -- and record each run:

    # terminal 1, 1x reference
    SSBU_FIXED_SEED=12345 SSBU_SPEED=1      ./play.sh ...
    .venv/bin/python -m rl.scripts.turbo_check record \\
        --out /tmp/ref.json --frames 600 --timeout 60

    # terminal 2, turbo
    SSBU_FIXED_SEED=12345 SSBU_SPEED=8      ./play.sh ...
    .venv/bin/python -m rl.scripts.turbo_check record \\
        --out /tmp/turbo.json --frames 600 --timeout 60

    .venv/bin/python -m rl.scripts.turbo_check compare \\
        /tmp/ref.json /tmp/turbo.json --max-missing 0
    .venv/bin/python -m rl.scripts.turbo_check compare \\
        /tmp/ref.json /tmp/turbo.json --json

``compare`` exits 0 on PASS, 1 on a hash divergence or too many missing
frames, and 2 on bad input.  ``--self-test`` runs a small in-process sanity
check.  ``record`` polls the block at ~1 kHz (``--poll``), records only
``fight_on`` frames with ``hash_valid`` set, dedupes by ``game_frame``, and
stops at ``--frames``, when ``fight_on`` falls after having been set, or at
``--timeout``.  It skips boot/menu frames on purpose: their relative frames
differ between runs.  A live frame with ``seed=0`` means the fight was never
re-seeded (no ``SSBU_FIXED_SEED``) and is rejected; ``--expect-speed`` fails
when the block says 1000 but the run was meant to be turbo (``SSBU_SPEED`` did
not reach the game).

The runtime writes this process's pid (low 24 bits) into ``flags`` bits 8..31.
``record`` looks up the running game and ignores a block whose epoch belongs to
another process, and starts over if the epoch changes, so a stale block left by
a previous launch cannot poison a run.

Important
---------
* A meaningful comparison requires ``SSBU_FIXED_SEED`` and the same match,
  fighters, CPU levels, stage and inputs on both runs.  Compare
  ``SSBU_SPEED=1`` against the turbo value; if the seeds differ, the runs are
  not comparable even when every hash happens to match.
* The hash covers the fighter simulation only.  Particles, camera, audio and
  other rendering-side state are not part of it, so a PASS does not validate
  those systems.
* Runs are aligned by *relative* game frame (each run's first frame is 0), so
  the two recordings may start at different absolute frame numbers.
* The control block has no sequence lock for the v2 fields.  A sample can be
  torn while the runtime writes it; duplicate ``game_frame`` values are
  dropped (first sample wins).

No third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

FORMAT = "ssbu_turbo_v1"
COMPARE_FORMAT = "ssbu_turbo_compare_v1"
DEFAULT_PATH = "/dev/shm/ssbu_control.bin"

MAGIC = b"SSBC"
SIZE = 80

VERSION_OFFSET = 4
FLAGS_OFFSET = 20
GAME_FRAME_OFFSET = 24
STATE_HASH_OFFSET = 28
SEED_OFFSET = 32
SPEED_MILLI_OFFSET = 68
MATCH_START_OFFSET = 76

FLAG_FIGHT_ON = 1 << 0
FLAG_HASH_VALID = 1 << 6
#: Flags bits 8..31 carry the writer's epoch (the game pid's low 24 bits).
#: A block left behind by a previous process has a different epoch, so a
#: recorder can refuse it instead of hashing a stale match.
EPOCH_SHIFT = 8
EPOCH_MASK = 0xFFFFFF

MIN_VERSION = 2

_U32 = struct.Struct("<I")


class TurboCheckError(RuntimeError):
    """Recording or parsing the control block failed."""


class BadInput(ValueError):
    """A run JSON file is missing, malformed or not a ssbu_turbo_v1 run."""


@dataclass(frozen=True)
class Sample:
    """One decoded v2 control-block sample."""

    version: int
    flags: int
    game_frame: int
    state_hash: int
    seed: int
    speed_milli: int
    match_start: int = 0

    @property
    def fight_on(self) -> bool:
        return bool(self.flags & FLAG_FIGHT_ON)

    @property
    def hash_valid(self) -> bool:
        return bool(self.flags & FLAG_HASH_VALID)

    @property
    def epoch(self) -> int:
        """The writer's epoch (game pid low 24 bits), 0 for older runtimes."""
        return (self.flags >> EPOCH_SHIFT) & EPOCH_MASK


def _u32(data, offset: int) -> int:
    return _U32.unpack_from(data, offset)[0]


def parse_sample(data) -> Sample:
    """Decode an 80-byte v2 control block.

    ``data`` may be ``bytes`` or a writable/read-only mmap.  Raises
    :class:`TurboCheckError` for a short block, bad magic or a pre-v2 version.
    """
    try:
        if len(data) < SIZE:
            raise TurboCheckError(
                f"control block is {len(data)} bytes, need {SIZE}")
        if data[:4] != MAGIC:
            raise TurboCheckError(
                f"bad control block magic {bytes(data[:4])!r}, expected {MAGIC!r}")
        version = _u32(data, VERSION_OFFSET)
        if version < MIN_VERSION:
            raise TurboCheckError(
                f"control block version {version} has no hash fields; "
                f"need version >= {MIN_VERSION}")
        return Sample(
            version=version,
            flags=_u32(data, FLAGS_OFFSET),
            game_frame=_u32(data, GAME_FRAME_OFFSET),
            state_hash=_u32(data, STATE_HASH_OFFSET),
            seed=_u32(data, SEED_OFFSET),
            speed_milli=_u32(data, SPEED_MILLI_OFFSET),
            match_start=_u32(data, MATCH_START_OFFSET),
        )
    except struct.error as exc:  # pragma: no cover - len() is checked above
        raise TurboCheckError(f"short control block: {exc}") from None


# ---------------------------------------------------------------------------
# recording


@dataclass
class RecordResult:
    payload: dict
    stop_reason: str
    samples: list[Sample] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.payload["game_frame"])


def _open_block(path: str, deadline: float, poll: float, clock, sleep):
    """Wait for a readable v2 block; return (mmap, first sample)."""
    last_error = "file missing"
    while True:
        try:
            fd = os.open(path, os.O_RDONLY)
        except FileNotFoundError:
            last_error = "file does not exist"
        except OSError as exc:
            last_error = str(exc)
        else:
            try:
                size = os.fstat(fd).st_size
                if size < SIZE:
                    last_error = f"file is {size} bytes, need {SIZE}"
                else:
                    mm = mmap.mmap(fd, SIZE, access=mmap.ACCESS_READ)
                    try:
                        sample = parse_sample(mm)
                    except TurboCheckError as exc:
                        last_error = str(exc)
                        mm.close()
                    else:
                        return mm, sample
            except OSError as exc:
                last_error = str(exc)
            finally:
                os.close(fd)
        if clock() >= deadline:
            raise TurboCheckError(
                f"timed out waiting for a v2 control block at {path}: {last_error}")
        sleep(poll)


def record_run(out: str, frames: int = 0, timeout: float = 30.0,
               path: str = DEFAULT_PATH, poll: float = 0.001,
               expect_pid: int | None = None,
               expect_speed: int | None = None,
               allow_seed_zero: bool = False,
               settle: int = 0,
               end_grace: float = 0.0,
               stable_reads: int = 1,
               clock=time.monotonic, sleep=time.sleep) -> RecordResult:
    """Poll ``path`` and save up to ``frames`` live-fight samples to ``out``.

    Only ``fight_on`` frames are recorded: boot and menu frames carry a hash
    too, but their relative frames differ between runs and would misalign the
    comparison.  The seed is set at the ready-go, so a live frame also has the
    real ``seed`` (a menu frame has 0).

    ``expect_pid`` rejects a block left by another game process (the runtime
    puts the pid's low 24 bits in flags bits 8..31); if the epoch changes, the
    recorder starts over, so a tool started before a restart still works.
    ``expect_speed`` fails early when ``SSBU_SPEED`` did not reach the game
    (the block would say 1000).  ``frames=0`` records until the fight ends or
    ``timeout`` expires.
    """
    if frames < 0:
        raise ValueError("frames must be >= 0")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    poll = max(float(poll), 0.0)

    started_ns = time.time_ns()
    deadline = clock() + timeout
    mm, sample = _open_block(path, deadline, poll, clock, sleep)

    seen_frames: set[int] = set()
    game_frames: list[int] = []
    state_hashes: list[int] = []
    collected: list[Sample] = []
    first_valid: Sample | None = None
    seen_fight_on = False
    seen_epoch: int | None = None
    first_live_frame: int | None = None
    match_start = 0
    off_since: float | None = None
    stop_reason = "timeout"
    stale_bad = 0
    pending_key: tuple | None = None
    pending_n = 0

    def stable(sample: Sample) -> bool:
        """True once the same sample has been read ``stable_reads`` times.

        The v2 block has no seqlock; the runtime writes ``game_frame`` last,
        but a reader can still catch a half-updated flags/hash pair.  Two
        agreeing reads (the poll is much faster than a presented frame) drop
        those torn samples.
        """
        nonlocal pending_key, pending_n
        if stable_reads <= 1:
            return True
        key = (sample.game_frame, sample.state_hash, sample.flags,
               sample.seed, sample.speed_milli)
        if key == pending_key:
            pending_n += 1
        else:
            pending_key, pending_n = key, 1
        return pending_n >= stable_reads

    try:
        while True:
            if not stable(sample):
                if clock() >= deadline:
                    stop_reason = "timeout"
                    break
                sleep(poll)
                sample = parse_sample(mm)  # type: ignore[arg-type]
                continue
            epoch = sample.epoch
            stale = False
            if epoch:
                if expect_pid is not None and epoch != (int(expect_pid) & EPOCH_MASK):
                    stale = True
                elif seen_epoch is not None and epoch != seen_epoch:
                    # a new game wrote the block: start the recording over
                    seen_frames.clear()
                    game_frames.clear()
                    state_hashes.clear()
                    collected.clear()
                    first_valid = None
                    seen_fight_on = False
                    first_live_frame = None
                    match_start = 0
                    off_since = None
                if not stale:
                    seen_epoch = epoch
            if stale:
                stale_bad += 1
            else:
                if sample.fight_on:
                    ms = sample.match_start
                    if ms:
                        if match_start and ms != match_start:
                            # a rematch: start the recording over at its start
                            seen_frames.clear()
                            game_frames.clear()
                            state_hashes.clear()
                            collected.clear()
                            first_valid = None
                            off_since = None
                        match_start = ms
                        first_live_frame = ms
                    off_since = None
                    seen_fight_on = True
                    if first_live_frame is None:
                        first_live_frame = sample.game_frame
                    settled = (first_live_frame is not None and
                               sample.game_frame - first_live_frame >= settle)
                else:
                    settled = False
                    if seen_fight_on and first_live_frame is not None:
                        if off_since is None:
                            off_since = clock()
                        elif clock() - off_since >= end_grace:
                            stop_reason = "fight_off"
                            break
                if (settled and sample.fight_on and sample.hash_valid
                        and sample.game_frame not in seen_frames):
                    seen_frames.add(sample.game_frame)
                    game_frames.append(sample.game_frame)
                    state_hashes.append(sample.state_hash)
                    collected.append(sample)
                    if first_valid is None:
                        first_valid = sample
                        if (expect_speed is not None
                                and sample.speed_milli != int(expect_speed)):
                            raise TurboCheckError(
                                f"the game reports speed_milli="
                                f"{sample.speed_milli}, expected "
                                f"{int(expect_speed)}; SSBU_SPEED is not in the "
                                f"game's environment")
                        if sample.seed == 0 and not allow_seed_zero:
                            raise TurboCheckError(
                                "the first live frame has seed=0; launch the "
                                "game with SSBU_FIXED_SEED (the runtime seeds "
                                "the fight at the ready-go)")
                    if frames and len(game_frames) >= frames:
                        stop_reason = "frames"
                        break

            if clock() >= deadline:
                stop_reason = "timeout"
                break
            sleep(poll)
            sample = parse_sample(mm)  # type: ignore[arg-type]
    finally:
        mm.close()

    if first_valid is None:
        if stale_bad:
            raise TurboCheckError(
                f"the control block at {path} belongs to another process "
                f"(epoch {sample.epoch}, game pid "
                f"{None if expect_pid is None else int(expect_pid) & EPOCH_MASK}); "
                f"it is stale and no live frame arrived within {timeout:g}s")
        raise TurboCheckError(
            f"no live fight frames recorded from {path} within {timeout:g}s "
            f"(record starts at fight_on and needs flags bit 6)")

    payload = {
        "format": FORMAT,
        "game_frame": game_frames,
        "state_hash": state_hashes,
        "seed": first_valid.seed,
        "speed_milli": first_valid.speed_milli,
        "match_start": int(match_start or first_live_frame or 0),
        "epoch": int(seen_epoch or 0),
        "source": path,
        "started_ns": started_ns,
    }
    _write_json(out, payload)
    return RecordResult(payload=payload, stop_reason=stop_reason,
                        samples=collected)


def _write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        raise TurboCheckError(f"output directory does not exist: {directory}")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# comparing


@dataclass(frozen=True)
class Run:
    path: str
    seed: int
    speed_milli: int
    frames: tuple[int, ...]
    hashes: tuple[int, ...]
    match_start: int = 0

    @property
    def first_frame(self) -> int:
        return self.frames[0]

    @property
    def last_frame(self) -> int:
        return self.frames[-1]


@dataclass(frozen=True)
class Divergence:
    rel_frame: int
    a_hash: int
    b_hash: int
    a_frame: int
    b_frame: int


@dataclass(frozen=True)
class CompareResult:
    run_a: Run
    run_b: Run
    max_missing: int
    common: int
    only_a: int
    only_b: int
    divergence: Divergence | None
    warnings: tuple[str, ...]
    shift: int = 0

    @property
    def missing(self) -> int:
        return self.only_a + self.only_b

    @property
    def matched(self) -> int:
        return self.common - (1 if self.divergence is not None else 0)

    @property
    def allowed_missing(self) -> int:
        """Edge frames a phase shift may leave outside the overlap (both ends)."""
        return self.max_missing + 2 * abs(self.shift)

    @property
    def passed(self) -> bool:
        return self.divergence is None and self.missing <= self.allowed_missing

    def as_dict(self) -> dict:
        def run_info(run: Run) -> dict:
            return {
                "path": run.path,
                "frames": len(run.frames),
                "first_frame": run.first_frame,
                "last_frame": run.last_frame,
                "seed": run.seed,
                "speed_milli": run.speed_milli,
                "match_start": run.match_start,
            }

        div = None
        if self.divergence is not None:
            div = {
                "rel_frame": self.divergence.rel_frame,
                "a_hash": self.divergence.a_hash,
                "b_hash": self.divergence.b_hash,
                "a_frame": self.divergence.a_frame,
                "b_frame": self.divergence.b_frame,
            }
        return {
            "format": COMPARE_FORMAT,
            "pass": self.passed,
            "max_missing": self.max_missing,
            "a": run_info(self.run_a),
            "b": run_info(self.run_b),
            "common": self.common,
            "only_a": self.only_a,
            "only_b": self.only_b,
            "shift": self.shift,
            "missing": self.missing,
            "matched": self.matched,
            "divergence": div,
            "warnings": list(self.warnings),
        }


def _require_int(value, name: str) -> int:
    if type(value) is not int:  # reject bools and floats
        raise BadInput(f"{name} must be an integer, got {type(value).__name__}")
    return value


def _dedupe(frames: Sequence[int], hashes: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    seen: set[int] = set()
    kept_frames: list[int] = []
    kept_hashes: list[int] = []
    for frame, value in zip(frames, hashes):
        if frame in seen:
            continue
        seen.add(frame)
        kept_frames.append(frame)
        kept_hashes.append(value)
    return tuple(kept_frames), tuple(kept_hashes)


def run_from_payload(payload, path: str = "<memory>") -> Run:
    """Validate a ``ssbu_turbo_v1`` payload and build a :class:`Run`."""
    if not isinstance(payload, dict):
        raise BadInput("run JSON must be an object")
    if payload.get("format") != FORMAT:
        raise BadInput(
            f"{path}: format {payload.get('format')!r}, expected {FORMAT!r}")
    frames = payload.get("game_frame")
    hashes = payload.get("state_hash")
    if not isinstance(frames, list) or not isinstance(hashes, list):
        raise BadInput(f"{path}: game_frame and state_hash must be lists")
    if len(frames) != len(hashes):
        raise BadInput(
            f"{path}: game_frame has {len(frames)} entries, "
            f"state_hash has {len(hashes)}")
    if not frames:
        raise BadInput(f"{path}: no game_frame samples")
    frames = [_require_int(v, f"{path}: game_frame[{i}]")
              for i, v in enumerate(frames)]
    hashes = [_require_int(v, f"{path}: state_hash[{i}]")
              for i, v in enumerate(hashes)]
    seed = _require_int(payload.get("seed"), f"{path}: seed")
    speed = _require_int(payload.get("speed_milli"), f"{path}: speed_milli")
    match_start = payload.get("match_start", 0)
    match_start = _require_int(match_start, f"{path}: match_start")
    frames_t, hashes_t = _dedupe(frames, hashes)
    return Run(path=path, seed=seed, speed_milli=speed,
               frames=frames_t, hashes=hashes_t, match_start=match_start)


def load_run(path: str) -> Run:
    """Load and validate a recorded run JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        raise BadInput(f"{path}: file not found") from None
    except OSError as exc:
        raise BadInput(f"{path}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise BadInput(f"{path}: invalid JSON: {exc}") from None
    return run_from_payload(payload, path)


def _relative_map(run: Run) -> dict[int, tuple[int, int]]:
    base = run.match_start if run.match_start else run.frames[0]
    return {frame - base: (frame, value)
            for frame, value in zip(run.frames, run.hashes)}


def _compare_at_shift(map_a, map_b, shift: int):
    common = sorted(k for k in map_a if k + shift in map_b)
    only_a = sorted(k for k in map_a if k + shift not in map_b)
    only_b = sorted(k for k in map_b if k - shift not in map_a)
    divergence = None
    for k in common:
        a_hash = map_a[k][1]
        b_hash = map_b[k + shift][1]
        if a_hash != b_hash:
            divergence = Divergence(
                rel_frame=k, a_hash=a_hash, b_hash=b_hash,
                a_frame=map_a[k][0], b_frame=map_b[k + shift][0])
            break
    return common, only_a, only_b, divergence


def compare_runs(run_a: Run, run_b: Run, max_missing: int = 0,
                 max_shift: int = 0) -> CompareResult:
    """Align two runs by relative frame and compare every shared hash.

    ``max_shift`` tolerates a small constant phase offset between the two
    recordings (the live-fight anchor can land one frame apart); a shift is
    reported in the result and in the text output.  0 keeps the strict
    index-by-index comparison.
    """
    if max_missing < 0:
        raise ValueError("max_missing must be >= 0")
    if max_shift < 0:
        raise ValueError("max_shift must be >= 0")
    map_a = _relative_map(run_a)
    map_b = _relative_map(run_b)
    shifts = [0]
    for k in range(1, max_shift + 1):
        shifts.extend((-k, k))
    chosen = None
    fallback = None
    for shift in shifts:
        common, only_a, only_b, div = _compare_at_shift(map_a, map_b, shift)
        missing = len(only_a) + len(only_b)
        if fallback is None or (div is None and fallback[4] is not None):
            fallback = (shift, common, only_a, only_b, div, missing)
        if div is None and missing <= max_missing + 2 * abs(shift):
            chosen = (shift, common, only_a, only_b, div, missing)
            break
    if chosen is None:
        chosen = fallback
    shift, common, only_a, only_b, divergence, missing = chosen
    common = len(common)
    only_a = len(only_a)
    only_b = len(only_b)

    warnings: list[str] = []
    if run_a.seed != run_b.seed:
        warnings.append(
            f"seeds differ (A={run_a.seed}, B={run_b.seed}); runs are not "
            "comparable. Restart both with the same SSBU_FIXED_SEED.")
    if run_a.speed_milli == run_b.speed_milli:
        warnings.append(
            f"both runs report the same speed (speed_milli={run_a.speed_milli}); "
            "a turbo proof needs one run at SSBU_SPEED=1 and the other at the "
            "turbo value.")
    if shift:
        warnings.append(
            f"the runs are phase-shifted by {shift:+d} relative frame(s); "
            "the hashes match after that alignment")

    return CompareResult(
        run_a=run_a,
        run_b=run_b,
        max_missing=max_missing,
        common=common,
        only_a=only_a,
        only_b=only_b,
        divergence=divergence,
        warnings=tuple(warnings),
        shift=shift,
    )


def _format_run(tag: str, run: Run) -> str:
    return (f"  {tag}  {run.path}  frames={len(run.frames)} "
            f"first={run.first_frame} last={run.last_frame} "
            f"seed={run.seed} speed_milli={run.speed_milli}")


def format_result(result: CompareResult) -> str:
    """Short human-readable comparison summary."""
    lines = [
        "turbo_check compare",
        _format_run("A", result.run_a),
        _format_run("B", result.run_b),
        f"  common={result.common} only_a={result.only_a} "
        f"only_b={result.only_b} missing={result.missing} "
        f"(max {result.max_missing})"
        + (f"  phase shift={result.shift:+d}" if result.shift else ""),
        f"  hashes: {result.matched}/{result.common} matched",
    ]
    if result.divergence is not None:
        div = result.divergence
        lines.append(
            f"  first divergence: relative frame {div.rel_frame} "
            f"(A frame {div.a_frame} hash {div.a_hash}, "
            f"B frame {div.b_frame} hash {div.b_hash})")
    else:
        lines.append("  first divergence: none")
    for warning in result.warnings:
        lines.append(f"  note: {warning}")
    if result.passed:
        lines.append("  PASS: fighter simulation hashes match on all common frames")
    elif result.divergence is not None:
        lines.append("  FAIL: fighter simulation diverged")
    else:
        lines.append(
            f"  FAIL: {result.missing} frame(s) present in only one run "
            f"(max {result.max_missing})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# self test


def _synthetic(frames: Iterable[int], hashes: Iterable[int],
               seed: int = 1234, speed: int = 4000,
               path: str = "<synthetic>") -> Run:
    return Run(path=path, seed=seed, speed_milli=speed,
               frames=tuple(frames), hashes=tuple(hashes))


def run_self_test(stream=None) -> int:
    stream = sys.stdout if stream is None else stream
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            raise AssertionError(message)

    try:
        a = _synthetic([100, 101, 102, 103], [10, 11, 12, 13])
        b = _synthetic([500, 501, 502, 503], [10, 11, 12, 13])
        result = compare_runs(a, b)
        check(result.passed, "equal relative runs must pass")
        check(result.common == 4, "all relative frames must be common")
        check(result.divergence is None, "equal runs must not diverge")

        bad = _synthetic([100, 101, 102, 103], [10, 99, 12, 13])
        result = compare_runs(a, bad)
        check(not result.passed, "a differing hash must fail")
        check(result.divergence is not None, "a differing hash must report a divergence")
        check(result.divergence.rel_frame == 1, "divergence must be at relative frame 1")
        check(result.divergence.a_hash == 11 and result.divergence.b_hash == 99,
              "divergence must report both hashes")

        short = _synthetic([100, 101, 102, 103], [10, 11, 12, 13])
        shorter = _synthetic([500, 501, 502], [10, 11, 12])
        check(not compare_runs(short, shorter, max_missing=0).passed,
              "missing frames must fail at max_missing=0")
        check(compare_runs(short, shorter, max_missing=1).passed,
              "missing frames must pass within tolerance")

        deduped = run_from_payload({
            "format": FORMAT,
            "game_frame": [7, 7, 8, 8, 9],
            "state_hash": [70, 999, 80, 999, 90],
            "seed": 1,
            "speed_milli": 1000,
        })
        check(len(deduped.frames) == 3, "duplicate game frames must be deduped")
        check(deduped.hashes == (70, 80, 90), "first duplicate sample must win")

        block = bytearray(SIZE)
        block[:4] = MAGIC
        struct.pack_into("<I", block, VERSION_OFFSET, 2)
        struct.pack_into("<I", block, FLAGS_OFFSET, FLAG_FIGHT_ON | FLAG_HASH_VALID)
        struct.pack_into("<I", block, GAME_FRAME_OFFSET, 42)
        struct.pack_into("<I", block, STATE_HASH_OFFSET, 0xDEADBEEF)
        struct.pack_into("<I", block, SEED_OFFSET, 777)
        struct.pack_into("<I", block, SPEED_MILLI_OFFSET, 8000)
        sample = parse_sample(block)
        check(sample.game_frame == 42 and sample.state_hash == 0xDEADBEEF,
              "sample fields must decode at contract offsets")
        check(sample.fight_on and sample.hash_valid, "flag bits must decode")

        with tempfile.TemporaryDirectory() as tmp:
            block_path = os.path.join(tmp, "control.bin")
            out_path = os.path.join(tmp, "run.json")
            with open(block_path, "wb") as fh:
                fh.write(block)
            recorded = record_run(out_path, frames=1, timeout=1.0,
                                  path=block_path, poll=0.0005)
            check(recorded.count == 1, "record must save the requested sample")
            loaded = load_run(out_path)
            check(loaded.frames == (42,), "recorded frame must round-trip")
            check(loaded.hashes == (0xDEADBEEF,), "recorded hash must round-trip")
    except (AssertionError, BadInput, TurboCheckError, ValueError) as exc:
        print(f"turbo_check self-test FAILED: {exc}", file=stream)
        return 1

    print(f"turbo_check self-test OK ({checks} checks)", file=stream)
    return 0


# ---------------------------------------------------------------------------
# CLI


def _add_record_parser(sub) -> None:
    p = sub.add_parser(
        "record", help="record game_frame/state_hash samples from the block",
        description="Poll the v2 control block and save per-frame hashes.")
    p.add_argument("--out", required=True, help="output run JSON path")
    p.add_argument("--frames", type=int, default=0, metavar="N",
                   help="stop after N samples (default 0 = until end/timeout)")
    p.add_argument("--timeout", type=float, default=30.0, metavar="SEC",
                   help="stop after SEC seconds (default 30)")
    p.add_argument("--path", default=DEFAULT_PATH,
                   help=f"control block path (default {DEFAULT_PATH})")
    p.add_argument("--poll", type=float, default=0.001, metavar="SEC",
                   help="poll interval (default 0.001 = ~1 kHz)")
    p.add_argument("--pid", type=int, default=None,
                   help="game pid; reject a control block left by another "
                        "process (default: look up the running ssbu_native)")
    p.add_argument("--expect-speed", type=int, default=None, metavar="MILLI",
                   help="require this speed_milli (1000 = 1x, 8000 = 8x); "
                        "fails if SSBU_SPEED did not reach the game")
    p.add_argument("--settle", type=int, default=10, metavar="N",
                   help="skip N game frames after fight_on before recording "
                        "(default 10; avoids the ready-go transition)")
    p.add_argument("--end-grace", type=float, default=0.3, metavar="SEC",
                   help="ignore a fight_on dropout shorter than SEC "
                        "(default 0.3)")
    p.add_argument("--stable-reads", type=int, default=2, metavar="N",
                   help="accept a sample only after N agreeing reads "
                        "(default 2; drops torn v2 reads)")
    p.add_argument("--allow-seed-zero", action="store_true",
                   help="accept a live frame with seed=0 (no SSBU_FIXED_SEED)")
    p.add_argument("--json", action="store_true",
                   help="print the saved run JSON to stdout")
    p.set_defaults(func=_cmd_record)


def _add_compare_parser(sub) -> None:
    p = sub.add_parser(
        "compare", help="compare two recorded runs by relative game frame",
        description="Align two runs by relative game frame and compare hashes.")
    p.add_argument("a", metavar="A.json", help="reference run (SSBU_SPEED=1)")
    p.add_argument("b", metavar="B.json", help="turbo run (SSBU_SPEED=N)")
    p.add_argument("--max-missing", type=int, default=0, metavar="N",
                   help="tolerate N frames present in only one run (default 0)")
    p.add_argument("--max-shift", type=int, default=0, metavar="N",
                   help="tolerate a constant phase offset of up to N relative "
                        "frames between the two recordings (default 0 = strict)")
    p.add_argument("--json", action="store_true", help="print a JSON report")
    p.set_defaults(func=_cmd_compare)


def _find_game_pid() -> int | None:
    """The running ssbu_native pid, or None (the recorder is standalone)."""
    try:
        from rl.memory.memapi import find_game_pid
        return int(find_game_pid())
    except Exception:  # noqa: BLE001
        return None


def _cmd_record(args) -> int:
    try:
        result = record_run(out=args.out, frames=args.frames,
                            timeout=args.timeout, path=args.path,
                            poll=args.poll,
                            expect_pid=(args.pid if args.pid is not None
                                        else _find_game_pid()),
                            expect_speed=args.expect_speed,
                            allow_seed_zero=args.allow_seed_zero,
                            settle=args.settle,
                            end_grace=args.end_grace,
                            stable_reads=args.stable_reads)
    except (TurboCheckError, ValueError, OSError) as exc:
        print(f"turbo_check record: {exc}", file=sys.stderr)
        return 1
    payload = result.payload
    count = result.count
    print(f"recorded {count} frame(s) -> {args.out} ({result.stop_reason})")
    print(f"  first game_frame={payload['game_frame'][0]} "
          f"last game_frame={payload['game_frame'][-1]}")
    print(f"  seed={payload['seed']} speed_milli={payload['speed_milli']} "
          f"epoch={payload['epoch']}")
    if args.json:
        print(json.dumps(payload, separators=(",", ":")))
    return 0


def _cmd_compare(args) -> int:
    try:
        run_a = load_run(args.a)
        run_b = load_run(args.b)
        result = compare_runs(run_a, run_b, max_missing=args.max_missing,
                              max_shift=args.max_shift)
    except BadInput as exc:
        print(f"turbo_check compare: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"turbo_check compare: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(format_result(result))
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    epilog = (
        "A meaningful PASS requires SSBU_FIXED_SEED, the same match/fighters/"
        "stage and identical inputs on both runs, recorded at SSBU_SPEED=1 and "
        "the turbo speed.  The hash covers the fighter simulation only; "
        "particles/camera are not included.")
    parser = argparse.ArgumentParser(
        prog="turbo_check",
        description="Record and compare v2 control-block state hashes to prove "
                    "SSBU_SPEED is simulation-correct.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true",
                        help="run built-in sanity checks and exit")
    sub = parser.add_subparsers(dest="command", metavar="{record,compare}")
    _add_record_parser(sub)
    _add_compare_parser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
