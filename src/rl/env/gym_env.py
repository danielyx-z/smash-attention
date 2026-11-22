#!/usr/bin/env python3
"""Gymnasium environment for the live SSBU recomp (state bridge v7).

Data path::

    /dev/shm/ssbu_state.bin            (reader.py, 60 Hz, v7 layout)
            |
    RawSnapshotReader (odd/even seq)   <-- this module
            |
    SSBUGymEnv.step(): decode action -> gym_input (uinput) -> wait 5 frames
                       -> new snapshot -> reward/termination -> obs

Layout (CONTRACTS.md, version 7): 80-byte header, two 128-byte fighter slots
(32 fields each), block size 336 bytes::

    header: magic "SSBU", version, n_fighters, seq, ts_ns, fps, reader_pid,
            fight_on, match_state, 4 blast floats, stage_id, game_frame, flags,
            3 pad words
    slot i: 32 little-endian words, see FIGHTER_FIELDS

Slot order is player order: slot 0 = entry 0 (P1), slot 1 = entry 1 (P2).
The odd/even protocol is the standard single-writer/many-reader one: ``seq``
is odd while the body is being written, even when complete.  A reader copies
the body between two equal, even reads; otherwise it retries.

The environment is *live* and non-deterministic: the same action does not
produce the same next observation.  It is still a valid Gymnasium env; see
``test_gym_env.py`` for how ``check_env`` is invoked (the env is marked
``nondeterministic`` through its spec because the fixture/writer keeps moving).

Input / reset adapters
----------------------
``gym_input.py`` and ``gym_reset.py`` are owned by Agent D and are imported
lazily, so this module has no hard dependency on them.  Interface assumption:

* input: the module (or an injected adapter object) exposes a callable that
  accepts **one canonical action dict** (see :mod:`gym_action`):
  ``apply_action`` / ``write_action`` / ``send_action`` / ``set_action`` /
  ``apply``, or an ``InputWriter``/``ActionWriter``/``VirtualPad`` class with
  one of those methods.  It must be non-blocking (write a new uinput report).
* reset: ``gym_reset`` exposes ``reset()`` / ``request_reset()`` /
  ``reset_game()`` / ``do_reset()`` (or a ``ResetClient``/``GameReset``/
  ``ResetWriter`` class).  It is called with no arguments; the env then waits
  for ``fight_on == 1`` itself.

Anything not found degrades to a no-op input adapter (actions are ignored) and
a "wait for the game to already be in a match" reset.  Both can be injected
for tests: ``SSBUGymEnv(input_adapter=..., reset_fn=...)``.
"""

from __future__ import annotations

import importlib
import mmap
import os
import struct
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:               # allow `python tools/mem/...` imports
    sys.path.insert(0, _HERE)

try:                                    # normal use: tools/mem on sys.path
    from . import gym_action, gym_obs, gym_reward
except ImportError:                     # pragma: no cover - script/test import
    from rl.env import gym_action
    from rl.env import gym_obs
    from rl.env import gym_reward

import gymnasium as gym
from gymnasium import spaces

# --------------------------------------------------------------------------
# v7 shared-memory layout (CONTRACTS.md)
# --------------------------------------------------------------------------
MAGIC = b"SSBU"
VERSION = 7
HEADER_SIZE = 80
N_FIGHTER_FIELDS = 32
FIGHTER_SIZE = N_FIGHTER_FIELDS * 4        # 128
MAX_FIGHTERS = 2
BLOCK_SIZE = HEADER_SIZE + FIGHTER_SIZE * MAX_FIGHTERS   # 336

SHM_PATH = "/dev/shm/ssbu_state.bin"
SEQ_OFFSET = 12

#: (name, struct fmt) in exact file order.
HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("magic", "4s"),
    ("version", "I"),
    ("n_fighters", "I"),
    ("seq", "I"),
    ("ts_ns", "Q"),
    ("fps", "f"),
    ("reader_pid", "I"),
    ("fight_on", "I"),
    ("match_state", "I"),
    ("blast_left", "f"),
    ("blast_right", "f"),
    ("blast_top", "f"),
    ("blast_bottom", "f"),
    ("stage_id", "i"),
    ("game_frame", "I"),
    ("flags", "I"),
    ("pad0", "I"),
    ("pad1", "I"),
    ("pad2", "I"),
)

#: Exactly the 32 fighter-slot fields, in the contract's index order.
FIGHTER_FIELDS: tuple[tuple[str, str], ...] = (
    ("entry_id", "i"),
    ("kind", "i"),
    ("stock", "i"),
    ("damage", "f"),
    ("x", "f"),
    ("y", "f"),
    ("facing", "f"),
    ("action_state", "i"),
    ("action_frame", "f"),
    ("on_ground", "i"),
    ("is_actionable", "i"),
    ("jumps_left", "i"),
    ("vx", "f"),
    ("vy", "f"),
    ("dist_to_blast", "f"),
    ("dx", "f"),
    ("dy", "f"),
    ("facing_rel", "i"),
    ("is_cpu", "i"),
    ("is_on_rebirth", "i"),
    ("shield", "i"),
    ("shield_size", "f"),
    ("is_enable_cancel", "i"),
    ("hitstun", "i"),
    ("buttons", "I"),
    ("stick_x", "f"),
    ("stick_y", "f"),
    ("trigger", "f"),
    ("cstick_x", "f"),
    ("cstick_y", "f"),
    ("raw_flags", "I"),
    ("pad", "I"),
)

HEADER_NAMES = tuple(name for name, _fmt in HEADER_FIELDS)
FIGHTER_NAMES = tuple(name for name, _fmt in FIGHTER_FIELDS)

HEADER_FMT = "<" + "".join(fmt for _name, fmt in HEADER_FIELDS)
FIGHTER_FMT = "<" + "".join(fmt for _name, fmt in FIGHTER_FIELDS)

_HEADER = struct.Struct(HEADER_FMT)
_FIGHTER = struct.Struct(FIGHTER_FMT)
_SEQ = struct.Struct("<I")

assert _HEADER.size == HEADER_SIZE, _HEADER.size
assert _FIGHTER.size == FIGHTER_SIZE, _FIGHTER.size
assert len(FIGHTER_FIELDS) == N_FIGHTER_FIELDS


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
class SnapshotError(RuntimeError):
    """The shared-memory block is unreadable or has the wrong layout."""


class SchemaMismatch(SnapshotError):
    """A valid block, but not the v7 layout this code understands."""


class SnapshotTimeout(SnapshotError, TimeoutError):
    """No valid frame arrived within the deadline."""


class _NotReady(Exception):
    """Internal: the file exists but has not been initialised/published yet."""


# --------------------------------------------------------------------------
# snapshot
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RawSnapshot:
    """One complete, consistent v7 frame."""

    version: int
    seq: int
    ts_ns: int
    fps: float
    reader_pid: int
    fight_on: int
    match_state: int
    n_fighters: int
    stage_id: int
    game_frame: int
    flags: int
    blast: tuple[float, float, float, float]
    fighters: tuple[dict, ...]      # always MAX_FIGHTERS entries (empty slots zeroed)
    raw: np.ndarray                 # (MAX_FIGHTERS, 32) float64 copy of the slots

    def fighter(self, index: int) -> dict:
        return self.fighters[index]

    @property
    def header(self) -> dict:
        return self.header_dict()

    def header_dict(self) -> dict:
        return {
            "version": self.version,
            "n_fighters": self.n_fighters,
            "seq": self.seq,
            "ts_ns": self.ts_ns,
            "fps": self.fps,
            "reader_pid": self.reader_pid,
            "fight_on": self.fight_on,
            "match_state": self.match_state,
            "blast_left": self.blast[0],
            "blast_right": self.blast[1],
            "blast_top": self.blast[2],
            "blast_bottom": self.blast[3],
            "stage_id": self.stage_id,
            "game_frame": self.game_frame,
            "flags": self.flags,
        }

    def to_info(self) -> dict:
        """Raw extras for ``info``; every call returns fresh containers."""
        return {
            "header": self.header_dict(),
            "raw_fighters": [dict(f) for f in self.fighters],
            "raw": self.raw.copy(),
        }


def fighter_valid(fighter: dict) -> bool:
    """Mirror of ``gym_obs.fighter_valid`` (kept here for env/reward use)."""
    return gym_obs.fighter_valid(fighter)


# --------------------------------------------------------------------------
# reader
# --------------------------------------------------------------------------
class RawSnapshotReader:
    """Read ``/dev/shm/ssbu_state.bin`` with the odd/even sequence protocol.

    The file is mapped read-only and reopened automatically if the writer
    re-creates it (different inode) or has not finished initialising it yet.
    """

    def __init__(self, path: str = SHM_PATH, timeout: float = 1.0,
                 poll: float = 0.0005):
        self.path = str(path)
        self.timeout = float(timeout)
        self.poll = max(1e-5, float(poll))
        self._mm: mmap.mmap | None = None
        self._dev: int | None = None
        self._ino: int | None = None
        self._lock = threading.RLock()

    # -- lifecycle --------------------------------------------------------
    def _close_mm(self) -> None:
        mm, self._mm = self._mm, None
        self._dev = self._ino = None
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):   # pragma: no cover
                pass

    def close(self) -> None:
        with self._lock:
            self._close_mm()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def is_open(self) -> bool:
        return self._mm is not None

    # -- low level --------------------------------------------------------
    def _ensure_open(self) -> None:
        stat = os.stat(self.path)               # FileNotFoundError -> retry
        if (self._mm is not None and stat.st_dev == self._dev
                and stat.st_ino == self._ino):
            return
        self._close_mm()
        fd = os.open(self.path, os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            if stat.st_size < BLOCK_SIZE:
                # A v6 block was only 224 bytes.  Give a precise error instead
                # of a generic timeout when the magic/version identify it.
                head = os.pread(fd, 8, 0) if stat.st_size >= 8 else b""
                if len(head) == 8 and head[:4] == MAGIC:
                    version = struct.unpack_from("<I", head, 4)[0]
                    if version not in (0, VERSION):
                        raise SchemaMismatch(
                            f"shared-memory layout v{version} ({stat.st_size} "
                            f"bytes) at {self.path}, this env needs "
                            f"v{VERSION} ({BLOCK_SIZE} bytes); update "
                            "tools/mem/schema.py+reader.py")
                raise _NotReady(
                    f"{self.path} is {stat.st_size} bytes; expected {BLOCK_SIZE}")
            mm = mmap.mmap(fd, BLOCK_SIZE, prot=mmap.PROT_READ)
        finally:
            os.close(fd)
        self._mm = mm
        self._dev = stat.st_dev
        self._ino = stat.st_ino

    def _read_seq(self) -> int:
        return _SEQ.unpack_from(self._mm, SEQ_OFFSET)[0]

    @staticmethod
    def _parse(body: bytes, seq: int) -> RawSnapshot | None:
        vals = _HEADER.unpack_from(body, 0)
        header = dict(zip(HEADER_NAMES, vals))

        magic = header["magic"]
        if magic == b"\0\0\0\0":
            return None                              # writer not initialised
        if magic != MAGIC:
            raise SnapshotError(
                f"bad shared-memory magic {magic!r}; expected {MAGIC!r}")
        version = int(header["version"])
        if version == 0:
            return None                              # header not complete yet
        if version != VERSION:
            raise SchemaMismatch(
                f"shared-memory layout v{version}, this env needs v{VERSION}; "
                "update tools/mem/schema.py+reader.py or check the reader")
        if seq == 0:
            return None                              # nothing published yet

        fighters: list[dict] = []
        raw = np.empty((MAX_FIGHTERS, N_FIGHTER_FIELDS), dtype=np.float64)
        for i in range(MAX_FIGHTERS):
            slot = _FIGHTER.unpack_from(body, HEADER_SIZE + FIGHTER_SIZE * i)
            fighters.append(dict(zip(FIGHTER_NAMES, slot)))
            raw[i, :] = slot

        return RawSnapshot(
            version=version,
            seq=int(seq),
            ts_ns=int(header["ts_ns"]),
            fps=float(header["fps"]),
            reader_pid=int(header["reader_pid"]),
            fight_on=int(header["fight_on"]),
            match_state=int(header["match_state"]),
            n_fighters=int(header["n_fighters"]),
            stage_id=int(header["stage_id"]),
            game_frame=int(header["game_frame"]),
            flags=int(header["flags"]),
            blast=(float(header["blast_left"]), float(header["blast_right"]),
                   float(header["blast_top"]), float(header["blast_bottom"])),
            fighters=tuple(fighters),
            raw=raw,
        )

    # -- public -----------------------------------------------------------
    def read_latest(self, timeout: float | None = None) -> RawSnapshot:
        """Return the newest complete frame, waiting up to ``timeout`` seconds.

        Raises :class:`SnapshotTimeout` (a ``TimeoutError``) when no valid
        frame arrives, and :class:`SchemaMismatch` immediately on a wrong
        layout version.
        """
        limit = self.timeout if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + limit
        note = "no frame published yet"
        while True:
            try:
                with self._lock:
                    self._ensure_open()
                    seq1 = self._read_seq()
                    if not (seq1 & 1):               # even: writer is idle
                        body = self._mm[0:BLOCK_SIZE]
                        seq2 = self._read_seq()
                        if seq1 == seq2 and not (seq2 & 1):
                            snap = self._parse(body, seq2)
                            if snap is not None:
                                return snap
            except _NotReady as exc:
                note = str(exc)
            except SnapshotError:
                raise                                # schema errors are fatal
            except (FileNotFoundError, OSError, ValueError, BufferError) as exc:
                note = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    self._close_mm()
            now = time.monotonic()
            if now >= deadline:
                raise SnapshotTimeout(
                    f"no valid v{VERSION} frame from {self.path} within "
                    f"{limit:.3f}s ({note}); is reader.py running?")
            time.sleep(min(self.poll, max(0.0, deadline - now)))

    def wait_frames(self, n: int = 1, timeout: float | None = None,
                    start: RawSnapshot | None = None) -> RawSnapshot:
        """Wait until the writer has published ``n`` frames past ``start``.

        ``start`` defaults to the latest frame at call time.  The publisher
        bumps ``seq`` twice per frame (odd while writing, even when done), so
        the target is ``start.seq + 2*n``.  A writer restart (seq goes
        backwards) is treated as a new epoch and returned immediately.
        """
        n = int(n)
        if n < 0:
            raise ValueError("n must be >= 0")
        limit = self.timeout if timeout is None else max(0.0, float(timeout))
        deadline = time.monotonic() + limit
        start = start or self.read_latest(timeout=limit)
        if n == 0:
            return start
        start_seq = start.seq
        required = 2 * n
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            snap = self.read_latest(timeout=remaining)
            if snap.seq == start_seq:
                pass
            elif snap.seq < start_seq:
                return snap                       # writer restarted
            elif snap.seq - start_seq >= required:
                return snap
            now = time.monotonic()
            if now >= deadline:
                raise SnapshotTimeout(
                    f"writer advanced only {snap.seq - start_seq} of "
                    f"{required} sequence ticks before the {limit:.3f}s timeout")
            time.sleep(min(self.poll, max(0.0, deadline - now)))


#: Backwards-friendly alias.
SnapshotReader = RawSnapshotReader


# --------------------------------------------------------------------------
# input adapters
# --------------------------------------------------------------------------
class NoopInputAdapter:
    """Adapter used when ``gym_input`` is unavailable (actions are ignored)."""

    name = "noop"
    available = False

    def apply(self, action: dict) -> None:       # noqa: ARG002
        return None

    def close(self) -> None:
        return None


class _CallableAdapter:
    available = True

    def __init__(self, fn: Callable, name: str = "callable", closer=None):
        self._fn = fn
        self.name = name
        self._closer = closer

    def apply(self, action: dict) -> None:
        self._fn(action)

    def close(self) -> None:
        if callable(self._closer):
            self._closer()


class _MethodAdapter(_CallableAdapter):
    def __init__(self, obj: Any, method_name: str):
        super().__init__(getattr(obj, method_name),
                         name=f"{type(obj).__name__}.{method_name}",
                         closer=getattr(obj, "close", None))


_INPUT_METHODS = ("apply_action", "write_action", "send_action", "set_action",
                  "apply", "write")
_INPUT_CLASSES = ("InputWriter", "ActionWriter", "VirtualPad", "GymInput",
                  "VirtualGamepad")

_RESET_FUNCS = ("reset", "request_reset", "reset_game", "do_reset", "reset_match")
_RESET_CLASSES = ("ResetClient", "GameReset", "ResetWriter")


def _adapt_input(target: Any) -> Any:
    """Wrap a module/object/callable into an adapter, or return None."""
    if target is None:
        return None
    if isinstance(target, NoopInputAdapter):
        return target                          # type: ignore[return-value]
    for method in _INPUT_METHODS:
        fn = getattr(target, method, None)
        if callable(fn) and not isinstance(target, type):
            # Keep an injected adapter object as-is when it already conforms
            # to our protocol (apply + name), so its close()/name survive.
            if method == "apply" and hasattr(target, "name"):
                return target
            return _MethodAdapter(target, method)
    if callable(target) and not isinstance(target, type):
        # A plain function is called as fn(canonical_action_dict).
        return _CallableAdapter(target, name=getattr(target, "__name__", "callable"))
    for cls_name in _INPUT_CLASSES:
        cls = getattr(target, cls_name, None)
        if isinstance(cls, type):
            try:
                instance = cls()
            except Exception:                  # pragma: no cover - optional dep
                continue
            adapted = _adapt_input(instance)
            if adapted is not None:
                return adapted
    return None


def default_input_adapter() -> tuple[Any, str | None]:
    """Lazily import ``gym_input``; return ``(adapter, error_or_None)``."""
    try:
        gym_input = importlib.import_module("rl.input.gym_input")
    except Exception as exc:                  # missing module or missing uinput
        return NoopInputAdapter(), f"gym_input unavailable ({type(exc).__name__}: {exc})"
    adapter = _adapt_input(gym_input)
    if adapter is None:
        return NoopInputAdapter(), "gym_input has no recognised action entry point"
    return adapter, None


def _adapt_reset(target: Any) -> Callable | None:
    if target is None:
        return None
    if callable(target) and not isinstance(target, type):
        return target
    for method in _RESET_FUNCS:
        fn = getattr(target, method, None)
        if callable(fn) and not isinstance(target, type):
            return fn
    for cls_name in _RESET_CLASSES:
        cls = getattr(target, cls_name, None)
        if isinstance(cls, type):
            try:
                instance = cls()
            except Exception:                  # pragma: no cover - optional dep
                continue
            fn = _adapt_reset(instance)
            if fn is not None:
                return fn
    return None


def default_reset_fn() -> Callable | None:
    """Lazily import ``gym_reset``; return the reset callable or None."""
    try:
        gym_reset = importlib.import_module("rl.input.gym_reset")
    except Exception:
        return None
    return _adapt_reset(gym_reset)


# --------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------
class SSBUGymEnv(gym.Env):
    """Live SSBU Gymnasium environment.

    Parameters
    ----------
    shm_path:
        Shared-memory file written by ``reader.py`` (v7 layout).
    frame_skip:
        Reader frames per decision; 5 -> 12 Hz at 60 Hz (contract).
    max_steps:
        Truncate after this many decisions; ``None`` disables.
    reward_mode:
        ``"none"`` (default, 0.0, for BC) or ``"combat"``.
    reward_config:
        :class:`gym_reward.RewardConfig` for the combat reward.
    normalize_obs:
        ``False`` (default) = raw physical values; ``True`` = profile v1
        divisors (see :mod:`gym_obs`).
    damage_mode:
        ``"raw"`` (default) = damage in percent; ``"bucket"`` = 10 % bucket
        index; ``"none"`` = zeroed.  Must match the checkpoint's training mode.
    macro_mode:
        Use ``Discrete(len(MACROS))`` instead of the buttons/stick Dict space.
    ego_slot:
        0 = P1 (slot 0) is "self"; 1 = P2.  Used by the combat reward.
    frame_timeout:
        Seconds allowed for one action's 5 frames.
    reset_timeout:
        Seconds allowed for a reset (fight_on + stable spawn frames).
    spawn_stable_frames:
        Consecutive fresh frames with ``fight_on == 1`` required by reset.
    require_positive_stock:
        Reset readiness requires each fighter's stock to be positive **when
        that slot's stock signal is usable**.  Slots whose stock stays 0 for
        ``stock_zero_frames`` live frames are treated as unavailable and are
        ignored (fight_on + readable slots are enough).  Default True.
    stock_termination:
        Terminate on a usable fighter's stock reaching 0.  Slots with an
        unusable stock signal are never used for termination; ``fight_on``
        1->0 still terminates.  Default True.
    stock_zero_frames:
        Consecutive live frames (``fight_on == 1``, readable slots) with
        stock == 0 before a slot's stock signal is declared unusable.
        Default 10 (about 0.17 s at 60 Hz).
    input_adapter / reset_fn:
        Optional injection points (tests, or a different uinput client).
    """

    metadata = {"render_modes": [], "render_fps": 12}

    def __init__(
        self,
        shm_path: str = SHM_PATH,
        frame_skip: int = 5,
        max_steps: int | None = 1800,
        reward_mode: str = "none",
        reward_config: gym_reward.RewardConfig | None = None,
        normalize_obs: bool = False,
        damage_mode: str = "raw",
        macro_mode: bool = False,
        ego_slot: int = 0,
        frame_timeout: float = 2.0,
        reset_timeout: float = 15.0,
        reader_timeout: float = 1.0,
        spawn_stable_frames: int = 2,
        require_positive_stock: bool = True,
        stock_termination: bool = True,
        stock_zero_frames: int = 10,
        input_adapter: Any = None,
        reset_fn: Callable | None = None,
    ):
        super().__init__()
        if int(frame_skip) < 1:
            raise ValueError("frame_skip must be >= 1")
        if reward_mode not in gym_reward.REWARD_MODES:
            raise ValueError(
                f"reward_mode must be one of {gym_reward.REWARD_MODES}, "
                f"got {reward_mode!r}")
        if int(ego_slot) not in (0, 1):
            raise ValueError("ego_slot must be 0 or 1")
        if int(spawn_stable_frames) < 1:
            raise ValueError("spawn_stable_frames must be >= 1")
        if int(stock_zero_frames) < 1:
            raise ValueError("stock_zero_frames must be >= 1")
        if damage_mode not in gym_obs.DAMAGE_MODES:
            raise ValueError(f"damage_mode must be one of "
                             f"{gym_obs.DAMAGE_MODES}, got {damage_mode!r}")

        self.shm_path = str(shm_path)
        self.frame_skip = int(frame_skip)
        self.max_steps = None if max_steps is None else int(max_steps)
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 or None")

        self.reward_mode = str(reward_mode)
        self.reward_config = reward_config or gym_reward.RewardConfig()
        self.ego_slot = int(ego_slot)
        self.normalize_obs = bool(normalize_obs)
        self.damage_mode = str(damage_mode)
        self.macro_mode = bool(macro_mode)
        self.frame_timeout = float(frame_timeout)
        self.reset_timeout = float(reset_timeout)
        self.reader_timeout = float(reader_timeout)
        self.spawn_stable_frames = int(spawn_stable_frames)
        self.require_positive_stock = bool(require_positive_stock)
        self.stock_termination = bool(stock_termination)
        self.stock_zero_frames = int(stock_zero_frames)

        # Per-slot stock-signal tracking.  True = a positive stock was seen
        # (stock 0 later means a real KO), False = 0 for stock_zero_frames
        # without ever being positive (signal unavailable), None = unknown.
        self._slot_stock_usable: list[bool | None] = [None] * MAX_FIGHTERS
        self._slot_zero_streak: list[int] = [0] * MAX_FIGHTERS
        self._slot_stock_warned: list[bool] = [False] * MAX_FIGHTERS
        self._last_stock_stamp: tuple[int, int] | None = None

        self.obs_builder = gym_obs.PolicyObservation(
            normalize=self.normalize_obs, damage_mode=self.damage_mode)
        self.observation_space: spaces.Box = gym_obs.observation_space(self.normalize_obs)
        self.action_space: spaces.Space = gym_action.make_action_space(self.macro_mode)
        self.render_mode = None

        self.reader = RawSnapshotReader(self.shm_path, timeout=self.reader_timeout)

        # lockstep (see rl.input.gate): the game is parked between decisions
        # and every step advances exactly ``frame_skip`` game frames.
        self._gate = None
        self._lockstep = False
        self._lock_reset = None
        self._lock_seed = None
        self._lock_park = 1

        self._input_adapter = None
        self._input_error: str | None = None
        self._input_resolved = False
        if input_adapter is not None:
            adapted = _adapt_input(input_adapter)
            if adapted is None:
                raise TypeError(
                    "input_adapter must be callable or expose apply_action/"
                    "write_action/send_action/set_action/apply")
            self._input_adapter = adapted
            self._input_resolved = True

        self._reset_fn = None
        self._reset_resolved = False
        if reset_fn is not None:
            adapted_reset = _adapt_reset(reset_fn)
            if adapted_reset is None:
                raise TypeError(
                    "reset_fn must be callable or expose reset()/request_reset()")
            self._reset_fn = adapted_reset
            self._reset_resolved = True
        self._last_snapshot: RawSnapshot | None = None
        self._step_count = 0
        self._closed = False
        self._episodes = 0

    # -- adapters ---------------------------------------------------------
    def _get_input_adapter(self) -> Any:
        if not self._input_resolved:
            self._input_resolved = True
            if self._input_adapter is None:
                adapter, error = default_input_adapter()
                self._input_adapter = adapter
                self._input_error = error
                if error:
                    warnings.warn(f"{error}; actions will be no-ops",
                                  RuntimeWarning, stacklevel=2)
            else:
                adapted = _adapt_input(self._input_adapter)
                if adapted is None:
                    raise TypeError(
                        "input_adapter must be callable or expose apply_action/"
                        "write_action/send_action/set_action/apply")
                self._input_adapter = adapted
        return self._input_adapter

    def _get_reset_fn(self) -> Callable | None:
        if not self._reset_resolved:
            self._reset_resolved = True
            if self._reset_fn is None:
                self._reset_fn = default_reset_fn()
            elif not callable(self._reset_fn):
                adapted = _adapt_reset(self._reset_fn)
                if adapted is None:
                    raise TypeError(
                        "reset_fn must be callable or expose reset()/"
                        "request_reset()")
                self._reset_fn = adapted
        return self._reset_fn

    # -- stock signal usability ------------------------------------------
    @property
    def stock_usable_slots(self) -> list[bool | None]:
        """Per-slot stock signal: True usable, False unavailable, None unknown."""
        return list(self._slot_stock_usable)

    @property
    def stock_usable(self) -> bool | None:
        """Global stock signal: True if any slot is usable, False if none are.

        ``None`` while undecided (fewer than ``stock_zero_frames`` live frames
        and no positive stock observed yet).
        """
        slots = self._slot_stock_usable
        if any(flag is True for flag in slots):
            return True
        if all(flag is False for flag in slots):
            return False
        return None

    def _observe_stock(self, snap: RawSnapshot) -> None:
        """Update per-slot stock usability from one fresh live frame.

        Only frames with ``fight_on == 1`` and readable slots count.  Once a
        slot showed a positive stock it is usable forever (stock 0 later is a
        real KO); a slot that is 0 for ``stock_zero_frames`` consecutive live
        frames without ever being positive is declared unavailable and only
        ``fight_on`` is trusted for it.  A later positive stock revives it.

        The same frame is only observed once (dedup by ``seq`` + timestamp).
        """
        if not self.require_positive_stock and not self.stock_termination:
            return
        stamp = (int(snap.seq), int(snap.ts_ns))
        if stamp == self._last_stock_stamp:
            return
        self._last_stock_stamp = stamp
        if int(snap.fight_on) != 1:
            return
        for i in range(MAX_FIGHTERS):
            fighter = snap.fighters[i]
            if not gym_obs.fighter_valid(fighter):
                continue
            stock = int(fighter.get("stock", 0))
            if stock > 0:
                self._slot_stock_usable[i] = True
                self._slot_zero_streak[i] = 0
                continue
            if self._slot_stock_usable[i] is not None:
                continue                    # usable (KO) or already unavailable
            self._slot_zero_streak[i] += 1
            if self._slot_zero_streak[i] >= self.stock_zero_frames:
                self._slot_stock_usable[i] = False
                if not self._slot_stock_warned[i]:
                    self._slot_stock_warned[i] = True
                    warnings.warn(
                        f"stock signal for fighter slot {i} is unusable: "
                        f"stock==0 for {self.stock_zero_frames} live frames "
                        "with fight_on=1.  That slot's stock will be ignored "
                        "for reset readiness and termination (fight_on 1->0 "
                        "still ends the episode).  Set require_positive_stock/"
                        "stock_termination=False to opt out explicitly.",
                        RuntimeWarning, stacklevel=3)

    # -- info -------------------------------------------------------------
    def _base_info(self, snap: RawSnapshot, phase: str) -> dict:
        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        info = {
            "phase": phase,
            "step": self._step_count,
            "episode": self._episodes,
            "frame_skip": self.frame_skip,
            "seq": snap.seq,
            "age_ns": max(0, now_ns - snap.ts_ns),
            "game_frame": snap.game_frame,
            "fps": snap.fps,
            "fight_on": snap.fight_on,
            "match_state": snap.match_state,
            "stage_id": snap.stage_id,
            "n_fighters": snap.n_fighters,
            "flags": snap.flags,
            "blast": tuple(snap.blast),
            "stocks": [int(f.get("stock", 0)) for f in snap.fighters],
            "damages": [float(f.get("damage", 0.0)) for f in snap.fighters],
            "stock_usable": self.stock_usable,
            "stock_usable_slots": self.stock_usable_slots,
        }
        info.update(snap.to_info())
        return info

    # -- reset ------------------------------------------------------------
    def _ready_for_reset(self, snap: RawSnapshot) -> bool:
        if int(snap.fight_on) != 1 or int(snap.n_fighters) < MAX_FIGHTERS:
            return False
        for i, f in enumerate(snap.fighters[:MAX_FIGHTERS]):
            if not gym_obs.fighter_valid(f):
                return False
            if not self.require_positive_stock:
                continue
            if self._slot_stock_usable[i] is False:
                continue                    # stock unavailable: trust fight_on
            if int(f.get("stock", 0)) <= 0:
                return False
        return True

    def _neutralise_input(self) -> None:
        """Release every button and the sticks before a reset.

        The pad keeps the last action it was given, so an episode that ended
        mid-run used to start the next one with the stick still held.
        """
        try:
            adapter = self._get_input_adapter()
            adapter.apply(gym_action.decode_action(
                {"buttons": np.zeros(gym_action.N_BUTTONS, np.int8),
                 "stick": np.zeros(2), "cstick": np.zeros(2)}))
        except Exception as exc:                  # noqa: BLE001
            self._input_error = f"neutralise before reset failed: {exc}"

    # -- lockstep ---------------------------------------------------------
    def enable_lockstep(self, timeout: float = 10.0, seed: int | None = None,
                        park_after: int = 1,
                        gate_path: str | None = None) -> None:
        """Switch to frame-exact stepping.

        Needs the game started with ``SSBU_GATE=1`` and ``SSBU_SPEED != 1`` and
        the reader running.  The gate is released here; the next ``reset()``
        parks the game on a frame boundary and every ``step()`` then advances
        exactly ``frame_skip`` frames.
        """
        from rl.input.gate import FrameGate
        from rl.input.gym_reset import ResetClient
        self._gate = (FrameGate(gate_path) if gate_path else FrameGate()).open(timeout)
        self._gate.release()
        # protocol-only reset client: request + ack, no wait on the free-running
        # state stream (the game is parked after the reset, so it has none)
        self._lock_reset = ResetClient(timeout=self.reset_timeout,
                                       state_reader=None)
        self._lock_seed = seed
        self._lock_park = int(park_after)
        self._lockstep = True

    @property
    def lockstep(self) -> bool:
        return self._lockstep

    #: Seconds a lockstep step may wait for the game before it is treated as a
    #: hang.  A stalled presentation (the window being dragged, a compositor
    #: hiccup) frees itself; a dead game does not.
    stall_grace = 60.0

    def _wait_tagged(self, frame: int, timeout: float | None = None) -> RawSnapshot:
        """Wait for the snapshot the reader tagged with game frame ``frame``."""
        limit = self.frame_timeout if timeout is None else float(timeout)
        start = time.monotonic()
        deadline = start + limit
        grace_deadline = start + max(limit, self.stall_grace)
        want = int(frame) & 0xFFFFFFFF
        warned = False
        while True:
            snap = self.reader.read_latest(
                timeout=max(0.001, deadline - time.monotonic()))
            if int(snap.game_frame) == want:
                return snap
            now = time.monotonic()
            if now >= deadline:
                if self._gate is not None and now < grace_deadline:
                    if not warned:
                        warnings.warn(
                            f"lockstep: no frame {want} after {limit:.1f}s "
                            "(game stalled?); waiting up to "
                            f"{self.stall_grace:.1f}s", RuntimeWarning)
                        warned = True
                    deadline = grace_deadline
                    time.sleep(0.01)
                    continue
                gs = self._gate.read() if self._gate is not None else None
                raise SnapshotTimeout(
                    f"no snapshot tagged game frame {want} within "
                    f"{now - start:.1f}s (latest tag {snap.game_frame}, "
                    f"gate {gs}); is the reader running and the game "
                    "started with SSBU_GATE=1?")
            time.sleep(0.0002)

    def _press_plus_once(self) -> None:
        """Tap PLUS (pause toggle) for two frames, then release."""
        adapter = self._get_input_adapter()
        buttons = np.zeros(gym_action.N_BUTTONS, np.int8)
        buttons[gym_action.BUTTONS.index("PLUS")] = 1
        adapter.apply(gym_action.decode_action(
            {"buttons": buttons, "stick": np.zeros(2), "cstick": np.zeros(2)}))
        self._wait_tagged(self._gate.advance(2), timeout=self.reset_timeout)
        self._neutralise_input()

    def _clean_start(self, snap: RawSnapshot) -> bool:
        """Is this a state an episode can sensibly start from?

        The rematch keeps a fighter's action status, so a reset right after a
        KO can hand back a fighter that is still falling or standing on the
        revival platform.  Wait (deterministically, frame by frame) until the
        ego is on the ground, actionable and not on the platform, and the
        opponent is not on it either.
        """
        fighters = snap.fighters
        if len(fighters) <= max(self.ego_slot, 1):
            return False
        ego = fighters[self.ego_slot]
        opp = fighters[1 - self.ego_slot]
        return (int(ego.get("on_ground", 0)) == 1
                and int(ego.get("is_actionable", 0)) == 1
                and int(ego.get("is_on_rebirth", 0)) == 0
                and int(opp.get("is_on_rebirth", 0)) == 0)

    def _reset_lockstep(self, t0: float):
        """Reset with the game parked on deterministic frames.

        The runtime performs the rematch, re-seeds the RNG (if a seed was
        given) and parks at reset frame + ``park_after`` itself.  The env then
        advances one frame at a time until the fight is live, so the first
        observation of every episode is the same frame of the same state,
        whatever the wall clock did.
        """
        gate = self._gate
        gate.release()                          # the rematch needs a running game
        gate.arm_reset(self._lock_park, self._lock_seed)
        self._lock_reset.reset(timeout=self.reset_timeout)
        frame = gate.wait_parked(timeout=self.reset_timeout)
        snap = self._wait_tagged(frame, timeout=self.reset_timeout)
        advanced = 0
        max_frames = int(self.reset_timeout * 240)
        still, last_sig, plus_presses, seen_motion = 0, None, 0, False
        while True:
            # Liveness: a live match changes the fighters' state every frame
            # (even idle animations advance).  A paused match (a pad was
            # disconnected) shows the same frame forever and looks like a
            # perfectly clean start, so require motion and press PLUS (a
            # toggle) if there is none for 8 frames.
            sig = tuple((int(f.get("action_state", -1)),
                         float(f.get("action_frame", 0.0)),
                         float(f.get("x", 0.0)), float(f.get("y", 0.0)))
                        for f in snap.fighters[:2])
            if last_sig is not None:
                if sig == last_sig:
                    still += 1
                else:
                    still, seen_motion = 0, True
            last_sig = sig
            if (seen_motion and self._ready_for_reset(snap)
                    and self._clean_start(snap)):
                break
            if advanced >= max_frames:
                gate.release()
                raise SnapshotTimeout(
                    f"lockstep reset: no live fight after {advanced} frames "
                    f"(fight_on={snap.fight_on}, fighters={snap.n_fighters}, "
                    f"moving={seen_motion}, PLUS pressed {plus_presses}x)")
            if still >= 8 and plus_presses < 6:
                self._press_plus_once()
                plus_presses += 1
                still, last_sig = 0, None
                self._lock_resumed = getattr(self, "_lock_resumed", 0) + 1
            snap = self._wait_tagged(gate.advance(1), timeout=self.reset_timeout)
            advanced += 1
        self._observe_stock(snap)
        self._last_snapshot = snap
        self._step_count = 0
        self._episodes += 1
        obs = self.obs_builder.build(snap)
        info = self._base_info(snap, "reset")
        info.update({
            "reset_fn_used": True,
            "reset_timeout_s": self.reset_timeout,
            "timing": {"reset_s": time.perf_counter() - t0},
            "lockstep": True,
            "reset_frames_advanced": advanced,
            "input_adapter": (
                getattr(self._input_adapter, "name", "unresolved")
                if self._input_resolved else "unresolved"),
        })
        return obs, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Request/await a fresh match and return ``(obs, info)``.

        With ``gym_reset`` available it is called first; otherwise the env
        waits for an already-running match.  A frame is "ready" when
        ``fight_on == 1``, both slots are readable, and each slot's stock is
        positive unless that slot's stock signal was declared unusable
        (``stock_usable_slots[i] is False``; then only ``fight_on`` and the
        readable slots are required).  Raises :class:`SnapshotTimeout` (a
        ``TimeoutError``) if ``spawn_stable_frames`` such fresh frames do not
        arrive within ``reset_timeout``.
        """
        super().reset(seed=seed)
        del options                            # no per-reset options yet
        if self._closed:
            raise RuntimeError("environment is closed")

        t0 = time.perf_counter()
        self._neutralise_input()
        if self._lockstep:
            return self._reset_lockstep(t0)
        reset_fn = self._get_reset_fn()
        reset_used = False
        if reset_fn is not None:
            reset_fn()
            reset_used = True

        deadline = time.monotonic() + self.reset_timeout
        stable = 0
        last: RawSnapshot | None = None
        # Only frames published after the reset request count towards the
        # fresh/stable spawn check; a stale fight_on=1 frame from before the
        # reset must not satisfy reset().
        try:
            baseline = self.reader.read_latest(
                timeout=max(0.0, deadline - time.monotonic()))
        except SnapshotTimeout as exc:
            raise SnapshotTimeout(
                f"reset timed out after {self.reset_timeout:.3f}s waiting for "
                f"the first state frame (reset_fn="
                f"{'yes' if reset_used else 'no'}): {exc}") from exc
        last_seq = baseline.seq
        while True:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                snap = self.reader.read_latest(timeout=remaining)
            except SnapshotTimeout as exc:
                raise SnapshotTimeout(
                    f"reset timed out after {self.reset_timeout:.3f}s waiting "
                    f"for fight_on==1 with two readable fighters "
                    f"(reset_fn={'yes' if reset_used else 'no'}): {exc}") from exc

            if snap.seq < last_seq:            # writer restarted: new epoch
                stable = 0
                last_seq = 0
            fresh = snap.seq > last_seq
            if fresh:
                self._observe_stock(snap)
            if self._ready_for_reset(snap):
                if fresh:
                    stable += 1
                    last_seq = snap.seq
                if stable >= self.spawn_stable_frames:
                    break
            else:
                stable = 0
                if fresh:
                    last_seq = snap.seq
            last = snap

            now = time.monotonic()
            if now >= deadline:
                last_txt = (f"fight_on={last.fight_on}, "
                            f"n_fighters={last.n_fighters}, seq={last.seq}"
                            if last is not None else "no frame")
                raise SnapshotTimeout(
                    f"reset timed out after {self.reset_timeout:.3f}s waiting "
                    f"for fight_on==1 with two readable fighters "
                    f"(reset_fn={'yes' if reset_used else 'no'}; {last_txt})")
            if not fresh:
                time.sleep(min(0.0005, max(0.0, deadline - now)))

        self._last_snapshot = snap
        self._step_count = 0
        self._episodes += 1

        obs = self.obs_builder.build(snap)
        info = self._base_info(snap, "reset")
        info.update({
            "reset_fn_used": reset_used,
            "reset_timeout_s": self.reset_timeout,
            "timing": {"reset_s": time.perf_counter() - t0},
            "input_adapter": (
                getattr(self._input_adapter, "name", "unresolved")
                if self._input_resolved else "unresolved"),
        })
        return obs, info

    # -- step -------------------------------------------------------------
    def step(self, action):
        """Apply one action, advance ``frame_skip`` reader frames, return 5-tuple.

        The action is written once (held for the whole decision interval); the
        wait is measured from the first frame published *after* the write, so
        the action is guaranteed to be held for at least ``frame_skip`` fresh
        frames.  The reward/termination compares the last pre-action snapshot
        with the post-step one.
        """
        if self._closed:
            raise RuntimeError("environment is closed")

        t0 = time.perf_counter()
        canonical = gym_action.decode_action(action, macro=self.macro_mode)
        adapter = self._get_input_adapter()

        prev = self._last_snapshot
        if prev is None:
            prev = self.reader.read_latest(timeout=self.reader_timeout)
        self._observe_stock(prev)

        t_write = time.perf_counter()
        adapter.apply(canonical)
        t_written = time.perf_counter()

        if self._lockstep:
            # The game is parked on a frame boundary: the input written above
            # is what its next poll sees, and exactly frame_skip frames run.
            baseline = prev
            target = self._gate.advance(self.frame_skip)
            curr = self._wait_tagged(target)
        else:
            # Baseline after the write guarantees the action is held for at
            # least frame_skip freshly published frames.
            baseline = self.reader.read_latest(timeout=self.frame_timeout)
            curr = self.reader.wait_frames(self.frame_skip,
                                           timeout=self.frame_timeout,
                                           start=baseline)
        t_read = time.perf_counter()
        self._observe_stock(curr)

        reward = 0.0
        if self.reward_mode == "combat":
            reward = gym_reward.compute_reward(
                prev, curr, self.reward_config, ego_slot=self.ego_slot)

        # termination: fight_on 1->0, or a *usable* fighter's stock reaching 0.
        # A slot whose stock never read positive (signal unavailable) is
        # ignored; fight_on 1->0 still ends the episode.
        end_reason = None
        terminated = False
        if int(prev.fight_on) == 1 and int(curr.fight_on) == 0:
            terminated = True
            end_reason = "fight_off"
        if not terminated and self.stock_termination:
            for i, f in enumerate(curr.fighters[:MAX_FIGHTERS]):
                if not gym_obs.fighter_valid(f):
                    continue
                if self._slot_stock_usable[i] is not True:
                    continue
                if int(f.get("stock", 0)) <= 0:
                    terminated = True
                    end_reason = "stock"
                    break

        self._step_count += 1
        truncated = (self.max_steps is not None
                     and self._step_count >= self.max_steps
                     and not terminated)
        if truncated:
            end_reason = "max_steps"

        self._last_snapshot = curr

        obs = self.obs_builder.build(curr)
        info = self._base_info(curr, "step")
        info.update({
            "action": _copy_action(canonical),
            "reward": float(reward),
            "reward_mode": self.reward_mode,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "end_reason": end_reason,
            "prev_seq": prev.seq,
            "baseline_seq": baseline.seq,
            "seq_delta": max(0, curr.seq - baseline.seq),
            "frames_advanced": (
                int(curr.game_frame) - int(prev.game_frame)
                if self._lockstep else max(0, curr.seq - baseline.seq) // 2),
            "lockstep": self._lockstep,
            "input_adapter": getattr(adapter, "name", type(adapter).__name__),
            "input_error": self._input_error,
            "timing": {
                "decode_s": t_write - t0,
                "write_s": t_written - t_write,
                "read_s": t_read - t_written,
                "total_s": time.perf_counter() - t0,
            },
        })
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -- gym plumbing -----------------------------------------------------
    def render(self):
        """No-op renderer (the game window is the only display)."""
        return None

    def close(self):
        """Release the mmap and the input adapter (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self._gate is not None:
            try:
                self._gate.close()             # releases the game
            except Exception:                  # pragma: no cover
                pass
            self._gate = None
        self.reader.close()
        adapter = self._input_adapter
        if adapter is not None and callable(getattr(adapter, "close", None)):
            try:
                adapter.close()
            except Exception:                  # pragma: no cover - optional dep
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- convenience ------------------------------------------------------
    @property
    def last_snapshot(self) -> RawSnapshot | None:
        return self._last_snapshot

    @property
    def step_count(self) -> int:
        return self._step_count


def _copy_action(canonical: dict) -> dict:
    out = dict(canonical)
    for key, value in canonical.items():
        if isinstance(value, np.ndarray):
            out[key] = value.copy()
    return out


# --------------------------------------------------------------------------
# mock shared memory for tests
# --------------------------------------------------------------------------
def make_fighter(
    entry_id: int = 0,
    kind: int = 0,
    stock: int = 1,
    damage: float = 0.0,
    x: float = 0.0,
    y: float = 0.0,
    facing: float = 1.0,
    action_state: int = 0,
    action_frame: float = 0.0,
    on_ground: int = 1,
    is_actionable: int = 1,
    jumps_left: int = 2,
    vx: float = 0.0,
    vy: float = 0.0,
    dist_to_blast: float = 120.0,
    dx: float = 0.0,
    dy: float = 0.0,
    facing_rel: int = 0,
    is_cpu: int = 1,
    is_on_rebirth: int = 0,
    shield: int = 0,
    shield_size: float = 0.0,
    is_enable_cancel: int = 0,
    hitstun: int = 0,
    buttons: int = 0,
    stick_x: float = 0.0,
    stick_y: float = 0.0,
    trigger: float = 0.0,
    cstick_x: float = 0.0,
    cstick_y: float = 0.0,
    raw_flags: int = 0xFFFFFFFF,
    pad: int = 0,
) -> dict:
    """Build one complete synthetic fighter slot (all 32 v7 fields)."""
    return {
        "entry_id": int(entry_id),
        "kind": int(kind),
        "stock": int(stock),
        "damage": float(damage),
        "x": float(x),
        "y": float(y),
        "facing": float(facing),
        "action_state": int(action_state),
        "action_frame": float(action_frame),
        "on_ground": int(on_ground),
        "is_actionable": int(is_actionable),
        "jumps_left": int(jumps_left),
        "vx": float(vx),
        "vy": float(vy),
        "dist_to_blast": float(dist_to_blast),
        "dx": float(dx),
        "dy": float(dy),
        "facing_rel": int(facing_rel),
        "is_cpu": int(is_cpu),
        "is_on_rebirth": int(is_on_rebirth),
        "shield": int(shield),
        "shield_size": float(shield_size),
        "is_enable_cancel": int(is_enable_cancel),
        "hitstun": int(hitstun),
        "buttons": int(buttons),
        "stick_x": float(stick_x),
        "stick_y": float(stick_y),
        "trigger": float(trigger),
        "cstick_x": float(cstick_x),
        "cstick_y": float(cstick_y),
        "raw_flags": int(raw_flags),
        "pad": int(pad),
    }


def default_fighters() -> list[dict]:
    """Synthetic P1 (Mario) vs P2 (Donkey Kong) at the FD spawn positions."""
    return [
        make_fighter(entry_id=0, kind=0, x=-20.0, y=0.0, facing=1.0,
                     dx=40.0, dy=0.0, facing_rel=1, dist_to_blast=120.0),
        make_fighter(entry_id=1, kind=1, x=20.0, y=0.0, facing=-1.0,
                     dx=-40.0, dy=0.0, facing_rel=1, dist_to_blast=120.0),
    ]


class ShmFixture:
    """Mock ``ssbu_state.bin`` writer for unit tests (no game needed).

    Example::

        fix = ShmFixture()
        fix.set_fighter(0, x=10.0, action_state=2)
        fix.start_publisher(hz=500)     # keep seq advancing for env.step()
        env = SSBUGymEnv(shm_path=fix.path, input_adapter=NoopInputAdapter())
        env.reset()
        env.step(env.action_space.sample())
        fix.close()

    Frames are published with the same odd/even protocol as reader.py.  The
    fixture does not start a publisher unless asked, so reader tests can write
    exactly one deterministic frame.
    """

    def __init__(
        self,
        path: str | None = None,
        version: int = VERSION,
        fight_on: int = 1,
        match_state: int = 1,
        fps: float = 60.0,
        stage_id: int = 454,               # Final Destination
        bounds: tuple[float, float, float, float] = (-240.0, 240.0, 180.0, -140.0),
        publish: bool = True,
    ):
        if path is None:
            fd, path = tempfile.mkstemp(prefix="ssbu_state_", suffix=".bin")
            os.close(fd)
            self._owns_path = True
        else:
            self._owns_path = False
        self.path = str(path)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o666)
        try:
            os.ftruncate(fd, BLOCK_SIZE)
            self._mm = mmap.mmap(fd, BLOCK_SIZE)
        finally:
            os.close(fd)

        self.version = int(version)
        self.fps = float(fps)
        self.stage_id = int(stage_id)
        self.bounds = tuple(float(v) for v in bounds)
        self._fight_on = int(fight_on)
        self._match_state = int(match_state)
        self._n_fighters = MAX_FIGHTERS
        self._fighters = default_fighters()
        self._seq = 0
        self._frame = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.publish_error: BaseException | None = None
        if publish:
            self.publish()

    # -- state ------------------------------------------------------------
    def set_fighter(self, slot: int, **fields) -> None:
        """Update fields of fighter ``slot`` (partial dict allowed)."""
        if slot not in (0, 1):
            raise ValueError("slot must be 0 or 1")
        with self._lock:
            unknown = set(fields) - set(FIGHTER_NAMES)
            if unknown:
                raise ValueError(f"unknown fighter fields: {sorted(unknown)}")
            self._fighters[slot].update(fields)

    def set_fighters(self, fighters: Sequence[dict]) -> None:
        if len(fighters) > MAX_FIGHTERS:
            raise ValueError(f"at most {MAX_FIGHTERS} fighters")
        with self._lock:
            for slot, fields in enumerate(fighters):
                unknown = set(fields) - set(FIGHTER_NAMES)
                if unknown:
                    raise ValueError(f"unknown fighter fields: {sorted(unknown)}")
                self._fighters[slot].update(fields)

    def set_state(self, **fields) -> None:
        """Update header-level state: fight_on, match_state, n_fighters,
        stage_id, bounds, fps, version."""
        allowed = {"fight_on", "match_state", "n_fighters", "stage_id",
                   "bounds", "fps", "version"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown fixture state fields: {sorted(unknown)}")
        with self._lock:
            for key, value in fields.items():
                if key == "fight_on":
                    self._fight_on = int(value)
                elif key == "match_state":
                    self._match_state = int(value)
                elif key == "n_fighters":
                    self._n_fighters = int(value)
                elif key == "stage_id":
                    self.stage_id = int(value)
                elif key == "bounds":
                    self.bounds = tuple(float(v) for v in value)
                elif key == "fps":
                    self.fps = float(value)
                elif key == "version":
                    self.version = int(value)

    # -- writing ----------------------------------------------------------
    def publish(self, game_frame: int | None = None) -> int:
        """Write one complete frame; return its (even) sequence number.

        ``game_frame`` overrides the header's frame number (lockstep tests tag
        each frame with the game frame it belongs to).
        """
        with self._lock:
            self._seq = (self._seq + 1) | 1          # odd: body being written
            seq = self._seq
            self._mm[SEQ_OFFSET:SEQ_OFFSET + 4] = _SEQ.pack(seq)

            self._frame += 1
            tag = self._frame if game_frame is None else int(game_frame)
            header = _HEADER.pack(
                MAGIC, self.version, self._n_fighters, seq,
                time.clock_gettime_ns(time.CLOCK_MONOTONIC), self.fps,
                os.getpid(), self._fight_on, self._match_state,
                self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3],
                self.stage_id, tag, 0, 0, 0, 0,
            )
            self._mm[0:HEADER_SIZE] = header
            for slot in range(MAX_FIGHTERS):
                off = HEADER_SIZE + FIGHTER_SIZE * slot
                if slot < self._n_fighters:
                    values = [self._fighters[slot][name] for name in FIGHTER_NAMES]
                else:
                    values = [0] * N_FIGHTER_FIELDS
                self._mm[off:off + FIGHTER_SIZE] = _FIGHTER.pack(*values)

            self._seq += 1                           # even: frame complete
            self._mm[SEQ_OFFSET:SEQ_OFFSET + 4] = _SEQ.pack(self._seq)
            return self._seq

    def publish_frames(self, n: int = 1) -> int:
        seq = self._seq
        for _ in range(int(n)):
            seq = self.publish()
        return seq

    def write_seq(self, seq: int) -> None:
        """Test helper: force ``seq`` without touching the body (torn frame)."""
        with self._lock:
            self._seq = int(seq) & 0xFFFFFFFF
            self._mm[SEQ_OFFSET:SEQ_OFFSET + 4] = _SEQ.pack(self._seq)

    # -- background publisher --------------------------------------------
    def start_publisher(self, hz: float = 60.0) -> "ShmFixture":
        """Publish frames from a daemon thread at ``hz`` (tests use 200-500)."""
        if hz <= 0:
            raise ValueError("hz must be > 0")
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()

        def loop():
            period = 1.0 / float(hz)
            next_at = time.perf_counter()
            while not self._stop.is_set():
                try:
                    self.publish()
                except BaseException as exc:      # keep the test failure visible
                    self.publish_error = exc
                    break
                next_at += period
                delay = next_at - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_at = time.perf_counter()

        self._thread = threading.Thread(target=loop, name="ShmFixture-publisher",
                                        daemon=True)
        self._thread.start()
        return self

    def stop_publisher(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def game_frame(self) -> int:
        with self._lock:
            return self._frame

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self.stop_publisher()
        mm, self._mm = getattr(self, "_mm", None), None
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):    # pragma: no cover
                pass
        if self._owns_path:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


__all__ = [
    "MAGIC", "VERSION", "HEADER_SIZE", "FIGHTER_SIZE", "BLOCK_SIZE",
    "MAX_FIGHTERS", "N_FIGHTER_FIELDS", "SHM_PATH", "SEQ_OFFSET",
    "HEADER_FIELDS", "FIGHTER_FIELDS", "HEADER_NAMES", "FIGHTER_NAMES",
    "HEADER_FMT", "FIGHTER_FMT",
    "SnapshotError", "SchemaMismatch", "SnapshotTimeout",
    "RawSnapshot", "RawSnapshotReader", "SnapshotReader", "fighter_valid",
    "SSBUGymEnv", "NoopInputAdapter", "default_input_adapter", "default_reset_fn",
    "ShmFixture", "make_fighter", "default_fighters",
]
