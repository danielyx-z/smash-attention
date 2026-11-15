#!/usr/bin/env python3
"""Observer-mode recorder for CPU-vs-CPU expert demonstrations.

The recorder never touches the game: it only reads the state bridge
``/dev/shm/ssbu_state.bin`` (layout v7, see CONTRACTS.md) at 60 Hz while a CPU
fight is running, and writes one ``.npz`` per episode plus a ``manifest.json``.

    # inside the game lock (game_session.py already holds it):
    python3 tools/mem/gym_record.py --episodes 10 --out data/E/mario_vs_dk

Episode lifecycle
-----------------
* ``fight_on`` 0 -> 1 starts an episode.
* ``fight_on`` 1 -> 0 stops it, saves it atomically, and (unless
  ``--no-reset``) asks D's reset client (``gym_reset`` / ``control_block``) to
  start the next fight.
* Existing final ``episode_*.npz`` files are skipped and the target
  ``--episodes`` counts the total in the output directory, so a killed run
  resumes where it stopped.
* Every ``--flush-every`` frames the in-progress episode is written to
  ``episode_NNNN.partial.npz``; after a crash that file is either resumed (if
  the fight is still running and the sequence gap is small) or finalised.

Dataset schema (CONTRACTS.md), one row per reader frame::

    seq            (N,) uint64
    ts_ns          (N,) uint64
    raw_fighters   (N, 2, 32) float32   # v7 slot, field order from CONTRACTS
    buttons_raw    (N, 2) uint32
    stick          (N, 2, 2) float32
    cstick         (N, 2, 2) float32
    trigger        (N, 2) float32
    frame_skip     int (5)
    meta           dict  (stored as ``meta_json`` because npz cannot hold dicts)

Extra keys kept for evidence/debugging: ``stage_id``, ``game_frame``,
``bounds``, ``result_json``, ``seq_gaps``.

Assumptions
-----------
* Layout v7 offsets are taken literally from CONTRACTS.md.  A v6 block (still
  written by the pre-v7 reader) is read on a best-effort basis: the 18 policy
  fields are filled, control fields are zero, and a warning is printed.
* ``raw_fighters`` stores every 32-bit word as float32.  All values the
  dataset uses (state ids, 12-bit button masks, co-ordinates) are exactly
  representable.
* Reset is optional and best-effort; the recorder works with ``--no-reset``
  and a manually restarted match.
"""

from __future__ import annotations

import argparse
import importlib
import json
import mmap
import os
import signal
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

SHM_PATH = "/dev/shm/ssbu_state.bin"
FRAME_SKIP = 5
DEFAULT_HZ = 60.0
DEFAULT_STAGE = os.environ.get("SSBU_STAGE", "final_destination")
DEFAULT_CPU_LEVEL = 9

# Canonical button order (CONTRACTS.md).  The recorder stores the raw mask and
# the decoded vector; B's schema.BUTTON_BITS is the authoritative bit order.
BUTTONS = ("A", "B", "X", "Y", "L", "R", "ZL", "ZR",
           "PLUS", "MINUS", "DPAD_L", "DPAD_R")
_LOCAL_BUTTON_BITS = (0x0001, 0x0002, 0x0004, 0x0008, 0x0020, 0x0400,
                      0x1000, 0x2000, 0x4000, 0x0010, 0x0040, 0x0080)
try:
    from rl.memory import schema as _schema  # type: ignore

    BUTTON_BITS = tuple(int(_schema.BUTTON_BITS[name]) for name in BUTTONS)
except Exception:
    BUTTON_BITS = _LOCAL_BUTTON_BITS


def decode_buttons_mask(mask) -> np.ndarray:
    """Raw ControlModule mask(s) -> canonical 12-vector(s)."""
    m = np.asarray(mask)
    out = np.zeros(m.shape + (len(BUTTONS),), dtype=np.uint8)
    for i, bit in enumerate(BUTTON_BITS):
        out[..., i] = ((m & bit) != 0).astype(np.uint8)
    return out

# ---- v7 layout -----------------------------------------------------------
MAGIC = b"SSBU"
VERSION = 7

HEADER_FMT_V7 = "<4sIIIQfIIIffffiIIIII"
HEADER_SIZE_V7 = 80
FIGHTER_FMT_V7 = "<iiiffffifiiifffffiiiifiiIfffffII"
FIGHTER_SIZE_V7 = 128
FIGHTER_WORDS = 32
MAX_FIGHTERS = 2
BLOCK_SIZE_V7 = HEADER_SIZE_V7 + FIGHTER_SIZE_V7 * MAX_FIGHTERS  # 336

SEQ_OFFSET = 12
TS_OFFSET = 16
FPS_OFFSET = 24
FIGHT_OFFSET = 32
MATCH_STATE_OFFSET = 36
STAGE_OFFSET = 56
GAME_FRAME_OFFSET = 60
BLAST_OFFSET = 40

_HEADER_V7 = struct.Struct(HEADER_FMT_V7)
_FIGHTER_V7 = struct.Struct(FIGHTER_FMT_V7)
_SEQ = struct.Struct("<I")

# v6 fallback (reader.py before B's v7 update).  Do not use for new data.
HEADER_FMT_V6 = "<4sIIIQfIIIffffII"
HEADER_SIZE_V6 = 64
FIGHTER_FMT_V6 = "<iiiffffifiiifffffiII"
FIGHTER_SIZE_V6 = 80
BLOCK_SIZE_V6 = HEADER_SIZE_V6 + FIGHTER_SIZE_V6 * MAX_FIGHTERS  # 224
_HEADER_V6 = struct.Struct(HEADER_FMT_V6)
_FIGHTER_V6 = struct.Struct(FIGHTER_FMT_V6)

assert _HEADER_V7.size == HEADER_SIZE_V7
assert _FIGHTER_V7.size == FIGHTER_SIZE_V7
assert _HEADER_V6.size == HEADER_SIZE_V6
assert _FIGHTER_V6.size == FIGHTER_SIZE_V6

# v7 field index -> name, exactly CONTRACTS.md.
FIELDS = (
    "entry_id", "kind", "stock", "damage", "x", "y", "facing",
    "action_state", "action_frame", "on_ground", "is_actionable",
    "jumps_left", "vx", "vy", "dist_to_blast", "dx", "dy", "facing_rel",
    "is_cpu", "is_on_rebirth", "shield", "shield_size", "is_enable_cancel",
    "hitstun", "buttons", "stick_x", "stick_y", "trigger", "cstick_x",
    "cstick_y", "raw_flags", "pad",
)
FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}
BUTTONS_INDEX = FIELD_INDEX["buttons"]
STICK_X_INDEX = FIELD_INDEX["stick_x"]
STICK_Y_INDEX = FIELD_INDEX["stick_y"]
TRIGGER_INDEX = FIELD_INDEX["trigger"]
CSTICK_X_INDEX = FIELD_INDEX["cstick_x"]
CSTICK_Y_INDEX = FIELD_INDEX["cstick_y"]
RAW_FLAGS_INDEX = FIELD_INDEX["raw_flags"]

# v6 field index -> v7 index, for the 18 policy fields + raw_flags.
_V6_TO_V7 = tuple(FIELD_INDEX[n] for n in (
    "entry_id", "kind", "stock", "damage", "x", "y", "facing",
    "action_state", "action_frame", "on_ground", "is_actionable",
    "jumps_left", "vx", "vy", "dist_to_blast", "dx", "dy", "facing_rel",
    "raw_flags",
))

CHAR_NAMES = {0: "mario", 1: "donkey_kong"}
STAGE_NAMES = {
    # stage ids are not published as names yet; keep the hook for when B
    # documents them.  The debug match always uses Final Destination.
    0: DEFAULT_STAGE,
    454: "final_destination",          # ShmFixture / debug-match stage id
}


def char_name(kind: int) -> str:
    return CHAR_NAMES.get(int(kind), f"kind_{int(kind)}")


def stage_name(stage_id: int) -> str:
    return STAGE_NAMES.get(int(stage_id), f"stage_{int(stage_id)}")


def _jsonable(value):
    """Convert numpy scalars / arrays to plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(obj), fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def detect_build() -> str:
    """Best-effort build id: env override, then the git revision."""
    env = os.environ.get("SSBU_BUILD")
    if env:
        return env
    git = os.path.join(REPO, ".git")
    try:
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as fh:
            head = fh.read().strip()
        if head.startswith("ref: "):
            ref = head[5:]
            with open(os.path.join(git, ref), encoding="utf-8") as fh:
                rev = fh.read().strip()
        else:
            rev = head
        return f"ssbu-recomp@{rev[:12]}"
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# shared memory reader
# ---------------------------------------------------------------------------
def _fighter_array(buf, offset: int, version: int) -> np.ndarray:
    """Return the 32 raw words of one slot as float32."""
    out = np.zeros(FIGHTER_WORDS, dtype=np.float32)
    if version == 7:
        vals = _FIGHTER_V7.unpack_from(buf, offset)
        out[:] = vals
    else:
        vals = _FIGHTER_V6.unpack_from(buf, offset)
        for src, dst in enumerate(_V6_TO_V7):
            out[dst] = vals[src]
    return out


def parse_frame(buf, version: int) -> dict:
    """Decode one complete block.  ``buf`` must be a bytes-like copy."""
    if version == 7:
        (magic, ver, n_fighters, seq, ts_ns, fps, reader_pid, fight_on,
         match_state, bl, br, bt, bb, stage_id, game_frame, flags,
         _p0, _p1, _p2) = _HEADER_V7.unpack_from(buf, 0)
        header_size, fighter_size = HEADER_SIZE_V7, FIGHTER_SIZE_V7
    elif version == 6:
        (magic, ver, n_fighters, seq, ts_ns, fps, reader_pid, fight_on,
         match_state, bl, br, bt, bb, _p0, _p1) = _HEADER_V6.unpack_from(buf, 0)
        header_size, fighter_size = HEADER_SIZE_V6, FIGHTER_SIZE_V6
        stage_id, game_frame, flags = 0, 0, 0
    else:
        raise ValueError(f"unsupported state layout v{version}")
    if magic != MAGIC:
        raise ValueError(f"bad state magic {magic!r} (expected {MAGIC!r})")

    fighters = np.stack([
        _fighter_array(buf, header_size + fighter_size * i, version)
        for i in range(MAX_FIGHTERS)
    ])  # (2, 32)

    return {
        "version": int(ver),
        "n_fighters": int(n_fighters),
        "seq": int(seq),
        "ts_ns": int(ts_ns),
        "fps": float(fps),
        "reader_pid": int(reader_pid),
        "fight_on": int(fight_on),
        "match_state": int(match_state),
        "stage_id": int(stage_id),
        "game_frame": int(game_frame),
        "flags": int(flags),
        "bounds": (float(bl), float(br), float(bt), float(bb)),
        "fighters": fighters,
        "buttons_raw": fighters[:, BUTTONS_INDEX].astype(np.uint32),
        "stick": fighters[:, [STICK_X_INDEX, STICK_Y_INDEX]].astype(np.float32),
        "cstick": fighters[:, [CSTICK_X_INDEX, CSTICK_Y_INDEX]].astype(np.float32),
        "trigger": fighters[:, TRIGGER_INDEX].astype(np.float32),
    }


class ShmReader:
    """Map the reader block once and return each complete published frame."""

    def __init__(self, path: str = SHM_PATH):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        size = os.fstat(self.fd).st_size
        self.mm = mmap.mmap(self.fd, size, prot=mmap.PROT_READ)
        os.close(self.fd)
        self.fd = None
        self.version = struct.unpack_from("<I", self.mm, 4)[0]
        if self.version == 7:
            self.block_size = BLOCK_SIZE_V7
            self.fighter_size = FIGHTER_SIZE_V7
        elif self.version == 6:
            self.block_size = BLOCK_SIZE_V6
            self.fighter_size = FIGHTER_SIZE_V6
        else:
            self.close()
            raise ValueError(
                f"{path}: unsupported state layout v{self.version} "
                "(expected 7; reader.py may be out of date)")
        if len(self.mm) < self.block_size:
            self.close()
            raise ValueError(
                f"{path}: block is {len(self.mm)} bytes, layout v{self.version} "
                f"needs {self.block_size}")

    def latest_seq(self) -> int:
        return struct.unpack_from("<I", self.mm, SEQ_OFFSET)[0]

    def read(self, retries: int = 8) -> dict | None:
        """Latest complete frame, or None on a torn read."""
        for _ in range(retries):
            seq1 = self.latest_seq()
            if seq1 == 0 or (seq1 & 1):
                continue
            body = self.mm[:self.block_size]      # one copy, then verify
            if self.latest_seq() != seq1:
                continue
            try:
                frame = parse_frame(body, self.version)
            except (struct.error, ValueError):
                continue
            if frame["seq"] == seq1:
                return frame
        return None

    def close(self) -> None:
        mm = getattr(self, "mm", None)
        if mm is not None:
            try:
                mm.close()
            except (BufferError, ValueError):
                pass
            self.mm = None
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None


# ---------------------------------------------------------------------------
# reset client (Agent D's files; lazy and optional)
# ---------------------------------------------------------------------------
def load_reset_fn(verbose: bool = True):
    """Return ``(callable, name)`` for the match reset client, or ``(None, None)``.

    Tries ``gym_reset`` then ``control_block`` and several plausible entry
    points.  All of them are optional: the recorder still works when the
    match is restarted by hand.
    """
    candidates = []
    # Module-level helpers first: control_block.request_reset() opens and closes
    # the block itself.  A class instance must be opened before it can be used
    # (an unopened ControlBlock raises "control block is not open").
    for modname in ("control_block", "gym_reset"):
        try:
            mod = importlib.import_module(f"rl.input.{modname}")
        except Exception:
            continue
        for name in ("reset_match", "request_reset", "do_reset", "reset"):
            fn = getattr(mod, name, None)
            if callable(fn):
                candidates.append((fn, f"{modname}.{name}"))
        for cls_name in ("ControlBlock", "Control", "ResetClient"):
            cls = getattr(mod, cls_name, None)
            if cls is None:
                continue
            try:
                obj = cls()
                opener = getattr(obj, "open", None)
                if callable(opener):
                    obj = opener() or obj
            except Exception:
                continue
            for name in ("request_reset", "reset", "do_reset"):
                fn = getattr(obj, name, None)
                if callable(fn):
                    candidates.append((fn, f"{modname}.{cls_name}().{name}"))
    if not candidates:
        if verbose:
            print("[record] no reset client (gym_reset/control_block); "
                  "use --no-reset or restart matches by hand", flush=True)
        return None, None
    fn, name = candidates[0]
    if verbose:
        print(f"[record] reset client: {name}", flush=True)
    return fn, name


_reset_failures = 0


def call_reset(fn, timeout: float = 20.0) -> bool:
    """Call a reset function, tolerating its calling convention."""
    global _reset_failures

    def _failed(exc: Exception) -> bool:
        global _reset_failures
        _reset_failures += 1
        if _reset_failures <= 3:
            print(f"[record] reset failed: {type(exc).__name__}: {exc}"
                  + (" (further failures are not logged)" if
                     _reset_failures == 3 else ""), flush=True)
        return False

    for call in (lambda: fn(), lambda: fn(timeout=timeout)):
        try:
            result = call()
        except TypeError:
            continue
        except Exception as exc:
            return _failed(exc)
        if result is False:
            return False
        _reset_failures = 0
        return True
    # Last resort: call it bare and report the failure.
    try:
        fn()
        _reset_failures = 0
        return True
    except Exception as exc:
        return _failed(exc)


# ---------------------------------------------------------------------------
# episode buffer + files
# ---------------------------------------------------------------------------
class Episode:
    """One episode's 60 Hz rows, built incrementally."""

    def __init__(self, index: int, meta: dict | None = None):
        self.index = int(index)
        self.meta = dict(meta or {})
        self.seq: list[int] = []
        self.ts_ns: list[int] = []
        self.raw: list[np.ndarray] = []
        self.buttons: list[np.ndarray] = []
        self.stick: list[np.ndarray] = []
        self.cstick: list[np.ndarray] = []
        self.trigger: list[np.ndarray] = []
        self.game_frame: list[int] = []
        self.seq_gaps = 0
        self.started_ns = time.time_ns()
        self.bounds = (0.0, 0.0, 0.0, 0.0)
        self.stage_id = 0
        self.result: dict = {}
        self._last_seq: int | None = None

    def __len__(self) -> int:
        return len(self.seq)

    @property
    def last_seq(self) -> int | None:
        return self._last_seq

    def append(self, frame: dict) -> None:
        seq = frame["seq"]
        if self._last_seq is not None and seq != self._last_seq + 2:
            # The reader bumps seq by 2 per publish; anything else is a missed
            # frame (recorded here so the BC stage can warn about it).
            if seq > self._last_seq:
                self.seq_gaps += 1
        self._last_seq = seq
        self.seq.append(seq)
        self.ts_ns.append(frame["ts_ns"])
        self.raw.append(frame["fighters"])
        self.buttons.append(frame["buttons_raw"])
        self.stick.append(frame["stick"])
        self.cstick.append(frame["cstick"])
        self.trigger.append(frame["trigger"])
        self.game_frame.append(frame["game_frame"])
        self.bounds = frame["bounds"]
        self.stage_id = frame["stage_id"]
        if not self.meta.get("p1_char"):
            self.meta.update({
                "p1_char": char_name(int(frame["fighters"][0, FIELD_INDEX["kind"]])),
                "p2_char": char_name(int(frame["fighters"][1, FIELD_INDEX["kind"]])),
                "stage": self.meta.get("stage") or stage_name(frame["stage_id"]),
                "stage_id": int(frame["stage_id"]),
            })

    def arrays(self, frame_skip: int = FRAME_SKIP) -> dict:
        if not self.seq:
            raise ValueError("empty episode")
        raw = np.stack(self.raw).astype(np.float32)
        # npz cannot store a Python dict without pickle; keep the JSON string
        # under both names so either convention loads without allow_pickle.
        meta_json = json.dumps(_jsonable(self.meta), sort_keys=True)
        result_json = json.dumps(_jsonable(self.result), sort_keys=True)
        return {
            "seq": np.asarray(self.seq, dtype=np.uint64),
            "ts_ns": np.asarray(self.ts_ns, dtype=np.uint64),
            "raw_fighters": raw,
            "buttons_raw": np.stack(self.buttons).astype(np.uint32),
            # decoded canonical order (CONTRACTS.md: raw mask + decoded vector)
            "buttons": decode_buttons_mask(np.stack(self.buttons)),
            "stick": np.stack(self.stick).astype(np.float32),
            "cstick": np.stack(self.cstick).astype(np.float32),
            "trigger": np.stack(self.trigger).astype(np.float32),
            "frame_skip": np.asarray(frame_skip, dtype=np.int64),
            "meta_json": np.asarray(meta_json),
            "meta": np.asarray(meta_json),
            "result_json": np.asarray(result_json),
            "result": np.asarray(result_json),
            "stage_id": np.asarray(self.stage_id, dtype=np.int64),
            "game_frame": np.asarray(self.game_frame, dtype=np.uint64),
            "bounds": np.asarray(self.bounds, dtype=np.float32),
            "seq_gaps": np.asarray(self.seq_gaps, dtype=np.int64),
        }

    def save(self, path: str, frame_skip: int = FRAME_SKIP) -> None:
        arrays = self.arrays(frame_skip)
        # np.savez appends .npz only when the name lacks it; keep it explicit so
        # the temporary file has exactly the name we replace.
        tmp = path + ".tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Episode":
        with np.load(path, allow_pickle=False) as data:
            meta = {}
            if "meta_json" in data:
                meta = json.loads(str(data["meta_json"]))
            elif "meta" in data:
                meta = json.loads(str(data["meta"]))
            ep = cls(index=int(meta.get("index", -1)), meta=meta)
            ep.seq = [int(v) for v in data["seq"]]
            ep.ts_ns = [int(v) for v in data["ts_ns"]]
            ep.raw = [row for row in data["raw_fighters"]]
            ep.buttons = [row for row in data["buttons_raw"]]
            ep.stick = [row for row in data["stick"]]
            ep.cstick = [row for row in data["cstick"]]
            ep.trigger = [row for row in data["trigger"]]
            if "game_frame" in data:
                ep.game_frame = [int(v) for v in data["game_frame"]]
            else:
                ep.game_frame = [0] * len(ep.seq)
            if "stage_id" in data:
                ep.stage_id = int(data["stage_id"])
            if "bounds" in data:
                ep.bounds = tuple(float(v) for v in data["bounds"])
            if "result_json" in data:
                ep.result = json.loads(str(data["result_json"]))
            if "seq_gaps" in data:
                ep.seq_gaps = int(data["seq_gaps"])
            ep._last_seq = ep.seq[-1] if ep.seq else None
            ep.started_ns = int(ep.ts_ns[0]) if ep.ts_ns else time.time_ns()
        return ep


def episode_path(out: str, index: int) -> str:
    return os.path.join(out, f"episode_{index:04d}.npz")


def partial_path(out: str, index: int) -> str:
    return os.path.join(out, f"episode_{index:04d}.partial.npz")


def read_manifest(out: str) -> dict:
    path = os.path.join(out, "manifest.json")
    if not os.path.exists(path):
        return {"format": "ssbu_gym_record_v1", "episodes": []}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_manifest(out: str, manifest: dict) -> None:
    _atomic_write_json(os.path.join(out, "manifest.json"), manifest)


def scan_existing(out: str) -> dict:
    """Return ``{index: path}`` for completed episodes in ``out``."""
    found = {}
    if not os.path.isdir(out):
        return found
    for name in os.listdir(out):
        if name.startswith("episode_") and name.endswith(".npz") \
                and ".partial." not in name:
            stem = name[len("episode_"):-len(".npz")]
            try:
                found[int(stem)] = os.path.join(out, name)
            except ValueError:
                continue
    return found


# ---------------------------------------------------------------------------
# recorder core (no sleeping, so tests can drive it)
# ---------------------------------------------------------------------------
class RecorderCore:
    """Frame-driven state machine: feed it every published frame."""

    def __init__(self, out: str, episodes: int = 10, frame_skip: int = FRAME_SKIP,
                 meta: dict | None = None, reset: bool = True,
                 resume_window_s: float = 5.0, min_frames: int = 2,
                 flush_every: int = 300, episode_frames: int = 0, log=print):
        self.out = out
        self.target = int(episodes)
        self.frame_skip = int(frame_skip)
        self.meta = dict(meta or {})
        self.reset = bool(reset)
        self.resume_window = float(resume_window_s)
        self.min_frames = int(min_frames)
        self.flush_every = max(1, int(flush_every))
        self.episode_frames = max(0, int(episode_frames))
        self.log = log
        self.manifest = read_manifest(out)
        self.done = scan_existing(out)
        self.manifest["episodes"] = [
            e for e in self.manifest.get("episodes", [])
            if int(e.get("index", -1)) in self.done
        ]
        self._sync_manifest()
        self.ep: Episode | None = None
        self._next_index = max(self.done) + 1 if self.done else 0
        self.reset_requested_at = 0.0
        self._resume_attempted = False
        self.finished = len(self.done) >= self.target
        self.stopped = False
        self.dropped = 0
        self.last_fight = 0

    # -- helpers ---------------------------------------------------------
    def _sync_manifest(self) -> None:
        """Recover manifest entries for finals found without one."""
        known = {int(e.get("index", -1))
                 for e in self.manifest.get("episodes", [])}
        changed = False
        for index, path in sorted(self.done.items()):
            if index in known:
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    meta = {}
                    for key in ("meta_json", "meta"):
                        if key in data:
                            meta = json.loads(str(data[key]))
                            break
                    result = {}
                    for key in ("result_json", "result"):
                        if key in data:
                            result = json.loads(str(data[key]))
                            break
                    seq = [int(v) for v in data["seq"]]
                    if "seq_gaps" in data:
                        meta["seq_gaps"] = int(data["seq_gaps"])
            except Exception as exc:
                self.log(f"[record] cannot index {path}: {exc}")
                meta, result, seq = {}, {}, []
            entry = {
                "index": int(index),
                "file": os.path.basename(path),
                "frames": len(seq),
                "duration_s": len(seq) / 60.0,
                "seq_start": seq[0] if seq else None,
                "seq_end": seq[-1] if seq else None,
                "seq_gaps": int(meta.get("seq_gaps", 0)),
                "stage_id": int(meta.get("stage_id", 0)),
                "meta": meta,
                "result": result,
                "recovered": True,
            }
            self.manifest.setdefault("episodes", []).append(entry)
            changed = True
        if changed:
            self.manifest["episodes"].sort(key=lambda e: int(e["index"]))
            self.manifest.update({
                "format": "ssbu_gym_record_v1", "hz": DEFAULT_HZ,
                "frame_skip": self.frame_skip,
                "updated_ns": time.time_ns(),
            })
            write_manifest(self.out, self.manifest)

    def _episode_meta(self) -> dict:
        meta = dict(self.meta)
        meta.setdefault("cpu_level", DEFAULT_CPU_LEVEL)
        meta.setdefault("build", detect_build())
        meta["index"] = self._next_index
        return meta

    def _start(self, frame: dict) -> None:
        self.ep = Episode(self._next_index, self._episode_meta())
        self.ep.started_ns = frame["ts_ns"]
        self.log(f"[record] episode {self._next_index} started "
                 f"(seq {frame['seq']})", )
        self._flush(partial=True)

    def _finish(self, last: dict | None = None) -> None:
        ep = self.ep
        if ep is None:
            return
        self.ep = None
        if len(ep) < self.min_frames:
            self.log(f"[record] episode {ep.index} discarded "
                     f"({len(ep)} frames < {self.min_frames})")
            try:
                os.unlink(partial_path(self.out, ep.index))
            except OSError:
                pass
            return
        if last is not None:
            # The stop frame is the first fight_on=0 frame; the last recorded
            # row is the most reliable final state.
            rows = ep.raw[-1] if ep.raw else last["fighters"]
            stocks = [int(rows[i, FIELD_INDEX["stock"]]) for i in (0, 1)]
            damage = [float(rows[i, FIELD_INDEX["damage"]]) for i in (0, 1)]
            ep.result = {
                "frames": len(ep),
                "stocks": stocks,
                "damage": damage,
                "fight_on": int(last["fight_on"]),
                "match_state": int(last["match_state"]),
                "winner": 0 if stocks[0] > stocks[1] else
                          (1 if stocks[1] > stocks[0] else None),
                "seq_start": int(ep.seq[0]),
                "seq_end": int(ep.seq[-1]),
                "seq_gaps": ep.seq_gaps,
                "duration_s": len(ep) / 60.0,
                "finished_ns": time.time_ns(),
            }
        final = episode_path(self.out, ep.index)
        ep.save(final, self.frame_skip)
        try:
            os.unlink(partial_path(self.out, ep.index))
        except OSError:
            pass
        self.done[ep.index] = final
        entry = {
            "index": ep.index,
            "file": os.path.basename(final),
            "frames": len(ep),
            "duration_s": len(ep) / 60.0,
            "seq_start": int(ep.seq[0]),
            "seq_end": int(ep.seq[-1]),
            "seq_gaps": ep.seq_gaps,
            "stage_id": ep.stage_id,
            "meta": ep.meta,
            "result": ep.result,
        }
        eps = [e for e in self.manifest.get("episodes", [])
               if int(e.get("index", -1)) != ep.index]
        eps.append(entry)
        eps.sort(key=lambda e: int(e["index"]))
        self.manifest["episodes"] = eps
        self.manifest.update({
            "format": "ssbu_gym_record_v1",
            "hz": DEFAULT_HZ,
            "frame_skip": self.frame_skip,
            "updated_ns": time.time_ns(),
        })
        write_manifest(self.out, self.manifest)
        self._next_index = max(self.done) + 1
        self.log(f"[record] episode {ep.index} saved: {len(ep)} frames, "
                 f"{ep.seq_gaps} seq gaps -> {final}")
        if len(self.done) >= self.target:
            self.finished = True
            self.log(f"[record] target reached: {len(self.done)}/{self.target}")

    def _flush(self, partial: bool) -> None:
        if self.ep is None or not len(self.ep):
            return
        if partial:
            self.ep.save(partial_path(self.out, self.ep.index), self.frame_skip)

    def _maybe_resume(self, frame: dict) -> bool:
        """Resume a crash-left partial if the same fight is still running."""
        if self._resume_attempted:
            return False
        self._resume_attempted = True
        best = None
        if os.path.isdir(self.out):
            for name in os.listdir(self.out):
                if not (name.startswith("episode_") and
                        name.endswith(".partial.npz")):
                    continue
                try:
                    idx = int(name[len("episode_"):-len(".partial.npz")])
                except ValueError:
                    continue
                if idx in self.done:
                    try:
                        os.unlink(os.path.join(self.out, name))
                    except OSError:
                        pass
                    continue
                if best is None or idx > best:
                    best = idx
        if best is None:
            return False
        path = partial_path(self.out, best)
        try:
            ep = Episode.load(path)
        except Exception as exc:
            self.log(f"[record] cannot load {path}: {exc}")
            return False
        if ep.index < 0:                     # partial without meta.index
            ep.index = best
            ep.meta["index"] = best
        if frame["fight_on"] != 1:
            # The fight already ended while we were down: finalise the partial.
            self.ep = ep
            self._finish(frame)
            self._resume_attempted = True
            return False
        gap = frame["seq"] - (ep.last_seq or 0)
        if gap < 0 or gap > 2 * self.resume_window * DEFAULT_HZ:
            self.log(f"[record] partial {path} is stale (seq gap {gap}); "
                     "finalising it and starting a new episode")
            self.ep = ep
            self._finish(frame)
            return False
        self.ep = ep
        self._next_index = ep.index
        self.log(f"[record] resumed partial episode {ep.index} "
                 f"({len(ep)} frames)", )
        return True

    # -- frame API -------------------------------------------------------
    def feed(self, frame: dict) -> str:
        """Consume one published frame.  Returns an event name for tests."""
        fight = int(frame["fight_on"])
        event = "skip"
        if self.finished or self.stopped:
            return "done"

        if self.ep is None:
            if fight == 1:
                resumed = False
                if not self._resume_attempted:
                    resumed = self._maybe_resume(frame)
                if self.ep is None:
                    self._start(frame)
                if self.ep is not None and \
                        (not resumed or frame["seq"] > (self.ep.last_seq or -1)):
                    self.ep.append(frame)
                event = "resume" if resumed else "start"
            else:
                event = "wait"
        else:
            if fight == 1:
                seq = frame["seq"]
                if self.ep.last_seq is None or seq > self.ep.last_seq:
                    self.ep.append(frame)
                    if self.episode_frames and len(self.ep) >= self.episode_frames:
                        # Cut here and ask the runtime for a rematch while the fight
                        # is still live (R2 direct reset); the results screen is
                        # never shown and the next episode starts on the new fight.
                        self._finish(frame)
                        return "split"
                    if len(self.ep) % self.flush_every == 0:
                        self._flush(partial=True)
                    event = "append"
                else:
                    self.dropped += 1
                    event = "duplicate"
            else:
                event = "stop"
                self._finish(frame)

        self.last_fight = fight
        return event

    def stop(self) -> None:
        """Save the in-progress episode as a partial and stop."""
        self.stopped = True
        self._flush(partial=True)
        if self.ep is not None:
            self.log(f"[record] stopped with partial episode {self.ep.index} "
                     f"({len(self.ep)} frames) at "
                     f"{partial_path(self.out, self.ep.index)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _acquire_lock():
    if os.environ.get("SSBU_GAME_LOCK_HELD") == "1":
        return None
    try:
        from rl.utils.locks import game_lock
    except Exception:
        return None
    lock = game_lock(timeout=60.0)
    lock.acquire()
    return lock


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "E", "recordings"),
                    help="output directory (one .npz per episode + manifest.json)")
    ap.add_argument("--episodes", type=int, default=10,
                    help="total episodes wanted in --out (existing ones count)")
    ap.add_argument("--hz", type=float, default=DEFAULT_HZ,
                    help="reader polling rate (frames recorded are the reader's)")
    ap.add_argument("--no-reset", action="store_true",
                    help="never ask gym_reset/control_block for a new match")
    ap.add_argument("--shm", default=SHM_PATH, help="state block path")
    ap.add_argument("--stage", default=DEFAULT_STAGE,
                    help="stage label stored in metadata")
    ap.add_argument("--cpu-level", type=int, default=DEFAULT_CPU_LEVEL)
    ap.add_argument("--flush-every", type=int, default=300,
                    help="frames between crash-safe partial saves")
    ap.add_argument("--resume-window", type=float, default=5.0,
                    help="max seconds of missing frames to resume a partial")
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--episode-frames", type=int, default=0,
                    help="cut an episode after N reader frames and rematch in-fight (0 = off)")
    ap.add_argument("--no-lock", action="store_true",
                    help="do not take /tmp/ssbu_game.lock (default: take it)")
    args = ap.parse_args(argv)

    if args.hz <= 0:
        ap.error("--hz must be positive")
    if args.episodes < 1:
        ap.error("--episodes must be >= 1")

    os.makedirs(args.out, exist_ok=True)
    try:
        lock = None if args.no_lock else _acquire_lock()
    except TimeoutError as exc:
        print(f"[record] game lock is busy ({exc}). Run under "
              "tools/mem/game_session.py or pass --no-lock.", file=sys.stderr)
        return 3

    try:
        try:
            reader = ShmReader(args.shm)
        except (OSError, ValueError) as exc:
            print(f"[record] cannot open state block: {exc}", file=sys.stderr)
            return 2
        if reader.version != 7:
            print(f"[record] warning: state layout v{reader.version}; "
                  "control fields (buttons/stick/cstick/trigger) are only "
                  "available in v7 recordings", flush=True)

        reset_fn = None if args.no_reset else load_reset_fn()[0]
        core = RecorderCore(
            args.out, episodes=args.episodes, frame_skip=FRAME_SKIP,
            meta={"stage": args.stage, "cpu_level": args.cpu_level,
                  "build": detect_build()},
            reset=not args.no_reset, resume_window_s=args.resume_window,
            min_frames=args.min_frames, flush_every=args.flush_every,
            episode_frames=args.episode_frames,
            log=lambda *a: print(*a, flush=True))
        if core.finished:
            print(f"[record] {len(core.done)} episodes already in {args.out}; "
                  f"target {args.episodes} reached", flush=True)
            reader.close()
            return 0
        print(f"[record] {len(core.done)}/{args.episodes} episodes already in "
              f"{args.out}; writing up to {core.target}", flush=True)

        stop = {"flag": False}

        def on_signal(_sig, _frm):
            stop["flag"] = True

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        poll_period = 1.0 / (args.hz * 2.0)  # poll at 2x to catch every publish
        next_at = time.monotonic()
        last_seq = -1
        reset_interval = 3.0
        last_status = 0.0
        try:
            while not stop["flag"] and not core.finished:
                frame = reader.read()
                now = time.monotonic()
                if frame is None or frame["seq"] == last_seq:
                    if core.ep is None and frame is not None \
                            and frame["fight_on"] == 0 and reset_fn is not None \
                            and now - core.reset_requested_at >= reset_interval:
                        core.reset_requested_at = now
                        call_reset(reset_fn)
                else:
                    last_seq = frame["seq"]
                    event = core.feed(frame)
                    if event == "split" and reset_fn is not None \
                            and not core.finished:
                        core.reset_requested_at = time.monotonic()
                        print("[record] cut at %d frames: requesting an in-fight "
                              "rematch" % args.episode_frames, flush=True)
                        call_reset(reset_fn)
                    if event == "stop" and reset_fn is not None \
                            and not core.finished:
                        core.reset_requested_at = time.monotonic()
                        call_reset(reset_fn)
                    if now - last_status >= 5.0:
                        last_status = now
                        done = len(core.done)
                        extra = (f" ep {core.ep.index}: {len(core.ep)} frames"
                                 if core.ep is not None else "")
                        print(f"[record] {done}/{core.target} episodes, "
                              f"fight_on={frame['fight_on']}{extra}", flush=True)

                next_at += poll_period
                delay = next_at - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_at = time.monotonic()
        finally:
            core.stop()
            reader.close()
            print(f"[record] done: {len(core.done)}/{core.target} episodes in "
                  f"{args.out}", flush=True)
        return 0 if core.finished else 130
    finally:
        if lock is not None:
            lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
