#!/usr/bin/env python3
"""Behaviour cloning for the SSBU gym (non-naive: GRU + action chunks).

Pipeline
--------
1. Read the observer-mode recordings written by ``gym_record.py`` (60 Hz).
2. Resample to the 12 Hz decision rate (every ``frame_skip`` = 5th frame).
3. Build windows of ``T`` = 8 decisions plus the previous action; the target is
   an action chunk of ``H`` = 4 decisions.
4. Train a GRU encoder (hidden 256) with heads for
       * 12 buttons            -> BCEWithLogits (per-button pos_weight)
       * 9x9 left-stick heatmap -> cross entropy (soft-argmax at inference)
       * 9x9 C-stick heatmap    -> cross entropy, optional
       * trigger               -> BCEWithLogits
5. Mirror augmentation (both observation and action: x, vx, dx, facing,
   stick_x, C-stick x, D-pad L/R) is applied at the episode level, so windows
   and targets stay aligned.
6. Episode-level 80/10/10 split, observation normalization stats stored in the
   checkpoint, per-button thresholds calibrated on the validation split to
   match the expert positive rate.
7. Early stopping on validation loss (``--patience``, ``--min-epochs``); the
   best-val weights are restored and saved, and metrics are computed there.
8. CPU guard frames (status 8/9/10) carry the CPU mask ``0x8004``, which
   decodes as a spurious X.  The training target relabels those frames
   ``ZL=1, X=0`` (``--no-shield-from-status`` disables).  The raw mask in the
   recording is untouched.
9. Inference uses a temporal ensemble (re-plan every ``execute`` = 2 actions
   and blend overlapping chunk predictions).

CLI
---
    .venv/bin/python tools/mem/gym_bc.py --data data/E/mario_vs_dk \
        --epochs 10 --out data/E/bc.pt
    .venv/bin/python tools/mem/gym_bc.py --overfit

Assumptions
-----------
* The expert action cloned is ``--player`` (default P1, slot 0), while the
  observation contains both fighters (CONTRACTS.md profile v1).
* Button bit ``i`` of the raw ControlModule mask is canonical button ``i``
  (A=bit 0, ..., DPAD_R=bit 11).  B/D own the final bit positions; override
  with ``BUTTON_BITS`` if their evidence differs.
* ``raw_fighters`` is (N, 2, 32) float32 in the v7 slot order.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import copy
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# ---------------------------------------------------------------------------
# contract constants
# ---------------------------------------------------------------------------
BUTTONS = ("A", "B", "X", "Y", "L", "R", "ZL", "ZR",
           "PLUS", "MINUS", "DPAD_L", "DPAD_R")
N_BUTTONS = len(BUTTONS)
BUTTON_INDEX = {name: i for i, name in enumerate(BUTTONS)}
# The ControlModule mask in CPU recordings is the CPU AI's virtual button
# state.  Guard frames carry 0x8004, whose 0x4 bit decodes as X; that is a
# spurious jump (A's check_button table has no 0x8000 entry).  The reliable
# shield signal is the status kind, so the dataset builder relabels those
# target frames ZL=1/X=0 (see BCConfig.shield_from_status).
GUARD_STATES = (8, 9, 10)          # GUARD_START, GUARD_ON, GUARD_OFF
SHIELD_BUTTON = BUTTON_INDEX["ZL"]
# Raw ControlModule bit positions, evidence in memmap.json / schema.py:
#   A=0x1 B=0x2 X=0x4 Y=0x8 L=0x20 R=0x400 ZL=0x1000 ZR=0x2000
#   PLUS=0x4000 MINUS=0x10 DPAD_L=0x40 DPAD_R=0x80
# Agent B's schema.BUTTON_BITS is authoritative when importable.
_LOCAL_BUTTON_BITS = (0x0001, 0x0002, 0x0004, 0x0008, 0x0020, 0x0400,
                      0x1000, 0x2000, 0x4000, 0x0010, 0x0040, 0x0080)
try:
    from rl.memory import schema as _schema  # type: ignore

    BUTTON_BITS = tuple(int(_schema.BUTTON_BITS[name]) for name in BUTTONS)
except Exception:  # standalone fallback (same values)
    BUTTON_BITS = _LOCAL_BUTTON_BITS

FIELDS = (
    "entry_id", "kind", "stock", "damage", "x", "y", "facing",
    "action_state", "action_frame", "on_ground", "is_actionable",
    "jumps_left", "vx", "vy", "dist_to_blast", "dx", "dy", "facing_rel",
    "is_cpu", "is_on_rebirth", "shield", "shield_size", "is_enable_cancel",
    "hitstun", "buttons", "stick_x", "stick_y", "trigger", "cstick_x",
    "cstick_y", "raw_flags", "pad",
)
FIELD_INDEX = {name: i for i, name in enumerate(FIELDS)}

STICK_BINS = 9
N_ACTIONS = STICK_BINS * STICK_BINS
FRAME_SKIP = 5
#: Damage is recorded as a percentage; "bucket" mode feeds the 10 % bucket
#: index instead of the raw value so that 83 % and 87 % are the same input.
DAMAGE_BUCKET = 10.0
DAMAGE_BUCKETS = 20

# Per-fighter observation layout (CONTRACTS.md profile v1).  Agent C's
# gym_obs.py is the single source of truth for the value scaling and the
# curated action_state one-hot; import it when present and fall back to the
# same table so training and the live env produce identical vectors.
_LOCAL_SCALARS = (
    ("x", 100.0), ("y", 100.0), ("vx", 50.0), ("vy", 50.0),
    ("damage", 100.0), ("facing", 1.0), ("on_ground", 1.0),
    ("is_actionable", 1.0), ("jumps_left", 2.0), ("action_frame", 60.0),
    ("shield", 1.0), ("dist_to_blast", 300.0), ("hitstun", 1.0),
    ("is_on_rebirth", 1.0),
)  # 14
_LOCAL_RELATIVE = (
    ("dx", 100.0), ("dy", 100.0), ("distance", 200.0), ("facing_rel", 1.0),
)
# same curated order as gym_obs.CURATED_ACTION_STATES (31 states + OTHER)
_LOCAL_CURATED_STATES = (
    0, 1, 2, 4, 6, 9, 11, 12, 13, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 35, 36, 37, 38, 43, 54, 55, 56, 63, 74,
)
# same families as gym_obs.FAMILY_FLAGS (status ids from schema.ACTION_STATE_NAMES)
_LOCAL_FAMILIES = (
    ("is_escape", (78, 79, 80)),
    ("is_tech", (71, 72)),
    ("is_ledge", (73, 74, 75, 76, 77)),
    ("is_catch", (43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53)),
    ("is_landing", (84, 85, 86)),
    ("is_shieldbreak", (124, 125, 126, 127, 128, 129, 130)),
)
try:  # Agent C's module is the source of truth when it is importable.
    from rl.env import gym_obs as _gym_obs  # type: ignore

    SCALAR_FIELDS = tuple((n, float(s)) for n, s in _gym_obs.SCALAR_FIELDS)
    RELATIVE_FIELDS = tuple((n, float(s)) for n, s in _gym_obs.RELATIVE_FIELDS)
    CURATED_ACTION_STATES = tuple(
        int(s) for _n, s in _gym_obs.CURATED_ACTION_STATES)
    FAMILY_FLAGS = tuple((str(n), tuple(int(x) for x in ids))
                         for n, ids in _gym_obs.FAMILY_FLAGS)
    DAMAGE_BUCKET = float(getattr(_gym_obs, "DAMAGE_BUCKET", DAMAGE_BUCKET))
    DAMAGE_BUCKETS = int(getattr(_gym_obs, "DAMAGE_BUCKETS", DAMAGE_BUCKETS))
    OBS_PROFILE = "gym_obs"
except Exception:  # standalone fallback (identical values)
    SCALAR_FIELDS = _LOCAL_SCALARS
    RELATIVE_FIELDS = _LOCAL_RELATIVE
    CURATED_ACTION_STATES = _LOCAL_CURATED_STATES
    FAMILY_FLAGS = _LOCAL_FAMILIES
    OBS_PROFILE = "local"

ACTION_ONEHOT_SIZE = 32
OTHER_INDEX = ACTION_ONEHOT_SIZE - 1
_STATE_LOOKUP = np.full(512, OTHER_INDEX, dtype=np.int64)
for _i, _state in enumerate(CURATED_ACTION_STATES):
    if 0 <= _state < len(_STATE_LOOKUP):
        _STATE_LOOKUP[_state] = _i
del _i, _state

_ONEHOT_SLICE = slice(14, 14 + ACTION_ONEHOT_SIZE)
_DX = 46
_DY = 47
_DIST = 48
_FACING_REL = 49
OBS_PER_FIGHTER = (len(SCALAR_FIELDS) + ACTION_ONEHOT_SIZE
                   + len(RELATIVE_FIELDS) + len(FAMILY_FLAGS))
OBS_DIM = 2 * OBS_PER_FIGHTER
OBS_LIMIT = 1.0e6
# Fields negated by a horizontal mirror (per fighter block).
_MIRROR_NEG = (0, 2, 5, _DX)   # x, vx, facing, dx
MIRROR_BUTTONS = (BUTTON_INDEX["DPAD_L"], BUTTON_INDEX["DPAD_R"])

EPISODE_RE = re.compile(r"^episode_(\d+)\.npz$")


class _JsonEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def jsonable(obj):
    return json.loads(json.dumps(obj, cls=_JsonEncoder))


# ---------------------------------------------------------------------------
# action helpers
# ---------------------------------------------------------------------------
def decode_buttons(mask, bits=BUTTON_BITS) -> np.ndarray:
    """Decode one or many raw masks into the canonical 12-vector."""
    m = np.asarray(mask)
    out = np.zeros(m.shape + (N_BUTTONS,), dtype=np.float32)
    for i, bit in enumerate(bits):
        out[..., i] = ((m & bit) != 0).astype(np.float32)
    return out


def encode_buttons(vector, bits=BUTTON_BITS) -> np.ndarray:
    v = np.asarray(vector)
    mask = np.zeros(v.shape[:-1], dtype=np.uint32)
    for i, bit in enumerate(bits):
        mask = mask | np.where(v[..., i] > 0.5, np.uint32(bit), np.uint32(0))
    return mask.astype(np.uint32)


def bin_centers(bins: int = STICK_BINS) -> np.ndarray:
    return np.linspace(-1.0, 1.0, bins, dtype=np.float32)


def stick_to_bin(v, bins: int = STICK_BINS) -> np.ndarray:
    """Map continuous (..., 2) stick values to flat bin indices."""
    v = np.asarray(v, dtype=np.float32)
    x = np.clip((v[..., 0] + 1.0) * 0.5, 0.0, 1.0) * (bins - 1)
    y = np.clip((v[..., 1] + 1.0) * 0.5, 0.0, 1.0) * (bins - 1)
    xi = np.rint(x).astype(np.int64)
    yi = np.rint(y).astype(np.int64)
    return xi * bins + yi


#: Coarse stick scheme: neutral + 8 compass directions (the expert stick is
#: binary in practice: 58.7 % |v| < 0.05, 37.8 % |v| >= 0.7, 3.5 % between).
STICK_DIRS = 9
STICK_MAGS = 3
STICK_NEUTRAL = 0.05
STICK_FULL = 0.70
MAG_SCALE = (0.0, 0.5, 1.0)
_DIR_VEC = np.array([[0.0, 0.0], [1.0, 0.0], [0.7071, 0.7071], [0.0, 1.0],
                     [-0.7071, 0.7071], [-1.0, 0.0], [-0.7071, -0.7071],
                     [0.0, -1.0], [0.7071, -0.7071]], np.float32)


def stick_to_dir(v):
    """(..., 2) stick -> (direction 0..8, magnitude 0..2).

    Direction 0 is neutral; 1..8 are E, NE, N, NW, W, SW, S, SE (45 degree
    sectors).  Magnitude is 0 neutral, 1 partial, 2 full deflection.
    """
    v = np.asarray(v, np.float32)
    x, y = v[..., 0], v[..., 1]
    m = np.hypot(x, y)
    x = np.where(np.isfinite(x), x, 0.0)
    y = np.where(np.isfinite(y), y, 0.0)
    ang = np.degrees(np.arctan2(y, x))
    # the +22.5 offset can land on exactly 360 for a boundary angle; clamp the
    # sector so the direction stays inside 0..8
    sector = np.clip(np.floor(((ang + 22.5) % 360.0) / 45.0).astype(np.int64),
                     0, 7)
    neutral = np.hypot(x, y) < STICK_NEUTRAL
    d = np.where(neutral, 0, sector + 1).astype(np.int64)
    mag = np.where(neutral, 0, np.where(m >= STICK_FULL, 2, 1)).astype(np.int64)
    return d, mag


def dir_to_xy(d, mag) -> np.ndarray:
    d = np.asarray(d, np.int64)
    mag = np.asarray(mag, np.int64)
    return _DIR_VEC[d] * np.asarray(MAG_SCALE, np.float32)[mag, None]


def stick_targets(v, mode: str = "grid"):
    """Target pair (primary, magnitude-or-None) for a stick array and a mode."""
    if mode == "dir":
        d, m = stick_to_dir(v)
        return d, m
    return stick_to_bin(v), None


def bin_to_xy(flat, bins: int = STICK_BINS) -> np.ndarray:
    flat = np.asarray(flat, dtype=np.int64)
    centers = bin_centers(bins)
    return np.stack([centers[flat // bins], centers[flat % bins]], axis=-1)


def soft_argmax_xy(logits: torch.Tensor, bins: int = STICK_BINS,
                   temperature: float = 1.0) -> torch.Tensor:
    """Expected (x, y) of a (..., bins*bins) heatmap logits tensor."""
    shape = logits.shape[:-1]
    p = torch.softmax(logits.reshape(*shape, bins * bins) / temperature, dim=-1)
    p = p.reshape(*shape, bins, bins)
    centers = torch.linspace(-1.0, 1.0, bins, dtype=logits.dtype,
                             device=logits.device)
    x = (p.sum(dim=-1) * centers).sum(dim=-1)
    y = (p.sum(dim=-2) * centers).sum(dim=-1)
    return torch.stack([x, y], dim=-1)


# ---------------------------------------------------------------------------
# observation helpers
# ---------------------------------------------------------------------------
def damage_bucket(value):
    """10 % damage bucket index, clipped to ``DAMAGE_BUCKETS - 1``."""
    v = np.asarray(value, np.float32)
    b = np.floor(np.nan_to_num(v, nan=0.0) / DAMAGE_BUCKET)
    return np.clip(b, 0.0, DAMAGE_BUCKETS - 1.0).astype(np.float32)


def obs_from_raw(raw, normalize: bool = False,
                 damage_mode: str = "raw") -> np.ndarray:
    """Build the 100-dim policy observation from (..., 2, 32) raw slots.

    Mirrors ``gym_obs.PolicyObservation(normalize=normalize)`` exactly:
    raw physical values by default (Agent C's default, and therefore what the
    live env feeds the policy), or the CONTRACTS.md divisors with
    ``normalize=True``.  ``action_state`` uses C's curated 32-entry one-hot
    (31 named states + OTHER), followed by the six family flags
    (escape/tech/ledge/catch/landing/shieldbreak); an unreadable slot is all
    zeros.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-2:] != (2, len(FIELDS)):
        raise ValueError(f"raw_fighters must be (..., 2, {len(FIELDS)}); "
                         f"got {raw.shape}")
    if len(SCALAR_FIELDS) + ACTION_ONEHOT_SIZE + len(RELATIVE_FIELDS) + \
            len(FAMILY_FLAGS) != OBS_PER_FIGHTER:
        raise RuntimeError("observation profile does not match OBS_PER_FIGHTER")
    lead = raw.shape[:-2]
    blocks = []
    for k in range(2):
        vals = raw[..., k, :]
        # gym_obs.fighter_valid: raw_flags != 0, or any field other than
        # entry_id/kind is non-zero.  That is exactly "any of words 2..31".
        valid = np.any(vals[..., 2:] != 0, axis=-1)
        b = np.zeros(lead + (OBS_PER_FIGHTER,), dtype=np.float32)

        pos = 0
        for name, scale in SCALAR_FIELDS:
            value = vals[..., FIELD_INDEX[name]]
            if name == "damage" and damage_mode != "raw":
                if damage_mode == "bucket":
                    b[..., pos] = damage_bucket(value)
                # "none": the slot stays zero
            else:
                b[..., pos] = value / scale if normalize else value
            pos += 1

        states = vals[..., FIELD_INDEX["action_state"]].astype(np.int64)
        idx = np.full(states.shape, OTHER_INDEX, dtype=np.int64)
        in_range = (states >= 0) & (states < len(_STATE_LOOKUP))
        idx[in_range] = _STATE_LOOKUP[states[in_range]]
        oh = np.zeros(lead + (ACTION_ONEHOT_SIZE,), dtype=np.float32)
        np.put_along_axis(oh, idx[..., None], 1.0, axis=-1)
        b[..., _ONEHOT_SLICE] = oh
        pos += ACTION_ONEHOT_SIZE

        dx = vals[..., FIELD_INDEX["dx"]]
        dy = vals[..., FIELD_INDEX["dy"]]
        distance = np.hypot(dx, dy)
        rel = (dx, dy, distance, vals[..., FIELD_INDEX["facing_rel"]])
        for value, (name, scale) in zip(rel, RELATIVE_FIELDS):
            del name
            b[..., pos] = value / scale if normalize else value
            pos += 1

        for _name, ids in FAMILY_FLAGS:            # family flags from the status id
            b[..., pos] = np.isin(states, ids).astype(np.float32)
            pos += 1

        np.clip(b, -OBS_LIMIT, OBS_LIMIT, out=b)
        b[~valid] = 0.0
        blocks.append(b)
    return np.concatenate(blocks, axis=-1).astype(np.float32)


def mirror_obs(obs: np.ndarray) -> np.ndarray:
    """Horizontal mirror of one or many observation vectors."""
    o = np.array(obs, dtype=np.float32, copy=True)
    for base in (0, OBS_PER_FIGHTER):
        for idx in _MIRROR_NEG:
            o[..., base + idx] = -o[..., base + idx]
    return o


def mirror_action_arrays(acts: dict) -> dict:
    """Mirror action arrays (same shapes as the episode action dict)."""
    out = {k: np.array(v, copy=True) for k, v in acts.items()}
    if "buttons" in out:
        out["buttons"] = _swap_dpad(out["buttons"])
    if "stick" in out:
        out["stick"][..., 0] *= -1.0
    if "cstick" in out:
        out["cstick"][..., 0] *= -1.0
    return out


# ---------------------------------------------------------------------------
# joint action vocabulary
# ---------------------------------------------------------------------------
#: The learned token that means "emit the previous action again".  It is a
#: token like any other, so the policy can *choose* to hold instead of jitter.
REPEAT_TOKEN = 0
#: Compass mirror across the vertical axis: E<->W, NE<->NW, SW<->SE.
_MIRROR_DIR = np.array([0, 5, 4, 3, 2, 1, 8, 7, 6], np.int64)


def action_keys(acts: dict, trigger_threshold: float = 0.5) -> np.ndarray:
    """One integer per decision: buttons | dir<<12 | mag<<16 | trigger<<19.

    The 12 button bits are dense (canonical order), so a key is a complete,
    discrete action.  ``--stick-mode dir`` already makes the expert's stick
    discrete (96 % of frames are neutral or full deflection), so this loses
    almost nothing and turns the action space into a small vocabulary.
    """
    btn = np.asarray(acts["buttons"], np.float32)
    if btn.ndim != 2 or btn.shape[1] != N_BUTTONS:
        raise ValueError(f"buttons must be (N, {N_BUTTONS}), got {btn.shape}")
    on = btn >= 0.5
    mask = np.zeros(len(btn), np.int64)
    for i in range(N_BUTTONS):
        mask |= on[:, i].astype(np.int64) << i
    d, m = stick_to_dir(acts["stick"])
    trig = np.asarray(acts["trigger"], np.float32).reshape(-1) >= trigger_threshold
    return (mask | (d.astype(np.int64) << 12) | (m.astype(np.int64) << 16)
            | (trig.astype(np.int64) << 19))


def key_parts(key):
    """``(buttons, dir, mag, trigger)`` for one key or an array of keys."""
    k = np.asarray(key, np.int64)
    return k & 0xFFF, (k >> 12) & 0xF, (k >> 16) & 0x3, (k >> 19) & 0x1


def mirror_key(key):
    """Key of the horizontally mirrored action (DPAD L/R swap, stick x flip)."""
    k = np.asarray(key, np.int64)
    mask, d, m, t = key_parts(k)
    left = np.int64(1) << BUTTON_INDEX["DPAD_L"]
    right = np.int64(1) << BUTTON_INDEX["DPAD_R"]
    new = (mask & ~(left | right)) | ((mask & left) << 1) | ((mask & right) >> 1)
    md = _MIRROR_DIR[np.clip(d, 0, 8)]
    return new | (md << 12) | (m << 16) | (t << 19)


def _key_distance(a, b) -> np.ndarray:
    """Rough action distance: buttons dominate, then direction, then magnitude."""
    am, ad, amg, at = key_parts(a)
    bm, bd, bmg, bt = key_parts(b)
    xor = am ^ bm
    hamming = np.zeros_like(xor)
    for i in range(N_BUTTONS):
        hamming = hamming + ((xor >> i) & 1)
    dd = np.abs(ad - bd)
    dd = np.minimum(dd, 8 - dd)
    return 4.0 * hamming + 2.0 * dd + 3.0 * np.abs(amg - bmg) + np.abs(at - bt)


@dataclass
class ActionVocab:
    """The observed action tuples plus the repeat token (index 0).

    Built from the *training* episodes only, over both the recorded and the
    horizontally mirrored actions, so the mirror augmentation can never produce
    a token the vocabulary cannot express.
    """

    keys: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    def __post_init__(self):
        self.keys = np.asarray(self.keys, np.int64).reshape(-1)
        order = np.argsort(self.keys)
        self._sorted = self.keys[order]
        self._tokens = (order + 1).astype(np.int64)

    @property
    def size(self) -> int:
        return len(self.keys) + 1

    @classmethod
    def build(cls, episodes, size: int = 64) -> "ActionVocab":
        counts: dict[int, int] = {}
        for ep in episodes:
            for acts in (ep.acts, mirror_action_arrays(ep.acts)):
                for key, n in zip(*np.unique(action_keys(acts),
                                             return_counts=True)):
                    counts[int(key)] = counts.get(int(key), 0) + int(n)
        room = max(1, int(size) - 1)
        ordered = sorted(counts, key=lambda k: -counts[k])
        kept = ordered[:room]
        kept_set = set(kept)
        for key in ordered[:room]:
            mirror = int(mirror_key(key))
            if mirror not in kept_set and len(kept) < room:
                kept.append(mirror)
                kept_set.add(mirror)
        keys = np.asarray(kept, np.int64)
        return cls(keys=keys)

    def encode(self, keys) -> np.ndarray:
        """Keys -> token indices; unseen keys snap to the nearest known one."""
        k = np.asarray(keys, np.int64)
        tok = np.zeros(k.shape, np.int64)
        if not len(self.keys):
            return tok
        pos = np.clip(np.searchsorted(self._sorted, k), 0, len(self._sorted) - 1)
        hit = self._sorted[pos] == k
        tok[hit] = self._tokens[pos[hit]]
        if not hit.all():
            missing = np.unique(k[~hit])
            remap = {int(v): self._nearest(int(v)) for v in missing}
            tok[~hit] = np.asarray([remap[int(v)] for v in k[~hit]], np.int64)
        return tok

    def _nearest(self, key: int) -> int:
        return int(np.argmin(_key_distance(self.keys, np.int64(key)))) + 1

    def decode(self, token: int) -> np.ndarray:
        """Token -> 17-vector ``[buttons(12), stick(2), cstick(2), trigger(1)]``.

        ``REPEAT_TOKEN`` decodes to zeros; callers resolve it against the
        previous action.
        """
        out = np.zeros(12 + 2 + 2 + 1, np.float32)
        if int(token) == REPEAT_TOKEN or not len(self.keys):
            return out
        key = self.keys[min(int(token) - 1, len(self.keys) - 1)]
        mask, d, m, t = key_parts(key)
        for i in range(N_BUTTONS):
            out[i] = float((int(mask) >> i) & 1)
        out[12:14] = dir_to_xy(np.asarray([d]), np.asarray([m]))[0]
        out[16] = float(t)
        return out

    def to_dict(self) -> dict:
        return {"keys": [int(k) for k in self.keys]}

    @classmethod
    def from_dict(cls, payload: dict) -> "ActionVocab":
        return cls(keys=np.asarray(payload.get("keys", []), np.int64))


def decode_tokens(tokens, prev, vocab: ActionVocab) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tokens -> ``(buttons (N,12), dir (N,), mag (N,))``; repeat uses ``prev``."""
    tokens = np.asarray(tokens, np.int64).reshape(-1)
    prev = np.asarray(prev, np.float32)
    n = len(tokens)
    if prev.ndim == 1:
        prev = prev[None, :]
    if len(prev) == 1 and n > 1:
        prev = np.repeat(prev, n, axis=0)
    btn = np.zeros((n, N_BUTTONS), np.float32)
    d = np.zeros(n, np.int64)
    mag = np.zeros(n, np.int64)
    rep = tokens == REPEAT_TOKEN
    if rep.any():
        btn[rep] = (prev[rep, :N_BUTTONS] >= 0.5).astype(np.float32)
        d[rep], mag[rep] = stick_to_dir(prev[rep, N_BUTTONS:N_BUTTONS + 2])
    used = ~rep
    if used.any():
        key = np.asarray(vocab.keys)[np.clip(tokens[used] - 1, 0,
                                             max(0, len(vocab.keys) - 1))]
        mask, dd, mm, _t = key_parts(key)
        for i in range(N_BUTTONS):
            btn[used, i] = ((mask >> i) & 1).astype(np.float32)
        d[used] = dd
        mag[used] = mm
    return btn, d, mag


def action_vector(action: dict, cfg) -> np.ndarray:
    """The ``prev`` vector for an action dict (layout of ``prev_action_vectors``)."""
    parts = [np.asarray(action["buttons"], np.float32),
             np.asarray(action["stick"], np.float32)]
    if getattr(cfg, "cstick", False):
        parts.append(np.asarray(action.get("cstick", np.zeros(2)), np.float32))
    if getattr(cfg, "trigger", True):
        parts.append(np.asarray([action.get("trigger", 0.0)], np.float32))
    return np.concatenate(parts).astype(np.float32)


# ---------------------------------------------------------------------------
# episode loading / dataset construction
# ---------------------------------------------------------------------------
@dataclass
class EpisodeData:
    index: int
    path: str
    meta: dict
    raw: np.ndarray           # (N, 2, 32)
    buttons_raw: np.ndarray   # (N, 2)
    stick: np.ndarray         # (N, 2, 2)
    cstick: np.ndarray        # (N, 2, 2)
    trigger: np.ndarray       # (N, 2)
    player: int = 0
    shield_from_status: bool = True
    damage_mode: str = "raw"
    obs: np.ndarray = field(default=None, repr=False)
    acts: dict = field(default=None, repr=False)

    def __post_init__(self):
        if self.obs is None:
            self.obs = obs_from_raw(self.raw, damage_mode=self.damage_mode)
        if self.acts is None:
            self.acts = self._actions()

    def guard_mask(self) -> np.ndarray:
        """True on frames whose status is the GUARD family (8/9/10)."""
        states = self.raw[:, self.player,
                          FIELD_INDEX["action_state"]].astype(np.int64)
        return np.isin(states, GUARD_STATES)

    def _actions(self) -> dict:
        p = self.player
        buttons = decode_buttons(self.buttons_raw[:, p])
        if self.shield_from_status:
            guard = self.guard_mask()
            if np.any(guard):
                # The CPU mask's 0x8004 carries a spurious X (jump) on guard
                # frames; deploy the shield as ZL (the virtual pad maps ZL to
                # the Pro Controller ZL).  The raw mask is not modified.
                buttons[guard, SHIELD_BUTTON] = 1.0
                buttons[guard, BUTTON_INDEX["X"]] = 0.0
        return {
            "buttons": buttons,
            "stick": self.stick[:, p].astype(np.float32),
            "cstick": self.cstick[:, p].astype(np.float32),
            "trigger": self.trigger[:, p].astype(np.float32),
        }

    def with_player(self, player: int) -> "EpisodeData":
        return EpisodeData(self.index, self.path, self.meta, self.raw,
                           self.buttons_raw, self.stick, self.cstick,
                           self.trigger, player=player,
                           shield_from_status=self.shield_from_status,
                           damage_mode=self.damage_mode,
                           obs=self.obs, acts=None)


def _read_meta(data, path: str) -> dict:
    meta = {}
    for key in ("meta_json", "meta"):
        if key in data:
            try:
                meta = json.loads(str(data[key]))
            except (ValueError, TypeError):
                meta = {}
            break
    match = EPISODE_RE.match(os.path.basename(path))
    file_index = int(match.group(1)) if match else -1
    if not meta:
        manifest = os.path.join(os.path.dirname(path), "manifest.json")
        if os.path.exists(manifest):
            try:
                with open(manifest, encoding="utf-8") as fh:
                    m = json.load(fh)
                for entry in m.get("episodes", []):
                    if int(entry.get("index", -1)) == file_index:
                        meta = dict(entry.get("meta", {}))
                        meta["result"] = entry.get("result", {})
                        break
            except (OSError, ValueError, TypeError):
                pass
    meta.setdefault("index", file_index)
    meta.setdefault("file", os.path.basename(path))
    return meta


def load_episode(path: str, player: int = 0, normalize_obs: bool = False,
                 shield_from_status: bool = True,
                 damage_mode: str = "raw") -> EpisodeData:
    with np.load(path, allow_pickle=False) as data:
        for key in ("raw_fighters", "buttons_raw", "stick", "cstick",
                    "trigger", "seq"):
            if key not in data:
                raise ValueError(f"{path}: missing key {key!r}")
        raw = data["raw_fighters"].astype(np.float32)
        buttons_raw = data["buttons_raw"].astype(np.uint32)
        stick = data["stick"].astype(np.float32)
        cstick = data["cstick"].astype(np.float32)
        trigger = data["trigger"].astype(np.float32)
        seq = data["seq"].astype(np.uint64)
        meta = _read_meta(data, path)
        if "seq_gaps" in data:
            meta["seq_gaps"] = int(data["seq_gaps"])
        else:
            meta["seq_gaps"] = int(np.count_nonzero(np.diff(seq) != 2)) \
                if len(seq) > 1 else 0
    if raw.ndim != 3 or raw.shape[1:] != (2, len(FIELDS)):
        raise ValueError(f"{path}: raw_fighters shape {raw.shape}, "
                         f"expected (N, 2, {len(FIELDS)})")
    if buttons_raw.shape != (len(raw), 2):
        raise ValueError(f"{path}: buttons_raw shape {buttons_raw.shape}")
    return EpisodeData(meta["index"], path, meta, raw, buttons_raw, stick,
                       cstick, trigger, player=player,
                       shield_from_status=shield_from_status,
                       damage_mode=damage_mode,
                       obs=obs_from_raw(raw, normalize=normalize_obs,
                                        damage_mode=damage_mode))


def load_episodes(data_dir: str, player: int = 0, limit: int | None = None,
                  normalize_obs: bool = False, shield_from_status: bool = True,
                  damage_mode: str = "raw", dedupe: bool = True,
                  log=print) -> list[EpisodeData]:
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data dir not found: {data_dir}")
    paths = []
    for name in sorted(os.listdir(data_dir)):
        if EPISODE_RE.match(name):
            paths.append(os.path.join(data_dir, name))
    if not paths:
        raise FileNotFoundError(f"no episode_*.npz recordings in {data_dir}")
    if dedupe:
        # Identical recordings happen (the same match captured twice); left in,
        # a duplicate pair can straddle the train/val split and make validation
        # optimistic.  Compare the published fighter frames, not the file bytes
        # (timestamps differ).
        keep, seen = [], set()
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                digest = hashlib.sha1(
                    np.ascontiguousarray(data["raw_fighters"]).tobytes()
                ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            keep.append(path)
        if len(keep) != len(paths):
            log(f"[bc] dedupe: dropped {len(paths) - len(keep)} of "
                f"{len(paths)} episodes with identical content")
        paths = keep
    if limit is not None:
        paths = paths[:limit]
    episodes = []
    for path in paths:
        try:
            ep = load_episode(path, player=player,
                              normalize_obs=normalize_obs,
                              shield_from_status=shield_from_status,
                              damage_mode=damage_mode)
        except Exception as exc:
            log(f"[bc] skipping {path}: {exc}")
            continue
        if len(ep.raw) < FRAME_SKIP:
            log(f"[bc] skipping {path}: only {len(ep.raw)} frames")
            continue
        episodes.append(ep)
    if not episodes:
        raise ValueError(f"no usable episodes in {data_dir}")
    n_btn = int(np.count_nonzero(episodes[0].buttons_raw))
    if n_btn == 0:
        log("[bc] warning: all raw button masks are zero; the recording has no "
            "v7 control fields (or no expert input), actions are no-ops")
    return episodes


def resample_indices(n: int, frame_skip: int = FRAME_SKIP) -> np.ndarray:
    return np.arange(0, n, frame_skip, dtype=np.int64)


def hold_actions(acts: dict, frame_skip: int = FRAME_SKIP) -> dict:
    """Label each decision with what the expert did over the whole hold.

    The env writes one action and advances ``frame_skip`` frames, so the policy
    physically cannot change input inside that window.  Labelling the decision
    with a single frame instead loses every tap that lands on the other frame
    (23 % of A presses, 56 % of B) and makes the target look noisy.  Buttons are
    OR-ed; the stick and C-stick take the largest deflection in the window.
    """
    n = len(acts["buttons"])
    skip = max(1, int(frame_skip))
    m = (n + skip - 1) // skip
    need = m * skip

    def pad(x):
        x = np.asarray(x)
        if need == n:
            return x
        return np.concatenate([x, np.repeat(x[-1:], need - n, axis=0)], axis=0)

    btn = pad(acts["buttons"]).reshape(m, skip, -1)
    st = pad(acts["stick"]).reshape(m, skip, 2)
    cs = pad(acts["cstick"]).reshape(m, skip, 2)
    tr = pad(np.asarray(acts["trigger"]).reshape(-1, 1)).reshape(m, skip)
    rows = np.arange(m)
    out = {
        "buttons": btn.max(axis=1).astype(np.float32),
        "stick": st[rows, np.argmax(np.hypot(st[..., 0], st[..., 1]), axis=1)],
        "cstick": cs[rows, np.argmax(np.hypot(cs[..., 0], cs[..., 1]), axis=1)],
        "trigger": tr.max(axis=1).astype(np.float32),
    }
    return out


def resample_episode(ep: EpisodeData, frame_skip: int = FRAME_SKIP,
                     label_mode: str = "single") -> EpisodeData:
    idx = resample_indices(len(ep.raw), frame_skip)
    if label_mode == "hold":
        acts = hold_actions(ep.acts, frame_skip)
    elif label_mode == "single":
        acts = {k: v[idx] for k, v in ep.acts.items()}
    else:
        raise ValueError(f"label_mode must be single or hold, got {label_mode!r}")
    return EpisodeData(
        ep.index, ep.path, ep.meta, ep.raw[idx], ep.buttons_raw[idx],
        ep.stick[idx], ep.cstick[idx], ep.trigger[idx], player=ep.player,
        shield_from_status=ep.shield_from_status,
        damage_mode=ep.damage_mode,
        obs=ep.obs[idx],
        acts=acts,
    )


def prev_action_vectors(acts: dict, include_cstick: bool,
                        include_trigger: bool) -> np.ndarray:
    parts = [acts["buttons"], acts["stick"]]
    if include_cstick:
        parts.append(acts["cstick"])
    if include_trigger:
        parts.append(acts["trigger"][:, None])
    vec = np.concatenate(parts, axis=-1).astype(np.float32)
    return np.vstack([np.zeros((1, vec.shape[1]), np.float32), vec[:-1]])


def decision_rewards(ep: "EpisodeData", scale: float = 0.01,
                     ego: int | None = None) -> np.ndarray:
    """Damage differential per decision, vectorised off ``ep.raw``.

    ``scale * (damage dealt - damage taken)``: a KO resets damage to 0, so the
    clamp at 0 stops it counting as healing.  Stock/KO/win terms are off by
    design — with 255-stock recordings they never fire, and damage is the
    progress signal we want ("damage is stocks").

    This is the offline twin of ``rl.env.gym_reward.compute_reward`` with the
    same weights; ``test_rewards_match_gym_reward`` pins them together.
    """
    dmg = np.asarray(ep.raw, np.float32)[:, :, FIELD_INDEX["damage"]]
    d = np.diff(np.vstack([dmg[:1], dmg]), axis=0)
    ego = int(ep.player if ego is None else ego)
    opp = 1 - ego
    dealt = np.clip(d[:, opp], 0.0, None)
    taken = np.clip(d[:, ego], 0.0, None)
    return (float(scale) * (dealt - taken)).astype(np.float32)


def return_to_go(rewards: np.ndarray, gamma: float = 0.995) -> np.ndarray:
    """Discounted return from each decision to the end of the episode."""
    g = np.zeros(len(rewards), np.float32)
    acc = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        acc = float(rewards[i]) + float(gamma) * acc
        g[i] = acc
    return g


def return_weights(rewards: np.ndarray, gamma: float = 0.995, beta: float = 4.0,
                   quantile: float = 0.5, w_min: float = 0.1,
                   w_max: float = 10.0) -> np.ndarray:
    """Conservative return weighting for behaviour cloning.

    ``w = clip(exp(beta * (G - quantile(G))), w_min, w_max)``, normalised to
    mean 1.  The quantile baseline (instead of the mean) and the clip are what
    make it conservative: a handful of lucky decisions cannot dominate the
    gradient, and no sample is dropped entirely.
    """
    if len(rewards) == 0:
        return np.zeros(0, np.float32)
    g = return_to_go(rewards, gamma)
    base = float(np.quantile(g, float(quantile)))
    w = np.exp(float(beta) * (g - base))
    w = np.clip(w, float(w_min), float(w_max))
    mean = float(w.mean())
    return (w / mean if mean > 0 else w).astype(np.float32)


def _empty_targets(horizon: int, stick_mode: str) -> dict:
    t = {"buttons": np.zeros((0, horizon, N_BUTTONS), np.float32),
         "stick": np.zeros((0, horizon), np.int64),
         "cstick": np.zeros((0, horizon), np.int64),
         "trigger": np.zeros((0, horizon), np.float32),
         "action": np.zeros((0, horizon), np.int64)}
    if stick_mode == "dir":
        t["stick_mag"] = np.zeros((0, horizon), np.int64)
    return t


def window_samples(obs: np.ndarray, acts: dict, window: int = 8,
                   horizon: int = 4, include_cstick: bool = False,
                   include_trigger: bool = True, stick_mode: str = "grid",
                   frame_weights: np.ndarray | None = None,
                   window_dtype: Any = np.float32,
                   vocab: ActionVocab | None = None) -> dict:
    """Build aligned samples for one episode (never crosses an episode).

    Window at decision ``t`` = observations ``t-window+1 .. t``; previous
    action = ``a[t-1]``; targets = ``a[t .. t+horizon-1]``.
    """
    n = len(obs)
    if window < 1 or horizon < 1:
        raise ValueError("window and horizon must be >= 1")
    count = n - window - horizon + 2
    if count <= 0:
        empty = np.zeros((0, window, obs.shape[1]), np.float32)
        out = {"windows": empty,
               "prev": np.zeros((0, 12 + 2 + (2 if include_cstick else 0) +
                                 (1 if include_trigger else 0)), np.float32),
               "targets": _empty_targets(horizon, stick_mode)}
        if frame_weights is not None:
            out["weights"] = np.zeros(0, np.float32)
        return out
    t = np.arange(window - 1, n - horizon + 1)
    windows = np.stack([obs[i - window + 1:i + 1] for i in t]).astype(window_dtype)
    prev = prev_action_vectors(acts, include_cstick, include_trigger)[t]
    buttons = np.stack([acts["buttons"][i:i + horizon] for i in t]).astype(np.float32)
    cstick = np.stack([stick_to_bin(acts["cstick"][i:i + horizon]) for i in t])
    trigger = np.stack([acts["trigger"][i:i + horizon] for i in t]).astype(np.float32)
    targets = {"buttons": buttons,
               "cstick": cstick, "trigger": trigger}
    if stick_mode == "dir":
        d, m = stick_to_dir(acts["stick"])
        targets["stick"] = np.stack([d[i:i + horizon] for i in t])
        targets["stick_mag"] = np.stack([m[i:i + horizon] for i in t])
    else:
        targets["stick"] = np.stack([stick_to_bin(acts["stick"][i:i + horizon]) for i in t])
    if vocab is not None:
        tokens = vocab.encode(action_keys(acts))
        # Label an unchanged decision as the repeat token.  78 % of expert
        # decisions at 30 Hz are the same tuple as the one before, so this
        # turns "hold" into a single class instead of hundreds of near-equal
        # tuples, and it is what lets the head *choose* to hold rather than
        # re-predict the whole tuple every decision.
        same = np.zeros(len(tokens), dtype=bool)
        same[1:] = tokens[1:] == tokens[:-1]
        tokens = np.where(same, REPEAT_TOKEN, tokens)
        targets["action"] = np.stack([tokens[i:i + horizon] for i in t]).astype(np.int64)
    out = {
        "windows": windows,
        "prev": prev.astype(np.float32),
        "targets": targets,
    }
    if frame_weights is not None:
        # the weight of the decision being predicted, i.e. the first target
        out["weights"] = np.asarray(frame_weights, np.float32)[t]
    return out


def concat_samples(parts: list[dict]) -> dict:
    if not parts:
        raise ValueError("no sample parts")
    total = sum(len(p["windows"]) for p in parts)
    windows = np.empty((total,) + parts[0]["windows"].shape[1:],
                       parts[0]["windows"].dtype)
    at = 0
    for p in parts:                      # one copy, then each part can go
        n = len(p["windows"])
        windows[at:at + n] = p["windows"]
        at += n
    out = {
        "windows": windows,
        "prev": np.concatenate([p["prev"] for p in parts], axis=0),
        "targets": {
            key: np.concatenate([p["targets"][key] for p in parts], axis=0)
            for key in parts[0]["targets"]
        },
    }
    if "weights" in parts[0]:
        out["weights"] = np.concatenate([p["weights"] for p in parts], axis=0)
    return out


def ego_swap_obs(obs: np.ndarray) -> np.ndarray:
    """Swap the two fighter blocks (the other player becomes the ego).

    ``dx``/``dy``/``distance``/``facing_rel`` are per-fighter (each block is
    relative to its own opponent), so a block swap is complete: no field needs
    a sign change.  The action target must come from the other player, which
    :func:`ego_swap_episode` does.
    """
    o = np.array(obs, copy=True)
    o[..., :OBS_PER_FIGHTER] = np.asarray(obs)[..., OBS_PER_FIGHTER:]
    o[..., OBS_PER_FIGHTER:] = np.asarray(obs)[..., :OBS_PER_FIGHTER]
    return o


def ego_swap_episode(ep: EpisodeData) -> EpisodeData:
    """Same episode seen from the opponent's seat: blocks swapped, actions from
    the other player.  Doubles the data of a two-player recording; on a mirror
    match (X vs X) it is the same behaviour, so it is free."""
    return EpisodeData(ep.index, ep.path, ep.meta, ep.raw, ep.buttons_raw,
                       ep.stick, ep.cstick, ep.trigger, player=1 - ep.player,
                       shield_from_status=ep.shield_from_status,
                       damage_mode=ep.damage_mode,
                       obs=ego_swap_obs(ep.obs), acts=None)


def mirror_episode(ep: EpisodeData) -> EpisodeData:
    return EpisodeData(ep.index, ep.path, ep.meta, ep.raw, ep.buttons_raw,
                       ep.stick, ep.cstick, ep.trigger, player=ep.player,
                       shield_from_status=ep.shield_from_status,
                       damage_mode=ep.damage_mode,
                       obs=mirror_obs(ep.obs),
                       acts=mirror_action_arrays(ep.acts))


def samples_for(episodes: list[EpisodeData], cfg: "BCConfig",
                mirror: bool = False, ego_swap: bool = False,
                return_beta: float = 0.0, return_gamma: float = 0.995,
                weight_quantile: float = 0.5, weight_min: float = 0.1,
                weight_max: float = 10.0,
                reward_scale: float = 0.01,
                window_dtype: Any = np.float32,
                vocab: ActionVocab | None = None) -> dict:
    """Windows for these episodes; with ``return_beta > 0`` each sample also
    carries a conservative return weight (see :func:`return_weights`)."""
    parts = []
    for ep in episodes:
        variants = [ep]
        if ego_swap:
            variants.append(ego_swap_episode(ep))
        for v in variants:
            fw = None
            if return_beta > 0:
                fw = return_weights(decision_rewards(v, reward_scale),
                                    gamma=return_gamma, beta=return_beta,
                                    quantile=weight_quantile, w_min=weight_min,
                                    w_max=weight_max)
            parts.append(window_samples(v.obs, v.acts, cfg.window, cfg.horizon,
                                        cfg.cstick, cfg.trigger,
                                        stick_mode=cfg.stick_mode,
                                        frame_weights=fw,
                                        window_dtype=window_dtype,
                                        vocab=vocab))
            if mirror:
                mv = mirror_episode(v)
                parts.append(window_samples(mv.obs, mv.acts, cfg.window,
                                            cfg.horizon, cfg.cstick, cfg.trigger,
                                            stick_mode=cfg.stick_mode,
                                            frame_weights=fw,
                                            window_dtype=window_dtype,
                                            vocab=vocab))
    return concat_samples(parts)


def split_episodes(n: int, seed: int = 0,
                   fractions=(0.8, 0.1, 0.1)) -> tuple[np.ndarray, ...]:
    """Deterministic episode-level split with at least one val/test episode."""
    if n < 1:
        raise ValueError("need at least one episode")
    perm = np.random.default_rng(seed).permutation(n)
    if n == 1:
        return perm[:1], perm[:1], perm[:1]
    if n == 2:
        return perm[:1], perm[1:], perm[1:]
    n_val = max(1, int(round(fractions[1] * n)))
    n_test = max(1, int(round(fractions[2] * n)))
    while n - n_val - n_test < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    n_train = n - n_val - n_test
    return (perm[:n_train], perm[n_train:n_train + n_val],
            perm[n_train + n_val:n_train + n_val + n_test])


def norm_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # float16 storage is fine for the network input, not for the statistics
    flat = np.asarray(windows, np.float32).reshape(-1, windows.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(windows: np.ndarray, mean, std) -> np.ndarray:
    return ((windows - mean) / std).astype(np.float32)


def shield_label_stats(episodes: list[EpisodeData]) -> dict:
    """GUARD-family decisions in already-resampled episodes.

    ``frames`` is the number of decisions whose status is 8/9/10 (these become
    ZL=1/X=0 training targets when ``shield_from_status`` is on);
    ``spurious_x_cleared`` counts how many of them had the CPU mask's 0x4 (X)
    bit set in the raw record.
    """
    by_state = {str(s): 0 for s in GUARD_STATES}
    total = 0
    x_cleared = 0
    x_bit = BUTTON_BITS[BUTTON_INDEX["X"]]
    for ep in episodes:
        states = ep.raw[:, ep.player,
                        FIELD_INDEX["action_state"]].astype(np.int64)
        guard = np.isin(states, GUARD_STATES)
        total += int(guard.sum())
        for state in GUARD_STATES:
            by_state[str(state)] += int(np.count_nonzero(states == state))
        if np.any(guard):
            masks = ep.buttons_raw[guard, ep.player]
            x_cleared += int(np.count_nonzero((masks & x_bit) != 0))
    return {"frames": total, "by_state": by_state,
            "spurious_x_cleared": x_cleared}


def merge_shield_stats(stats: list[dict]) -> dict:
    by_state: dict[str, int] = {}
    for item in stats:
        for key, value in item["by_state"].items():
            by_state[key] = by_state.get(key, 0) + int(value)
    return {
        "frames": sum(int(s["frames"]) for s in stats),
        "by_state": by_state,
        "spurious_x_cleared": sum(int(s["spurious_x_cleared"]) for s in stats),
    }


def dataset_action_stats(samples: dict) -> dict:
    """Button positive rates and stick statistics for a sample set."""
    t = samples["targets"]
    b = t["buttons"].reshape(-1, N_BUTTONS) if t["buttons"].size else \
        np.zeros((0, N_BUTTONS), np.float32)
    rate = b.mean(axis=0) if len(b) else np.zeros(N_BUTTONS)
    stick = bin_to_xy(t["stick"].ravel()) if t["stick"].size else \
        np.zeros((0, 2), np.float32)
    cstick = bin_to_xy(t["cstick"].ravel()) if t["cstick"].size else \
        np.zeros((0, 2), np.float32)
    trig = t["trigger"].ravel() if t["trigger"].size else np.zeros(0)
    return {
        "button_rate": {n: float(rate[i]) for i, n in enumerate(BUTTONS)},
        "stick_mean": [float(stick[:, 0].mean()), float(stick[:, 1].mean())]
                      if len(stick) else [0.0, 0.0],
        "stick_abs_mean": [float(np.abs(stick[:, 0]).mean()),
                           float(np.abs(stick[:, 1]).mean())]
                          if len(stick) else [0.0, 0.0],
        "cstick_mean": [float(cstick[:, 0].mean()), float(cstick[:, 1].mean())]
                       if len(cstick) else [0.0, 0.0],
        "trigger_rate": float(trig.mean()) if len(trig) else 0.0,
    }


# ---------------------------------------------------------------------------
# config / model
# ---------------------------------------------------------------------------
@dataclass
class BCConfig:
    obs_dim: int = OBS_DIM
    window: int = 8
    horizon: int = 4
    frame_skip: int = FRAME_SKIP
    hidden: int = 256
    gru_layers: int = 1
    bins: int = STICK_BINS
    stick_mode: str = "grid"          # "grid" = 9x9 heatmap, "dir" = 9 dirs + 3 magnitudes
    cstick: bool = False
    trigger: bool = True
    player: int = 0
    ego_swap: bool = False
    transition_weight: float = 1.0
    obs_normalize: bool = False
    dropout: float = 0.0
    # CPU recordings put a spurious X (0x4) in the guard mask 0x8004; relabel
    # guard-family frames as ZL=1/X=0 for the training target.
    shield_from_status: bool = True
    # "heads": per-button sigmoid + separate stick heads (the original).
    # "vocab": one softmax over the observed joint action tuples + repeat.
    action_mode: str = "heads"
    vocab_size: int = 64
    # "raw": damage in percent, "bucket": 10 % buckets, "none": zeroed out.
    damage_mode: str = "raw"
    # "single": label = the action at the decision frame.
    # "hold": OR over the ``frame_skip`` frames the action is actually held for.
    label_mode: str = "single"

    @property
    def prev_dim(self) -> int:
        return 12 + 2 + (2 if self.cstick else 0) + (1 if self.trigger else 0)

    def to_dict(self) -> dict:
        return asdict(self)


class BCPolicyNet(nn.Module):
    def __init__(self, cfg: BCConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.GRU(cfg.obs_dim, cfg.hidden,
                              num_layers=cfg.gru_layers, batch_first=True)
        self.trunk = nn.Sequential(
            nn.Linear(cfg.hidden + cfg.prev_dim, cfg.hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.ReLU(),
            nn.Dropout(cfg.dropout),
        )
        h, H, bins = cfg.hidden, cfg.horizon, cfg.bins
        if cfg.action_mode == "vocab":
            # One logit per action token: a mutually-exclusive choice, so the
            # distribution can be multimodal without averaging the modes.
            self.action_head = nn.Linear(h, H * int(cfg.vocab_size))
            return
        self.button_head = nn.Linear(h, H * N_BUTTONS)
        self.stick_out = STICK_DIRS if cfg.stick_mode == "dir" else bins * bins
        self.stick_head = nn.Linear(h, H * self.stick_out)
        self.stick_mag_head = nn.Linear(h, H * STICK_MAGS) if cfg.stick_mode == "dir" else None
        self.cstick_head = nn.Linear(h, H * bins * bins) if cfg.cstick else None
        self.trigger_head = nn.Linear(h, H) if cfg.trigger else None

    def forward(self, obs: torch.Tensor, prev: torch.Tensor) -> dict:
        # obs (B, T, obs_dim), prev (B, prev_dim)
        _, h_n = self.encoder(obs)
        z = self.trunk(torch.cat([h_n[-1], prev], dim=-1))
        cfg = self.cfg
        B, H, bins = z.shape[0], cfg.horizon, cfg.bins
        if cfg.action_mode == "vocab":
            return {"action": self.action_head(z).reshape(B, H,
                                                          int(cfg.vocab_size))}
        out = {
            "buttons": self.button_head(z).reshape(B, H, N_BUTTONS),
            "stick": self.stick_head(z).reshape(B, H, self.stick_out),
        }
        if self.stick_mag_head is not None:
            out["stick_mag"] = self.stick_mag_head(z).reshape(B, H, STICK_MAGS)
        if self.cstick_head is not None:
            out["cstick"] = self.cstick_head(z).reshape(B, H, bins * bins)
        if self.trigger_head is not None:
            out["trigger"] = self.trigger_head(z).reshape(B, H)
        return out


class WindowDataset(Dataset):
    """Windows normalized in ``__getitem__``.

    The normalization used to copy the whole array up front, which doubled the
    dataset's memory; a 30 Hz set (1.7M windows x 8 frames x 112) is several GB
    on its own and the copy was what the OOM killer took.
    """

    def __init__(self, samples: dict, mean, std):
        self.windows = samples["windows"]      # the only copy
        self.prev = samples["prev"].astype(np.float32)
        self.targets = samples["targets"]
        w = samples.get("weights")
        self.weights = None if w is None else np.asarray(w, np.float32)
        self.mean = np.asarray(mean, np.float32)
        self.std = np.asarray(std, np.float32)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int):
        # every target key travels (stick_mag exists in the direction stick mode)
        targets = {k: torch.from_numpy(v[i]) for k, v in self.targets.items()}
        window = ((self.windows[i] - self.mean) / self.std).astype(np.float32,
                                                                  copy=False)
        w = 1.0 if self.weights is None else float(self.weights[i])
        return (torch.from_numpy(window), torch.from_numpy(self.prev[i]),
                targets, torch.tensor(w))

    def __len__(self) -> int:
        return len(self.windows)


def compute_pos_weight(buttons: np.ndarray, max_weight: float = 20.0) -> np.ndarray:
    b = buttons.reshape(-1, N_BUTTONS)
    if len(b) == 0:
        return np.ones(N_BUTTONS, np.float32)
    pos = b.sum(axis=0)
    neg = len(b) - pos
    weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    return np.clip(weight, 0.1, max_weight).astype(np.float32)


# ---------------------------------------------------------------------------
# inference policy + temporal ensemble
# ---------------------------------------------------------------------------
def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def soft_argmax_np(logits: np.ndarray, bins: int = STICK_BINS) -> np.ndarray:
    """Expected (x, y) for (..., bins*bins) logits."""
    p = _softmax_np(np.asarray(logits, dtype=np.float32))
    p = p.reshape(p.shape[:-1] + (bins, bins))
    centers = bin_centers(bins)
    x = (p.sum(axis=-1) * centers).sum(axis=-1)
    y = (p.sum(axis=-2) * centers).sum(axis=-1)
    return np.stack([x, y], axis=-1)


def probs_to_action(button_prob: np.ndarray, stick: np.ndarray,
                    cstick: np.ndarray, trigger: float,
                    thresholds: np.ndarray | None = None) -> dict:
    button_prob = np.asarray(button_prob, dtype=np.float32).reshape(N_BUTTONS)
    th = np.full(N_BUTTONS, 0.5, np.float32) if thresholds is None \
        else np.asarray(thresholds, np.float32)
    return {
        "buttons_prob": button_prob,
        "buttons": (button_prob >= th).astype(np.float32),
        "stick": np.asarray(stick, dtype=np.float32).reshape(2),
        "cstick": np.asarray(cstick, dtype=np.float32).reshape(2),
        "trigger": float(trigger),
        "thresholds": th,
    }


def action_to_env(action: dict, kind: str = "array"):
    """Native gym action: a Dict sample, or the 16-vector gym_action accepts.

    ``action_to_env(..., "array")`` is ``[buttons(12), stick(2), cstick(2)]``
    because ``gym_action.decode_action`` rejects longer vectors; the trigger
    is only carried in the dict form (C's action space always uses 0 anyway).
    """
    buttons = np.asarray(action["buttons"], np.float32).reshape(N_BUTTONS)
    stick = np.asarray(action["stick"], np.float32).reshape(2)
    cstick = np.asarray(action.get("cstick", np.zeros(2)), np.float32).reshape(2)
    trigger = float(np.asarray(action.get("trigger", 0.0)).reshape(-1)[0])
    if kind == "dict":
        return {"buttons": buttons, "stick": stick, "cstick": cstick,
                "trigger": trigger}
    return np.concatenate([buttons, stick, cstick]).astype(np.float32)


class BCPolicy:
    """Checkpoint-backed policy; feed it the 100-dim observation each step."""

    def __init__(self, checkpoint: dict, device: str = "cpu",
                 sample: bool = False, temperature: float = 1.0):
        if isinstance(checkpoint, str):
            checkpoint = load_checkpoint(checkpoint)
        cfg = BCConfig(**{k: v for k, v in checkpoint["config"].items()
                          if k in BCConfig.__dataclass_fields__})
        self.cfg = cfg
        self.device = torch.device(device)
        #: ``sample`` decodes by drawing from the predicted distribution instead
        #: of taking the argmax at a tuned threshold.  A BC policy that has
        #: drifted off the expert's states predicts "neutral" almost surely, so
        #: the argmax stands still in closed loop; sampling keeps it moving.
        self.sample = bool(sample)
        #: Softmax temperature for token sampling (<1 = more committed).
        self.temperature = max(1e-3, float(temperature))
        payload = checkpoint.get("vocab")
        self.vocab = ActionVocab.from_dict(payload) if payload else None
        if cfg.action_mode == "vocab" and self.vocab is None:
            raise ValueError("checkpoint has action_mode=vocab but no vocabulary")
        self.net = BCPolicyNet(cfg).to(self.device)
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.eval()
        self.mean = torch.as_tensor(checkpoint["norm"]["mean"],
                                    dtype=torch.float32, device=self.device)
        self.std = torch.as_tensor(checkpoint["norm"]["std"],
                                   dtype=torch.float32, device=self.device)
        self.thresholds = np.asarray(
            checkpoint.get("thresholds", [0.5] * N_BUTTONS), dtype=np.float32)
        self.checkpoint = checkpoint
        self.reset()

    def reset(self) -> None:
        self._hist: collections.deque = collections.deque(
            maxlen=self.cfg.window)
        self._prev = np.zeros(self.cfg.prev_dim, np.float32)

    def observe(self, obs) -> None:
        o = np.asarray(obs, dtype=np.float32).reshape(-1)
        if o.shape[0] != self.cfg.obs_dim:
            raise ValueError(f"observation has {o.shape[0]} values; "
                             f"checkpoint expects {self.cfg.obs_dim}")
        self._hist.append(o)

    def _window(self) -> torch.Tensor:
        if not self._hist:
            raise RuntimeError("observe() an observation before plan()")
        items = list(self._hist)
        while len(items) < self.cfg.window:
            items.insert(0, items[0])
        w = torch.from_numpy(np.stack(items)).to(self.device)
        w = (w - self.mean) / self.std
        return w.unsqueeze(0)

    def set_prev(self, action: dict | None) -> None:
        if action is None:
            self._prev = np.zeros(self.cfg.prev_dim, np.float32)
            return
        self._prev = action_vector(action, self.cfg)

    def plan(self) -> list[dict]:
        """Predict the next H actions from the current observation history."""
        obs = self._window()
        return self._plan_from(obs)

    def _token_action(self, token: int, prev: np.ndarray) -> dict:
        """One token -> an action dict; the repeat token re-emits ``prev``."""
        if int(token) == REPEAT_TOKEN:
            buttons = (np.asarray(prev[:N_BUTTONS]) >= 0.5).astype(np.float32)
            stick = np.asarray(prev[N_BUTTONS:N_BUTTONS + 2], np.float32).copy()
            off = N_BUTTONS + 2
            cstick = (np.asarray(prev[off:off + 2], np.float32).copy()
                      if self.cfg.cstick else np.zeros(2, np.float32))
            if self.cfg.cstick:
                off += 2
            trigger = (float(prev[off]) if self.cfg.trigger and len(prev) > off
                       else 0.0)
        else:
            vec = self.vocab.decode(int(token))
            buttons = (vec[:N_BUTTONS] >= 0.5).astype(np.float32)
            stick = vec[N_BUTTONS:N_BUTTONS + 2].copy()
            cstick = vec[N_BUTTONS + 2:N_BUTTONS + 4].copy()
            trigger = float(vec[N_BUTTONS + 4])
        return {"buttons": buttons, "stick": stick, "cstick": cstick,
                "trigger": trigger}

    def _decode_chunk(self, tokens) -> list[dict]:
        prev = self._prev.copy()
        out = []
        for token in np.asarray(tokens, np.int64).reshape(-1):
            action = self._token_action(int(token), prev)
            prev = action_vector(action, self.cfg)
            out.append(action)
        return out

    def _plan_from(self, obs: torch.Tensor) -> list[dict]:
        prev = torch.from_numpy(self._prev).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(obs, prev)
        if self.cfg.action_mode == "vocab":
            logits = out["action"][0]
            if self.sample:
                probs = torch.softmax(logits / self.temperature, dim=-1)
                tokens = torch.multinomial(probs, 1).squeeze(-1).cpu().numpy()
            else:
                tokens = torch.argmax(logits, dim=-1).cpu().numpy()
            return self._decode_chunk(tokens)
        bp = torch.sigmoid(out["buttons"])[0].cpu().numpy()      # (H,12)
        if self.sample:
            rng = np.random.default_rng()
            bp = (rng.random(bp.shape) < bp).astype(np.float32)
        if self.cfg.stick_mode == "dir":
            if self.sample:
                d = torch.multinomial(torch.softmax(out["stick"][0], dim=-1),
                                      1).squeeze(-1).cpu().numpy()
                m = torch.multinomial(torch.softmax(out["stick_mag"][0], dim=-1),
                                      1).squeeze(-1).cpu().numpy()
            else:
                d = torch.argmax(out["stick"][0], dim=-1).cpu().numpy()
                m = torch.argmax(out["stick_mag"][0], dim=-1).cpu().numpy()
            stick = dir_to_xy(d, m)                                # (H,2)
        else:
            stick = soft_argmax_xy(out["stick"][0], self.cfg.bins,
                                   ).cpu().numpy()                 # (H,2)
        if "cstick" in out:
            cstick = soft_argmax_xy(out["cstick"][0], self.cfg.bins,
                                    ).cpu().numpy()
        else:
            cstick = np.zeros((self.cfg.horizon, 2), np.float32)
        if "trigger" in out:
            trigger = torch.sigmoid(out["trigger"][0]).cpu().numpy()
        else:
            trigger = np.zeros(self.cfg.horizon, np.float32)
        if self.sample:                     # bp is already a 0/1 mask
            ones = np.ones(N_BUTTONS, np.float32)
            return [probs_to_action(bp[h], stick[h], cstick[h], trigger[h], ones)
                    for h in range(self.cfg.horizon)]
        return [probs_to_action(bp[h], stick[h], cstick[h], trigger[h],
                                self.thresholds)
                for h in range(self.cfg.horizon)]

    def act(self, obs) -> dict:
        """Convenience one-step policy (no ensemble)."""
        self.observe(obs)
        action = self.plan()[0]
        self.set_prev(action)
        return action


def blend_actions(old: dict, new: dict, weight_old: float = 0.5) -> dict:
    """Temporal-ensemble blend of two predictions for the same step."""
    w = float(weight_old)
    bp = w * old["buttons_prob"] + (1.0 - w) * new["buttons_prob"]
    stick = w * old["stick"] + (1.0 - w) * new["stick"]
    cstick = w * old.get("cstick", np.zeros(2)) + \
        (1.0 - w) * new.get("cstick", np.zeros(2))
    trigger = w * old.get("trigger", 0.0) + (1.0 - w) * new.get("trigger", 0.0)
    th = new.get("thresholds")
    return probs_to_action(bp, stick, cstick, trigger, th)


def _swap_dpad(buttons: np.ndarray) -> np.ndarray:
    order = list(range(N_BUTTONS))
    order[BUTTON_INDEX["DPAD_L"]] = BUTTON_INDEX["DPAD_R"]
    order[BUTTON_INDEX["DPAD_R"]] = BUTTON_INDEX["DPAD_L"]
    return buttons[..., order]


class TemporalEnsemble:
    """ACT-style action queue: plan H, execute the first ``execute`` actions.

    When ``blend`` > 0 the new chunk is averaged with the not-yet-executed
    tail of the previous chunk (temporal ensemble).
    """

    def __init__(self, policy: BCPolicy, execute: int = 2,
                 blend: float = 0.5):
        self.policy = policy
        self.execute = max(1, int(execute))
        # Averaging whole actions can invent a combination the token head
        # never chose (A + a direction), so vocab checkpoints never blend.
        if getattr(policy.cfg, "action_mode", "heads") == "vocab":
            blend = 0.0
        self.blend = float(blend)
        self.reset()

    def reset(self) -> None:
        self.policy.reset()
        self.queue: list[dict] = []
        self._since_plan = 0
        self.steps = 0

    def act(self, obs, kind: str = "dict"):
        self.policy.observe(obs)
        if not self.queue or self._since_plan >= self.execute:
            chunk = self.policy.plan()
            if self.queue and self.blend > 0:
                k = min(len(self.queue), len(chunk))
                for i in range(k):
                    chunk[i] = blend_actions(self.queue[i], chunk[i],
                                             self.blend)
            self.queue = chunk
            self._since_plan = 0
        action = self.queue.pop(0)
        self._since_plan += 1
        self.policy.set_prev(action)
        self.steps += 1
        return action if kind == "dict" else action_to_env(action, kind)


# ---------------------------------------------------------------------------
# threshold calibration + metrics
# ---------------------------------------------------------------------------
def calibrate_thresholds(probs: np.ndarray, targets: np.ndarray,
                         eps: float = 1e-6) -> np.ndarray:
    """Per-button threshold so predicted positive rate == expert rate."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    th = np.empty(probs.shape[1], np.float64)
    for i in range(probs.shape[1]):
        rate = float(targets[:, i].mean()) if len(targets) else 0.0
        if rate <= 0.0:
            th[i] = 1.0 + eps
        elif rate >= 1.0:
            th[i] = 0.0
        else:
            th[i] = float(np.quantile(probs[:, i], 1.0 - rate))
    return np.clip(th, 0.0, 1.0 + eps).astype(np.float32)


def _f1(pred: np.ndarray, target: np.ndarray) -> float:
    tp = float(np.sum((pred > 0.5) & (target > 0.5)))
    fp = float(np.sum((pred > 0.5) & (target < 0.5)))
    fn = float(np.sum((pred <= 0.5) & (target > 0.5)))
    denom = 2 * tp + fp + fn
    # 0/0 (button never pressed and never predicted) is a perfect score; only
    # a non-empty error set can make F1 zero.
    return 2 * tp / denom if denom > 0 else 1.0


def compute_metrics(pred: dict, target: dict,
                    thresholds: np.ndarray | None = None,
                    prev: np.ndarray | None = None,
                    mode: str = "grid") -> dict:
    """Per-button F1, stick accuracy, and action-rate divergence."""
    bp = np.asarray(pred["buttons"], np.float32)
    tb = np.asarray(target["buttons"], np.float32)
    if bp.ndim == 3:
        bp = bp[:, 0]
    if tb.ndim == 3:
        tb = tb[:, 0]
    th = np.full(N_BUTTONS, 0.5, np.float32) if thresholds is None \
        else np.asarray(thresholds, np.float32)
    pred_bin = (bp >= th).astype(np.float32)
    f1_cal = {name: _f1(pred_bin[:, i], tb[:, i])
              for i, name in enumerate(BUTTONS)}
    f1_raw = {name: _f1(bp[:, i], tb[:, i]) for i, name in enumerate(BUTTONS)}
    rate_pred = pred_bin.mean(axis=0) if len(pred_bin) else np.zeros(N_BUTTONS)
    rate_expert = tb.mean(axis=0) if len(tb) else np.zeros(N_BUTTONS)
    divergence = {name: float(abs(rate_pred[i] - rate_expert[i]))
                  for i, name in enumerate(BUTTONS)}

    if mode == "dir":
        t_dir = np.asarray(target["stick"], np.int64)
        t_mag = np.asarray(target["stick_mag"], np.int64)
        p_dir = np.asarray(pred["stick_dir"], np.int64)
        p_mag = np.asarray(pred["stick_mag"], np.int64)
        if t_dir.ndim == 2:
            t_dir, t_mag = t_dir[:, 0], t_mag[:, 0]
        stick_true = t_dir
        stick_pred = p_dir
        stick_acc = float(np.mean(p_dir == t_dir)) if len(t_dir) else 0.0
        stick_mae = float(np.mean(np.abs(dir_to_xy(p_dir, p_mag) -
                                             dir_to_xy(t_dir, t_mag))))             if len(t_dir) else 0.0
    else:
        stick_pred = np.asarray(pred["stick_bin"], np.int64)
        stick_true = np.asarray(target["stick"], np.int64)
        if stick_true.ndim == 2:
            stick_true = stick_true[:, 0]
            stick_pred = stick_pred[:, 0] if stick_pred.ndim == 2 else stick_pred
        stick_acc = float(np.mean(stick_pred == stick_true)) if len(stick_true) else 0.0
        stick_xy = np.asarray(pred["stick"], np.float32)
        true_xy = bin_to_xy(stick_true)
        if stick_xy.ndim == 3:
            stick_xy = stick_xy[:, 0]
        stick_mae = float(np.mean(np.abs(stick_xy - true_xy))) if len(true_xy) else 0.0

    active = [name for i, name in enumerate(BUTTONS)
              if (rate_expert[i] > 0 or rate_pred[i] > 0)]
    out = {
        "n": int(len(tb)),
        "per_button_f1": f1_cal,
        "mean_f1": float(np.mean(list(f1_cal.values()))) if f1_cal else 0.0,
        "mean_f1_active": float(np.mean([f1_cal[n] for n in active]))
                          if active else 1.0,
        "per_button_f1_uncalibrated": f1_raw,
        "mean_f1_uncalibrated": float(np.mean(list(f1_raw.values())))
                                if f1_raw else 0.0,
        "button_pred_rate": {name: float(rate_pred[i]) for i, name
                             in enumerate(BUTTONS)},
        "button_expert_rate": {name: float(rate_expert[i]) for i, name
                               in enumerate(BUTTONS)},
        "action_rate_divergence": divergence,
        "mean_action_rate_divergence": float(np.mean(list(divergence.values())))
                                       if divergence else 0.0,
        "stick_acc": stick_acc,
        "stick_mae": stick_mae,
        "thresholds": [float(v) for v in th],
    }
    if mode == "dir":
        out["stick_mode"] = "dir"
        out["stick_parts"] = int(STICK_DIRS)
        out["stick_mag_acc"] = float(np.mean(p_mag == t_mag)) if len(t_mag) else 0.0
        if len(t_dir):
            d_xy = np.abs(dir_to_xy(p_dir, p_mag) - dir_to_xy(t_dir, t_mag))
            out["stick_within"] = float(np.mean(d_xy.max(axis=-1) < 0.25))
        else:
            out["stick_within"] = 0.0
    else:
        out["stick_parts"] = int(STICK_BINS * STICK_BINS)
    if prev is not None:
        # Decision points: target differs from the previous action.  This is the
        # 16.5 % of samples where the behaviour actually changes; the rest is
        # "hold", which a copy-last model gets for free.
        pv = np.asarray(prev, np.float32)
        pbtn = pv[:, :12] >= 0.5
        if mode == "dir":
            p_dir, p_mag = stick_to_dir(pv[:, 12:14])
            change = (tb != pbtn).any(axis=1) | (stick_true != p_dir) | \
                (t_mag != p_mag)
        else:
            pstick = stick_to_bin(pv[:, 12:14])
            tstick = np.asarray(target["stick"], np.int64)
            if tstick.ndim == 2:
                tstick = tstick[:, 0]
            change = (tb != pbtn).any(axis=1) | (tstick != pstick)
        nch = int(change.sum())
        f1_ch = {name: _f1(pred_bin[change, i], tb[change, i])
                 for i, name in enumerate(BUTTONS)} if nch else {n: 1.0 for n in BUTTONS}
        active_ch = [name for name in BUTTONS
                     if nch and tb[change, BUTTON_INDEX[name]].mean() > 0]
        out["change_frames"] = nch
        out["change_rate"] = float(nch / max(1, len(tb)))
        out["change_per_button_f1"] = f1_ch
        out["change_mean_f1"] = float(np.mean(list(f1_ch.values()))) if nch else 1.0
        out["change_mean_f1_active"] = float(np.mean([f1_ch[n] for n in active_ch])) \
            if active_ch else 1.0
        if mode == "dir" and nch:
            out["change_dir_acc"] = float(np.mean(stick_pred[change] ==
                                                  stick_true[change]))
    if "cstick_bin" in target and "cstick_bin" in pred:
        cp = np.asarray(pred["cstick_bin"], np.int64)
        ct = np.asarray(target["cstick_bin"], np.int64)
        if ct.ndim == 2:
            ct = ct[:, 0]
            cp = cp[:, 0] if cp.ndim == 2 else cp
        out["cstick_acc"] = float(np.mean(cp == ct)) if len(ct) else 0.0
    if "trigger" in pred:
        tp = np.asarray(pred["trigger"], np.float32)
        tt = np.asarray(target["trigger"], np.float32)
        if tp.ndim == 2:
            tp = tp[:, 0]
        if tt.ndim == 2:
            tt = tt[:, 0]
        out["trigger_bce"] = float(np.mean(
            -(tt * np.log(np.clip(tp, 1e-6, 1 - 1e-6)) +
              (1 - tt) * np.log(np.clip(1 - tp, 1e-6, 1 - 1e-6))))) \
            if len(tt) else 0.0
        out["trigger_pred_rate"] = float(tp.mean()) if len(tp) else 0.0
        out["trigger_expert_rate"] = float(tt.mean()) if len(tt) else 0.0
    return out


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean over the sample axis, weighted when weights are given."""
    per_sample = values.reshape(values.shape[0], -1).mean(dim=1)
    if weights is None:
        return per_sample.mean()
    w = weights.reshape(-1).to(per_sample.dtype)
    return (per_sample * w).sum() / w.sum().clamp_min(1e-8)


def _loss(out: dict, targets: dict, pos_weight: torch.Tensor,
          cfg: BCConfig, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    if cfg.action_mode == "vocab":
        logits = out["action"]
        el = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             targets["action"].reshape(-1),
                             reduction="none").reshape(logits.shape[:2])
        loss = _weighted_mean(el, weights)
        return loss, {"action": float(loss.detach())}
    btn = F.binary_cross_entropy_with_logits(
        out["buttons"], targets["buttons"], pos_weight=pos_weight,
        reduction="none")
    loss_btn = _weighted_mean(btn, weights)
    if cfg.stick_mode == "dir":
        stick_el = F.cross_entropy(out["stick"].reshape(-1, STICK_DIRS),
                                   targets["stick"].reshape(-1),
                                   reduction="none").reshape(out["stick"].shape[:2])
        mag_el = F.cross_entropy(out["stick_mag"].reshape(-1, STICK_MAGS),
                                 targets["stick_mag"].reshape(-1),
                                 reduction="none").reshape(out["stick_mag"].shape[:2])
        loss_stick = _weighted_mean(stick_el, weights)
        loss_mag = _weighted_mean(mag_el, weights)
        loss = loss_btn + loss_stick + 0.5 * loss_mag
        parts = {"buttons": float(loss_btn.detach()),
                 "stick": float(loss_stick.detach()),
                 "stick_mag": float(loss_mag.detach())}
    else:
        loss_stick = F.cross_entropy(
            out["stick"].reshape(-1, out["stick"].shape[-1]),
            targets["stick"].reshape(-1))
        loss = loss_btn + loss_stick
        parts = {"buttons": float(loss_btn.detach()),
                 "stick": float(loss_stick.detach())}
    if cfg.cstick and "cstick" in out:
        lc = F.cross_entropy(out["cstick"].reshape(-1, out["cstick"].shape[-1]),
                             targets["cstick"].reshape(-1))
        loss = loss + 0.5 * lc
        parts["cstick"] = float(lc.detach())
    if cfg.trigger and "trigger" in out:
        lt = F.binary_cross_entropy_with_logits(out["trigger"],
                                                targets["trigger"])
        loss = loss + 0.5 * lt
        parts["trigger"] = float(lt.detach())
    return loss, parts


def collect_predictions(model: BCPolicyNet, samples: dict, mean, std,
                        device: str = "cpu", batch: int = 256,
                        vocab: ActionVocab | None = None) -> dict:
    """Predictions for the first action of every chunk."""
    ds = WindowDataset(samples, mean, std)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    model.eval()
    if model.cfg.action_mode == "vocab":
        if vocab is None:
            raise ValueError("vocab predictions need the ActionVocab")
        toks, pvs, tvs = [], [], []
        with torch.no_grad():
            for obs, prev, targets, _w in loader:
                out = model(obs.to(device), prev.to(device))
                toks.append(torch.argmax(out["action"][:, 0], dim=-1).cpu().numpy())
                pvs.append(prev.numpy())
                tvs.append(np.asarray(targets["action"])[:, 0])
        tokens = np.concatenate(toks) if toks else np.zeros(0, np.int64)
        pvec = (np.concatenate(pvs) if pvs
                else np.zeros((0, model.cfg.prev_dim), np.float32))
        btns, d, m = decode_tokens(tokens, pvec, vocab)
        return {"buttons": btns, "stick_dir": d, "stick_mag": m,
                "stick_bin": d, "stick": dir_to_xy(d, m),
                "tokens": tokens,
                "target_tokens": (np.concatenate(tvs) if tvs
                                  else np.zeros(0, np.int64))}
    mode = model.cfg.stick_mode
    bp, sb, cb, tr, sl, sm = [], [], [], [], [], []
    with torch.no_grad():
        for obs, prev, targets, _w in loader:
            obs = obs.to(device)
            prev = prev.to(device)
            out = model(obs, prev)
            bp.append(torch.sigmoid(out["buttons"][:, 0]).cpu().numpy())
            sb.append(torch.argmax(out["stick"][:, 0], dim=-1).cpu().numpy())
            if mode == "dir":
                sm.append(torch.argmax(out["stick_mag"][:, 0], dim=-1).cpu().numpy())
            else:
                sl.append(out["stick"][:, 0].cpu().numpy())
            if "cstick" in out:
                cb.append(torch.argmax(out["cstick"][:, 0], dim=-1).cpu().numpy())
            if "trigger" in out:
                tr.append(torch.sigmoid(out["trigger"][:, 0]).cpu().numpy())
    pred = {
        "buttons": np.concatenate(bp) if bp else np.zeros((0, N_BUTTONS)),
        "stick_bin": np.concatenate(sb) if sb else np.zeros(0, np.int64),
    }
    if mode == "dir":
        pred["stick_dir"] = pred["stick_bin"]
        pred["stick_mag"] = np.concatenate(sm) if sm else np.zeros(0, np.int64)
        pred["stick"] = dir_to_xy(pred["stick_dir"], pred["stick_mag"])
    else:
        pred["stick_logits"] = np.concatenate(sl) if sl else np.zeros((0, N_ACTIONS))
        pred["stick"] = soft_argmax_np(pred["stick_logits"]) \
            if len(pred["stick_logits"]) else np.zeros((0, 2), np.float32)
    if cb:
        pred["cstick_bin"] = np.concatenate(cb)
    if tr:
        pred["trigger"] = np.concatenate(tr)
    return pred


def copy_last_pred(samples: dict, mode: str = "grid") -> dict:
    """The trivial policy: repeat the previous action (no learning at all)."""
    prev = np.asarray(samples["prev"], np.float32)
    stick = prev[:, 12:14]
    if mode == "dir":
        d, m = stick_to_dir(stick)
        return {"buttons": prev[:, :12], "stick_dir": d, "stick_bin": d,
                "stick_mag": m, "stick": dir_to_xy(d, m)}
    return {
        "buttons": prev[:, :12],
        "stick": stick,
        "stick_bin": stick_to_bin(stick),
    }


def copy_last_metrics(samples: dict, mode: str = "grid") -> dict:
    """Metrics of the copy-last policy on the same samples (baseline)."""
    target = first_actions(samples)
    # binary inputs: no calibration, a 0.5 threshold is exact
    return compute_metrics(copy_last_pred(samples, mode), target,
                           thresholds=np.full(N_BUTTONS, 0.5, np.float32),
                           prev=samples["prev"], mode=mode)


def transition_mask(samples: dict, mode: str = "grid") -> np.ndarray:
    """True where the target action differs from the previous one."""
    prev = np.asarray(samples["prev"], np.float32)
    tb = samples["targets"]["buttons"][:, 0]
    pbtn = prev[:, :12] >= 0.5
    if mode == "dir":
        p_dir, p_mag = stick_to_dir(prev[:, 12:14])
        t_dir = samples["targets"]["stick"][:, 0]
        t_mag = samples["targets"]["stick_mag"][:, 0]
        return (tb != pbtn).any(axis=1) | (t_dir != p_dir) | (t_mag != p_mag)
    pstick = stick_to_bin(prev[:, 12:14])
    tstick = samples["targets"]["stick"][:, 0]
    return (tb != pbtn).any(axis=1) | (tstick != pstick)


def first_actions(samples: dict) -> dict:
    t = samples["targets"]
    out = {
        "buttons": t["buttons"][:, 0],
        "stick": t["stick"][:, 0],
        "stick_bin": t["stick"][:, 0],
        "cstick_bin": t["cstick"][:, 0],
        "trigger": t["trigger"][:, 0],
    }
    if "stick_mag" in t:
        out["stick_mag"] = t["stick_mag"][:, 0]
    return out


def _apply_threads(threads: int, batch: int) -> int:
    """Pick sane CPU threads; small batches oversubscribe badly in this box."""
    if threads > 0:
        n = int(threads)
    elif batch <= 128:
        n = 1
    else:
        n = min(4, os.cpu_count() or 1)
    old = torch.get_num_threads()
    torch.set_num_threads(max(1, n))
    return old


def train(data_dir: str, out: str, cfg: BCConfig | None = None,
          epochs: int = 10, batch: int = 64, lr: float = 1e-3,
          device: str = "cpu", seed: int = 0, mirror: bool = True,
          ego_swap: bool = False, select: str = "loss",
          limit: int | None = None, threads: int = 0,
          weight_decay: float = 1e-4, dropout: float | None = None,
          patience: int = 5, min_epochs: int = 1,
          init_from: str | None = None, return_beta: float = 0.0,
          return_gamma: float = 0.995, weight_quantile: float = 0.5,
          weight_min: float = 0.1, weight_max: float = 10.0,
          reward_scale: float = 0.01, window_dtype: str = "float32",
          dedupe: bool = True,
          log=print) -> tuple[dict, dict]:
    """Train from a recording directory; returns ``(checkpoint, metrics)``.

    Early stopping: training stops once validation loss has not improved for
    ``patience`` epochs (0 disables), never before ``min_epochs``.  The best-val
    weights are restored and that state is what the checkpoint stores.
    """
    old = _apply_threads(threads, batch)
    try:
        return _train(data_dir, out, cfg=cfg, epochs=epochs, batch=batch, lr=lr,
                      device=device, seed=seed, mirror=mirror,
                      ego_swap=ego_swap, select=select, limit=limit,
                      weight_decay=weight_decay, dropout=dropout,
                      patience=patience, min_epochs=min_epochs,
                      init_from=init_from, return_beta=return_beta,
                      return_gamma=return_gamma,
                      weight_quantile=weight_quantile, weight_min=weight_min,
                      weight_max=weight_max, reward_scale=reward_scale,
                      window_dtype=window_dtype, dedupe=dedupe, log=log)
    finally:
        torch.set_num_threads(old)


def _train(data_dir: str, out: str, cfg: BCConfig | None = None,
           epochs: int = 10, batch: int = 64, lr: float = 1e-3,
           device: str = "cpu", seed: int = 0, mirror: bool = True,
           ego_swap: bool = False, select: str = "loss",
           limit: int | None = None, weight_decay: float = 1e-4,
           dropout: float | None = None, patience: int = 5,
           min_epochs: int = 1, init_from: str | None = None,
           return_beta: float = 0.0, return_gamma: float = 0.995,
           weight_quantile: float = 0.5, weight_min: float = 0.1,
           weight_max: float = 10.0, reward_scale: float = 0.01,
           window_dtype: str = "float32", dedupe: bool = True,
           log=print) -> tuple[dict, dict]:
    cfg = cfg or BCConfig()
    if cfg.action_mode not in ("heads", "vocab"):
        raise ValueError(f"action_mode must be heads or vocab, got "
                         f"{cfg.action_mode!r}")
    if cfg.damage_mode not in ("raw", "bucket", "none"):
        raise ValueError(f"damage_mode must be raw, bucket or none, got "
                         f"{cfg.damage_mode!r}")
    if cfg.label_mode not in ("single", "hold"):
        raise ValueError(f"label_mode must be single or hold, got "
                         f"{cfg.label_mode!r}")
    if cfg.action_mode == "vocab" and cfg.stick_mode != "dir":
        # The vocabulary encodes discrete directions; grid stick decoding
        # would also break the stick metrics.
        cfg = replace(cfg, stick_mode="dir")
    if dropout is not None:
        cfg = replace(cfg, dropout=float(dropout))
    if not (0.0 <= cfg.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {cfg.dropout}")
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if patience < 0:
        raise ValueError(f"patience must be >= 0, got {patience}")
    if min_epochs < 0:
        raise ValueError(f"min_epochs must be >= 0, got {min_epochs}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = str(device)
    episodes = load_episodes(data_dir, player=cfg.player, limit=limit,
                             normalize_obs=cfg.obs_normalize,
                             shield_from_status=cfg.shield_from_status,
                             damage_mode=cfg.damage_mode, dedupe=dedupe,
                             log=log)
    gaps = sum(int(ep.meta.get("seq_gaps", 0)) for ep in episodes)
    if gaps:
        log(f"[bc] warning: {gaps} reader-frame sequence gaps across the "
            "recordings; every-5th-sample resampling is approximate there")
    episodes = [resample_episode(ep, cfg.frame_skip, label_mode=cfg.label_mode)
                for ep in episodes]
    episodes = [ep for ep in episodes
                if len(ep.obs) >= cfg.window + cfg.horizon - 1]
    if not episodes:
        raise ValueError("no episode is long enough for the requested window")
    n = len(episodes)
    tr_idx, va_idx, te_idx = split_episodes(n, seed=seed)
    # Remap the split positions to the filtered episode list.
    train_eps = [episodes[i] for i in tr_idx]
    val_eps = [episodes[i] for i in va_idx]
    test_eps = [episodes[i] for i in te_idx]
    log(f"[bc] {n} episodes: train {len(train_eps)}, val {len(val_eps)}, "
        f"test {len(test_eps)} (episode-level split)")

    vocab = None
    if cfg.action_mode == "vocab":
        # Built from the training actions (recorded and mirrored) so the
        # vocabulary can express anything the mirror augmentation produces.
        vocab = ActionVocab.build(train_eps, size=cfg.vocab_size)
        cfg = replace(cfg, vocab_size=vocab.size)
        log(f"[bc] action vocabulary: {vocab.size - 1} observed tuples + "
            f"repeat token ({vocab.size} classes)")

    shield_labels = {
        "from_status": bool(cfg.shield_from_status),
        "train": shield_label_stats(train_eps),
        "val": shield_label_stats(val_eps),
        "test": shield_label_stats(test_eps),
    }
    shield_labels["total"] = merge_shield_stats(
        [shield_labels["train"], shield_labels["val"], shield_labels["test"]])
    if cfg.shield_from_status:
        log(f"[bc] shield-from-status: {shield_labels['total']['frames']} "
            f"guard decisions relabeled ZL=1/X=0 "
            f"(train {shield_labels['train']['frames']}, "
            f"{shield_labels['train']['spurious_x_cleared']} spurious X)")
    else:
        log(f"[bc] shield-from-status OFF: {shield_labels['total']['frames']} "
            "guard decisions keep their raw mask")

    dtype = np.float16 if str(window_dtype) == "float16" else np.float32
    weighted = dict(return_beta=return_beta, return_gamma=return_gamma,
                    weight_quantile=weight_quantile, weight_min=weight_min,
                    weight_max=weight_max, reward_scale=reward_scale,
                    window_dtype=dtype)
    train_samples = samples_for(train_eps, cfg, mirror=mirror,
                                ego_swap=ego_swap, vocab=vocab, **weighted)
    if return_beta > 0 and "weights" in train_samples:
        w = np.asarray(train_samples["weights"], np.float32)
        log(f"[bc] conservative return weighting: beta {return_beta:g}, gamma "
            f"{return_gamma:g}, quantile {weight_quantile:g}, clip "
            f"[{weight_min:g}, {weight_max:g}] -> weight mean {w.mean():.3f} "
            f"p10 {np.percentile(w, 10):.3f} p90 {np.percentile(w, 90):.3f} "
            f"max {w.max():.3f}")

    val_samples = samples_for(val_eps, cfg, mirror=False, vocab=vocab,
                              **weighted)
    test_samples = samples_for(test_eps, cfg, mirror=False, vocab=vocab,
                               **weighted)
    log(f"[bc] samples: train {len(train_samples['windows'])}, "
        f"val {len(val_samples['windows'])}, test {len(test_samples['windows'])}")

    mean, std = norm_stats(train_samples["windows"])
    ds = WindowDataset(train_samples, mean, std)
    if cfg.transition_weight and float(cfg.transition_weight) != 1.0:
        mask = transition_mask(train_samples, cfg.stick_mode)
        w = np.where(mask, float(cfg.transition_weight), 1.0)
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                        num_samples=len(w), replacement=True)
        loader = DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=0,
                            drop_last=False)
        log(f"[bc] transition weighting x{cfg.transition_weight:.2f}: "
            f"{int(mask.sum())}/{len(mask)} samples are decision points")
    else:
        loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0,
                            drop_last=False)
    pos_weight = torch.as_tensor(
        compute_pos_weight(train_samples["targets"]["buttons"]),
        dtype=torch.float32, device=device)
    model = BCPolicyNet(cfg).to(device)
    if init_from:
        prior = load_checkpoint(init_from)
        try:
            model.load_state_dict(prior["state_dict"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"cannot warm start from {init_from}: its architecture does not "
                f"match this config ({exc})") from exc
        log(f"[bc] warm start from {init_from} "
            f"(best epoch {prior.get('best_epoch')}, "
            f"{sum(p.numel() for p in model.parameters())} params, "
            f"lr {lr:g})")
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    if select not in ("loss", "f1_active", "f1_change"):
        raise ValueError(f"select must be loss/f1_active/f1_change, got {select!r}")
    best_val = float("inf")
    best_score = float("-inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0
    epochs_run = 0
    stopped_early = False
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for obs, prev, targets, w in loader:
            obs = obs.to(device)
            prev = prev.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            w = w.to(device)
            pred = model(obs, prev)
            loss, parts = _loss(pred, targets, pos_weight, cfg, weights=w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(obs)
            seen += len(obs)
        sched.step()
        epochs_run = epoch
        train_loss = total / max(1, seen)
        val_loss = _eval_loss(model, val_samples, mean, std, pos_weight,
                              cfg, device, batch)
        vpred = collect_predictions(model, val_samples, mean, std, device,
                                    batch, vocab=vocab)
        vmet = compute_metrics(vpred, first_actions(val_samples), None,
                               prev=val_samples["prev"], mode=cfg.stick_mode)
        f1_act = float(vmet["mean_f1_active"])
        f1_chg = float(vmet.get("change_mean_f1_active", 0.0))
        score = {"loss": -val_loss, "f1_active": f1_act,
                 "f1_change": f1_chg}[select]
        improved = score > best_score
        if improved:
            best_score = score
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_f1_active": f1_act,
                        "val_f1_change": f1_chg, "best": bool(improved),
                        "bad_epochs": bad_epochs})
        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs \
                or improved:
            log(f"[bc] epoch {epoch:3d}/{epochs}: train {train_loss:.4f} "
                f"val {val_loss:.4f}  f1_active {f1_act:.3f}  "
                f"f1_change {f1_chg:.3f}"
                + (" *" if improved else ""))
        if patience > 0 and epoch >= max(1, min_epochs) \
                and bad_epochs >= patience:
            stopped_early = True
            log(f"[bc] early stop at epoch {epoch}: no val improvement for "
                f"{bad_epochs} epochs (best epoch {best_epoch}, "
                f"val {best_val:.4f})")
            break

    if best_epoch == 0 and history:      # val never finite/improved
        best_epoch = history[-1]["epoch"]
        best_val = history[-1]["val_loss"]
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    log(f"[bc] selected epoch {best_epoch}/{epochs_run} "
        f"(val {best_val:.4f}, {len(history)} epochs run"
        + (", early stop)" if stopped_early else ")"))
    val_pred = collect_predictions(model, val_samples, mean, std, device,
                                   batch, vocab=vocab)
    train_pred = collect_predictions(model, train_samples, mean, std, device,
                                     batch, vocab=vocab)
    test_pred = collect_predictions(model, test_samples, mean, std, device,
                                    batch, vocab=vocab)
    if cfg.action_mode == "vocab":
        # The token head is already a calibrated categorical distribution;
        # per-button thresholds would only distort the decoded action.
        thresholds = np.full(N_BUTTONS, 0.5, np.float32)
    else:
        thresholds = calibrate_thresholds(val_pred["buttons"],
                                         first_actions(val_samples)["buttons"])
    metrics = {
        "train": compute_metrics(
            train_pred, first_actions(train_samples), thresholds,
            prev=train_samples["prev"], mode=cfg.stick_mode),
        "val": compute_metrics(val_pred, first_actions(val_samples), thresholds,
                               prev=val_samples["prev"], mode=cfg.stick_mode),
        "test": compute_metrics(
            test_pred, first_actions(test_samples), thresholds,
            prev=test_samples["prev"], mode=cfg.stick_mode),
        "baseline": {
            "train": copy_last_metrics(train_samples, cfg.stick_mode),
            "val": copy_last_metrics(val_samples, cfg.stick_mode),
            "test": copy_last_metrics(test_samples, cfg.stick_mode),
        },
        "history": history,
        "expert_stats": dataset_action_stats(train_samples),
        "split": {"train": [int(i) for i in tr_idx],
                  "val": [int(i) for i in va_idx],
                  "test": [int(i) for i in te_idx]},
        "n_episodes": n,
        "alignment": {"seq_gaps": gaps, "ok": gaps == 0,
                      "frame_skip": cfg.frame_skip},
        "shield_from_status": bool(cfg.shield_from_status),
        "damage_mode": cfg.damage_mode,
        "label_mode": cfg.label_mode,
        "action_mode": cfg.action_mode,
        "vocab_size": int(cfg.vocab_size),
        "dedupe": bool(dedupe),
        "shield_labels": shield_labels,
        "pos_weight": pos_weight.cpu().numpy().tolist(),
        # The train/val/test metrics above are all computed with the restored
        # best-validation weights.
        "select": select,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "best_score": float(best_score),
        "epochs_run": int(epochs_run),
        "stopped_early": bool(stopped_early),
        "patience": int(patience),
        "min_epochs": int(min_epochs),
        "weight_decay": float(weight_decay),
        "dropout": float(cfg.dropout),
    }
    if cfg.action_mode == "vocab":
        # Token-level view of the same predictions (repeat = "hold").
        split_tokens = {}
        for name, pred in (("train", train_pred), ("val", val_pred),
                           ("test", test_pred)):
            got = np.asarray(pred.get("tokens", []), np.int64)
            want = np.asarray(pred.get("target_tokens", []), np.int64)
            split_tokens[name] = {
                "accuracy": float(np.mean(got == want)) if len(want) else 0.0,
                "pred_repeat_rate": float(np.mean(got == REPEAT_TOKEN))
                                    if len(got) else 0.0,
                "expert_repeat_rate": float(np.mean(want == REPEAT_TOKEN))
                                      if len(want) else 0.0,
            }
        metrics["action_tokens"] = split_tokens
        log("[bc] token accuracy: " + "  ".join(
            f"{name} {d['accuracy']:.3f} (repeat pred {d['pred_repeat_rate']:.3f}"
            f"/ expert {d['expert_repeat_rate']:.3f})"
            for name, d in split_tokens.items()))
    checkpoint = {
        "format": "ssbu_bc_v1",
        "config": cfg.to_dict(),
        "state_dict": model.state_dict(),
        "norm": {"mean": mean.tolist(), "std": std.tolist()},
        "thresholds": thresholds.tolist(),
        "vocab": vocab.to_dict() if vocab is not None else None,
        "pos_weight": metrics["pos_weight"],
        "metrics": {k: metrics[k] for k in ("train", "val", "test")},
        "expert_stats": metrics["expert_stats"],
        "split": metrics["split"],
        "alignment": metrics["alignment"],
        "shield_from_status": metrics["shield_from_status"],
        "damage_mode": metrics["damage_mode"],
        "label_mode": metrics["label_mode"],
        "action_mode": metrics["action_mode"],
        "vocab_size": metrics["vocab_size"],
        "dedupe": metrics["dedupe"],
        "action_tokens": metrics.get("action_tokens"),
        "shield_labels": metrics["shield_labels"],
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "epochs_run": metrics["epochs_run"],
        "stopped_early": metrics["stopped_early"],
        "history": metrics["history"],
        "training": {
            "epochs_requested": int(epochs),
            "patience": int(patience),
            "min_epochs": int(min_epochs),
            "weight_decay": float(weight_decay),
            "dropout": float(cfg.dropout),
            "lr": float(lr),
            "batch": int(batch),
            "init_from": os.path.abspath(init_from) if init_from else None,
            "return_beta": float(return_beta),
            "return_gamma": float(return_gamma),
            "weight_quantile": float(weight_quantile),
            "weight_clip": [float(weight_min), float(weight_max)],
            "reward_scale": float(reward_scale),
            "window_dtype": str(np.dtype(dtype)),
        },
        "meta": {
            "data": os.path.abspath(data_dir),
            "player": cfg.player,
            "mirror": bool(mirror),
            "ego_swap": bool(ego_swap),
            "transition_weight": float(cfg.transition_weight),
            "select": select,
            "trained_ns": time.time_ns(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    torch.save(checkpoint, out)
    log(f"[bc] saved {out}")
    return checkpoint, metrics


def _eval_loss(model, samples, mean, std, pos_weight, cfg, device,
               batch) -> float:
    if len(samples["windows"]) == 0:
        return float("inf")
    ds = WindowDataset(samples, mean, std)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for obs, prev, targets, w in loader:
            obs = obs.to(device)
            prev = prev.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            out = model(obs, prev)
            loss, _ = _loss(out, targets, pos_weight, cfg, weights=w.to(device))
            total += float(loss) * len(obs)
            seen += len(obs)
    return total / max(1, seen)


# ---------------------------------------------------------------------------
# tiny-batch overfit (used by the unit tests and --overfit)
# ---------------------------------------------------------------------------
def synthetic_episode(n: int = 80, seed: int = 0, player: int = 0,
                      index: int = 0) -> EpisodeData:
    """A small deterministic recording whose actions depend on the state."""
    rng = np.random.default_rng(seed)
    raw = np.zeros((n, 2, len(FIELDS)), np.float32)
    x = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    y = np.sin(np.linspace(0, 6.0, n, dtype=np.float32)).astype(np.float32)
    for k in range(2):
        raw[:, k, FIELD_INDEX["x"]] = 50.0 * (x if k == 0 else -x)
        raw[:, k, FIELD_INDEX["y"]] = 10.0 * y
        raw[:, k, FIELD_INDEX["vx"]] = 20.0 * x
        raw[:, k, FIELD_INDEX["damage"]] = 20.0 * (k + 1) + 5.0 * np.abs(y)
        raw[:, k, FIELD_INDEX["facing"]] = np.where(x >= 0, 1.0, -1.0)
        raw[:, k, FIELD_INDEX["on_ground"]] = 1.0
        raw[:, k, FIELD_INDEX["is_actionable"]] = 1.0
        raw[:, k, FIELD_INDEX["action_state"]] = 0.0
        raw[:, k, FIELD_INDEX["dist_to_blast"]] = 200.0
        raw[:, k, FIELD_INDEX["dx"]] = -100.0 * x
        raw[:, k, FIELD_INDEX["dy"]] = 5.0 * y
        raw[:, k, FIELD_INDEX["facing_rel"]] = 1.0
    buttons = np.zeros((n, 2), np.uint32)
    stick = np.zeros((n, 2, 2), np.float32)
    cstick = np.zeros((n, 2, 2), np.float32)
    trigger = np.zeros((n, 2), np.float32)
    # Easily learnable expert: A when x > 0, B when damage > 25, stick = (x, y).
    for k in range(2):
        sign = x if k == 0 else -x
        mask = np.zeros(n, np.uint32)
        mask[sign > 0] |= BUTTON_BITS[BUTTON_INDEX["A"]]
        mask[np.abs(y) > 0.5] |= BUTTON_BITS[BUTTON_INDEX["B"]]
        buttons[:, k] = mask
        stick[:, k, 0] = sign
        stick[:, k, 1] = y
    meta = {"index": index, "p1_char": "mario", "p2_char": "donkey_kong",
            "stage": "final_destination", "build": "synthetic"}
    return EpisodeData(index, f"synthetic_{index}", meta, raw, buttons, stick,
                       cstick, trigger, player=player)


def train_overfit(steps: int = 200, seed: int = 0, device: str = "cpu",
                  log=print, hidden: int = 64) -> tuple[BCPolicyNet, dict]:
    """Fit a tiny deterministic dataset; returns the model and metrics."""
    cfg = BCConfig(hidden=hidden, cstick=False, trigger=True)
    # Tiny batches are slower with many intra-op threads (oversubscription:
    # 100 steps took 1.6 s on 1 thread and 20 s on 4 in this container).
    n_threads = torch.get_num_threads()
    if n_threads > 1:
        torch.set_num_threads(1)
    try:
        return _train_overfit(cfg, steps, seed, device, log)
    finally:
        torch.set_num_threads(n_threads)


def _train_overfit(cfg: BCConfig, steps: int, seed: int, device: str,
                   log) -> tuple[BCPolicyNet, dict]:
    torch.manual_seed(seed)
    episodes = [resample_episode(synthetic_episode(80, seed=s, index=s),
                                 cfg.frame_skip)
                for s in range(2)]
    samples = samples_for(episodes, cfg, mirror=False)
    mean, std = norm_stats(samples["windows"])
    ds = WindowDataset(samples, mean, std)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=0)
    model = BCPolicyNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    pos_weight = torch.as_tensor(
        compute_pos_weight(samples["targets"]["buttons"]),
        dtype=torch.float32, device=device)
    for step in range(1, steps + 1):
        model.train()
        for obs, prev, targets, w in loader:
            obs = obs.to(device)
            prev = prev.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            out = model(obs, prev)
            loss, _ = _loss(out, targets, pos_weight, cfg, weights=w.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    pred = collect_predictions(model, samples, mean, std, device)
    metrics = compute_metrics(pred, first_actions(samples),
                              calibrate_thresholds(pred["buttons"],
                                                   first_actions(samples)["buttons"]))
    metrics["train_loss"] = _eval_loss(model, samples, mean, std, pos_weight,
                                       cfg, device, 256)
    metrics["norm"] = {"mean": mean.tolist(), "std": std.tolist()}
    log(f"[bc] overfit: loss {metrics['train_loss']:.4f}, "
        f"mean F1 {metrics['mean_f1']:.3f}, stick acc {metrics['stick_acc']:.3f}")
    return model, metrics


# ---------------------------------------------------------------------------
# checkpoint helpers
# ---------------------------------------------------------------------------
def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_policy(path: str, device: str = "cpu",
                sample: bool = False,
                temperature: float = 1.0) -> BCPolicy:
    return BCPolicy(load_checkpoint(path), device=device, sample=sample,
                    temperature=temperature)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="recording directory from gym_record.py")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(REPO, "data", "E", "bc.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--player", type=int, default=0,
                    help="expert slot to clone (default 0 = P1)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-episodes", type=int, default=None)
    ap.add_argument("--cstick", action="store_true",
                    help="add the C-stick 9x9 head")
    ap.add_argument("--obs-normalize", action="store_true",
                    help="use gym_obs' divisor mode (default: raw physical "
                         "values, matching the env's normalize_obs=False)")
    ap.add_argument("--no-shield-from-status", action="store_true",
                    help="keep the raw CPU mask on GUARD frames (default: "
                         "relabel them ZL=1/X=0 from the status kind)")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--ego-swap", action="store_true",
                    help="also train from the opponent's seat (blocks swapped, "
                         "actions from the other player): 2x more data")
    ap.add_argument("--stick-mode", choices=("dir", "grid"), default="dir",
                    help="dir = 9 directions + 3 magnitudes (the expert stick is "
                         "binary); grid = the old 9x9 heatmap (vocab forces dir)")
    ap.add_argument("--action-mode", choices=("heads", "vocab"), default="heads",
                    help="heads = per-button sigmoids + stick heads; vocab = one "
                         "softmax over the observed joint action tuples plus a "
                         "repeat token (mutually exclusive, multimodal)")
    ap.add_argument("--vocab-size", type=int, default=64,
                    help="max observed tuples kept in the vocabulary (the repeat "
                         "token is extra)")
    ap.add_argument("--damage-mode", choices=("raw", "bucket", "none"),
                    default="raw",
                    help="raw percent; bucket = 10%% buckets (20 values); "
                         "none = zeroed")
    ap.add_argument("--label-mode", choices=("single", "hold"), default="single",
                    help="single = label the decision frame; hold = OR the "
                         "buttons over the frames the pad is actually held for")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="keep episodes with identical recorded content (the "
                         "default drops them so duplicates cannot straddle the "
                         "train/val split)")
    ap.add_argument("--transition-weight", type=float, default=1.0,
                    help="sampling weight for decisions that differ from the "
                         "previous action (1.0 = off)")
    ap.add_argument("--select-metric", choices=("loss", "f1_active", "f1_change"),
                    default="loss",
                    help="model selection / early stopping metric on validation "
                         "(loss, active-button F1, or change-point F1)")
    ap.add_argument("--return-weight", type=float, default=0.0,
                    help="conservative return weighting: beta on the discounted "
                         "damage-differential return (0 = plain BC)")
    ap.add_argument("--return-gamma", type=float, default=0.995,
                    help="discount per decision for the return weighting")
    ap.add_argument("--weight-quantile", type=float, default=0.5,
                    help="quantile of the return used as the weight baseline")
    ap.add_argument("--weight-clip", type=float, nargs=2, default=(0.1, 10.0),
                    metavar=("MIN", "MAX"), help="clip on the per-sample weight")
    ap.add_argument("--window-dtype", choices=("float32", "float16"),
                    default="float32",
                    help="storage dtype of the observation windows: float16 "
                         "halves the dataset memory for a 30 Hz run")
    ap.add_argument("--reward-scale", type=float, default=0.01,
                    help="reward units per percent of damage")
    ap.add_argument("--init-from", default=None,
                    help="warm start from an existing checkpoint (same "
                         "architecture) instead of random init; use with a "
                         "smaller --lr to keep training on new data")
    ap.add_argument("--patience", type=int, default=5,
                    help="early-stop after N epochs without val improvement "
                         "(0 disables)")
    ap.add_argument("--min-epochs", type=int, default=1,
                    help="never early-stop before this epoch")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0,
                    help="trunk dropout probability (0 = off)")
    ap.add_argument("--overfit", action="store_true",
                    help="run the tiny-batch overfit smoke test instead")
    ap.add_argument("--overfit-steps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = auto: 1 for batch <= 128)")
    ap.add_argument("--json", action="store_true", help="print metrics JSON")
    args = ap.parse_args(argv)

    cfg = BCConfig(window=args.window, horizon=args.horizon,
                   frame_skip=args.frame_skip, hidden=args.hidden,
                   cstick=args.cstick, player=args.player,
                   ego_swap=bool(args.ego_swap),
                   stick_mode=args.stick_mode,
                   transition_weight=float(args.transition_weight),
                   obs_normalize=args.obs_normalize, dropout=args.dropout,
                   shield_from_status=not args.no_shield_from_status,
                   action_mode=args.action_mode, vocab_size=args.vocab_size,
                   damage_mode=args.damage_mode, label_mode=args.label_mode)
    # Progress goes to stderr so `--json` stays machine-readable on stdout.
    log = lambda *a: print(*a, file=sys.stderr, flush=True)  # noqa: E731
    if args.overfit:
        _, metrics = train_overfit(steps=args.overfit_steps, seed=args.seed,
                                   device=args.device, log=log)
        if args.json:
            print(json.dumps(jsonable(metrics), indent=2))
        return 0 if metrics["mean_f1"] > 0.8 and metrics["stick_acc"] > 0.8 else 1

    if not args.data:
        ap.error("--data is required unless --overfit is used")
    if args.epochs < 1:
        ap.error("--epochs must be >= 1")
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.patience < 0:
        ap.error("--patience must be >= 0")
    if args.min_epochs < 0:
        ap.error("--min-epochs must be >= 0")
    if args.weight_decay < 0:
        ap.error("--weight-decay must be >= 0")
    if not (0.0 <= args.dropout < 1.0):
        ap.error("--dropout must be in [0, 1)")
    ckpt, metrics = train(args.data, args.out, cfg=cfg, epochs=args.epochs,
                          batch=args.batch, lr=args.lr, device=args.device,
                          seed=args.seed, mirror=not args.no_mirror,
                          ego_swap=bool(args.ego_swap),
                          select=args.select_metric,
                          limit=args.limit_episodes, threads=args.threads,
                          weight_decay=args.weight_decay,
                          dropout=args.dropout, patience=args.patience,
                          min_epochs=args.min_epochs,
                          init_from=args.init_from,
                          return_beta=args.return_weight,
                          return_gamma=args.return_gamma,
                          weight_quantile=args.weight_quantile,
                          weight_min=args.weight_clip[0],
                          weight_max=args.weight_clip[1],
                          reward_scale=args.reward_scale,
                          window_dtype=args.window_dtype,
                          dedupe=not args.no_dedupe, log=log)
    summary = {
        "checkpoint": os.path.abspath(args.out),
        "best_epoch": metrics["best_epoch"],
        "best_val_loss": metrics["best_val_loss"],
        "epochs_run": metrics["epochs_run"],
        "epochs_requested": args.epochs,
        "stopped_early": metrics["stopped_early"],
        "patience": metrics["patience"],
        "min_epochs": metrics["min_epochs"],
        "weight_decay": metrics["weight_decay"],
        "dropout": metrics["dropout"],
        "shield_from_status": metrics["shield_from_status"],
        "shield_labels": metrics["shield_labels"],
        "history": metrics["history"],
        "train": metrics["train"],
        "val": metrics["val"],
        "test": metrics["test"],
        "baseline": metrics["baseline"],
        "select": metrics["select"],
        "return_beta": args.return_weight,
        "expert_stats": metrics["expert_stats"],
        "split": metrics["split"],
        "alignment": metrics["alignment"],
    }
    print(json.dumps(jsonable(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
