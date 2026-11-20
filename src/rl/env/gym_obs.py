#!/usr/bin/env python3
"""Policy observation profile v1 for the SSBU gym environment.

Layout (CONTRACTS.md), flat ``Box(float32)``, P1 then P2, 56 values each
(112 total).  For one fighter::

    x/100, y/100, vx/50, vy/50, damage/100, facing, on_ground, is_actionable,
    jumps_left/2, action_frame/60, shield, dist_to_blast/300, hitstun,
    is_on_rebirth, action_state one-hot(32), dx/100, dy/100, distance/200,
    facing_rel, is_escape, is_tech, is_ledge, is_catch, is_landing,
    is_shieldbreak

``action_state`` is a curated one-hot of 32 entries: 31 common fighter status
kinds plus a final ``OTHER`` bucket for everything else.  The curated list is
fixed (ids are the game's status kinds, cross-referenced with schema.py) and
documented in :data:`CURATED_ACTION_STATES`.

Normalization is a separate, off-by-default option.  ``normalize=False``
(the default, used by BC pipelines) returns the raw physical values in the
same field order; ``normalize=True`` divides by the contract constants above.
The divisor of every field is listed in :data:`SCALAR_FIELDS` /
:data:`RELATIVE_FIELDS`.

The profile deliberately contains **no** stock, dead count, stage id, match
state, opponent id, control state, or projectile data.  Those stay in
``info["raw_*"]``.

An empty/unreadable fighter slot is encoded as all zeros (no fake WAIT
one-hot).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from gymnasium import spaces

from rl.memory import status_kinds

# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------
#: Damage is recorded in percent.  ``damage_mode="bucket"`` replaces it with
#: a 10 % bucket index (0..20) so 83 % and 87 % are the same input.  Everything
#: at 200 % or above shares the last bucket: knockback is already lethal there,
#: so the exact percent adds nothing.  ``"none"`` zeroes the slot.  Training
#: must use the same mode as the live env.
DAMAGE_BUCKET = 10.0
DAMAGE_BUCKETS = 21
DAMAGE_MODES = ("raw", "bucket", "none")

#: (field name, divisor used when ``normalize=True``) in profile order.
SCALAR_FIELDS: tuple[tuple[str, float], ...] = (
    ("x", 100.0),
    ("y", 100.0),
    ("vx", 50.0),
    ("vy", 50.0),
    ("damage", 100.0),
    ("facing", 1.0),
    ("on_ground", 1.0),
    ("is_actionable", 1.0),
    ("jumps_left", 2.0),
    ("action_frame", 60.0),
    ("shield", 1.0),
    ("dist_to_blast", 300.0),
    ("hitstun", 1.0),
    ("is_on_rebirth", 1.0),
)

#: Curated action states: (name, game status kind) in one-hot order.
#: The final (32nd) slot is OTHER.  The ids are this build's real status kinds
#: (see rl.memory.status_kinds): the recovery states (helpless fall, ledge,
#: KO, respawn, Up-B) first, then the most frequent states of the DK
#: recordings.  Only ids verified by a probe get a name; the rest are ``S<id>``
#: because ``schema.ACTION_STATE_NAMES`` is wrong for most of them.  The old
#: table covered 39 % of recorded frames; this one covers 80 %.
CURATED_ACTION_STATES: tuple[tuple[str, int], ...] = (
    ("WAIT", 0),
    ("S3", 3),
    ("S7", 7),
    ("GUARD", 10),
    ("S11", 11),
    ("S12", 12),
    ("S14", 14),
    ("HELPLESS_FALL", 16),
    ("S22", 22),
    ("S24", 24),
    ("S28", 28),
    ("S32", 32),
    ("S33", 33),
    ("S34", 34),
    ("S39", 39),
    ("S41", 41),
    ("S43", 43),
    ("S44", 44),
    ("S53", 53),
    ("DAMAGE", 54),
    ("S72", 72),
    ("S73", 73),
    ("S79", 79),
    ("LEDGE_CATCH_MOVE", 117),
    ("LEDGE_CATCH", 118),
    ("LEDGE_HANG", 119),
    ("REBIRTH_DESCENT", 181),
    ("REBIRTH_PLATFORM", 182),
    ("KO", 470),
    ("SPECIAL_HI", 478),
    ("S505", 505),
)

ACTION_ONEHOT_SIZE = 32
OTHER_INDEX = ACTION_ONEHOT_SIZE - 1
ACTION_STATE_ONEHOT: dict[int, int] = {
    state_id: index for index, (_name, state_id) in enumerate(CURATED_ACTION_STATES)
}
# Live evidence (Agent B): held shield is status 10 (GUARD); 8 GUARD_START and
# 9 GUARD_ON are the same shield family and share the GUARD one-hot slot.
for _alias in (8, 9):
    ACTION_STATE_ONEHOT.setdefault(_alias, ACTION_STATE_ONEHOT[10])
del _alias

assert len(CURATED_ACTION_STATES) == OTHER_INDEX, "curated list must fill 31 slots"

#: (derived field name, divisor) after the one-hot block.
RELATIVE_FIELDS: tuple[tuple[str, float], ...] = (
    ("dx", 100.0),
    ("dy", 100.0),
    ("distance", 200.0),
    ("facing_rel", 1.0),
)

#: Boolean families over the real status ids (rl.memory.status_kinds),
#: written after the relative block.  1.0 when the fighter's status belongs to
#: the family, else 0.0.  They keep the recovery-relevant distinctions visible
#: even for ids that share the OTHER one-hot bucket.
FAMILY_FLAGS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("is_ledge", tuple(sorted(status_kinds.LEDGE_KINDS))),       # 117-123
    ("is_special", tuple(sorted(status_kinds.SPECIAL_KINDS))),   # 476-479
    ("is_helpless", tuple(sorted(status_kinds.HELPLESS_KINDS))), # 16
    ("is_ko_rebirth", (status_kinds.KO,
                       *sorted(status_kinds.REBIRTH_KINDS))),    # 470, 181, 182
    ("is_damage", tuple(sorted(status_kinds.DAMAGE_KINDS))),     # 54-62
    ("is_down", tuple(sorted(status_kinds.DOWN_KINDS))),         # 505-511
)
FAMILY_FLAG_NAMES: tuple[str, ...] = tuple(name for name, _ids in FAMILY_FLAGS)

_HITSTUN_POS = [n for n, _d in SCALAR_FIELDS].index("hitstun")
PER_FIGHTER_SIZE = (len(SCALAR_FIELDS) + ACTION_ONEHOT_SIZE
                    + len(RELATIVE_FIELDS) + len(FAMILY_FLAGS))
N_FIGHTERS = 2
OBS_SIZE = PER_FIGHTER_SIZE * N_FIGHTERS
OBS_DTYPE = np.float32
#: Safety clamp so a torn/garbage frame can never leave the Box.  Raw SSBU
#: coordinates are |x|,|y| < 1e4 and derived velocities far below this.
OBS_LIMIT = 1.0e6

_ACTION_NAMES = tuple(name for name, _id in CURATED_ACTION_STATES) + ("OTHER",)


def _fighter_feature_names(fighter_index: int) -> list[str]:
    prefix = f"p{fighter_index + 1}_"
    names = [prefix + name for name, _scale in SCALAR_FIELDS]
    names += [prefix + "action_" + name for name in _ACTION_NAMES]
    names += [prefix + name for name, _scale in RELATIVE_FIELDS]
    names += [prefix + name for name in FAMILY_FLAG_NAMES]
    return names


#: Human-readable name of every observation component, length :data:`OBS_SIZE`.
FEATURE_NAMES: tuple[str, ...] = tuple(
    name for i in range(N_FIGHTERS) for name in _fighter_feature_names(i)
)

#: Component slices, for debugging and tests: name -> (start, stop).
SLICES: dict[str, slice] = {}
for _i in range(N_FIGHTERS):
    _base = _i * PER_FIGHTER_SIZE
    SLICES[f"p{_i + 1}_scalars"] = slice(_base, _base + len(SCALAR_FIELDS))
    SLICES[f"p{_i + 1}_action"] = slice(_base + len(SCALAR_FIELDS),
                                        _base + len(SCALAR_FIELDS) + ACTION_ONEHOT_SIZE)
    _rel = _base + len(SCALAR_FIELDS) + ACTION_ONEHOT_SIZE
    SLICES[f"p{_i + 1}_relative"] = slice(_rel, _rel + len(RELATIVE_FIELDS))
    SLICES[f"p{_i + 1}_family"] = slice(_rel + len(RELATIVE_FIELDS),
                                        _base + PER_FIGHTER_SIZE)
del _i, _base


def observation_space(normalize: bool = False) -> spaces.Box:
    """The v1 policy observation Box (flat float32)."""
    del normalize  # the space is the same; only the value scaling differs
    return spaces.Box(low=-OBS_LIMIT, high=OBS_LIMIT,
                      shape=(OBS_SIZE,), dtype=OBS_DTYPE)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fighter_valid(fighter: dict) -> bool:
    """True when the slot was readable this frame (raw_flags != 0)."""
    flags = int(fighter.get("raw_flags", 0) or 0)
    if flags:
        return True
    # Be liberal with hand-made test dicts that omit raw_flags.
    return any(bool(v) for k, v in fighter.items()
               if k not in ("raw_flags", "entry_id", "kind"))


def _number(value: Any) -> float:
    """Coerce a raw field to a finite, bounded float."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    # A finite but absurd value (some action frames carry 3.4e38) is a bad
    # read, not a large measurement: treat it as missing.  Clipping it would
    # still overflow float16 training windows.
    if not math.isfinite(out) or abs(out) > OBS_LIMIT:
        return 0.0
    return out


def _action_onehot(state: Any) -> np.ndarray:
    onehot = np.zeros(ACTION_ONEHOT_SIZE, dtype=OBS_DTYPE)
    try:
        state_int = int(state)
    except (TypeError, ValueError):
        state_int = -1
    onehot[ACTION_STATE_ONEHOT.get(state_int, OTHER_INDEX)] = 1.0
    return onehot


def damage_bucket(value: Any) -> float:
    """10 % damage bucket index, clipped to ``DAMAGE_BUCKETS - 1``."""
    v = _number(value)
    return float(min(max(math.floor(v / DAMAGE_BUCKET), 0), DAMAGE_BUCKETS - 1))


def _fighter_vector(fighter: dict, normalize: bool, out: np.ndarray,
                    damage_mode: str = "raw") -> None:
    if not fighter_valid(fighter):
        out.fill(0.0)          # empty slot: no WAIT one-hot, all zeros
        return

    pos = 0
    for name, scale in SCALAR_FIELDS:
        value = _number(fighter.get(name, 0.0))
        if name == "damage" and damage_mode != "raw":
            if damage_mode == "bucket":
                out[pos] = damage_bucket(value)
            # "none": leave the slot at zero
        else:
            out[pos] = value / scale if normalize else value
        pos += 1

    # `hitstun` is recomputed from the status id so recordings made with the
    # old reader (which flagged 476-479, the specials) agree with live frames.
    try:
        _st = int(fighter.get("action_state", -1))
    except (TypeError, ValueError):
        _st = -1
    out[_HITSTUN_POS] = 1.0 if _st in status_kinds.HITSTUN_KINDS else 0.0

    out[pos:pos + ACTION_ONEHOT_SIZE] = _action_onehot(fighter.get("action_state", -1))
    pos += ACTION_ONEHOT_SIZE

    dx = _number(fighter.get("dx", 0.0))
    dy = _number(fighter.get("dy", 0.0))
    distance = math.hypot(dx, dy)
    values = (dx, dy, distance, _number(fighter.get("facing_rel", 0)))
    for value, (name, scale) in zip(values, RELATIVE_FIELDS):
        del name
        out[pos] = value / scale if normalize else value
        pos += 1

    try:
        state_int = int(fighter.get("action_state", -1))
    except (TypeError, ValueError):
        state_int = -1
    for _name, ids in FAMILY_FLAGS:
        out[pos] = 1.0 if state_int in ids else 0.0
        pos += 1

    np.clip(out, -OBS_LIMIT, OBS_LIMIT, out=out)


def _fighter_list(snapshot: Any) -> Sequence[dict]:
    """Accept a RawSnapshot, a mapping with 'fighters', or a sequence of dicts."""
    fighters = getattr(snapshot, "fighters", None)
    if fighters is not None:
        return fighters
    if isinstance(snapshot, dict) and "fighters" in snapshot:
        return snapshot["fighters"]
    if isinstance(snapshot, (list, tuple)):
        return snapshot
    raise TypeError(f"cannot read fighters from {type(snapshot)!r}")


# --------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------
class PolicyObservation:
    """Build the flat v1 policy observation from a raw snapshot.

    Parameters
    ----------
    normalize:
        ``False`` (default) keeps raw physical values in the fixed profile
        order.  ``True`` applies the CONTRACTS.md divisors (x/100, vx/50, ...).
    """

    size = OBS_SIZE
    dtype = OBS_DTYPE

    def __init__(self, normalize: bool = False, damage_mode: str = "raw"):
        if damage_mode not in DAMAGE_MODES:
            raise ValueError(f"damage_mode must be one of {DAMAGE_MODES}, "
                             f"got {damage_mode!r}")
        self.normalize = bool(normalize)
        self.damage_mode = str(damage_mode)

    def per_fighter(self, fighter: dict) -> np.ndarray:
        out = np.zeros(PER_FIGHTER_SIZE, dtype=OBS_DTYPE)
        _fighter_vector(fighter, self.normalize, out, self.damage_mode)
        return out

    def build(self, snapshot: Any) -> np.ndarray:
        """Return the float32 observation for ``snapshot`` (P1 then P2)."""
        fighters = _fighter_list(snapshot)
        obs = np.zeros(OBS_SIZE, dtype=OBS_DTYPE)
        for i in range(N_FIGHTERS):
            fighter = fighters[i] if i < len(fighters) else {}
            _fighter_vector(fighter, self.normalize,
                            obs[i * PER_FIGHTER_SIZE:(i + 1) * PER_FIGHTER_SIZE],
                            self.damage_mode)
        return obs

    __call__ = build


def build_observation(snapshot: Any, normalize: bool = False,
                      damage_mode: str = "raw") -> np.ndarray:
    """Convenience wrapper around :class:`PolicyObservation`."""
    return PolicyObservation(normalize=normalize,
                             damage_mode=damage_mode).build(snapshot)


def observation_metadata() -> dict:
    """Small JSON-able description of profile v1 for docs/records."""
    return {
        "profile": "v1",
        "size": OBS_SIZE,
        "per_fighter": PER_FIGHTER_SIZE,
        "dtype": "float32",
        "normalize_default": False,
        "scalars": [{"name": n, "divisor": s} for n, s in SCALAR_FIELDS],
        "action_states": [{"name": n, "id": i} for n, i in CURATED_ACTION_STATES]
                         + [{"name": "OTHER", "id": None}],
        "relative": [{"name": n, "divisor": s} for n, s in RELATIVE_FIELDS],
        "family_flags": [{"name": n, "states": list(ids)} for n, ids in FAMILY_FLAGS],
        "excluded": ["stock", "dead", "stage", "match_state", "opponent_id",
                     "controls", "projectiles"],
    }
