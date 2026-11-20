"""Status-kind (``action_state``) ids that matter, verified on this build.

``schema.ACTION_STATE_NAMES`` was written from an older/different numbering
and is wrong for most ids (an "aerial attack" id sits on the ground, "item"
ids are the ledge, "tumble" ids are special moves).  Do not name states from
it.  Everything below was derived from live probes (scripted jump / Up-B off
the ledge) and from 2.16 M frames of recorded play, and is the common status
table of the game, so it is **not specific to Donkey Kong**; only the
fighter-specific sub-states (ids >= ~500) differ per character.

Evidence for each family is in ``findings/FINDINGS.md``:

* ledge 117-123: ``x = +-83.6, y = -27.44`` exactly while in 118/119 (a
  fixed hang point); 120-123 start there and rise (getup / attack / roll /
  jump).
* helpless 16: follows Up-B (478) in the air 191 times, not actionable,
  constant terminal fall speed (vy ~ -99.5).
* special 476-479: grounded and airborne special moves; 478 is Up-B (a probe
  that presses Up+B always enters it and rises).
* KO 470 / respawn 181 (descent), 182 (platform at x~0, y~84).
* damage 54-62 and down 505-511 are the hitstun / knocked-down families.
"""

from __future__ import annotations

WAIT = 0
GUARD = 10
HELPLESS_FALL = 16
DAMAGE = 54
LEDGE_CATCH_MOVE = 117
LEDGE_CATCH = 118
LEDGE_HANG = 119
REBIRTH_DESCENT = 181
REBIRTH_PLATFORM = 182
KO = 470
SPECIAL_HI = 478

#: Hanging on / climbing out of the ledge.  A fighter in one of these is safe.
LEDGE_KINDS = frozenset(range(117, 124))
#: Helpless after an Up Special: no jump, no attack, only a ledge or a hit.
HELPLESS_KINDS = frozenset({HELPLESS_FALL})
#: Special moves (N, S, Hi, Lw).  They are NOT damage reactions.
SPECIAL_KINDS = frozenset({476, 477, 478, 479})
#: Dead / respawning.
KO_KINDS = frozenset({KO})
REBIRTH_KINDS = frozenset({REBIRTH_DESCENT, REBIRTH_PLATFORM})
#: Damage reactions (hitstun) and knocked-down states.
DAMAGE_KINDS = frozenset(range(54, 63))
DOWN_KINDS = frozenset(range(505, 512))
HITSTUN_KINDS = DAMAGE_KINDS | DOWN_KINDS

#: Measured ledge hang point of Final Destination (stage coordinates).
LEDGE_X = 83.6
LEDGE_Y = -27.44


def is_ledge(state) -> bool:
    try:
        return int(state) in LEDGE_KINDS
    except (TypeError, ValueError):
        return False


def is_helpless(state) -> bool:
    try:
        return int(state) in HELPLESS_KINDS
    except (TypeError, ValueError):
        return False
