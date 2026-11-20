"""Status-kind (``action_state``) ids that matter, verified on this build.

``schema.ACTION_STATE_NAMES`` was written from another numbering and is wrong
for almost every id.  Everything here comes from the live scenario suite
(``rl.scripts.status_scenarios``: scripted pad input, ids recorded per
decision, ``status_ids_verified.json``) and from the 2.2 M-frame DK
recordings, and is the game's **common** status table, so it is not specific to
Donkey Kong.  Only fighter-specific sub-states (ids >= ~500, e.g. DK's special
sub-states 505-511) differ per character.

Verified live (scenario -> id):

* wait 0, dash 3, run 4, run-brake 5, jump-squat 10, jump 11, midair jump 12,
  fall 14, fall-after-jump 15, **helpless fall 16** (after Up-B), landing 22,
  aerial landing 24, helpless landing 25, **guard 27 (on) / 28 (hold) /
  29 (off)**, air dodge 34, jab 39, f/u/d-tilt 42/43/44, f-smash 45/47,
  d-smash 48/49/50, u-smash 51/53, **aerial attack 54** (all aerials),
  **specials 476 N, 477 S, 478 Hi (Up-B), 479 Lw**, DK sub-states 505-511.
* ledge: catch 117, 118, hang 119 at x=+-83.6, y=-27.44 exactly; exits:
  attack 120, roll 122, getup/jump 123, then the ledge-jump rise 124.
* KO: 181 (entry, 1 decision) then 470 (flight), respawn platform 182.
* damage reactions 71-75: the state entered one decision after the ego's
  damage rises (73 and 72 in 98 % of hits, then 71, 74, 75).

Old (wrong) beliefs this replaces: 10 = guard (it is jump-squat), 54-62 =
damage (54 is the aerial attack), 73-77 = ledge (73 is a damage state),
505-511 = down (DK special sub-states), 20/21 = helpless (16).
"""

from __future__ import annotations

WAIT = 0
JUMP_SQUAT = 10
HELPLESS_FALL = 16
GUARD_ON = 27
GUARD = 28
GUARD_OFF = 29
ATTACK_AIR = 54
LEDGE_CATCH_MOVE = 117
LEDGE_CATCH = 118
LEDGE_HANG = 119
DEAD_ENTRY = 181
REBIRTH_PLATFORM = 182
KO = 470
SPECIAL_HI = 478

#: Hanging on / leaving the ledge.  Verified live: catch 117/118, hang 119,
#: attack 120, roll 122, getup/jump 123, ledge-jump rise 124.
LEDGE_KINDS = frozenset(range(117, 125))
#: Helpless after an Up Special: no jump, no attack, only a ledge or a hit.
HELPLESS_KINDS = frozenset({HELPLESS_FALL})
#: Special moves N / S / Hi / Lw (ground and air share the ids).
SPECIAL_KINDS = frozenset({476, 477, 478, 479})
#: Shield: raising, held, lowering.  The reader's `shield` field and the BC
#: shield relabel are derived from GUARD_HELD_KINDS.
GUARD_KINDS = frozenset({GUARD_ON, GUARD, GUARD_OFF})
GUARD_HELD_KINDS = frozenset({GUARD_ON, GUARD})
#: Dead / respawning.
KO_KINDS = frozenset({DEAD_ENTRY, KO})
REBIRTH_KINDS = frozenset({REBIRTH_PLATFORM})
#: Damage reactions (hitstun), measured from the state entered after damage.
DAMAGE_KINDS = frozenset({71, 72, 73, 74, 75})
HITSTUN_KINDS = DAMAGE_KINDS
#: Fighter-specific sub-states start here (DK's special sub-states 505-511).
CHARACTER_SPECIFIC_MIN = 500

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
