"""Shared field layout for the SSBU state bridge.

reader.py writes this block, verify_gui.py and probe.py read it.  It lives in
POSIX shared memory (/dev/shm/ssbu_state.bin) as a plain mmap-able file, so a
reader maps it once and reads the latest frame with no copies and no locks.

Concurrency model: single writer, many readers, no locks.
  * the writer publishes `seq` as an ODD value, writes the body, then bumps
    `seq` to the next EVEN value.
  * a reader that sees an odd `seq` (or a changed `seq` across its read) knows
    it raced and retries.  Every field is a plain little-endian scalar at a
    fixed offset, so a torn read is always detectable.

Change VERSION when the layout changes; readers refuse a mismatched version.
"""

from __future__ import annotations

import mmap
import os
import struct

MAGIC = b"SSBU"
VERSION = 7

SHM_PATH = "/dev/shm/ssbu_state.bin"
MAX_FIGHTERS = 2          # we only care about player 0 and 1

# ---- header -------------------------------------------------------------
# magic, version, n_fighters, seq, ts_ns, fps, reader_pid, fight_on,
# match_state, blast_left, blast_right, blast_top, blast_bottom,
# stage_id, game_frame, flags, pad, pad, pad
HEADER_FIELDS = [
    ("magic",         "4s"),
    ("version",       "I"),
    ("n_fighters",    "I"),
    ("seq",           "I"),
    ("ts_ns",         "Q"),
    ("fps",           "f"),
    ("reader_pid",    "I"),
    ("fight_on",      "I"),
    ("match_state",   "I"),
    ("blast_left",    "f"),
    ("blast_right",   "f"),
    ("blast_top",     "f"),
    ("blast_bottom",  "f"),
    ("stage_id",      "i"),
    ("game_frame",    "I"),
    ("flags",         "I"),
    ("pad0",          "I"),
    ("pad1",          "I"),
    ("pad2",          "I"),
]
HEADER_NAMES = [name for name, _fmt in HEADER_FIELDS]
HEADER_FMT = "<" + "".join(fmt for _name, fmt in HEADER_FIELDS)
HEADER_SIZE = 80
SEQ_OFFSET = 12
TS_OFFSET = 16
FPS_OFFSET = 24
FIGHT_OFFSET = 32
NFIGHTERS_OFFSET = 8
MATCH_STATE_OFFSET = 36
BLAST_OFFSET = 40         # float32 left, right, top, bottom
STAGE_ID_OFFSET = 56
GAME_FRAME_OFFSET = 60
FLAGS_OFFSET = 64

#: header.flags bits
HEADER_FLAG_STAGE_ID = 1 << 0        # stage_id was read this frame
HEADER_FLAG_CONTROL_LAYOUT = 1 << 1  # the ControlModule virtual getters were resolved

# ---- one fighter --------------------------------------------------------
# 32 slots x 4 bytes = 128 bytes (CONTRACTS.md v7).
# FIELDS[i] is exactly the i-th value in FIGHTER_FMT.
FIELDS = [
    # (name,           dtype,     description)
    ("entry_id",        "int32",   "player slot 0..7"),
    ("kind",            "int32",   "fighter kind (Mario = 0, Donkey Kong = 1)"),
    ("stock",           "int32",   "remaining stocks (u32 at record+0xD8, stored signed)"),
    ("damage",          "float32", "damage percent shown on the HUD"),
    ("x",               "float32", "world position x"),
    ("y",               "float32", "world position y"),
    ("facing",          "float32", "facing direction (+1 right / -1 left)"),
    ("action_state",    "int32",   "fighter status kind"),
    ("action_frame",    "float32", "frames into the current action"),
    ("on_ground",       "int32",   "1 when standing on a surface"),
    ("is_actionable",   "int32",   "1 when the game's transition terms allow an action"),
    ("jumps_left",      "int32",   "air jumps remaining"),
    ("vx",              "float32", "velocity x (derived from position)"),
    ("vy",              "float32", "velocity y (derived from position)"),
    ("dist_to_blast",   "float32", "closest distance to any blast zone"),
    ("dx",              "float32", "opponent x - self x"),
    ("dy",              "float32", "opponent y - self y"),
    ("facing_rel",      "int32",   "+1 opponent in front, -1 behind, 0 unknown"),
    ("is_cpu",          "int32",   "1 when the record says CPU (record+0x92)"),
    ("is_on_rebirth",   "int32",   "1 while on the revival platform (record+0x38C)"),
    ("shield",          "int32",   "1 while in a GUARD status (v1, derived)"),
    ("shield_size",     "float32", "shield size (0 until a source is known)"),
    ("is_enable_cancel", "int32",  "CancelModule::is_enable_cancel (module+0x1C)"),
    ("hitstun",         "int32",   "1 while in a DAMAGE reaction status (derived)"),
    ("buttons",         "uint32",  "raw ControlModule button bitmask"),
    ("stick_x",         "float32", "left stick x [-1, 1]"),
    ("stick_y",         "float32", "left stick y [-1, 1]"),
    ("trigger",         "float32", "analog trigger/shoulder value [0, 1]"),
    ("cstick_x",        "float32", "C-stick x [-1, 1]"),
    ("cstick_y",        "float32", "C-stick y [-1, 1]"),
    ("raw_flags",       "uint32",  "bit i = field i was readable this frame"),
    ("pad",             "uint32",  "padding"),
]

FIELD_NAMES = [f[0] for f in FIELDS]
FIELD_INDEX = {name: i for i, name in enumerate(FIELD_NAMES)}
RAW_FLAGS_INDEX = FIELD_INDEX["raw_flags"]
PAD_INDEX = FIELD_INDEX["pad"]
FIGHTER_WORDS = len(FIELDS)
FIGHTER_SIZE = FIGHTER_WORDS * 4

_TYPE_CODE = {"int32": "i", "uint32": "I", "float32": "f"}
FIGHTER_FMT = "<" + "".join(_TYPE_CODE[dtype] for _name, dtype, _desc in FIELDS)

BLOCK_SIZE = HEADER_SIZE + FIGHTER_SIZE * MAX_FIGHTERS

_HEADER = struct.Struct(HEADER_FMT)
_FIGHTER = struct.Struct(FIGHTER_FMT)

assert len(FIELDS) == 32, len(FIELDS)
assert _HEADER.size == HEADER_SIZE, _HEADER.size
assert _FIGHTER.size == FIGHTER_SIZE, _FIGHTER.size
assert BLOCK_SIZE == 336, BLOCK_SIZE

# ---- canonical button order ---------------------------------------------
# CONTRACTS.md: A B X Y L R ZL ZR PLUS MINUS DPAD_L DPAD_R.
# The bit positions are the game's own ControlModule mask, proven by the
# check_button table the game reads at guest 0x4538fec (lua_bind
# ControlModule__check_button_trigger_impl: mask & s_button_bits[id]):
#   0 A=0x1, 1 B=0x2, 2 X=0x4, 3 Y=0x8, 4 L=0x20, 5 R=0x400,
#   6 ZL=0x1000, 7 ZR=0x2000, 8 PLUS=0x4000, 9 MINUS=0x10,
#   10 DPAD_L=0x40, 11 DPAD_R=0x80
BUTTON_ORDER = ("A", "B", "X", "Y", "L", "R", "ZL", "ZR",
                "PLUS", "MINUS", "DPAD_L", "DPAD_R")
BUTTON_BITS = {
    "A": 0x0001, "B": 0x0002, "X": 0x0004, "Y": 0x0008,
    "L": 0x0020, "R": 0x0400, "ZL": 0x1000, "ZR": 0x2000,
    "PLUS": 0x4000, "MINUS": 0x0010, "DPAD_L": 0x0040, "DPAD_R": 0x0080,
}
BUTTON_MASKS = tuple(BUTTON_BITS[n] for n in BUTTON_ORDER)


def decode_buttons(mask: int) -> tuple[int, ...]:
    """Raw ControlModule bitmask -> 12 ints in BUTTON_ORDER."""
    mask = int(mask or 0) & 0xFFFFFFFF
    return tuple(1 if mask & bit else 0 for bit in BUTTON_MASKS)


def buttons_pressed(mask: int) -> list[str]:
    """Names of the buttons set in a raw ControlModule bitmask."""
    mask = int(mask or 0) & 0xFFFFFFFF
    return [name for name in BUTTON_ORDER if mask & BUTTON_BITS[name]]


def encode_buttons(names) -> int:
    """Canonical button names (or a 12-element 0/1 vector) -> raw mask."""
    mask = 0
    if isinstance(names, dict):
        for name, on in names.items():
            if on and name in BUTTON_BITS:
                mask |= BUTTON_BITS[name]
        return mask
    for i, item in enumerate(names):
        if isinstance(item, str):
            if item in BUTTON_BITS:
                mask |= BUTTON_BITS[item]
        elif item and i < len(BUTTON_ORDER):
            mask |= BUTTON_MASKS[i]
    return mask


# ---- match_state enum ---------------------------------------------------
class MatchState:
    MENU      = 0
    IN_MATCH  = 1
    RESULTS   = 2


# ---- blast zone fallback (only used before the live bounds are known) ----
# The live bounds are read from the game's camera manager (see gamestate.py);
# these constants are a last resort so the value is never NaN.
FD_BLAST_LEFT   = -240.0
FD_BLAST_RIGHT  =  240.0
FD_BLAST_TOP    =  180.0
FD_BLAST_BOTTOM = -140.0

def dist_to_blastzone(x: float, y: float, bounds=None) -> float:
    """Closest distance to any blast zone boundary.

    `bounds` is (left, right, top, bottom) read live from the game's camera
    manager.  Falls back to the Final Destination constants when unknown.
    """
    left, right, top, bottom = bounds if bounds else (
        FD_BLAST_LEFT, FD_BLAST_RIGHT, FD_BLAST_TOP, FD_BLAST_BOTTOM)
    dx = min(x - left, right - x)
    dy = min(y - bottom, top - y)
    return max(0.0, min(dx, dy))


# ---- header helpers -----------------------------------------------------
def pack_header(n_fighters: int, seq: int, ts_ns: int, fps: float,
                reader_pid: int, fight_on: int, match_state: int,
                bounds=None, stage_id: int = 0, game_frame: int = 0,
                flags: int = 0) -> bytes:
    left, right, top, bottom = bounds if bounds else (0.0, 0.0, 0.0, 0.0)
    return _HEADER.pack(MAGIC, VERSION, n_fighters, seq, ts_ns, fps,
                        reader_pid, fight_on, match_state,
                        left, right, top, bottom, int(stage_id), int(game_frame),
                        int(flags) & 0xFFFFFFFF, 0, 0, 0)


def unpack_header(buf) -> dict:
    vals = _HEADER.unpack_from(buf, 0)
    return dict(zip(HEADER_NAMES, vals))


# ---- fighter helpers ----------------------------------------------------
_INT32_MIN, _INT32_MAX = -2 ** 31, 2 ** 31 - 1


def _coerce(dtype: str, value):
    """Keep a live value inside what the shared layout can hold."""
    if value is None:
        return 0.0 if dtype == "float32" else 0
    if dtype == "float32":
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return 0
    if dtype == "uint32":
        return max(0, min(2 ** 32 - 1, iv))
    return max(_INT32_MIN, min(_INT32_MAX, iv))


def pack_fighter(values: dict, raw_flags: int | None = None) -> bytes:
    if raw_flags is None:
        raw_flags = values.get("raw_flags", 0)
    raw_flags = int(raw_flags or 0) & 0xFFFFFFFF
    row = []
    for name, dtype, _desc in FIELDS:
        if name == "raw_flags":
            row.append(raw_flags)
        elif name == "pad":
            row.append(0)
        else:
            row.append(_coerce(dtype, values.get(name)))
    return _FIGHTER.pack(*row)


def unpack_fighter(buf, index: int) -> dict:
    off = HEADER_SIZE + FIGHTER_SIZE * index
    vals = _FIGHTER.unpack_from(buf, off)
    return dict(zip(FIELD_NAMES, vals))


def fighter_offset(index: int) -> int:
    return HEADER_SIZE + FIGHTER_SIZE * index


# ---- mapping helpers ----------------------------------------------------
def open_writer(path: str = SHM_PATH):
    """Create/truncate the block and return a writable mmap."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    os.ftruncate(fd, BLOCK_SIZE)
    mm = mmap.mmap(fd, BLOCK_SIZE)
    os.close(fd)
    return mm


def open_reader(path: str = SHM_PATH):
    """Map the block read-only. Raises FileNotFoundError if reader.py never ran."""
    fd = os.open(path, os.O_RDONLY)
    mm = mmap.mmap(fd, BLOCK_SIZE, prot=mmap.PROT_READ)
    os.close(fd)
    return mm


def describe() -> str:
    lines = [f"shm {SHM_PATH}  version {VERSION}  size {BLOCK_SIZE} bytes",
             f"header {HEADER_SIZE} bytes, {MAX_FIGHTERS} fighters x {FIGHTER_SIZE} bytes",
             f"match_state: 0=MENU 1=IN_MATCH 2=RESULTS",
             "buttons (canonical order / ControlModule bit): "
             + " ".join(f"{n}=0x{BUTTON_BITS[n]:x}" for n in BUTTON_ORDER)]
    for i, (name, dtype, desc) in enumerate(FIELDS):
        lines.append(f"  [{i:2d}] {name:<16} {dtype:<8} {desc}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe())

# ---- action_state human-readable names ------------------------------------
# WARNING: this table came from an older numbering and is WRONG for most ids
# (an "aerial attack" id sits on the ground, the ledge ids read as items, the
# specials read as tumble).  Only the ids overlaid by VERIFIED_STATUS_NAMES
# below are trustworthy; see status_kinds.py.  Do not name a state from here.
ACTION_STATE_NAMES = {
    0:  "WAIT",
    1:  "WALK",
    2:  "RUN",
    3:  "RUN_BRAKE",
    4:  "TURN",
    5:  "TURN_RUN",
    6:  "DASH",
    7:  "DASH_BACK",
    8:  "GUARD_START",
    9:  "GUARD_ON",
    10: "GUARD",
    11: "JUMP_SQUAT",
    12: "JUMP",
    13: "JUMP_AERIAL",
    14: "JUMP_F",
    15: "JUMP_B",
    16: "JUMP_AERIAL_F",
    17: "JUMP_AERIAL_B",
    18: "FALL",
    19: "FALL_AERIAL",
    20: "FALL_SPECIAL",
    21: "FALL_SPECIAL_AERIAL",
    22: "ATTACK",
    23: "ATTACK_DASH",
    24: "ATTACK_S3",
    25: "ATTACK_HI3",
    26: "ATTACK_LW3",
    27: "ATTACK_S4",
    28: "ATTACK_HI4",
    29: "ATTACK_LW4",
    30: "ATTACK_AIR",
    31: "ATTACK_AIR_F",
    32: "ATTACK_AIR_B",
    33: "ATTACK_AIR_HI",
    34: "ATTACK_AIR_LW",
    35: "SPECIAL_S",
    36: "SPECIAL_HI",
    37: "SPECIAL_LW",
    38: "SPECIAL_N",
    39: "SPECIAL_S_AIR",
    40: "SPECIAL_HI_AIR",
    41: "SPECIAL_LW_AIR",
    42: "SPECIAL_N_AIR",
    43: "THROW",
    44: "THROW_F",
    45: "THROW_B",
    46: "THROW_HI",
    47: "THROW_LW",
    48: "CATCH",
    49: "CATCH_WAIT",
    50: "CATCH_TURN",
    51: "CATCH_CUT",
    52: "CATCH_ATTACK",
    53: "THROW_CATCH",
    54: "DAMAGE",
    55: "DAMAGE_FALL",
    56: "DAMAGE_FLY",
    57: "DAMAGE_FLY_HI",
    58: "DAMAGE_FLY_LW",
    59: "DAMAGE_FLY_ROLL",
    60: "DAMAGE_FLY_REFLECT",
    61: "DAMAGE_FLY_ABSORB",
    62: "DAMAGE_GROUND",
    63: "DOWN",
    64: "DOWN_WAIT",
    65: "DOWN_DAMAGE",
    66: "DOWN_FORWARD",
    67: "DOWN_BACK",
    68: "DOWN_STAND_F",
    69: "DOWN_STAND_B",
    70: "DOWN_STAND",
    71: "PASSIVE",
    72: "PASSIVE_FB",
    73: "CLIFF_CATCH",
    74: "CLIFF_WAIT",
    75: "CLIFF_CLIMB_QUICK",
    76: "CLIFF_CLIMB_SLOW",
    77: "CLIFF_ATTACK",
    78: "ESCAPE_F",
    79: "ESCAPE_B",
    80: "ESCAPE_AIR",
    81: "REBOUND",
    82: "REBOUND_STOP",
    83: "OTTOTTO",
    84: "LANDING",
    85: "LANDING_AIR",
    86: "LANDING_FALL_SPECIAL",
    87: "ATTACK_AIR_CATCH",
    88: "CATCH_CUT_AIR",
    89: "CATCH_ATTACK_AIR",
    90: "THROW_ITEM",
    91: "THROW_ITEM_F",
    92: "THROW_ITEM_B",
    93: "THROW_ITEM_HI",
    94: "THROW_ITEM_LW",
    95: "THROW_ITEM_AIR",
    96: "THROW_ITEM_AIR_F",
    97: "THROW_ITEM_AIR_B",
    98: "THROW_ITEM_AIR_HI",
    99: "THROW_ITEM_AIR_LW",
    100: "DROP_ITEM",
    101: "DROP_ITEM_AIR",
    102: "ITEM_PICKUP",
    103: "ITEM_PICKUP_AIR",
    104: "ITEM_SWALLOW",
    105: "ITEM_THROW",
    106: "ITEM_THROW_F",
    107: "ITEM_THROW_B",
    108: "ITEM_THROW_HI",
    109: "ITEM_THROW_LW",
    110: "ITEM_THROW_AIR",
    111: "ITEM_THROW_AIR_F",
    112: "ITEM_THROW_AIR_B",
    113: "ITEM_THROW_AIR_HI",
    114: "ITEM_THROW_AIR_LW",
    115: "ITEM_DROP",
    116: "ITEM_DROP_AIR",
    117: "GET_ITEM",
    118: "GET_ITEM_AIR",
    119: "LOST_ITEM",
    120: "EAT_ITEM",
    121: "EAT_ITEM_AIR",
    122: "EAT_DRIVE_ITEM",
    123: "EAT_DRIVE_ITEM_AIR",
    124: "SHIELD_BREAK_FLY",
    125: "SHIELD_BREAK_FALL",
    126: "SHIELD_BREAK_DOWN_D",
    127: "SHIELD_BREAK_DOWN_U",
    128: "SHIELD_BREAK_STAND_D",
    129: "SHIELD_BREAK_STAND_U",
    130: "SHIELD_BREAK_TEETER",
    131: "SQUAT",
    132: "SQUAT_WAIT",
    133: "SQUAT_TURN",
    134: "SQUAT_WAIT_TURN",
    135: "OTTOTTO_WAIT",
    136: "OTTOTTO_TURN",
    137: "DOWN_STAND_F",
    138: "DOWN_STAND_B",
    139: "APPEAL_S",
    140: "APPEAL_HI",
    141: "APPEAL_LW",
    142: "APPEAL_S_R",
    143: "APPEAL_HI_R",
    144: "APPEAL_LW_R",
    145: "APPEAL_S_L",
    146: "APPEAL_HI_L",
    147: "APPEAL_LW_L",
    148: "SLEEP",
    149: "SLEEP_START",
    150: "SLEEP_END",
    151: "DOWN_REFLECT",
    152: "DOWN_REFLECT_D",
    153: "DOWN_REFLECT_U",
    154: "REBIRTH",
    155: "REBIRTH_WAIT",
    156: "REBIRTH_STAND",
    157: "APPEAL_WAIT",
    158: "APPEAL_WALK",
    159: "APPEAL_DASH",
    160: "APPEAL_TURN",
    161: "APPEAL_FALL",
    162: "APPEAL_LANDING",
    163: "APPEAL_PASS",
    164: "APPEAL_CATCH",
    165: "APPEAL_THROW",
    166: "APPEAL_ITEM",
    167: "APPEAL_ITEM_AIR",
    168: "DEAD_UP",
    169: "DEAD_DOWN",
    170: "DEAD_LEFT",
    171: "DEAD_RIGHT",
    172: "DEAD_UP_FALL",
    173: "DEAD_DOWN_FALL",
    174: "DEAD_LEFT_FALL",
    175: "DEAD_RIGHT_FALL",
    176: "ENTRY",
    177: "ENTRY_END",
    178: "ENTRY_START",
    179: "RESET",
    180: "REBIRTH_END",
    181: "REBIRTH_END_WAIT",
    182: "SPECIAL_HI_AIR",
    183: "SPECIAL_HI_AIR_LANDING",
    184: "SPECIAL_LW_AIR",
    185: "SPECIAL_LW_AIR_LANDING",
    186: "SPECIAL_N_AIR",
    187: "SPECIAL_N_AIR_LANDING",
    188: "SPECIAL_S_AIR",
    189: "SPECIAL_S_AIR_LANDING",
    190: "SPECIAL_S_AIR_LANDING",
    191: "FINAL",
    192: "FINAL_END",
    193: "FINAL_START",
    194: "FINAL_READY",
    195: "FINAL_SCENE01",
    196: "FINAL_SCENE02",
    197: "FINAL_SCENE03",
    198: "FINAL_SCENE04",
    199: "FINAL_SCENE05",
    200: "FINAL_SCENE06",
    201: "FINAL_SCENE07",
    202: "FINAL_SCENE09",
    203: "FINAL_SCENE10",
    204: "FINAL_SCENE11",
    205: "FINAL_SCENE12",
    206: "FINAL_SCENE13",
    207: "FINAL_SCENE14",
    208: "FINAL_SCENE15",
    209: "FINAL_SCENE16",
    210: "FINAL_SCENE17",
    211: "FINAL_SCENE18",
    212: "FINAL_SCENE19",
    213: "FINAL_SCENE20",
    214: "FINAL_SCENE21",
    215: "FINAL_SCENE22",
    216: "FINAL_SCENE23",
    217: "FINAL_SCENE24",
    218: "FINAL_SCENE25",
    219: "FINAL_SCENE26",
    220: "FINAL_SCENE27",
    221: "FINAL_SCENE28",
    222: "FINAL_SCENE29",
    223: "FINAL_SCENE30",
    224: "FINAL_SCENE31",
    225: "FINAL_SCENE32",
    226: "FINAL_SCENE33",
    227: "FINAL_SCENE34",
    228: "FINAL_SCENE35",
    229: "FINAL_SCENE36",
    230: "FINAL_SCENE37",
    231: "FINAL_SCENE38",
    232: "FINAL_SCENE39",
    233: "FINAL_SCENE40",
    234: "FINAL_SCENE41",
    235: "FINAL_SCENE42",
    236: "FINAL_SCENE43",
    237: "FINAL_SCENE44",
    238: "FINAL_SCENE45",
    239: "FINAL_SCENE46",
    240: "FINAL_SCENE47",
    241: "FINAL_SCENE48",
    242: "FINAL_SCENE49",
    243: "FINAL_SCENE50",
    244: "FINAL_SCENE51",
    245: "FINAL_SCENE52",
    246: "FINAL_SCENE53",
    247: "FINAL_SCENE54",
    248: "FINAL_SCENE55",
    249: "FINAL_SCENE56",
    250: "FINAL_SCENE57",
    251: "FINAL_SCENE58",
    252: "FINAL_SCENE59",
    263: "KO",
    264: "KO_ICE",
    265: "KO_FIRE",
    266: "KO_THUNDER",
    267: "KO_STAR",
    268: "KO_SCREEN",
    269: "KO_STAGE",
    270: "KO_FINAL",
    271: "KO_FINAL_ICE",
    272: "KO_FINAL_FIRE",
    273: "KO_FINAL_THUNDER",
    274: "KO_FINAL_STAR",
    275: "KO_FINAL_SCREEN",
    276: "KO_FINAL_STAGE",
    470: "KO_FINAL_DOWN",
    476: "TUMBLE",
    477: "TUMBLE_HIGH",
    478: "TUMBLE_LOW",
    479: "TUMBLE_AIR",
    505: "DOWN_FALL",
    506: "DOWN_FALL_ICE",
    507: "DOWN_FALL_FIRE",
    508: "DOWN_FALL_THUNDER",
    509: "DOWN_FALL_STAR",
    510: "DOWN_FALL_SCREEN",
    511: "DOWN_FALL_STAGE",
}

#: Names verified by live probes and the DK recordings (status_kinds.py).
VERIFIED_STATUS_NAMES = {
    0: "WAIT", 10: "GUARD", 16: "HELPLESS_FALL", 54: "DAMAGE",
    117: "LEDGE_CATCH_MOVE", 118: "LEDGE_CATCH", 119: "LEDGE_HANG",
    120: "LEDGE_EXIT_120", 121: "LEDGE_EXIT_121", 122: "LEDGE_EXIT_122",
    123: "LEDGE_EXIT_123", 181: "REBIRTH_DESCENT", 182: "REBIRTH_PLATFORM",
    470: "KO", 476: "SPECIAL_476", 477: "SPECIAL_477", 478: "SPECIAL_HI",
    479: "SPECIAL_479",
}
ACTION_STATE_NAMES.update(VERIFIED_STATUS_NAMES)

def action_state_name(state: int) -> str:
    return ACTION_STATE_NAMES.get(state, f"UNKNOWN({state})")
