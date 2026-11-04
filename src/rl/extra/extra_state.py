#!/usr/bin/env python3
"""Extra (raw-only) live game state for the SSBU recomp: reader + publisher.

This is the *extras* channel.  It never touches the policy observation and it
does not modify the core reader/schema/gamestate.  Everything here is
read-only: /proc/<pid>/mem reads plus the game's own symbol table.

Output: /dev/shm/ssbu_extra.bin, a self-describing block (see extra_map.json):

    header 32 bytes:  magic "SSBX", version, seq, ts_ns, count,
                      total_bytes, flags
    record 56 + value: name[48], type u8, verified u8, index u8, rsv u8,
                      length u32, value bytes padded to 8

Type tags: 0 i64, 1 f64, 2 bytes, 3 bool(u8), 4 u32, 5 vec3f, 6 vec4f, 7 str.

Evidence model
--------------
Every struct offset comes from disassembling the game's own lua_bind accessor
(see extra_map.json for the exact instruction).  In addition, the concrete
module getters are resolved at run time through the live vtable and their
machine code is decoded (a small AArch64 load-pattern decoder) so the field
offset is read out of the running build instead of being trusted blindly.
When the decoder succeeds the record is marked verified; when a field uses a
documented constant because the getter is not a simple load, the map says so.

Usage:
    python3 tools/mem/extra_state.py --print          # one snapshot, human
    python3 tools/mem/extra_state.py --json           # one snapshot, JSON
    python3 tools/mem/extra_state.py --duration 30    # publish at 60 Hz
    python3 tools/mem/extra_state.py --duration 30 --if-live   # only in a match
    python3 tools/mem/extra_state.py --no-shm --print # do not touch /dev/shm
"""

from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.memory.elfsym import ElfSymbols          # noqa: E402
from rl.memory.memapi import ProcessMemory, find_game_pid, guest_image_base  # noqa: E402

# ---------------------------------------------------------------------------
# format
# ---------------------------------------------------------------------------
MAGIC = b"SSBX"
VERSION = 1
SHM_PATH = "/dev/shm/ssbu_extra.bin"
MAX_BLOCK = 1 << 20                    # 1 MiB, fixed so readers can mmap once

HEADER_FMT = "<4sIIQIII"
_HEADER = struct.Struct(HEADER_FMT)
HEADER_SIZE = _HEADER.size                         # 32
RECORD_FMT = "<48sBBBBI"
_RECORD = struct.Struct(RECORD_FMT)
RECORD_SIZE = _RECORD.size                         # 56

T_I64, T_F64, T_BYTES, T_BOOL, T_U32, T_VEC3, T_VEC4, T_STR = range(8)
TYPE_NAMES = {
    T_I64: "i64", T_F64: "f64", T_BYTES: "bytes", T_BOOL: "bool",
    T_U32: "u32", T_VEC3: "vec3f", T_VEC4: "vec4f", T_STR: "str",
}
NAME_MAX = 47

FLAG_LIVE = 1 << 0                     # fight_on at the time of writing
FLAG_DECODED = 1 << 1                  # at least one getter decoded live


class Rec:
    """One self-describing extra value."""

    __slots__ = ("name", "type", "value", "index", "verified", "note")

    def __init__(self, name: str, type_: int, value, index: int = 0,
                 verified: bool = False, note: str = ""):
        self.name = name
        self.type = int(type_)
        self.value = value
        self.index = int(index)
        self.verified = bool(verified)
        self.note = note

    # -- encoding ---------------------------------------------------------
    def encoded(self) -> bytes:
        t = self.type
        if t == T_I64:
            data = struct.pack("<q", int(self.value))
        elif t == T_F64:
            data = struct.pack("<d", float(self.value))
        elif t == T_BYTES:
            data = bytes(self.value)
        elif t == T_BOOL:
            data = bytes([1 if self.value else 0])
        elif t == T_U32:
            data = struct.pack("<I", int(self.value) & 0xFFFFFFFF)
        elif t == T_VEC3:
            data = struct.pack("<3f", *[float(v) for v in self.value])
        elif t == T_VEC4:
            data = struct.pack("<4f", *[float(v) for v in self.value])
        elif t == T_STR:
            data = str(self.value).encode("utf-8", "replace")
        else:
            raise ValueError(f"unknown type {t}")
        return data

    def packed(self) -> bytes:
        data = self.encoded()
        name = self.name.encode("ascii", "replace")[:NAME_MAX]
        head = _RECORD.pack(name, self.type, 1 if self.verified else 0,
                            self.index & 0xFF, 0, len(data))
        pad = (-len(data)) % 8
        return head + data + b"\x00" * pad

    def value_python(self):
        if self.type == T_VEC3:
            return list(self.value)
        if self.type == T_VEC4:
            return list(self.value)
        if self.type == T_BYTES:
            return bytes(self.value).hex()
        return self.value

    def __repr__(self) -> str:
        return (f"Rec({self.name!r}, {TYPE_NAMES.get(self.type, self.type)}, "
                f"{self.value_python()!r}, idx={self.index}, "
                f"verified={self.verified})")


def pack_block(records: list[Rec], seq: int, ts_ns: int, flags: int = 0) -> bytes:
    body = b"".join(r.packed() for r in records)
    total = HEADER_SIZE + len(body)
    if total > MAX_BLOCK:
        raise ValueError(f"extra block too large: {total} > {MAX_BLOCK}")
    head = _HEADER.pack(MAGIC, VERSION, seq & 0xFFFFFFFF, int(ts_ns),
                           len(records), total, flags & 0xFFFFFFFF)
    return head + body


def unpack_block(data: bytes, strict: bool = True) -> dict:
    """Parse a block.  Returns {header..., 'records': [Rec, ...]}.

    ``strict`` verifies the 8-byte value padding; a torn block is rejected by
    checking total/count against the buffer.
    """
    if len(data) < HEADER_SIZE:
        raise ValueError("short extra block")
    (magic, version, seq, ts_ns, count, total, flags) = \
        _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported extra version {version}")
    if total < HEADER_SIZE or total > len(data):
        raise ValueError(f"bad total_bytes {total} (buffer {len(data)})")
    off = HEADER_SIZE
    records = []
    for _ in range(count):
        if off + RECORD_SIZE > total:
            raise ValueError("truncated record header")
        name_b, type_, ver, index, _rsv, length = \
            _RECORD.unpack_from(data, off)
        off += RECORD_SIZE
        if off + length > total:
            raise ValueError("truncated record value")
        raw = data[off:off + length]
        off += length
        pad = (-length) % 8
        if strict and any(data[off:off + pad]):
            raise ValueError("non-zero record padding")
        off += pad
        name = name_b.split(b"\x00", 1)[0].decode("ascii", "replace")
        records.append(Rec(name, type_, _decode_value(type_, raw),
                           index=index, verified=bool(ver)))
    if off != total:
        raise ValueError(f"record bytes {off} != total {total}")
    return {
        "magic": magic, "version": version, "seq": seq, "ts_ns": ts_ns,
        "count": count, "total_bytes": total, "flags": flags,
        "records": records,
    }


def _decode_value(type_: int, raw: bytes):
    if type_ == T_I64:
        return struct.unpack("<q", raw)[0]
    if type_ == T_F64:
        return struct.unpack("<d", raw)[0]
    if type_ == T_BYTES:
        return raw
    if type_ == T_BOOL:
        return bool(raw[0]) if raw else False
    if type_ == T_U32:
        return struct.unpack("<I", raw)[0]
    if type_ == T_VEC3:
        return struct.unpack("<3f", raw)
    if type_ == T_VEC4:
        return struct.unpack("<4f", raw)
    if type_ == T_STR:
        return raw.decode("utf-8", "replace")
    return raw


# ---------------------------------------------------------------------------
# AArch64 getter decoder (the subset the lua_bind getters use)
# ---------------------------------------------------------------------------
def _u32(code: bytes, i: int) -> int:
    return struct.unpack_from("<I", code, i * 4)[0]


def _imm12(insn: int) -> int:
    return (insn >> 10) & 0xFFF


def _next_is_ret(code: bytes, i: int) -> bool:
    """True when instruction i+1 is ret (or a nop then ret)."""
    n = len(code) // 4
    if i + 1 < n and _u32(code, i + 1) == 0xD65F03C0:
        return True
    if i + 2 < n and _u32(code, i + 1) == 0xD503201F \
            and _u32(code, i + 2) == 0xD65F03C0:
        return True
    return False


def decode_getter(code: bytes | None):
    """Decode a tiny concrete getter into (mode, off1, off2, kind).

    Patterns understood (all seen in the SSBU module vtables):

        ldrb w0,[x0,#B]                  -> ('direct', 0, B, 'u8')
        ldr  w0,[x0,#B] / ldrsw         -> ('direct', 0, B, 'u32'/'i32')
        ldr  s0,[x0,#B]                  -> ('direct', 0, B, 'f32')
        ldr  x8,[x0,#A]; ldrb w0,[x8,#B] -> ('deref', A, B, 'u8')
        ldr  x8,[x0,#A]; ldr  s0,[x8,#B] -> ('deref', A, B, 'f32')
        ldr  x8,[x0,#A]; ldr  q0,[x8,#B] -> ('deref', A, B, 'vec4')
        add  x0,x0,#A ; ret              -> ('ptr', A, 0, 'ptr')
        ldr  x0,[x0,#A] ; ret            -> ('ptr', A, 0, 'ptr')

    Returns None when the getter is not one of these shapes (array walkers,
    virtual dispatch, arithmetic).
    """
    if not code or len(code) < 8:
        return None
    ptr_off = None
    ptr_reg = None
    for i in range(min(16, len(code) // 4)):
        insn = _u32(code, i)
        if insn == 0xD65F03C0:                       # ret
            break
        if insn == 0xD503201F:                       # nop
            continue
        rn = (insn >> 5) & 0x1F
        rt = insn & 0x1F
        # ldr xT,[xN,#imm]  (64-bit, unsigned offset)
        if (insn & 0xFFC00000) == 0xF9400000:
            imm = _imm12(insn) * 8
            if rn == 0 and rt == 0 and ptr_off is None and _next_is_ret(code, i):
                # ldr x0,[x0,#imm] ; ret  -> returns the pointer at module+imm
                return ("loadptr", 0, imm, "ptr")
            if rn == 0 and ptr_off is None:
                ptr_off, ptr_reg = imm, rt
                continue
            if rn == ptr_reg and rt == 0 and _next_is_ret(code, i):
                # ldr x8,[x0,#A] ; ldr x0,[x8,#B] ; ret
                return ("loadptr", ptr_off, imm, "ptr")
            continue
        # ldr wT,[xN,#imm]
        if (insn & 0xFFC00000) == 0xB9400000:
            imm = _imm12(insn) * 4
            if rt == 0 and _next_is_ret(code, i):
                if rn == 0:
                    return ("direct", 0, imm, "u32")
                if rn == ptr_reg:
                    return ("deref", ptr_off, imm, "u32")
            continue
        # ldrsw xT/wT,[xN,#imm]
        if (insn & 0xFFC00000) == 0xB9800000:
            imm = _imm12(insn) * 4
            if rt == 0 and _next_is_ret(code, i):
                if rn == 0:
                    return ("direct", 0, imm, "i32")
                if rn == ptr_reg:
                    return ("deref", ptr_off, imm, "i32")
            continue
        # ldrb wT,[xN,#imm]
        if (insn & 0xFFC00000) == 0x39400000:
            imm = _imm12(insn)
            if rt == 0 and _next_is_ret(code, i):
                if rn == 0:
                    return ("direct", 0, imm, "u8")
                if rn == ptr_reg:
                    return ("deref", ptr_off, imm, "u8")
            continue
        # ldrsh wT,[xN,#imm]
        if (insn & 0xFFC00000) == 0x79C00000:
            imm = _imm12(insn) * 2
            if rt == 0 and _next_is_ret(code, i):
                if rn == 0:
                    return ("direct", 0, imm, "i16")
                if rn == ptr_reg:
                    return ("deref", ptr_off, imm, "i16")
            continue
        # ldr sT,[xN,#imm]
        if (insn & 0xFFC00000) == 0xBD400000:
            imm = _imm12(insn) * 4
            if rt == 0 and _next_is_ret(code, i):
                if rn == 0:
                    return ("direct", 0, imm, "f32")
                if rn == ptr_reg:
                    return ("deref", ptr_off, imm, "f32")
            continue
        # ldr qT,[xN,#imm]
        if (insn & 0xFFC00000) == 0x3DC00000:
            imm = _imm12(insn) * 16
            if rt == 0 and _next_is_ret(code, i):
                if rn == 0:
                    return ("direct", 0, imm, "vec4")
                if rn == ptr_reg:
                    return ("deref", ptr_off, imm, "vec4")
            continue
        # add xD,xN,#imm
        if (insn & 0xFFC00000) == 0x91000000:
            imm = _imm12(insn)
            if rn == 0 and rt == 0 and _next_is_ret(code, i):
                return ("addr", imm, 0, "addr")
            continue
    return None


def decode_vec_getter(code: bytes | None):
    """Decode a velocity-vector getter of the kinetic entry objects.

    Three shapes occur in the live build:
        add x0,x0,#imm ; ret
        ldr w8,[x0,#A] ; add x9,x0,#B ; add x10,x0,#C ; cmp w8,#0 ;
        csel x0,x9,x10,gt ; ret
    The second one returns one of two vectors depending on a signed counter.
    Returns ('addr', imm) or ('select', A, B, C) or None.
    """
    if not code or len(code) < 8:
        return None
    ins = [_u32(code, i) for i in range(min(16, len(code) // 4))]
    # add x0,x0,#imm ; ret
    if len(ins) >= 2 and (ins[0] & 0xFFC00000) == 0x91000000 \
            and ((ins[0] >> 5) & 0x1F) == 0 and (ins[0] & 0x1F) == 0 \
            and ins[1] == 0xD65F03C0:
        return ("addr", _imm12(ins[0]))
    # conditional select
    a = b = c = None
    for i, w in enumerate(ins):
        if (w & 0xFFC00000) == 0xB9400000 and ((w >> 5) & 0x1F) == 0 \
                and (w & 0x1F) == 8 and a is None:
            a = _imm12(w) * 4
        elif (w & 0xFFC00000) == 0x91000000 and ((w >> 5) & 0x1F) == 0:
            rt = w & 0x1F
            if rt == 9 and b is None:
                b = _imm12(w)
            elif rt == 10 and c is None:
                c = _imm12(w)
        elif w == 0xD65F03C0:
            break
    if a is not None and b is not None and c is not None:
        # csel x0,x9,x10,gt appears somewhere before the ret
        for w in ins:
            if (w & 0xFFE00C00) == 0x9A800120:   # csel x0,x9,x10,gt
                return ("select", a, b, c)
        # accept the observed encoding even if the mask above differs
        return ("select", a, b, c)
    return None


# ---------------------------------------------------------------------------
# documented offsets (disassembly evidence in extra_map.json)
# ---------------------------------------------------------------------------
# module pointers in BattleObjectModuleAccessor
ACC_POSTURE = 0x38
ACC_STATUS = 0x40
ACC_WORK = 0x50
ACC_GROUND = 0x58
ACC_KINETIC = 0x68
ACC_MOTION = 0x88
ACC_STOP = 0x90
ACC_DAMAGE = 0xA8
ACC_HIT = 0xB0
ACC_SHIELD = 0x100
ACC_CANCEL = 0x128

IMPL_READY_GO = 0xD2
IMPL_ENTRIES = 0x20
ENTRY_FIGHTER = 0x4160
ENTRY_INFO = 0xF0                 # FighterInformation object
ENTRY_RECORD = 0xF8               # its data record

POSTURE_POS = 0x30
STATUS_KIND = 0x98
STATUS_SITUATION = 0xDC
MOTION_FRAME = 0x70

DAMAGE_ARRAY = 0x18
DAMAGE_RECORD_STRIDE = 0x1C0
DAMAGE_AMOUNT = 0x00
DAMAGE_POWER_MAX = 0x0C
DAMAGE_REACTION = 0x10

SHIELD_PARTS = 0x18
SHIELD_PART_COUNT = 0x20
SHIELD_NODES = 0x28
SHIELD_NODE_STRIDE = 0x50
SHIELD_SLOTS = 0x50
SHIELD_SLOT_STRIDE = 0x2D0
SHIELD_PART_STRIDE = 0x34
SHIELD_NO_FLAG = 0x7F
SHIELD_ATTACK_LIMIT_GLOBAL = 0x80
SHIELD_HIT_STOP_MUL = 0x84
PART_SIZE = 0x06
PART_POS_X = 0x08
PART_LR = 0x10
PART_HOP_ANGLE = 0x14
PART_ATTACK_MUL = 0x18
PART_SPEED_MUL = 0x1C
PART_ATTACK_LIMIT = 0x20
PART_LIFE_MUL = 0x24

WORK_KIND_ARRAY = 0x18
WORK_TERMS = 0x28
WORK_INT_ARRAY = 0x20
WORK_FLOAT_ARRAY = 0x10
WORK_FLAGS = 0x30

INFO_KIND_PLUS1 = 0x30
INFO_COLOR = 0x51
INFO_IS_CPU = 0x92
INFO_GRAVITY = 0x36C
INFO_DEAD_COUNT = 0xCC            # u32[3], + 0xD4 total
INFO_STOCK = 0xD8
INFO_LAST_DEAD_SUICIDE = 0x1A3
INFO_ON_REBIRTH = 0x38C
INFO_CATEGORY = 0x3AC
INFO_HIT_POINT_MAX = 0x38         # f32[2]
INFO_LOG = 0x15E0                 # ptr[8] per log type, stride 0x11B0
INFO_LOG_STRIDE = 0x11B0

# global guest-image absolute addresses (debug_mode.c uses the same constants;
# they are 0x7100000000-relative, so _g() maps them onto the live base)
G_MATCH_BLOCK = 0x71052C41B0
G_RNG = 0x71052381C0
G_BOM_SINGLETON = 0x71052B6A00
G_BOM_GLOBAL = 0x71052B7548
G_STAGE_MANAGER = 0x71053299D8
G_CAMERA_GLOBAL = 0x71052B7EF8 + 8   # BossManager::instance_ + 8

BOM_POOL_STRIDES = (0xF940, 0x3C30, 0x60C0, 0x300, 0x2B0)
BOM_POOLS = 5

MATCH_RULE_KIND = 0x05
MATCH_STOCK = 0x0A
MATCH_ITEM_APPEAR = 0x54
MATCH_STAGE_ID = 0x420
MATCH_MELEE_MODE = 0x00

STAGE_IMPL_ID = 0x17C
STAGE_OBJ_ARRAY = 0x110

CAMERA_DEAD = 0xC30
CAMERA_RANGE = 0xC20
CAMERA_POS = 0xD30
CAMERA_TARGET = 0xD20
CAMERA_INTERNAL_POS = 0xEC0
CAMERA_INTERNAL_TARGET = 0xEB0
CAMERA_FOV = 0xDAC

# vtable slots (byte offsets already divided by 8)
SLOTS = {
    "damage_amount": 25, "damage_reaction": 26, "damage_power_max": 27,
    "damage_no_reaction": 38, "damage_capture_cut": 59,
    "damage_lock": 75, "damage_paralyze": 85, "damage_critical": 91,
    "ground_offset_x": 38, "ground_offset_y": 39,
    "ground_status_cliff": 72, "ground_can_entry_cliff": 76,
    "ground_shape": 111, "ground_z": 113, "ground_attach": 121,
    "ground_touch_flag": 141, "ground_center_pos": 180,
    "ground_down_pos": 176, "ground_attach_cliff": 226,
    "kinetic_type": 36, "kinetic_sum_x": 25, "kinetic_sum_y": 26,
    "motion_kind": 39, "motion_end": 57, "motion_looped": 58,
    "motion_flip": 114, "motion_blend": 37, "motion_changing": 38,
    "motion_rate": 40, "motion_prev_frame": 45, "motion_end_frame": 48,
    "motion_loop_flag": 63,
    "stop_is_stop": 17, "stop_is_hit": 19, "stop_is_damage": 20,
    "stop_hit_stop_frame": 25,
    "shield_is_shield": 17, "shield_part_size": 41, "shield_center_pos": 51,
    "shield_lr": 52, "shield_group": 53, "shield_life_mul": 34,
    "shield_attack_mul": 30, "shield_speed_mul": 32,
    "shield_attack_limit": 36, "shield_pos_x": 54, "shield_hop_angle": 24,
}

# ssbu-rl lives beside ssbu-recomp; four levels up from src/rl/extra.
DEFAULT_ELF = os.path.abspath(os.path.join(
    HERE, "..", "..", "..", "..", "ssbu-recomp",
    "build-linux", "binaries", "13.0.4", "main.elf"))


# ---------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------
class ExtraState:
    """Read-only extras view of one running game process."""

    def __init__(self, pid: int, elf_path: str | None = None):
        self.pid = int(pid)
        self.mem = ProcessMemory(self.pid)
        self.syms = ElfSymbols(elf_path or DEFAULT_ELF)
        self.base = guest_image_base(self.pid)
        self._figure = self.base + self.syms.offset(
            "_ZN3lib9SingletonIN3app14FighterManagerEE9instance_E")
        self._camera_global = self._g(G_CAMERA_GLOBAL)
        self._stage_global = self._g(G_STAGE_MANAGER)
        self._bom_singleton = self._g(G_BOM_SINGLETON)
        self._bom_global = self._g(G_BOM_GLOBAL)
        self._match = self._g(G_MATCH_BLOCK)
        self._rng = self._g(G_RNG)
        self._getter_cache: dict[tuple[int, int], tuple | None] = {}
        self.decoded_any = False
        self.resolved: list[tuple[str, str]] = []

    # -- lifecycle --------------------------------------------------------
    def close(self):
        self.mem.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- addresses --------------------------------------------------------
    def _g(self, guest_abs: int) -> int:
        """Map a guest-image absolute (as used by debug_mode.c) to our map."""
        return self.base + (guest_abs - 0x7100000000)

    def manager_impl(self) -> int | None:
        mgr = self.mem.u64(self._figure)
        return self.mem.u64(mgr) if mgr else None

    def fight_on(self) -> bool:
        impl = self.manager_impl()
        return bool(impl) and self.mem.u8(impl + IMPL_READY_GO) == 1

    # -- vtable getter resolution ----------------------------------------
    def _getter(self, vtable: int, slot: int):
        key = (vtable, slot)
        if key in self._getter_cache:
            return self._getter_cache[key]
        target = self.mem.u64(vtable + 8 * slot)
        result = None
        if target:
            code = self.mem.read(target, 64)
            result = decode_getter(code)
            if result is not None:
                self.decoded_any = True
        self._getter_cache[key] = result
        return result

    def _module_getter(self, module: int | None, slot: int):
        if not module:
            return None
        vt = self.mem.u64(module)
        if not vt:
            return None
        return self._getter(vt, slot)

    def _read_getter(self, module: int | None, slot: int):
        """Resolve and read a getter; returns (value, verified) or (None, False)."""
        dec = self._module_getter(module, slot)
        if dec is None:
            return None, False
        mode, off1, off2, kind = dec
        if mode == "addr":
            # pointer-returning getter: only _read_vector uses this shape
            return None, False
        if mode == "deref":
            base = self.mem.u64(module + off1)
            if not base:
                return None, False
            addr = base + off2
        elif mode == "loadptr":
            base = self.mem.u64(module + off1) if off1 else module
            if not base:
                return None, False
            addr = base + off2
            v = self.mem.u64(addr)
            return (v, True) if v else (None, False)
        elif mode == "direct":
            addr = module + off2
        else:
            return None, False
        if kind == "u8":
            v = self.mem.u8(addr)
            return (v, True) if v is not None else (None, False)
        if kind == "i16":
            d = self.mem.read(addr, 2)
            return (struct.unpack("<h", d)[0], True) if d and len(d) == 2 else (None, False)
        if kind in ("u32", "i32"):
            v = self.mem.u32(addr)
            if v is None:
                return None, False
            if kind == "i32" and v >= 2 ** 31:
                v -= 2 ** 32
            return v, True
        if kind == "f32":
            v = self.mem.f32(addr)
            if v is None or not math.isfinite(v):
                return None, False
            return v, True
        if kind == "vec4":
            d = self.mem.read(addr, 16)
            if not d or len(d) < 16:
                return None, False
            vals = struct.unpack("<4f", d)
            if not all(math.isfinite(v) for v in vals):
                return None, False
            return vals, True
        if kind == "ptr":
            v = self.mem.u64(addr)
            return (v, True) if v else (None, False)
        return None, False

    def _read_wide(self, module: int | None, slot: int):
        """Read 8 bytes at a scalar getter's address (Hash40 fields).

        `MotionModule::motion_kind` is a Hash40 (the live value decoded to
        hash40("wait") = 0x47dee83e5), so the u32 some getters return is only
        the low half.  This reads the full 8 bytes and sanity-checks the
        Hash40 shape (upper bits zero except the name length byte).
        """
        dec = self._module_getter(module, slot)
        if dec is None:
            return None, False
        mode, off1, off2, kind = dec
        if mode == "direct":
            addr = module + off2
        elif mode == "deref":
            b = self.mem.u64(module + off1)
            addr = (b or 0) + off2
        elif mode == "loadptr":
            # ldr x0,[x0,#off] returns the 8 bytes stored at module+off
            base = self.mem.u64(module + off1) if off1 else module
            addr = (base or 0) + off2
        else:
            return None, False
        d = self.mem.read(addr, 8)
        if not d or len(d) != 8:
            return None, False
        v = struct.unpack("<Q", d)[0]
        if v >= (1 << 63):
            return None, False
        # Hash40: 40-bit crc plus the name length in bits 32..39.
        if v >= (1 << 40):
            return None, False
        return v, True

    def _read_vector(self, module: int | None, slot: int):
        dec = self._module_getter(module, slot)
        if dec is None:
            return None, False
        mode, off1, off2, kind = dec
        if kind == "vec4":
            return self._read_getter(module, slot)
        if mode == "addr":
            addr = module + off1             # getter returns &module[off]
        elif mode == "loadptr":
            base = self.mem.u64(module + off1) if off1 else module
            if not base:
                return None, False
            addr = base + off2               # getter returns the stored pointer
        elif mode == "deref":
            b = self.mem.u64(module + off1)
            addr = (b or 0) + off2
        else:
            addr = module + off2
        if not addr:
            return None, False
        d = self.mem.read(addr, 12)
        if not d or len(d) < 12:
            return None, False
        vals = struct.unpack("<3f", d)
        if not all(math.isfinite(v) for v in vals):
            return None, False
        return vals, True

    # -- record helpers ---------------------------------------------------
    def _add(self, out: list[Rec], name: str, type_: int, value,
             index: int = 0, verified: bool = False):
        if value is None:
            return
        if len(name.encode("ascii", "replace")) > NAME_MAX:
            name = name[:NAME_MAX]
        out.append(Rec(name, type_, value, index=index, verified=verified))

    def _fv(self, out, name, module, slot, kind, index=0, scale=1.0):
        v, ok = self._read_getter(module, slot)
        if v is None:
            return
        if kind == "vec4":
            self._add(out, name, T_VEC4, v, index, ok)
        elif kind == "f32":
            self._add(out, name, T_F64, float(v) * scale, index, ok)
        elif kind == "bool":
            self._add(out, name, T_BOOL, bool(v), index, ok)
        else:
            self._add(out, name, T_I64, int(v), index, ok)

    # -- globals ----------------------------------------------------------
    def read_globals(self, out: list[Rec]):
        m = self.mem
        add = self._add
        add(out, "g.base", T_I64, self.base, 0, True)
        impl = self.manager_impl()
        if impl:
            add(out, "g.ready_go", T_BOOL,
                bool(m.u8(impl + IMPL_READY_GO)), 0, True)
            add(out, "g.fighter_manager", T_I64, impl, 0, True)

        # stage manager: singleton -> object -> impl -> stage id @0x17c
        first = m.u64(self._stage_global) if self._stage_global else None
        stage_impl = m.u64(first) if first else None
        if first:
            add(out, "g.stage_manager", T_I64, first, 0, True)
        if stage_impl:
            add(out, "g.stage_impl", T_I64, stage_impl, 0, True)
            sid = m.u32(stage_impl + STAGE_IMPL_ID)
            if sid is not None and sid != 0xFFFFFFFF and sid <= 0xFFFF:
                add(out, "g.stage_id", T_U32, sid, 0, True)
            idx = m.u32(stage_impl + STAGE_IMPL_ID)
            if idx is not None and idx < 8:
                for k in range(3):
                    so = m.u64(stage_impl + STAGE_OBJ_ARRAY + 8 * k)
                    if so:
                        add(out, "g.stage_obj%d" % k, T_I64, so, 0, True)
                so = m.u64(stage_impl + STAGE_OBJ_ARRAY + 8 * idx)
                if so:
                    add(out, "g.stage_obj_cur", T_I64, so, 0, True)
                    raw = m.read(so, 64)
                    if raw:
                        add(out, "g.stage_obj_cur.raw64", T_BYTES, raw, 0, True)

        # match block (rule header + stage block)
        if self._match:
            raw = m.read(self._match, 0x100)
            if raw:
                add(out, "g.match.raw0x100", T_BYTES, raw, 0, True)
            kind = m.u8(self._match + MATCH_RULE_KIND)
            add(out, "g.match.rule_kind", T_U32, kind, 0, kind is not None)
            stock = m.u8(self._match + MATCH_STOCK)
            add(out, "g.match.stock", T_U32, stock, 0, stock is not None)
            items = m.u32(self._match + MATCH_ITEM_APPEAR)
            add(out, "g.match.item_appear", T_U32, items, 0, items is not None)
            melee = m.u8(self._match + MATCH_MELEE_MODE)
            add(out, "g.match.melee_mode", T_U32, melee, 0, melee is not None)
            stageid = m.u32(self._match + MATCH_STAGE_ID)
            add(out, "g.match.stage_id_raw", T_U32, stageid, 0,
                stageid is not None)
            # a compact fingerprint of the whole block for change detection
            add(out, "g.match.fnv1a", T_U32,
                self._fnv1a(m.read(self._match, 0x180) or b""), 0, True)

        # RNG seed / state (debug_mode.c: the rollback generator)
        if self._rng:
            raw = m.read(self._rng, 16)
            if raw:
                add(out, "g.rng.raw16", T_BYTES, raw, 0, True)
                for k, v in enumerate(struct.unpack("<4I", raw)):
                    add(out, "g.rng.word%d" % k, T_U32, v, 0, True)

        # camera singleton + manager
        a = m.u64(self._camera_global) if self._camera_global else None
        cam = m.u64(a) if a else None
        if cam:
            add(out, "g.camera.manager", T_I64, cam, 0, True)
            for name, off, n in (
                    ("g.camera.dead", CAMERA_DEAD, 4),
                    ("g.camera.range", CAMERA_RANGE, 4),
                    ("g.camera.pos", CAMERA_POS, 4),
                    ("g.camera.target", CAMERA_TARGET, 4),
                    ("g.camera.internal_pos", CAMERA_INTERNAL_POS, 4),
                    ("g.camera.internal_target", CAMERA_INTERNAL_TARGET, 4)):
                d = m.read(cam + off, 4 * n)
                if d and len(d) == 4 * n:
                    vals = struct.unpack("<%df" % n, d)
                    if all(math.isfinite(v) for v in vals):
                        add(out, name, T_VEC4 if n == 4 else T_VEC3, vals, 0, True)
            fov = m.f32(cam + CAMERA_FOV)
            if fov is not None and math.isfinite(fov):
                add(out, "g.camera.fov", T_F64, fov, 0, True)

        # BattleObjectManager: singleton cell and the pool manager used by
        # is_active_find_battle_object (0x3ac560).
        add(out, "g.bom.singleton_cell", T_I64,
            m.u64(self._bom_singleton) if self._bom_singleton else 0, 0, False)
        bom = m.u64(self._bom_global) if self._bom_global else None
        if bom:
            add(out, "g.bom.manager", T_I64, bom, 0, True)
            raw = m.read(bom, 0x40)
            if raw:
                add(out, "g.bom.manager.raw64", T_BYTES, raw, 0, True)
            for k in range(BOM_POOLS):
                pool = m.u64(bom + 8 * k)
                if not pool:
                    continue
                cnt = m.u32(pool + 0x18)
                arr = m.u64(pool + 8)
                if cnt is not None:
                    add(out, "g.bom.pool%d.count" % k, T_U32, cnt, 0, True)
                if arr:
                    add(out, "g.bom.pool%d.array" % k, T_I64, arr, 0, True)
                add(out, "g.bom.pool%d.stride" % k, T_U32,
                    BOM_POOL_STRIDES[k], 0, True)

    @staticmethod
    def _fnv1a(data: bytes) -> int:
        h = 0x811C9DC5
        for b in data:
            h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
        return h

    # -- battle object enumeration ---------------------------------------
    def read_battle_objects(self, out: list[Rec]):
        """Enumerate the five BattleObjectManager pools (live-proven).

        manager = *(0x71052b7548); pool k = manager+8*k;
        pool+0x18 = count, pool+8 = array; strides F940/3C30/60C0/300/2B0.
        A live slot (verified during a Mario/DK match):
            +0x08 u32 object id       (P1 0, P2 0x10000)
            +0x10 u64 entry/slot no.  (0, 1)
            +0x20 u64 module accessor (= obj+0x20 of the fighter object)
            +0x3A u8  active status   (4 = active fighter, 0 = free)
            free slots carry id 0x50000000.
        """
        m = self.mem
        bom = m.u64(self._bom_global) if self._bom_global else None
        if not bom:
            return
        total = 0
        for k in range(BOM_POOLS):
            pool = m.u64(bom + 8 * k)
            if not pool:
                continue
            cnt = m.u32(pool + 0x18)
            arr = m.u64(pool + 8)
            if not cnt or not arr or cnt > 0x1000:
                continue
            stride = BOM_POOL_STRIDES[k]
            for j in range(min(cnt, 128)):
                slot = arr + stride * j
                oid = m.u32(slot + 8)
                status = m.u8(slot + 0x3A)
                if oid is None or status is None:
                    continue
                # free/poison slots: id 0x50000000 and status 0
                if oid == 0x50000000 or status <= 3:
                    continue
                idx = total & 0xFF
                total += 1
                add = self._add
                add(out, "bom.obj.pool", T_U32, k, idx, True)
                add(out, "bom.obj.id", T_U32, oid, idx, True)
                add(out, "bom.obj.entry", T_I64, m.u64(slot + 0x10), idx,
                    True)
                add(out, "bom.obj.status_3a", T_U32, status, idx, True)
                add(out, "bom.obj.category", T_U32, (oid >> 28) & 0xF, idx,
                    True)
                raw = m.read(slot, 0x60)
                if raw:
                    add(out, "bom.obj.raw0x60", T_BYTES, raw, idx, True)
                acc = m.u64(slot + 0x20)
                if acc and m.is_mapped(acc, 0x160):
                    add(out, "bom.obj.accessor", T_I64, acc, idx, True)
                    posture = m.u64(acc + ACC_POSTURE)
                    if posture:
                        d = m.read(posture + POSTURE_POS, 12)
                        if d and len(d) == 12:
                            add(out, "bom.obj.pos", T_VEC3,
                                struct.unpack("<3f", d), idx, True)
                    status_mod = m.u64(acc + ACC_STATUS)
                    if status_mod:
                        sk = m.u32(status_mod + STATUS_KIND)
                        add(out, "bom.obj.status_kind", T_U32, sk, idx, True)
                    motion = m.u64(acc + ACC_MOTION)
                    if motion:
                        kind, kok = self._read_wide(
                            motion, SLOTS["motion_kind"])
                        if kind is not None:
                            add(out, "bom.obj.motion_kind", T_I64, kind, idx,
                                kok)
                if total >= 64:
                    return

    # -- one fighter ------------------------------------------------------
    def read_fighter(self, impl: int, entry_id: int, out: list[Rec]):
        m = self.mem
        entry = m.u64(impl + IMPL_ENTRIES + 8 * entry_id)
        if not entry:
            return
        obj = m.u64(entry + ENTRY_FIGHTER)
        acc = m.u64(obj + 0x20) if obj else None
        idx = entry_id + 1
        add = self._add
        add(out, "f.entry", T_I64, entry, idx, True)
        if obj:
            add(out, "f.object", T_I64, obj, idx, True)
        if not acc:
            return
        add(out, "f.accessor", T_I64, acc, idx, True)
        mods = {}
        for off in (ACC_POSTURE, ACC_STATUS, ACC_WORK, ACC_GROUND, ACC_KINETIC,
                    ACC_MOTION, ACC_STOP, ACC_DAMAGE, ACC_HIT, ACC_SHIELD,
                    ACC_CANCEL):
            v = m.u64(acc + off)
            if v:
                mods[off] = v
                add(out, "f.module.%03x" % off, T_I64, v, idx, True)

        # status / posture (core fields, repeated here only for context)
        p = mods.get(ACC_POSTURE)
        if p:
            d = m.read(p + POSTURE_POS, 12)
            if d and len(d) == 12:
                add(out, "f.pos", T_VEC3, struct.unpack("<3f", d), idx, True)
            add(out, "f.facing", T_F64, m.f32(p + 0xAC), idx, True)
        s = mods.get(ACC_STATUS)
        if s:
            add(out, "f.status_kind", T_U32, m.u32(s + STATUS_KIND), idx, True)
            add(out, "f.situation_kind", T_U32,
                m.u32(s + STATUS_SITUATION), idx, True)

        for fn, args in (
                (self._read_shield, (mods.get(ACC_SHIELD), idx, out)),
                (self._read_damage, (mods.get(ACC_DAMAGE), idx, out)),
                (self._read_ground, (mods.get(ACC_GROUND), idx, out)),
                (self._read_kinetic, (mods.get(ACC_KINETIC), idx, out)),
                (self._read_motion, (mods.get(ACC_MOTION), idx, out)),
                (self._read_stop, (mods.get(ACC_STOP), idx, out)),
                (self._read_hit, (mods.get(ACC_HIT), idx, out)),
                (self._read_work, (mods.get(ACC_WORK),
                                   mods.get(ACC_CANCEL), idx, out)),
                (self._read_info, (entry, entry_id, idx, out))):
            try:
                fn(*args)
            except Exception:
                # one module being torn down must not hide the other groups
                continue

    def _read_shield(self, mod, idx, out):
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.shield.ptr", T_I64, mod, idx, True)
        noflag = m.u8(mod + SHIELD_NO_FLAG)
        add(out, "f.shield.no_flag", T_U32, noflag, idx, True)
        arr = m.u64(mod + SHIELD_PARTS)
        cnt = m.u32(mod + SHIELD_PART_COUNT)
        nodes = m.u64(mod + SHIELD_NODES)
        slots = m.u64(mod + SHIELD_SLOTS)
        if arr:
            add(out, "f.shield.parts", T_I64, arr, idx, True)
        if nodes:
            add(out, "f.shield.nodes", T_I64, nodes, idx, True)
        if slots:
            add(out, "f.shield.slots", T_I64, slots, idx, True)
        if cnt is not None and 0 < cnt <= 16:
            add(out, "f.shield.part_count", T_U32, cnt, idx, True)
        else:
            cnt = 0
        add(out, "f.shield.attack_limit_global", T_F64,
            m.f32(mod + SHIELD_ATTACK_LIMIT_GLOBAL), idx, True)
        add(out, "f.shield.hit_stop_mul", T_F64,
            m.f32(mod + SHIELD_HIT_STOP_MUL), idx, True)
        # get_lr / get_part_size / get_life_mul / ... all index the part array
        for i in range(min(cnt, 8)):
            part = arr + SHIELD_PART_STRIDE * i
            d = m.read(part, SHIELD_PART_STRIDE)
            if not d or len(d) < SHIELD_PART_STRIDE:
                continue
            pre = "f.shield.part%d." % i
            add(out, pre + "raw", T_BYTES, d, idx, True)
            add(out, pre + "size_s16", T_I64,
                struct.unpack_from("<h", d, PART_SIZE)[0], idx, True)
            add(out, pre + "pos_x", T_F64,
                struct.unpack_from("<f", d, PART_POS_X)[0], idx, True)
            add(out, pre + "lr", T_F64,
                struct.unpack_from("<f", d, PART_LR)[0], idx, True)
            add(out, pre + "hop_angle", T_F64,
                struct.unpack_from("<f", d, PART_HOP_ANGLE)[0], idx, True)
            add(out, pre + "attack_mul", T_F64,
                struct.unpack_from("<f", d, PART_ATTACK_MUL)[0], idx, True)
            add(out, pre + "speed_mul", T_F64,
                struct.unpack_from("<f", d, PART_SPEED_MUL)[0], idx, True)
            add(out, pre + "attack_limit", T_F64,
                struct.unpack_from("<f", d, PART_ATTACK_LIMIT)[0], idx, True)
            add(out, pre + "life_mul", T_F64,
                struct.unpack_from("<f", d, PART_LIFE_MUL)[0], idx, True)
        # node array (set_size writes node+0x20); node index = part[0].s16 + i
        if nodes and arr and cnt:
            nd = m.read(arr + 4, 2)
            if nd and len(nd) == 2:
                base_i = struct.unpack("<h", nd)[0]
                for i in range(min(cnt, 8)):
                    node = nodes + SHIELD_NODE_STRIDE * (base_i + i)
                    raw = m.read(node, SHIELD_NODE_STRIDE)
                    if not raw or len(raw) < SHIELD_NODE_STRIDE:
                        continue
                    add(out, "f.shield.node%d.raw" % i, T_BYTES, raw, idx, True)
                    add(out, "f.shield.node%d.size" % i, T_F64,
                        struct.unpack_from("<f", raw, 0x20)[0], idx, True)
        # is_shield proxy (the real getter walks three arrays; documented)
        if noflag is not None and cnt and arr and nodes:
            d4 = m.read(arr + 4, 2)
            if d4 and len(d4) == 2:
                n = struct.unpack("<h", d4)[0]
                node = m.u64(mod + SHIELD_NODES) + SHIELD_NODE_STRIDE * n
                active = m.u32(node + 0x40)
                add(out, "f.shield.likely", T_BOOL,
                    bool(noflag == 0 and active), idx, False)
        # center position of shield slot i (get_center_pos walks slot array)
        if slots and arr:
            first = m.read(arr, 4)
            if first and len(first) == 4:
                slot_i = struct.unpack("<i", first)[0]
                if 0 <= slot_i < 64:
                    d = m.read(slots + SHIELD_SLOT_STRIDE * slot_i + 0x30, 8)
                    if d and len(d) == 8:
                        objp = struct.unpack("<Q", d)[0]
                        if objp:
                            add(out, "f.shield.center_obj", T_I64, objp, idx, True)
                            # the getter calls vtable[4] on this object; the
                            # concrete implementation is normally
                            # `add x0,x0,#0x30 ; ret`
                            vt = m.u64(objp)
                            if vt:
                                tgt = m.u64(vt + 32)
                                code = m.read(tgt, 8) if tgt else None
                                dec = decode_getter(code)
                                cvec = None
                                if dec and dec[0] == "addr":
                                    cvec = objp + dec[1]
                                elif dec and dec[0] == "loadptr":
                                    b0 = m.u64(objp + dec[1]) if dec[1] else objp
                                    cvec = (b0 or 0) + dec[2]
                                if cvec:
                                    d2 = m.read(cvec, 12)
                                    if d2 and len(d2) == 12:
                                        add(out, "f.shield.center", T_VEC3,
                                            struct.unpack("<3f", d2), idx, True)

    def _read_damage(self, mod, idx, out):
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.damage.ptr", T_I64, mod, idx, True)
        arr = m.u64(mod + DAMAGE_ARRAY)
        if arr:
            add(out, "f.damage.array", T_I64, arr, idx, True)
            add(out, "f.damage.amount", T_F64, m.f32(arr + DAMAGE_AMOUNT),
                idx, True)
            add(out, "f.damage.power_max", T_F64,
                m.f32(arr + DAMAGE_POWER_MAX), idx, True)
            add(out, "f.damage.reaction", T_F64,
                m.f32(arr + DAMAGE_REACTION), idx, True)
            raw = m.read(arr, 0x40)
            if raw:
                add(out, "f.damage.rec0.raw0x40", T_BYTES, raw, idx, True)
        # simple boolean getters (the decoder handles the simple ones)
        for name, slot in (("f.damage.is_paralyze", SLOTS["damage_paralyze"]),
                           ("f.damage.is_capture_cut",
                            SLOTS["damage_capture_cut"]),
                           ("f.damage.no_reaction_perfect",
                            SLOTS["damage_no_reaction"])):
            v, ok = self._read_getter(mod, slot)
            if v is not None:
                add(out, name, T_BOOL, bool(v), idx, ok)
        # is_damage_lock (vtable[75], target 0x4006d0): the getter is
        #   if u8[+0x384] != 0: return 1
        #   return u8[+0x385] != 0
        b384 = m.u8(mod + 0x384)
        b385 = m.u8(mod + 0x385)
        if b384 is not None and b385 is not None:
            add(out, "f.damage.is_lock", T_BOOL,
                bool(b384 != 0 or b385 != 0), idx, True)
        # is_critical_hit (vtable[91], target 0x400ad0): u8[+0x38A]
        b38a = m.u8(mod + 0x38A)
        if b38a is not None:
            add(out, "f.damage.is_critical", T_BOOL, bool(b38a), idx, True)
        raw = m.read(mod + 0x380, 0x10)
        if raw:
            add(out, "f.damage.flags_raw", T_BYTES, raw, idx, True)
        # no_reaction bytes from the module itself (documented)
        for name, off in (("f.damage.no_reaction.byte44", 0x44),
                          ("f.damage.no_reaction.byte45", 0x45)):
            v = m.u8(mod + off)
            if v is not None:
                add(out, name, T_U32, v, idx, True)

    def _read_ground(self, mod, idx, out):
        """GroundModule at accessor+0x58.

        Exactly the fields the concrete getters use (extra_map.json):
            offset_x/y   module+0x50/+0x54 (get_offset_x/y)
            z            f32 module+0x80 (vtable[113])
            shape_kind   *(module+0x28) + 0x394 (vtable[111])
            inner        *(module+0x28)
            is_status_cliff   u32[inner+0x5A0] == 3
            is_attach         (u8[inner+0x5DB] >> 5) & 1
            is_attach_cliff   u64[inner+0x560] != 0
            is_touch(flag)    flag bit i -> u64[inner+0x10+0x30*i] != 0
            center       inner+0x4C0 (+ inner+0x3C0, or the two floats)
        """
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.ground.ptr", T_I64, mod, idx, True)
        add(out, "f.ground.offset_x", T_F64, m.f32(mod + 0x50), idx, True)
        add(out, "f.ground.offset_y", T_F64, m.f32(mod + 0x54), idx, True)
        add(out, "f.ground.z", T_F64, m.f32(mod + 0x80), idx, True)
        inner = m.u64(mod + 0x28)
        if inner:
            add(out, "f.ground.inner", T_I64, inner, idx, True)
            shape = m.u32(inner + 0x394)
            add(out, "f.ground.shape_kind", T_U32, shape, idx, True)
            st = m.u32(inner + 0x5A0)
            if st is not None:
                add(out, "f.ground.is_status_cliff", T_BOOL, st == 3,
                    idx, True)
            at = m.u8(inner + 0x5DB)
            if at is not None:
                add(out, "f.ground.is_attach", T_BOOL, bool((at >> 5) & 1),
                    idx, True)
            acc = m.u64(inner + 0x560)
            add(out, "f.ground.is_attach_cliff", T_BOOL, bool(acc), idx, True)
            add(out, "f.ground.cliff_obj", T_I64, acc, idx, True)
            # touch flags: the is_touch getter tests these pointers
            mask = 0
            for i in range(8):
                p = m.u64(inner + 0x10 + 0x30 * i)
                if p:
                    mask |= 1 << i
                add(out, "f.ground.touch%d.ptr" % i, T_I64, p, idx, True)
            add(out, "f.ground.touch_mask", T_U32, mask, idx, True)
            # center position: exact replication of the 0x4646a0 getter
            q = m.read(inner + 0x4C0, 16)
            if q and len(q) == 16:
                base = list(struct.unpack("<4f", q))
                mode = m.u32(inner + 0x39C)
                if mode == 1:
                    off = m.read(inner + 0x3C0, 16)
                    if off and len(off) == 16:
                        o = struct.unpack("<4f", off)
                        c = tuple(base[k] + o[k] for k in range(4))
                        add(out, "f.ground.center", T_VEC4, c, idx, True)
                else:
                    ox = m.f32(inner + 0x3D0)
                    oy = m.f32(inner + 0x3E4)
                    if ox is not None and oy is not None:
                        add(out, "f.ground.center", T_VEC4,
                            (base[0] + ox, base[1] + oy, 0.0, 0.0), idx,
                            True)
                    add(out, "f.ground.center_base", T_VEC4, base, idx, True)
            # the value added by the center getter when mode != 1
            off = m.read(inner + 0x3C0, 16)
            if off and len(off) == 16:
                add(out, "f.ground.center_offset_raw", T_VEC4,
                    struct.unpack("<4f", off), idx, False)
            raw = m.read(inner + 0x380, 0x280)
            if raw:
                add(out, "f.ground.inner_raw", T_BYTES, raw, idx, True)

    def _read_kinetic(self, mod, idx, out):
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.kinetic.ptr", T_I64, mod, idx, True)
        cnt = m.u32(mod + 0x20)
        if cnt is not None:
            add(out, "f.kinetic.count", T_U32, cnt, idx, True)
        else:
            return
        if cnt > 32:
            return
        sum_v = [0.0, 0.0, 0.0]
        decoded_all = True
        for i in range(min(cnt, 16)):
            ent = mod + 0x30 + 0x18 * i
            ob = m.u64(ent)
            fl = m.u32(ent + 0x0C)
            pre = "f.kinetic.entry%d." % i
            add(out, pre + "flags", T_U32, fl, idx, fl is not None)
            if not ob:
                continue
            add(out, pre + "object", T_I64, ob, idx, True)
            en = m.u8(ob + 0x30)
            add(out, pre + "enabled", T_U32, en, idx, True)
            vt = m.u64(ob)
            if not vt:
                decoded_all = False
                continue
            tgt = m.u64(vt + 32)          # vtable[4] returns &velocity
            code = m.read(tgt, 12) if tgt else None
            vd = decode_vec_getter(code)
            if vd and vd[0] == "addr":
                vec_addr = ob + vd[1]
            elif vd and vd[0] == "select":
                selcnt = m.i32(ob + vd[1])
                if selcnt is None:
                    decoded_all = False
                    continue
                vec_addr = ob + (vd[2] if selcnt > 0 else vd[3])
            else:
                decoded_all = False
                continue
            d = m.read(vec_addr, 12)
            if not d or len(d) < 12:
                decoded_all = False
                continue
            v = struct.unpack("<3f", d)
            if not all(math.isfinite(x) for x in v):
                decoded_all = False
                continue
            add(out, pre + "vel", T_VEC3, v, idx, True)
            if en:
                for k in range(3):
                    sum_v[k] += v[k]
        add(out, "f.kinetic.engine_vel", T_VEC3, tuple(sum_v), idx,
            decoded_all and cnt > 0)
        add(out, "f.kinetic.engine_speed", T_F64,
            math.sqrt(sum(x * x for x in sum_v)), idx,
            decoded_all and cnt > 0)
        # kinetic type getter (documented slot)
        v, ok = self._read_getter(mod, SLOTS["kinetic_type"])
        if v is not None:
            add(out, "f.kinetic.type", T_U32, int(v), idx, ok)

    def _read_motion(self, mod, idx, out):
        """MotionModule at accessor+0x88.

        Offsets are not guessed: every one below is the exact field read by
        the concrete getter (see extra_map.json -> evidence).
            frame        vtable[14]  f32 module+0x70
            motion_kind  vtable[39]  u64 module+0x68   (Hash40, live = "wait")
            is_changing  vtable[38]  u8  module+0x238
            is_loop_flag vtable[63]  bit1 of u8 module+0x234
            is_flip      vtable[114] u8  module+0x23A
            rate         vtable[40]  f32 module+0x7C
            prev_frame   vtable[45]  f32 module+0x74
            end_frame    vtable[48]  u32(end f32), derived exactly
            is_end       derived from frame >= end_frame
            is_looped    derived exactly from the getter's float window
        """
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.motion.ptr", T_I64, mod, idx, True)
        frame = m.f32(mod + MOTION_FRAME)
        add(out, "f.motion.frame", T_F64, frame, idx, True)
        kind, kok = self._read_wide(mod, SLOTS["motion_kind"])
        if kind is not None:
            add(out, "f.motion.kind", T_I64, kind, idx, kok)
            add(out, "f.motion.kind_low32", T_U32, kind & 0xFFFFFFFF,
                idx, kok)
        changing = m.u8(mod + 0x238)
        if changing is not None:
            add(out, "f.motion.is_changing", T_BOOL, bool(changing), idx, True)
        loopflag = m.u8(mod + 0x234)
        if loopflag is not None:
            add(out, "f.motion.is_loop_flag", T_BOOL,
                bool((loopflag >> 1) & 1), idx, True)
        flip = m.u8(mod + 0x23A)
        if flip is not None:
            add(out, "f.motion.is_flip", T_BOOL, bool(flip), idx, True)
        rate = m.f32(mod + 0x7C)
        add(out, "f.motion.rate", T_F64, rate, idx, True)
        prev = m.f32(mod + 0x74)
        add(out, "f.motion.prev_frame", T_F64, prev, idx, True)
        whole = m.f32(mod + 0x88)
        # end_frame: exact replication of the 0x49c140 getter (returns u32)
        use_alt = m.u8(mod + 0x239)
        end = None
        if use_alt:
            if m.u64(mod + 0x10):
                v = m.f32(mod + 0x8C)
                if v is not None and math.isfinite(v):
                    end = int(v) & 0xFFFFFFFF
        else:
            if m.u8(mod + 0x250) or m.u8(mod + 0x252):
                v = m.f32(mod + 0x228)
                if v is not None and math.isfinite(v):
                    end = int(v) & 0xFFFFFFFF
        if end is not None:
            add(out, "f.motion.end_frame", T_U32, end, idx, True)
            if frame is not None and math.isfinite(frame):
                add(out, "f.motion.is_end", T_BOOL,
                    bool(loopflag == 0 and frame >= float(end)), idx, True)
        # is_looped: exact replication of the 0x49cb70 getter
        if (loopflag and frame is not None and math.isfinite(frame)
                and frame >= 0.0 and rate is not None and whole is not None
                and prev is not None
                and math.isfinite(rate) and math.isfinite(whole)
                and math.isfinite(prev)):
            looped = bool(frame < rate * whole and frame < prev)
            add(out, "f.motion.is_looped", T_BOOL, looped, idx, True)
        # is_blend uses a virtual dispatch; keep the raw prefix only
        raw = m.read(mod, 0x40)
        if raw:
            add(out, "f.motion.raw0x40", T_BYTES, raw, idx, True)

    def _read_stop(self, mod, idx, out):
        """StopModule at accessor+0x90 (live vtable 0x7104f890e0).

        Concrete getters: is_stop u8+0xF0, is_hit u8+0xF4, is_damage u8+0xF8,
        get_hit_stop_real_frame u32+0x10.  hit_stop_frame is the engine's real
        hitlag length (live: 19 -> 12 -> 6 -> 0, and 4 on a shielded hit).
        """
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.stop.ptr", T_I64, mod, idx, True)
        add(out, "f.stop.is_stop", T_BOOL, bool(m.u8(mod + 0xF0)), idx, True)
        add(out, "f.stop.is_hit", T_BOOL, bool(m.u8(mod + 0xF4)), idx, True)
        add(out, "f.stop.is_damage", T_BOOL, bool(m.u8(mod + 0xF8)), idx, True)
        add(out, "f.stop.hit_stop_frame", T_U32, m.u32(mod + 0x10), idx, True)
        raw = m.read(mod + 0xF0, 0x10)
        if raw:
            add(out, "f.stop.flags_raw", T_BYTES, raw, idx, True)

    def _read_hit(self, mod, idx, out):
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.hit.ptr", T_I64, mod, idx, True)
        # raw prefix; invincibility getters vary by object, the frame setters
        # write at documented offsets in the concrete HitModule (map: partial)
        raw = m.read(mod, 0x80)
        if raw:
            add(out, "f.hit.raw0x80", T_BYTES, raw, idx, False)

    def _read_work(self, mod, cancel, idx, out):
        if not mod:
            return
        m = self.mem
        add = self._add
        add(out, "f.work.ptr", T_I64, mod, idx, True)
        terms_p = m.u64(mod + WORK_TERMS)
        if terms_p:
            add(out, "f.work.terms_ptr", T_I64, terms_p, idx, True)
            terms = m.read(terms_p, 256)
            if terms:
                add(out, "f.work.terms", T_BYTES, terms, idx, True)
                # 256-bit enabled mask, using the game's own test
                words = [0, 0, 0, 0]
                count = 0
                for t, b in enumerate(terms):
                    if b & 0x38:
                        en = 0
                    elif b & 0x02:
                        en = (b >> 2) & 1
                    else:
                        en = b & 1
                    if en:
                        count += 1
                        words[t >> 6] |= 1 << (t & 63)
                add(out, "f.work.term_count", T_U32, count, idx, True)
                add(out, "f.work.term_mask0", T_I64, words[0] - (1 << 64)
                    if words[0] >= 1 << 63 else words[0], idx, True)
                add(out, "f.work.term_mask1", T_I64, words[1] - (1 << 64)
                    if words[1] >= 1 << 63 else words[1], idx, True)
                add(out, "f.work.term_mask2", T_I64, words[2] - (1 << 64)
                    if words[2] >= 1 << 63 else words[2], idx, True)
                add(out, "f.work.term_mask3", T_I64, words[3] - (1 << 64)
                    if words[3] >= 1 << 63 else words[3], idx, True)
        ka = m.u64(mod + WORK_KIND_ARRAY)
        if ka:
            add(out, "f.work.kind_array", T_I64, ka, idx, True)
            kind0 = m.u64(ka)
            if kind0:
                add(out, "f.work.kind0", T_I64, kind0, idx, True)
                ints_p = m.u64(kind0 + WORK_INT_ARRAY)
                floats_p = m.u64(kind0 + WORK_FLOAT_ARRAY)
                flags_p = m.u64(kind0 + WORK_FLAGS)
                if ints_p:
                    add(out, "f.work.ints_ptr", T_I64, ints_p, idx, True)
                    d = m.read(ints_p, 64 * 4)
                    if d:
                        add(out, "f.work.ints", T_BYTES, d, idx, True)
                if floats_p:
                    add(out, "f.work.floats_ptr", T_I64, floats_p, idx, True)
                    d = m.read(floats_p, 32 * 4)
                    if d:
                        add(out, "f.work.floats", T_BYTES, d, idx, True)
                if flags_p:
                    add(out, "f.work.flags_ptr", T_I64, flags_p, idx, True)
                    d = m.read(flags_p, 64)
                    if d:
                        add(out, "f.work.flags", T_BYTES, d, idx, True)
        if cancel:
            add(out, "f.cancel.ptr", T_I64, cancel, idx, True)
            add(out, "f.cancel.enable", T_BOOL, bool(m.u8(cancel + 0x1C)),
                idx, True)

    def _read_info(self, entry, entry_id, idx, out):
        m = self.mem
        add = self._add
        info_obj = m.u64(entry + ENTRY_INFO)
        rec = m.u64(entry + ENTRY_RECORD)
        if info_obj:
            add(out, "f.info.object", T_I64, info_obj, idx, True)
        if not rec:
            return
        add(out, "f.info.record", T_I64, rec, idx, True)
        kind = m.u8(rec + INFO_KIND_PLUS1)
        if kind is not None and 0 < kind < 200:
            add(out, "f.info.kind", T_U32, kind - 1, idx, True)
        add(out, "f.info.stock", T_U32, m.u32(rec + INFO_STOCK), idx, True)
        add(out, "f.info.color", T_U32, m.u8(rec + INFO_COLOR), idx, True)
        add(out, "f.info.is_cpu", T_BOOL, bool(m.u8(rec + INFO_IS_CPU)),
            idx, True)
        add(out, "f.info.is_on_rebirth", T_BOOL,
            bool(m.u8(rec + INFO_ON_REBIRTH)), idx, True)
        add(out, "f.info.category", T_U32, m.u8(rec + INFO_CATEGORY),
            idx, True)
        add(out, "f.info.gravity", T_F64, m.f32(rec + INFO_GRAVITY), idx, True)
        add(out, "f.info.is_last_dead_suicide", T_BOOL,
            bool(m.u8(rec + INFO_LAST_DEAD_SUICIDE)), idx, True)
        d = m.read(rec + INFO_HIT_POINT_MAX, 8)
        if d and len(d) == 8:
            hp = struct.unpack("<2f", d)
            add(out, "f.info.hit_point_max", T_VEC3,
                (hp[0], hp[1], hp[0] + hp[1]), idx, True)
        dc = m.read(rec + INFO_DEAD_COUNT, 12)
        if dc and len(dc) == 12:
            per = struct.unpack("<3I", dc)
            add(out, "f.info.dead_total", T_U32, sum(per), idx, True)
            if entry_id < 3:
                add(out, "f.info.dead_count", T_U32, per[entry_id], idx, True)
        # damage log: pointer arrays at rec+0x15E0, stride 0x11B0 per type;
        # record contents are not decoded -> raw, verified=false for meaning.
        for t in range(3):
            p = m.u64(rec + INFO_LOG + INFO_LOG_STRIDE * t)
            if not p:
                continue
            add(out, "f.info.log%d.ptr" % t, T_I64, p, idx, False)
            raw = m.read(p, 0x40)
            if raw:
                add(out, "f.info.log%d.raw0x40" % t, T_BYTES, raw, idx,
                    False)

    # -- everything -------------------------------------------------------
    def read(self) -> list[Rec]:
        out: list[Rec] = []
        self.read_globals(out)
        impl = self.manager_impl()
        if impl:
            for e in range(2):
                try:
                    self.read_fighter(impl, e, out)
                except Exception:
                    # a torn-down fighter must never kill the extras channel
                    continue
            try:
                self.read_battle_objects(out)
            except Exception:
                pass
        return out


# ---------------------------------------------------------------------------
# publisher
# ---------------------------------------------------------------------------
class Publisher:
    """One writer, many readers; odd/even seq, fixed 1 MiB file."""

    def __init__(self, path: str = SHM_PATH):
        self.path = path
        self.fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(self.fd, MAX_BLOCK)
        self.mm = mmap.mmap(self.fd, MAX_BLOCK)
        self.seq = 0
        self.mm[0:HEADER_SIZE] = b"\x00" * HEADER_SIZE

    def publish(self, records: list[Rec], flags: int = 0) -> int:
        ts = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        self.seq = (self.seq + 1) | 1
        # header with odd seq, then body, then final header with count/total
        head = _HEADER.pack(MAGIC, VERSION, self.seq, ts, 0, 0, flags)
        self.mm[0:HEADER_SIZE] = head
        off = HEADER_SIZE
        for r in records:
            blob = r.packed()
            if off + len(blob) > MAX_BLOCK:
                break
            self.mm[off:off + len(blob)] = blob
            off += len(blob)
        total = off
        self.seq += 1
        self.mm[0:HEADER_SIZE] = _HEADER.pack(
            MAGIC, VERSION, self.seq, ts, len(records), total, flags)
        return ts

    def close(self):
        try:
            self.mm.close()
        finally:
            os.close(self.fd)


def read_block(path: str = SHM_PATH) -> dict:
    with open(path, "rb") as fh:
        return unpack_block(fh.read())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _describe(records: list[Rec]) -> str:
    lines = []
    for r in records:
        v = r.value_python()
        if isinstance(v, float):
            vs = f"{v:.4f}"
        elif isinstance(v, bytes):
            vs = f"<{len(v)} bytes>"
        elif isinstance(v, (list, tuple)):
            vs = "(" + ", ".join(f"{x:.3f}" for x in v) + ")"
        else:
            vs = str(v)
        lines.append(f"{'V' if r.verified else '?'} [{r.index}] "
                     f"{r.name:<40} {TYPE_NAMES.get(r.type, r.type):<6} {vs}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--elf", default=None)
    ap.add_argument("--shm", default=SHM_PATH)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="publish for N seconds (0 = one shot)")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--print", action="store_true", help="print the snapshot")
    ap.add_argument("--json", action="store_true", help="print JSON")
    ap.add_argument("--no-shm", action="store_true", help="do not write shm")
    ap.add_argument("--if-live", action="store_true",
                    help="only write the block while fight_on == 1")
    args = ap.parse_args()

    pid = args.pid or find_game_pid()
    if not pid:
        print("error: no ssbu_native process", file=sys.stderr)
        return 2
    ex = ExtraState(pid, args.elf)
    pub = None if args.no_shm else Publisher(args.shm)
    period = 1.0 / max(1.0, args.fps)
    start = time.perf_counter()
    end = start + args.duration if args.duration > 0 else start
    shows = 0
    try:
        while True:
            recs = ex.read()
            live = ex.fight_on()
            flags = (FLAG_LIVE if live else 0) | \
                    (FLAG_DECODED if ex.decoded_any else 0)
            if pub and (live or not args.if_live):
                pub.publish(recs, flags)
            if args.print:
                print(_describe(recs), flush=True)
                shows += 1
                if args.duration <= 0:
                    break
                if shows >= 1:
                    break
            elif args.json:
                print(json.dumps({
                    "pid": pid, "live": live, "seq": pub.seq if pub else 0,
                    "records": [
                        {"name": r.name, "type": TYPE_NAMES.get(r.type, r.type),
                         "index": r.index, "verified": r.verified,
                         "value": r.value_python()} for r in recs]},
                    indent=1), flush=True)
                break
            if args.duration <= 0:
                break
            if time.perf_counter() >= end:
                break
            time.sleep(period)
    finally:
        if pub:
            pub.close()
        ex.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
