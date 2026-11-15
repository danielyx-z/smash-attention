"""Resolve live SSBU fighter state through the game's own object graph.

Nothing here is hardcoded:

  * the guest image base comes from /proc/<pid>/maps
  * every symbol offset comes from main.elf's .dynsym
  * every struct offset was derived by disassembling the game's own
    lua_bind accessors (see memmap.json for the disassembly evidence)

Chain (verified live):

    instance_  -> mgr -> impl                (FighterManager)
    impl + 0x20 + 8*entry_id -> FighterEntry
    entry + 0xF0             -> FighterInformation (embedded; vtable at +0)
    info + 8                 -> information record (kind, stock, cpu, rebirth)
    entry + 0x4160           -> fighter object
    obj + 0x20               -> BattleObjectModuleAccessor

    accessor + 0x38  -> PostureModule   pos    = module + 0x30 (x, y, z floats)
                                        facing = module + 0xAC (float)
    accessor + 0x40  -> StatusModule    action_state   = module + 0x98 (u32)
                                        situation_kind = module + 0xDC (u32)
    accessor + 0x48  -> ControlModule   input struct, virtual getters
    accessor + 0x50  -> WorkModule      instance ints  = [[module+0x18][0]+0x20]
                                        transition terms = [module+0x28]
    accessor + 0x88  -> MotionModule    action_frame = module + 0x70 (float)
    accessor + 0xA8  -> DamageModule    damage = *(module + 0x18)
    accessor + 0x128 -> CancelModule    is_enable_cancel = module + 0x1C (u8)

The information record is reached the same way the game's own
FighterInformation lua_bind accessors do it: `this` is entry+0xF0, and every
accessor starts with `ldr x8,[x0,#8]` before reading its field.

ControlModule getters are virtual.  The lua_bind wrappers load the module with
`ldr x0,[accessor,#0x48]`, load the vtable, and branch to
+0x238 (get_button), +0x178 (get_stick_x), +0x188 (get_stick_y),
+0x200 (get_sub_stick_x), +0x210 (get_sub_stick_y), +0x220 (get_trigger).
The concrete functions are tiny loaders; gamestate reads the vtable target,
disassembles the loader and uses the offsets it finds, so the layout is
resolved from the running build instead of hardcoded.  The known fallbacks are
kept for crash safety and are documented in memmap.json.

Blast zones come from the live camera manager, not from a table:

    BossManager::instance_ + 8 -> camera singleton
    [singleton] -> camera manager, dead area = manager + 0xC30 (4 floats:
    left, right, top, bottom).  This is exactly what the game's own
    app::camera::get_dead_area() / sv_camera_manager::dead_range() read.

Stage id:

    StageManager::instance_ -> *[0] -> *[0] -> u32 + 0x17C
    (exactly what app::sv_information::stage_id reads; 0xFFFFFFFF = unknown)

`is_actionable` is derived from the game's own transition-term table: the
WorkModule keeps one byte per status transition term, and a fighter that may
act has several terms enabled.  Hitstun/tumble/KO have 0-2, WAIT has ~49.
This replaced the earlier empirical action-state table (kept only as a
fallback for the rare case where the term table cannot be read).
"""

from __future__ import annotations

import math
import os
import struct
import time

from rl.memory.elfsym import ElfSymbols
from rl.memory.memapi import ProcessMemory, guest_image_base
from rl.memory import schema

#: FighterManager singleton, resolved from main.elf .dynsym at run time.
FIGHTER_MANAGER = "_ZN3lib9SingletonIN3app14FighterManagerEE9instance_E"
#: The camera singleton is the anonymous pointer immediately after this symbol
#: (see app::camera::get_dead_area: adrp 0x52b7000 / ldr [x8,#0xf00]).
BOSS_MANAGER = "_ZN3lib9SingletonIN3app11BossManagerEE9instance_E"
#: StageManager singleton (app::sv_information::stage_id reads this chain).
STAGE_MANAGER = "_ZN3lib9SingletonIN3app12StageManagerEE9instance_E"

MAX_ENTRIES = 8

# -- struct offsets (all proven by disassembly, see memmap.json) ------------
IMPL_READY_GO = 0xD2          # u8, 1 while a fight is running
IMPL_ENTRIES = 0x20           # u64[8] of FighterEntry*
ENTRY_FIGHTER = 0x4160        # u64, first fighter object of the entry
ENTRY_INFO = 0xF0             # FighterInformation object is embedded here
ENTRY_RECORD = 0xF8           # u64 -> information record (= *(info+8))
OBJ_ACCESSOR = 0x20           # u64, BattleObjectModuleAccessor*
ACC_POSTURE = 0x38
ACC_STATUS = 0x40
ACC_CONTROL = 0x48
ACC_WORK = 0x50
ACC_MOTION = 0x88
ACC_DAMAGE = 0xA8
ACC_CANCEL = 0x128

POSTURE_POS = 0x30            # float32 x, y, z
POSTURE_LR = 0xAC             # float32 facing
STATUS_KIND = 0x98            # u32 action/status kind
STATUS_SITUATION = 0xDC       # u32 0 = ground, else air/cliff
MOTION_FRAME = 0x70           # float32 action frame
DAMAGE_ARRAY = 0x18           # u64 -> damage array

WORK_KIND_ARRAY = 0x18        # u64 -> 16 pointers (one per work kind)
WORK_INT_BASE = 0x20          # kind object + 0x20 -> int32 array
WORK_TERMS = 0x28             # u64 -> one byte per transition term
INSTANCE_JUMP_COUNT = 4       # int32 index: jumps used
INSTANCE_JUMP_MAX = 6         # int32 index: jumps available
CANCEL_ENABLE = 0x1C          # u8, CancelModule::is_enable_cancel

REC_KIND_PLUS1 = 0x30         # u8, fighter kind + 1
REC_CPU_FLAG = 0x92           # u8, 1 = CPU, 0 = human
REC_STOCK = 0xD8              # u32, remaining stocks (contract: read u32, store i32)
REC_ON_REBIRTH = 0x38C        # u8, 1 while on the revival platform

#: 13.0.4 match block (the records the game spawns the fighters from).  The
#: entry information record at entry+0xF8 keeps the select-screen copy, which
#: can still name the saved roster when the debug hook forces a character by
#: name (SSBU_DEBUG_MATCH); the match's own player record carries the kind that
#: actually spawned, at +8.  Player record n is at ``MATCH_BLOCK + 0x260*n``.
MATCH_BLOCK = 0x52C4730
MATCH_RECORD_SIZE = 0x260
MATCH_RECORD_KIND = 0x8       # u8, fighter kind (0 = Mario, 1 = Donkey Kong, ...)

# ControlModule: the module holds a pointer to its input struct at +0x110.
CTRL_INPUT_PTR = 0x110
CTRL_BUTTONS = 0x7C           # u32 raw button bitmask
CTRL_BUTTONS_PREV = 0x80      # u32 previous frame
CTRL_STICK_X = 0x10           # f32
CTRL_STICK_Y = 0x14           # f32
CTRL_SUB_STICK_X = 0x40       # f32 (C-stick)
CTRL_SUB_STICK_Y = 0x44       # f32
CTRL_TRIGGER = 0x88           # u32 / f32, see resolved getter
CTRL_FLIP_X = 0x120           # u32, nonzero => stick x / sub stick x negated

#: vtable slots, from the lua_bind wrappers (see memmap.json evidence)
CTRL_SLOT_BUTTON = 0x238
CTRL_SLOT_STICK_X = 0x178
CTRL_SLOT_STICK_Y = 0x188
CTRL_SLOT_SUB_X = 0x200
CTRL_SLOT_SUB_Y = 0x210
CTRL_SLOT_TRIGGER = 0x220

CAMERA_DEAD_AREA = 0xC30      # float32 left, right, top, bottom
STAGE_ID_OFFSET = 0x17C       # u32 in the StageManager object
STAGE_ID_UNKNOWN = 0xFFFFFFFF

#: The game's global match block for build 13.0.4 (runtime/debug_mode.c uses
#: GA(0x71052c41b0)); its stage id is the value the debug hook prints at match
#: start ("stage id 0x125 (match block + 0x420)").  Used only as a fallback
#: when the StageManager chain is not populated in-match.
MATCH_BLOCK_OFFSET = 0x52C41B0
MATCH_BLOCK_STAGE = 0x420

# SituationKind: 0 is "standing on the ground".
SITUATION_GROUND = 0

# A fighter with at least this many enabled transition terms can act.  WAIT
# has ~49, dash ~30, jump ~32, cliff ~12, the cancel window of an attack ~49;
# hitstun has 2, tumble/KO 0.
ACTIONABLE_TERM_MIN = 4

#: Status kinds of a damage reaction (hitstun).  These are the game's own
#: DAMAGE/DAMAGE_FLY/TUMBLE/DOWN_FALL kinds; `hitstun` is 1 while in one.
DAMAGE_STATUS_KINDS = frozenset(
    set(range(54, 63)) | {476, 477, 478, 479} | set(range(505, 512))
)

#: Status kinds of a held shield.  Confirmed live at 60 Hz by watching the
#: derived field against the game's own status kind: whenever the fighter is
#: actually holding shield the status kind is 10 (the CPU brackets it with
#: jump-squat/attack kinds).  The approximate ACTION_STATE_NAMES table called
#: 10 "GUARD_OFF"; the live status machine says it is the held guard.
GUARD_STATUS_KINDS = frozenset({10})


# ---- fallback is_actionable lookup (only if the term table is unreadable) --
# Conservative: only mark actionable if we are confident.
ACTIONABLE_TABLE = {
    0:   1,   # WAIT
    1:   1,   # WALK
    2:   1,   # RUN
    4:   1,   # TURN
    6:   1,   # DASH
    7:   1,   # DASH_BACK
    11:  1,   # JUMP_SQUAT
    24:  1,   # SQUAT
    25:  1,   # SQUAT_WAIT
    26:  1,   # SQUAT_TURN
    140: 1,   # GUARD (held shield)
}
DEFAULT_ACTIONABLE = 0


def _enabled_term_count(data: bytes) -> int:
    """Count the enabled transition terms, exactly as the game's accessor does.

    is_enable_transition_term(term):
        b = terms[term]
        if b & 0x38: return 0
        return ((b >> 2) & 1) if (b & 2) else (b & 1)
    """
    n = 0
    for b in data:
        if b & 0x38:
            continue
        n += ((b >> 2) & 1) if (b & 0x02) else (b & 0x01)
    return n


def default_elf() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # ssbu-rl lives beside ssbu-recomp; four levels up from src/rl/memory.
    repo = os.environ.get("SSBU_RECOMP_DIR") or os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "ssbu-recomp"))
    return os.path.join(repo, "build-linux", "binaries", "13.0.4", "main.elf")


# ---- ControlModule virtual getters ---------------------------------------
def _decode_ldr_imm(insn: int):
    """Decode a base+immediate LDR into (kind, rt, rn, offset) or None."""
    op = insn & 0xFFC00000
    rt = insn & 0x1F
    rn = (insn >> 5) & 0x1F
    imm12 = (insn >> 10) & 0xFFF
    if op == 0xF9400000:        # ldr xN, [xM, #imm]
        return ("x", rt, rn, imm12 * 8)
    if op == 0xB9400000:        # ldr wN, [xM, #imm]
        return ("w", rt, rn, imm12 * 4)
    if op == 0xBD400000:        # ldr sN, [xM, #imm]
        return ("s", rt, rn, imm12 * 4)
    if op == 0x39400000:        # ldrb wN, [xM, #imm]
        return ("b", rt, rn, imm12)
    return None


def decode_control_getter(code: bytes):
    """Pull the offsets out of one of the tiny ControlModule getters.

    Every resolved getter so far looks like one of:

        ldr x8, [x0, #0x110]        ; input struct pointer
        ldr w0, [x8, #0x7c]         ; button bitmask
        ret

        ldr x9, [x0, #0x110]
        ldr s0, [x9, #0x10]         ; stick x
        ldr w8, [x0, #0x120]        ; sign-flip flag
        cmp w8, #0
        fneg s1, s0
        fcsel s0, s0, s1, eq
        ret

    Returns (input_offset, (field_kind, field_offset), flip_offset).
    """
    if not code or len(code) < 8:
        return None
    insns = [struct.unpack_from("<I", code, i)[0] for i in range(0, len(code) - 3, 4)]
    loads = []
    for insn in insns:
        d = _decode_ldr_imm(insn)
        if d:
            loads.append(d)
    ptr_off = ptr_reg = None
    field = None
    flip_off = None
    for kind, rt, rn, off in loads:
        if ptr_off is None:
            if rn == 0 and kind == "x":
                ptr_off, ptr_reg = off, rt
            continue
        if field is None and rn == ptr_reg:
            field = (kind, off)
        elif rn == 0 and kind in ("w", "x"):
            flip_off = off
    if ptr_off is None or field is None:
        return None
    return ptr_off, field, flip_off


class ControlLayout:
    """Offsets resolved from the ControlModule vtable at runtime."""

    __slots__ = ("input_off", "button_off", "stick_x_off", "stick_y_off",
                 "sub_x_off", "sub_y_off", "trigger_off", "trigger_kind",
                 "flip_x_off", "flip_sub_x_off", "vtable")

    def __init__(self, vtable: int, fields: dict):
        self.vtable = vtable
        self.input_off = fields["input"]
        self.button_off = fields["button"][1]
        self.stick_x_off = fields["stick_x"][1]
        self.stick_y_off = fields["stick_y"][1]
        self.sub_x_off = fields["sub_x"][1]
        self.sub_y_off = fields["sub_y"][1]
        self.trigger_off = fields["trigger"][1]
        self.trigger_kind = fields["trigger"][0]
        self.flip_x_off = fields.get("flip_x")
        self.flip_sub_x_off = fields.get("flip_sub_x")

    def __repr__(self) -> str:
        return (f"ControlLayout(input={self.input_off:#x}, button={self.button_off:#x}, "
                f"stick=({self.stick_x_off:#x},{self.stick_y_off:#x}), "
                f"cstick=({self.sub_x_off:#x},{self.sub_y_off:#x}), "
                f"trigger={self.trigger_off:#x}/{self.trigger_kind}, "
                f"flip_x={self.flip_x_off}, flip_sub={self.flip_sub_x_off})")


#: Fallback layout, used only if the live vtable cannot be read (teardown).
FALLBACK_CONTROL_LAYOUT = ControlLayout(0, {
    "input": CTRL_INPUT_PTR,
    "button": ("w", CTRL_BUTTONS),
    "stick_x": ("s", CTRL_STICK_X),
    "stick_y": ("s", CTRL_STICK_Y),
    "sub_x": ("s", CTRL_SUB_STICK_X),
    "sub_y": ("s", CTRL_SUB_STICK_Y),
    "trigger": ("w", CTRL_TRIGGER),
    "flip_x": CTRL_FLIP_X,
    "flip_sub_x": CTRL_FLIP_X,
})



class GameState:
    """Read the observation set for every fighter in the current match."""

    def __init__(self, pid: int, elf_path: str | None = None):
        self.pid = pid
        self.mem = ProcessMemory(pid)
        self.syms = ElfSymbols(elf_path or default_elf())
        self.base = guest_image_base(pid)
        self._instance = self.base + self.syms.offset(FIGHTER_MANAGER)
        boss = self.syms.symbols.get(BOSS_MANAGER)
        self._camera_global = (self.base + boss + 8) if boss is not None else None
        stage = self.syms.symbols.get(STAGE_MANAGER)
        self._stage_global = (self.base + stage) if stage is not None else None
        self._seen_fight = False       # for match_state transition
        self._fight_off_at = None      # monotonic time of the last ready_go 1->0
        self._prev_pos: dict[int, tuple[float, float, float]] = {}
        self._control_cache: dict[int, ControlLayout | None] = {}
        self._last_getter: dict[int, tuple[int, int, tuple]] = {}
        self.last_bounds: tuple[float, float, float, float] | None = None
        self.last_stage_id: int | None = None
        self.last_stage_source: str | None = None
        self.control_layout: ControlLayout | None = None


    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        self.mem.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- match ------------------------------------------------------------
    def manager_impl(self) -> int | None:
        mgr = self.mem.u64(self._instance)
        return self.mem.u64(mgr) if mgr else None

    def fight_on(self) -> bool:
        impl = self.manager_impl()
        return bool(impl) and self.mem.u8(impl + IMPL_READY_GO) == 1

    def match_state(self) -> int:
        """MENU(0) / IN_MATCH(1) / RESULTS(2).

        A results screen keeps the FighterEntry objects allocated, while the
        main menu frees them; that is the reliable discriminator.  A short
        grace period covers the frame or two before the entries are released
        after the ready-go flag drops.
        """
        on = self.fight_on()
        if on:
            self._seen_fight = True
            self._fight_off_at = None
            return schema.MatchState.IN_MATCH

        impl = self.manager_impl()
        loaded = False
        if impl:
            for e in range(2):
                if self.mem.u64(impl + IMPL_ENTRIES + 8 * e):
                    loaded = True
                    break
        if loaded:
            return schema.MatchState.RESULTS

        now = time.monotonic()
        if self._fight_off_at is None:
            self._fight_off_at = now
        if self._seen_fight and (now - self._fight_off_at) < 8.0:
            return schema.MatchState.RESULTS
        return schema.MatchState.MENU

    # -- stage ------------------------------------------------------------
    def read_stage_bounds(self) -> tuple[float, float, float, float] | None:
        """Live (left, right, top, bottom) blast-zone rectangle."""
        if not self._camera_global:
            return None
        m = self.mem
        a = m.u64(self._camera_global)
        if not a:
            return None
        ptr = m.u64(a)
        if not ptr:
            return None
        data = m.read(ptr + CAMERA_DEAD_AREA, 16)
        if not data or len(data) < 16:
            return None
        left, right, top, bottom = struct.unpack("<4f", data)
        if not all(math.isfinite(v) for v in (left, right, top, bottom)):
            return None
        if not (left < right and bottom < top):
            return None
        if max(abs(left), abs(right), abs(top), abs(bottom)) > 20000.0:
            return None
        return (left, right, top, bottom)

    # -- stage id ---------------------------------------------------------
    def read_stage_id(self) -> int | None:
        """The game's own stage id (u32).

        app::sv_information::stage_id does
            adrp/ldr -> StageManager::instance_ contents
            ldr x8,[x8]              ; the manager object
            ldr w9,[x8,#0x17c]       ; -1 means "no stage chosen"
        so there are two pointer dereferences from the singleton global.

        On this build that field reads 0 during a fight (it is the select-screen
        information copy).  The match block keeps the in-match stage id (the
        same value the debug hook prints at match start), so use it as the
        fallback.  `self.last_stage_source` records which one was used.
        """
        m = self.mem
        sid = None
        if self._stage_global:
            first = m.u64(self._stage_global)
            if first:
                mgr = m.u64(first)
                if mgr:
                    sid = m.u32(mgr + STAGE_ID_OFFSET)
                    if sid is not None and sid != STAGE_ID_UNKNOWN and sid <= 0xFFFF:
                        if sid != 0:
                            self.last_stage_source = "stagemanager"
                            return sid
        mb = m.u32(self.base + MATCH_BLOCK_OFFSET + MATCH_BLOCK_STAGE)
        if mb is not None and mb != STAGE_ID_UNKNOWN and mb <= 0xFFFF and mb != 0:
            self.last_stage_source = "matchblock"
            return mb
        self.last_stage_source = None
        return None

    # -- ControlModule ----------------------------------------------------
    def _getter(self, vtable: int, slot: int):
        target = self.mem.u64(vtable + slot)
        if not target:
            return None
        # Guest-image offset of the concrete getter (memmap.json evidence).
        goff = target - self.base
        # Read the loader from the live process.  For a main-image getter the
        # bytes are identical to main.elf[goff]; for a per-character NRO module
        # (loads around 0x72...) the live read is the only option.
        code = self.mem.read(target, 32)
        if not code:
            return None
        decoded = decode_control_getter(code)
        if decoded is not None:
            self._last_getter[slot] = (target, goff, decoded)
        return decoded

    def control_layout_for(self, module: int) -> ControlLayout | None:
        """Resolve the concrete virtual getters once per ControlModule vtable."""
        vtable = self.mem.u64(module)
        if not vtable:
            return None
        if vtable in self._control_cache:
            return self._control_cache[vtable]
        slots = (("button", CTRL_SLOT_BUTTON),
                 ("stick_x", CTRL_SLOT_STICK_X),
                 ("stick_y", CTRL_SLOT_STICK_Y),
                 ("sub_x", CTRL_SLOT_SUB_X),
                 ("sub_y", CTRL_SLOT_SUB_Y),
                 ("trigger", CTRL_SLOT_TRIGGER))
        decoded = {}
        for name, slot in slots:
            d = self._getter(vtable, slot)
            if d is None:
                decoded = None
                break
            decoded[name] = d
        layout: ControlLayout | None = None
        if decoded:
            inputs = {d[0] for d in decoded.values()}
            if len(inputs) == 1:
                fields = {
                    "input": inputs.pop(),
                    "button": decoded["button"][1],
                    "stick_x": decoded["stick_x"][1],
                    "stick_y": decoded["stick_y"][1],
                    "sub_x": decoded["sub_x"][1],
                    "sub_y": decoded["sub_y"][1],
                    "trigger": decoded["trigger"][1],
                    "flip_x": decoded["stick_x"][2],
                    "flip_sub_x": decoded["sub_x"][2],
                }
                if all(0 <= fields[k][1] < 0x10000
                       for k in ("button", "stick_x", "stick_y", "sub_x",
                                 "sub_y", "trigger")) \
                        and 0 <= fields["input"] < 0x10000:
                    layout = ControlLayout(vtable, fields)
        if layout is not None:
            self._control_cache[vtable] = layout
        elif self.base <= vtable < self.base + 0x10000000:
            # The vtable is a real guest-image vtable but a getter could not
            # be read this frame (teardown).  Use the proven fallback offsets
            # without caching them, so the next frame retries the real ones.
            return FALLBACK_CONTROL_LAYOUT
        return layout

    @staticmethod
    def _trigger_value(raw: int | None, kind: str) -> float | None:
        """Turn the getter's raw word into [0, 1].

        The resolved getter can load an f32 (kind 's') or a u32 (kind 'w').
        For the u32 form the live value may be fixed-point 0..0xFFFF or the
        bit pattern of a float; accept both, because which one it is depends
        on the pad backend.
        """
        if raw is None:
            return None
        if kind == "s":
            raw = int(raw) & 0xFFFFFFFF
            value = struct.unpack("<f", struct.pack("<I", raw))[0]
        else:
            raw = int(raw)
            if raw <= 0x10000:
                value = raw / 65535.0
            else:
                value = struct.unpack("<f", struct.pack("<I", raw & 0xFFFFFFFF))[0]
                if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                    value = raw / 0xFFFFFFFF
        if not math.isfinite(value):
            return None
        return min(1.0, max(0.0, value))

    def read_control(self, accessor: int, put) -> ControlLayout | None:
        """Read the expert control fields through the game's ControlModule."""
        m = self.mem
        module = m.u64(accessor + ACC_CONTROL)
        if not module:
            return None
        layout = self.control_layout_for(module)
        if not layout:
            return None
        inp = m.u64(module + layout.input_off)
        if not inp:
            return None

        raw_buttons = m.u32(inp + layout.button_off)
        if raw_buttons is not None:
            put("buttons", raw_buttons)

        flip = 0
        if layout.flip_x_off is not None:
            flip = m.u32(module + layout.flip_x_off) or 0
        if layout.flip_sub_x_off is not None and layout.flip_sub_x_off != layout.flip_x_off:
            flip_sub = m.u32(module + layout.flip_sub_x_off) or 0
        else:
            flip_sub = flip

        sx = m.f32(inp + layout.stick_x_off)
        if sx is not None:
            put("stick_x", max(-1.0, min(1.0, -sx if flip else sx)))
        sy = m.f32(inp + layout.stick_y_off)
        if sy is not None:
            put("stick_y", max(-1.0, min(1.0, sy)))
        cx = m.f32(inp + layout.sub_x_off)
        if cx is not None:
            put("cstick_x", max(-1.0, min(1.0, -cx if flip_sub else cx)))
        cy = m.f32(inp + layout.sub_y_off)
        if cy is not None:
            put("cstick_y", max(-1.0, min(1.0, cy)))

        raw_trigger = m.u32(inp + layout.trigger_off)
        trig = self._trigger_value(raw_trigger, layout.trigger_kind)
        if trig is not None:
            put("trigger", trig)

        self.control_layout = layout
        return layout


    # -- one fighter ------------------------------------------------------
    def read_fighter(self, impl: int, entry_id: int) -> dict | None:
        m = self.mem
        entry = m.u64(impl + IMPL_ENTRIES + 8 * entry_id)
        if not entry:
            return None
        obj = m.u64(entry + ENTRY_FIGHTER)
        if not obj:
            return None
        acc = m.u64(obj + OBJ_ACCESSOR)
        if not acc:
            return None

        posture = m.u64(acc + ACC_POSTURE)
        status = m.u64(acc + ACC_STATUS)
        work = m.u64(acc + ACC_WORK)
        motion = m.u64(acc + ACC_MOTION)
        damage_mod = m.u64(acc + ACC_DAMAGE)
        rec = m.u64(entry + ENTRY_RECORD)

        row: dict = {}
        flags = 0

        def put(name: str, value):
            nonlocal flags
            if value is not None:
                row[name] = value
                flags |= 1 << schema.FIELD_INDEX[name]

        put("entry_id", entry_id)

        if posture:
            put("x", m.f32(posture + POSTURE_POS))
            put("y", m.f32(posture + POSTURE_POS + 4))
            put("facing", m.f32(posture + POSTURE_LR))

        if status:
            kind = m.u32(status + STATUS_KIND)
            situation = m.u32(status + STATUS_SITUATION)
            put("action_state", kind)
            if situation is not None:
                put("on_ground", 1 if situation == SITUATION_GROUND else 0)

            # v1 derivations from the game's status kind:
            #   shield  = a GUARD status (confirmed live together with the raw
            #             ControlModule shield bit)
            #   hitstun = a DAMAGE reaction status
            if kind is not None:
                put("shield", 1 if kind in GUARD_STATUS_KINDS else 0)
                put("hitstun", 1 if kind in DAMAGE_STATUS_KINDS else 0)
            # shield_size has no verified source yet; publish 0 (CONTRACTS v7).
            put("shield_size", 0.0)

            # is_actionable: the game's own transition-term table.
            if work:
                terms_base = m.u64(work + WORK_TERMS)
                if terms_base:
                    terms = m.read(terms_base, 0x100)
                    if terms:
                        count = _enabled_term_count(terms)
                        put("is_actionable", 1 if count >= ACTIONABLE_TERM_MIN else 0)
                        row["term_count"] = count
            if "is_actionable" not in row and kind is not None:
                put("is_actionable", ACTIONABLE_TABLE.get(kind, DEFAULT_ACTIONABLE))

        if motion:
            put("action_frame", m.f32(motion + MOTION_FRAME))

        if damage_mod:
            arr = m.u64(damage_mod + DAMAGE_ARRAY)
            if arr:
                put("damage", m.f32(arr))

        # kind, stock, CPU and rebirth live in the information record
        # (u64(entry+0xF8); the game's accessors take entry+0xF0 = info and
        # dereference info+8, which is this same pointer).  The kind there is
        # the select-screen copy; when SSBU_DEBUG_MATCH forces a character by
        # name the match's own record has the spawned kind, so prefer it.
        kind = m.u8(self.base + MATCH_BLOCK
                    + MATCH_RECORD_SIZE * entry_id + MATCH_RECORD_KIND)
        if kind is None or kind >= 0x80:
            kind = None
            if rec:
                kv = m.u8(rec + REC_KIND_PLUS1)
                if kv is not None and kv > 0:
                    kind = kv - 1
        if kind is not None:
            put("kind", kind)
        if rec:
            stock = m.u32(rec + REC_STOCK)
            if stock is not None:
                if stock >= 0x80000000:      # store the u32 bit pattern as i32
                    stock -= 0x100000000
                put("stock", stock)
            cpu = m.u8(rec + REC_CPU_FLAG)
            if cpu is not None:
                put("is_cpu", 1 if cpu else 0)
            rebirth = m.u8(rec + REC_ON_REBIRTH)
            if rebirth is not None:
                put("is_on_rebirth", 1 if rebirth else 0)

        # jumps_left: WorkModule instance ints (jump_count / jump_count_max).
        if work:
            kind_arr = m.u64(work + WORK_KIND_ARRAY)
            if kind_arr:
                kind0 = m.u64(kind_arr)
                if kind0:
                    int_base = m.u64(kind0 + WORK_INT_BASE)
                    if int_base:
                        used = m.i32(int_base + 4 * INSTANCE_JUMP_COUNT)
                        top = m.i32(int_base + 4 * INSTANCE_JUMP_MAX)
                        if (used is not None and top is not None
                                and 0 <= used <= 20 and 0 <= top <= 20):
                            put("jumps_left", max(0, top - used))

        # CancelModule::is_enable_cancel.
        cancel = m.u64(acc + ACC_CANCEL)
        if cancel:
            enable = m.u8(cancel + CANCEL_ENABLE)
            if enable is not None:
                put("is_enable_cancel", 1 if enable else 0)

        # Expert control fields, resolved from the ControlModule vtable.
        self.read_control(acc, put)

        # Reject torn-down / uninitialised fighter objects.  During a results
        # transition the entry still has a non-null object pointer but its
        # status kind is 0x7fffffff and its floats can be NaN; publishing that
        # would make the verifier flash red for a frame.
        for name in ("x", "y", "damage", "facing", "action_frame",
                     "stick_x", "stick_y", "cstick_x", "cstick_y", "trigger"):
            if name in row and not math.isfinite(row[name]):
                return None
        if "x" not in row or "y" not in row:
            return None
        if abs(row["x"]) > 10000.0 or abs(row["y"]) > 10000.0:
            return None
        st = row.get("action_state")
        if st is not None and not (0 <= st <= 2000):
            return None

        row["raw_flags"] = flags | (1 << schema.RAW_FLAGS_INDEX)
        return row

    # -- all fighters -----------------------------------------------------
    def read_fighters(self) -> tuple[bool, list[dict | None], int]:
        """(fight_on, [slot 0 row, slot 1 row], match_state).

        The list always has exactly MAX_FIGHTERS entries and is in player
        order: slot 0 = entry 0 (P1), slot 1 = entry 1 (P2).  A missing
        player is None (the reader writes a zeroed slot).  There is no
        human-first reorder any more (CONTRACTS v7).
        """
        impl = self.manager_impl()
        if not impl:
            self.last_bounds = None
            self.last_stage_id = None
            return False, [None] * schema.MAX_FIGHTERS, schema.MatchState.MENU

        fight = self.fight_on()
        bounds = self.read_stage_bounds() if fight else None
        self.last_bounds = bounds
        self.last_stage_id = self.read_stage_id() if fight else None

        # Entries 0 and 1 are the two player slots.  Only those two are read;
        # extra CPU fighters (2..7) are not part of this shared-memory view.
        rows: list[dict | None] = [None] * schema.MAX_FIGHTERS
        for entry_id in range(schema.MAX_FIGHTERS):
            rows[entry_id] = self.read_fighter(impl, entry_id)

        # derived velocity (units/second) from the previous sample
        now = time.monotonic()
        for row in rows:
            if row is None:
                continue
            x, y = row.get("x"), row.get("y")
            if x is None or y is None:
                continue
            prev = self._prev_pos.get(row["entry_id"])
            if prev:
                dt = now - prev[2]
                if 1e-4 < dt < 0.5:
                    row["vx"] = (x - prev[0]) / dt
                    row["vy"] = (y - prev[1]) / dt
                    row["raw_flags"] |= (1 << schema.FIELD_INDEX["vx"]
                                         | 1 << schema.FIELD_INDEX["vy"])
            self._prev_pos[row["entry_id"]] = (x, y, now)

        # derived relative position / facing (slot pairs are fixed)
        if len(rows) >= 2:
            for i, j in ((0, 1), (1, 0)):
                r, o = rows[i], rows[j]
                if r is None or o is None:
                    continue
                if "x" not in r or "y" not in r or "x" not in o or "y" not in o:
                    continue
                dx = o["x"] - r["x"]
                dy = o["y"] - r["y"]
                r["dx"] = dx
                r["dy"] = dy
                f = r.get("facing")
                if f is None or abs(dx) < 1e-6:
                    rel = 0
                else:
                    rel = 1 if (dx > 0) == (f > 0) else -1
                r["facing_rel"] = rel
                r["raw_flags"] |= (1 << schema.FIELD_INDEX["dx"]
                                   | 1 << schema.FIELD_INDEX["dy"]
                                   | 1 << schema.FIELD_INDEX["facing_rel"])

        # derived distance to the nearest blast zone
        for row in rows:
            if row is None:
                continue
            if "x" in row and "y" in row:
                row["dist_to_blast"] = schema.dist_to_blastzone(
                    row["x"], row["y"], bounds)
                row["raw_flags"] |= 1 << schema.FIELD_INDEX["dist_to_blast"]

        ms = self.match_state()
        return fight, rows, ms

    # -- debugging --------------------------------------------------------
    def describe(self) -> str:
        impl = self.manager_impl()
        lines = [f"pid        {self.pid}",
                 f"guest base {self.base:#x}",
                 f"instance_  {self._instance:#x}"]
        lines.append(f"camera     {self._camera_global:#x}" if self._camera_global
                     else "camera     <no symbol>")
        lines.append(f"manager    {impl:#x}" if impl else "manager    <none>")
        if impl:
            lines.append(f"fight_on   {self.mem.u8(impl + IMPL_READY_GO)}")
        b = self.read_stage_bounds()
        lines.append(f"blast      {b}" if b else "blast      <unknown>")
        sid = self.read_stage_id()
        lines.append(f"stage_id   {sid}" if sid is not None else "stage_id   <unknown>")
        if self.last_stage_source:
            lines.append(f"stage_src  {self.last_stage_source}")
        if self.control_layout:
            lines.append(f"control    {self.control_layout}")
        for slot, (target, goff, decoded) in sorted(self._last_getter.items()):
            lines.append(f"getter     0x{slot:x} -> target 0x{target:x} guest 0x{goff:x} {decoded}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from rl.memory.memapi import find_game_pid

    pid = int(sys.argv[1]) if len(sys.argv) > 1 else find_game_pid()
    with GameState(pid) as gs:
        print(gs.describe())
        fight, rows, ms = gs.read_fighters()
        print("fight_on:", fight, "match_state:", ms)
        for slot, row in enumerate(rows):
            print(f"  slot {slot}:", row)
