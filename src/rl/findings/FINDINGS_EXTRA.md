# Agent F — extra memory archaeology

Owner: F. Files: `tools/mem/extra_state.py`, `tools/mem/extra_map.json`,
`tools/mem/FINDINGS_EXTRA.md`, `tools/mem/test_extra.py`.

Goal: read as much additional live state as possible without touching the
policy observation, the core reader/bridge, or the runtime.  Output is a
separate, self-describing block: `/dev/shm/ssbu_extra.bin` (`SSBX`).

## What is published

One snapshot is a list of self-describing records.  Each record has a 48-byte
name, a type tag (`i64/f64/bytes/bool/u32/vec3f/vec4f/str`), an `index`
(0 global, 1 = P1/entry 0, 2 = P2/entry 1, 3+ = battle-object slot), a
`verified` byte, and the value.  The header carries magic/version/seq/ts/count/
total_bytes/flags and uses the same odd/even sequence protocol as the core
block.  The exact layout is in `extra_map.json` → `format`.

Groups:

- `g.*` — match/rule/stage/RNG/camera/BattleObjectManager globals.
- `f.info.*` — FighterInformation: kind, stock, dead count, rebirth,
  category, color, CPU flag, gravity, hit-point max, damage-log pointers.
- `f.shield.*` — ShieldModule at `accessor+0x100`: part array, part count,
  part size/pos/LR/hop/attack/speed/limit/life multipliers, bubble node size,
  raw part and node blobs, center position, and a `likely` shielding proxy.
- `f.damage.*` — DamageModule at `accessor+0xA8`: damage/power/reaction from
  the live hit record, damage-lock/paralyze/critical/capture-cut/no-reaction.
- `f.stop.*` — StopModule at `accessor+0x90`: hit-stop state and the engine's
  real hitlag frame count (`get_hit_stop_real_frame`).
- `f.ground.*` — GroundModule at `accessor+0x58`: offsets, touch/attach/
  cliff flags, shape, z, center/down contact points.
- `f.kinetic.*` — KineticModule at `accessor+0x68`: per-entry objects, flag
  masks, enabled bits, engine velocity vectors, and the replicated sum speed.
- `f.motion.*` — MotionModule at `accessor+0x88`: frame, motion kind, loop/
  end/blend/changing/flip flags, rate, previous/end frame.
- `f.work.*` — WorkModule at `accessor+0x50`: the full 256-byte transition
  term table, an enabled-term bitmask and count, the work flag array (command/
  cancel flags), and the instance int/float arrays. `f.cancel.enable` is the
  CancelModule flag.
- `bom.obj.*` — BattleObjectManager pools: object id, pool kind, raw entry,
  accessor/position/status when the entry maps.

## Method (no guessed offsets)

Every address is resolved from the binary and/or the live object graph:

1. Guest image base from `/proc/<pid>/maps` (`memapi.py`).
2. Symbol offsets from `main.elf` `.dynsym` (`elfsym.py`).
3. Module pointers from the `lua_bind *_impl` thunks: the first instruction is
   `ldr x0,[x0,#N]`, so the accessor offset is read out of the game's wrapper.
   This is how `StopModule=0x90` and `HitModule=0xB0` were added to the
   existing table.
4. The module's first qword is its vtable; the thunk's next instruction loads
   `vtable[slot]`.  Live vtables were dumped (e.g. Shield
   `0x7104f71a90`) and the concrete targets disassembled with
   `aarch64-linux-gnu-objdump`.
5. `extra_state.decode_getter()` repeats step 4 at run time: it reads the
   concrete getter's machine code and decodes the load pattern (`ldrb`, `ldr`,
   `ldrsh`, `ldr s`, `ldr q`, `ldr x0,[x0,#imm]`, `add x0,x0,#imm`).  It only
   accepts a load whose next instruction is `ret`, so complex array walkers and
   virtual dispatch are rejected instead of misread.  Successful decodes set
   `verified=1`; the map records the static disassembly as well.

## Verified offsets (static evidence)

All offsets below were read from the game's own accessors; the exact
instructions are in `extra_map.json` → `evidence`.

### ShieldModule (`*(accessor+0x100)`)

- `is_shield` = vtable slot 17; concrete target `0x4c5de0`:
  `ldrb w8,[x0,#127]` (no-shield flag), then part array at `module+0x18`
  (stride `0x34`), node array at `module+0x28` (stride `0x50`), node `+0x40`
  marks an active shielding part.
- `get_part_size` = slot 41, target `0x4c6190`:
  `ldr x8,[x0,#24]; smaddl x8,w1,0x34,x8; ldrsh w0,[x8,#6]`
  → **part size = i16 at part+0x06**.
- `get_lr` slot 52 → f32 part+0x10; `get_pos_x` slot 54 → f32 part+0x08;
  `get_hop_angle` slot 24 → f32 part+0x14; `get_attack_mul` slot 30 →
  f32 part+0x18; `get_speed_mul` slot 32 → f32 part+0x1C; `get_life_mul`
  slot 34 → f32 part+0x24; `get_attack_limit` slot 36 multiplies part+0x20 by
  `module+0x80`.
- `set_size` slot 18, target `0x4c5e20`:
  `*(module+0x28) + 0x50*(part[+4]+arg)` and `str s0,[node,#0x20]`
  → **bubble node size = f32 at node+0x20**.
- `get_center_pos` slot 51: slot node =
  `*(module+0x50) + 0x2D0*part[0].i32`; `node+0x30` is a position object whose
  vtable[4] returns the vector.
- `set_hit_stop_mul` slot 38 writes `module+0x84`; `get_group_num` slot 53
  returns `module+0x10`.

There is **no lua_bind getter for an absolute shield HP**.  The full Shield
surface (39 functions) was enumerated: size/life-mul/attack/speed/limit/pos/
LR/hop/status only.  What exists is published raw (part `0x34` bytes, node
`0x50` bytes) so HP can be correlated from a shielding frame later.

### DamageModule (`*(accessor+0xA8)`)

- hit record array = `*(module+0x18)`, stride `0x1C0`.
  `damage` slot 25 target `0x3ff9f0` → f32 `+0x00`;
  `reaction` slot 26 → f32 `+0x10` (hitstun/reaction length);
  `power_max` slot 27 → f32 `+0x0C`.
- `is_no_reaction_mode_perfect` slot 38 target `0x3ffcd0` → byte `module+0x45`;
  `check_no_reaction` also reads `module+0x44`.
- `is_damage_lock` slot 75, `is_paralyze` slot 85, `is_critical_hit` slot 91,
  `is_capture_cut` slot 59.  These are beyond the 64-entry prefix dumped in
  the first live probe; the run-time decoder resolves them (or the field is
  omitted).

### StopModule (`*(accessor+0x90)`) — the engine hitlag

- `is_stop` slot 17, `is_hit` slot 19, `is_damage` slot 20,
  `get_hit_stop_real_frame` slot 25.  The last one is the engine's real
  hit-stop/hitlag frame count, not a multiplier.

### GroundModule (`*(accessor+0x58)`)

- `offset_x` slot 38 target `0x45fdf0` → f32 `module+0x50`;
  `offset_y` slot 39 → f32 `module+0x54`.
- slots: `is_status_cliff` 72, `can_entry_cliff` 76, `shape_kind` 111,
  `get_z` 113, `is_attach` 121, `touch_flag` 141, `is_touch` 148,
  `center_pos` 180, `down_pos` 176, `is_attach_cliff` 226.
  The decoder reads the concrete field offset for each at run time.

### KineticModule (`*(accessor+0x68)`) — engine velocity

`get_sum_speed` slot 24, target `0x47a760`:

```
count = *(module+0x20)
entries = module+0x30, stride 0x18
entry+0x00 = kinetic object, entry+0x0C = flag mask
object+0x30 = enabled byte
object vtable[4] returns &velocity (normally add x0,x0,#0x30 ; ret)
sum = Σ velocity over entries whose (flags & type_mask) != 0 and enabled
get_sum_speed_x/y take components 0/1 of that sum.
```

`extra_state.py` replicates this read-only (mask = all) and publishes
`f.kinetic.engine_vel` plus each entry's vector.  This is the engine's own
velocity, unlike the core reader's position-delta estimate.

### MotionModule (`*(accessor+0x88)`)

`frame` = f32 `module+0x70` (getter `0x49bf50`).  Slots: `motion_kind` 39,
`is_blend` 37, `is_changing` 38, `rate` 40, `prev_frame` 45, `end_frame` 48,
`is_end` 57, `is_looped` 58, `is_loop_flag` 63, `is_flip` 114.

### WorkModule (`*(accessor+0x50)`)

- terms = `*(module+0x28)`, 256 bytes.  Enabled test exactly as
  `is_enable_transition_term`: `0` if `b&0x38`, else `(b>>2)&1` if `b&2`,
  else `b&1`.  Published as the raw bytes plus a 256-bit mask/count.
- kind array = `module+0x18` (16 pointers).  `kind0+0x20` = int array
  (B's jumps), `kind0+0x10` = float array, `kind0+0x30` = u32 flag array.
- `is_flag` target `0x4e4900` proves the flag id encoding:
  kind = `(id>>24)&0xF`, word = `(id>>5)&0x7FFFF`, bit = `id&0x1F`.
  The flag array (command/cancel flags) is published raw.

### FighterInformation

`entry+0xF0` is the FighterInformation object; `entry+0xF8` is its data
record (`*(object+8)`), proven by `FighterManager__get_fighter_information`
(`... add x0,x8,#0xF0`).  Offsets from the lua_bind getters:

- stock `+0xD8` u32; dead count `+0xCC/+0xD0/+0xD4`;
- color `+0x51`; is_operation_cpu `+0x92`; last-dead-suicide `+0x1A3`;
- gravity `+0x36C`; hit-point max `+0x38`/`+0x3C`;
- on-rebirth `+0x38C`; category `+0x3AC`;
- damage log: `record + 0x11B0*type + 8*index + 0x15E0` → pointer to a u32
  array.  Published raw; the log's field meaning (last hitter) is not decoded.

### BattleObjectManager

Singleton `_ZN3lib9SingletonIN3app19BattleObjectManagerEE9instance_E`
(`0x52b6a00`) exists, and `is_active_find_battle_object_impl` resolves through
a second global `0x52b7548`:

```
manager = *(0x52b7548)
pools   = manager + 8*k, k=0..4
pool+0x18 = count (u32), pool+8 = array
strides = 0xF940, 0x3C30, 0x60C0, 0x300, 0x2B0
slot+8 = u32 object id; slot+0x3A = byte tested by is_active
```

Enumeration is a read-only pointer walk.  The rest of the entry layout is not
fully decoded; the raw `0x60` bytes and the accessor (slot+0x20, the same
offset as FighterEntry's fighter object) are published only when they map and
sanity-check.

### Stage / match / camera / RNG

- StageManager (`0x53299d8`): cell → object → impl; `impl+0x17C` stage id;
  `impl+0x110+8*i` stage object pointers (proven by
  `is_discretion_final_enabled`).
- Match block (`0x52c41b0`, from `runtime/debug_mode.c`):
  `+0` melee mode, `+5` rule kind, `+0xA` stock, `+0x54` item appear,
  `+0x420` stage id, players at `+0x588`.  Published raw + fingerprint.
- RNG: `0x52381c0`, 16 bytes / 4 u32 (the rollback seed generator).
- Camera: `*(BossManager::instance_+8) → a → manager`;
  `+0xC30` dead area (`app::camera::get_dead_area`), `+0xC20` range,
  `+0xD20` target, `+0xD30` pos, `+0xEB0/+0xEC0` internal target/pos,
  `+0xDAC` fov (`sv_camera_manager::get_target/get_pos/get_fov`).

## What is still unknown

- **Absolute shield HP.**  No accessor exists; raw part/node blobs are
  published for later correlation.
- **Numeric cliff id.**  `can_entry_cliff` / `is_attach_cliff` are decoded;
  `can_entry_cliff_hang_data` returns a hang-data struct that is not decoded.
- **True distance to floor.**  `f.ground.floor_y` is the center-point y; a
  true contact distance needs `get_touch_pos` with a touch id.
- **Battle-object owner/category/entry id.**  Pools and ids are proven; the
  rest of the object layout is partial (raw blob published).
- **Match timer, platform geometry, guest frame counter.**  Not decoded;
  camera bounds/range/pos/target are the geometry proxy.  (The core block's
  `game_frame` is the reader's own counter.)
- **HitModule invincibility frames.**  Only setters (`set_invincible_frame_global`)
  exist; no getter.  A raw HitModule blob is published.

## Tests

`tools/mem/test_extra.py` (no lock, no game required):

- format round-trip for every type, name truncation, 8-byte record alignment,
  bad magic/version/truncation rejection, odd/even seq invariant;
- the getter decoder against real instruction sequences (direct u8/u32/f32,
  deref i16 for `get_part_size`, pointer add for the position getter, and two
  negative cases for complex getters);
- a live probe that runs only when a match is actually live (`fight_on==1`),
  otherwise it skips.  `SSBU_EXTRA_LIVE=1` turns a missing match into a
  failure so a CI run can require it.

Run: `python3 tools/mem/test_extra.py -v`
Live: `python3 tools/mem/game_session.py run --owner F --no-launch --no-reader -- python3 tools/mem/extra_state.py --no-shm --print`

## Live observations

(filled by the final live run; see the section below)
