# SSBU live-state bridge (`tools/mem`)

> **Repository note (2026-11-03).** This repository is the `ssbu-rl` package
> (`src/rl/`). The recompiled game itself is the separate upstream project
> [OpenSmash-Ultimate/ssbu-recomp](https://github.com/OpenSmash-Ultimate/ssbu-recomp)
> and is **not vendored here**. This toolkit needs one small runtime change in
> that tree — the R2 direct-reset hook and debug-match forcing in
> `runtime/debug_mode.c` — kept as
> [`patches/ssbu-recomp-rl.patch`](patches/ssbu-recomp-rl.patch).
> All credit for the recomp belongs to the upstream authors.
>
> The sections below predate the move out of `ssbu-recomp/tools/mem/` and still
> use `tools/mem/...` paths; see [`MIGRATION.md`](MIGRATION.md) for the current
> layout and commands.

## Upstream runtime change

The only runtime file this toolkit requires is `runtime/debug_mode.c`. Apply
the patch in a checkout of `ssbu-recomp`:

```bash
git -C /path/to/ssbu-recomp apply /path/to/ssbu-rl/patches/ssbu-recomp-rl.patch
```

It is off by default and activates only with `SSBU_RL=1`. It adds:

* the R2 direct-reset control block (`/dev/shm/ssbu_control.bin`, `SSBC` v1);
* `SSBU_DEBUG_MATCH` forcing for `p1=cpu|human`, `p2=...`, `players=N`,
  `cpu=N`, `stage=end`, and the fighter/stage selection used by the recorder.

No other runtime file is needed. The other local modifications in the
`ssbu-recomp` working tree are unrelated to this repository.

---

Reads Super Smash Bros. Ultimate's live fighter state out of the running
recompiled game **from another process**, publishes it to POSIX shared memory at
60 fps, and shows it in a second window so a human can confirm it matches the
game.

The game is never modified and no code is injected. Everything is read through
`/proc/<pid>/mem`.

```
tools/mem/
  elfsym.py      minimal ELF64 .dynsym reader (symbol offsets from main.elf)
  memapi.py      /proc/<pid>/mem access, maps parsing, base discovery
  gamestate.py   the object-graph walk that turns raw memory into fields
  schema.py      the shared-memory layout (single source of truth)
  probe.py       one-shot CLI probe
  reader.py      60 fps daemon -> /dev/shm/ssbu_state.bin
  verify_gui.py  second-window verifier (Tkinter)
  live.sh        one command: game + reader + verifier + focus keeper
  focus_game.sh  give the game window the X11 keyboard focus
  memmap.json    every discovered symbol, offset, type, verified flag, evidence
  seq_long_fight.txt  input sequence used during early probing (not needed to play)
```

---

## Quick start

```bash
cd /home/daniel/Projects/SSBU/ssbu-recomp

# one command: start the game, the 60 fps reader, the verifier HUD,
# and keep the keyboard focused on the game
tools/mem/live.sh

# keep an already-running game, just (re)start the reader + HUD
tools/mem/live.sh --attach

# if focus ever lands on the desktop, give it back to the game
tools/mem/focus_game.sh
```

The game is started with `SSBU_SKIP_MOVIES=1`, `SSBU_KEYBOARD_PLAYER=1` and
`SSBU_DEBUG_MATCH=p1=mario,p2=donkey,cpu=0` (a passive opponent, so there is
time to try moves and watch the state; use `cpu=1` or higher for a fight).
Override the match with `--match '...'` or the `SSBU_LIVE_MATCH` environment
variable. Logs: `~/smash.log`, `~/ssbu_reader.log`, `~/ssbu_verifier.log`.

> **One-time game setup (important).** The debug harness reuses the *saved*
> select-screen roster, and this save has player 1 as a **CPU** and a **0-stock**
> rule. That is why the keyboard appeared dead and matches ended after one KO.
> On the game's own character-select screen set the first panel to **Human**,
> player 2 to CPU (or remove it), and set the rules to **3 stocks**. The game
> remembers this. `SSBU_DEBUG_MATCH` can force the second panel's type
> (`p2=human`) but its fighter-name selection does not apply on this build, so
> the saved roster is authoritative.

> **Focus.** `xdotool windowfocus` alone can leave SDL's
> `SDL_GetKeyboardFocus()` NULL, which makes `GetForegroundWindow()` return
> NULL and the game ignore every key. `live.sh` therefore also sends one real
> click inside the game window. If you focus the game by hand and keys do
> nothing, click it once.

To run the pieces by hand:

```bash
DISPLAY=:0 ./play.sh --env SSBU_SKIP_MOVIES=1 --env SSBU_KEYBOARD_PLAYER=1
python3 tools/mem/reader.py        # publishes to /dev/shm/ssbu_state.bin
python3 tools/mem/verify_gui.py    # second window
python3 tools/mem/probe.py         # one frame, no shared memory
python3 tools/mem/probe.py --frames 20 --interval 0.1
python3 tools/mem/probe.py --json
```

The verifier window: `H` toggles the raw hex bytes of each row, `P` pauses
refresh (for screenshots), `Q` quits. It is **topmost** but never calls
`focus_set`, so starting it does not take the keyboard away from the game.

### Controls (default keyboard)

`J`/`L` = left/right, `I`/`K` = up/down, `Z` = A (jump), `X` = B, `C` = X,
`V` = Y, `Q`/`E` = L/R, `A`/`S` = ZL/ZR, `Enter` = +, `Backspace` = −.

The game only reads the keyboard while its **SDL window** has the X11 input
focus *and* SDL's own keyboard-focus flag. Under this mutter session
`xdotool windowactivate` (EWMH) is unreliable and `xdotool windowfocus`
(XSetInputFocus) alone can leave SDL's flag unset. `live.sh` raises, activates
and focuses the window, then sends one click inside the client area (which is
what makes SDL take keyboard focus) and starts a background keeper that
re-focuses the game whenever focus is on neither the game nor the HUD.
`focus_game.sh` does the same by hand.

---

## Permissions / `ptrace_scope`

`/proc/<pid>/mem` is guarded by the same check as `ptrace`. With the default
Yama setting (`/proc/sys/kernel/yama/ptrace_scope = 1`) a process may only read
the memory of **its own descendants**.

You have two options, and only one of them needs root:

**A. Let `reader.py` start the game (no root, recommended for a standalone
reader)**

```bash
python3 tools/mem/reader.py --spawn
```

`--spawn` runs `play.sh` as a child of the reader, so the game is a descendant
and the default policy allows the reads. The game log goes to
`~/smash_reader.log`.

**B. Attach to a game you started yourself (needs a sysctl)**

```bash
sudo sysctl -w kernel.yama.ptrace_scope=0
python3 tools/mem/reader.py
```

This lowers a system-wide security setting; it is not persistent and reverts on
reboot. `live.sh` uses this attach path, so on a fresh boot either run the
sysctl once or start the game with `reader.py --spawn`.

If neither is done, `reader.py` and `probe.py` exit immediately with:

```
error: cannot read pid NNNN: permission denied (ptrace_scope=1). Either start
the game as a child of reader.py (--spawn), or run 'sudo sysctl -w
kernel.yama.ptrace_scope=0'.
```

No root is needed to run the reader, the GUI, or the probe once access is
granted.

---

## How the addresses are found

Nothing is hardcoded.

* **Image base** — `/proc/<pid>/maps` is parsed and the recompiled guest image
  is identified as the anonymous, read-only, offset-0 region (it is the only
  one). On this build that is `0x7100000000`.
  Beware: the guest image lives at `0x71_00000000` (10 hex digits) while heap
  allocations live at `0x7100_xxxxxxxx` (12 digits); a careless range check
  conflates them.
* **Symbol offsets** — parsed straight out of `main.elf`'s `.dynsym` by
  `elfsym.py`. `nm ssbu_native` is useless here because the fighter state is in
  the *guest* image, not the host binary.
* **Struct offsets** — obtained by disassembling the game's own `lua_bind`
  accessors with `aarch64-linux-gnu-objdump`. Each accessor is a two- to
  ten-instruction function that loads one field, so the offset is exact. The
  disassembly for every field is recorded in `memmap.json` under `evidence`.

### The chain

```
instance_ (dynsym) -> mgr -> impl            FighterManager
impl + 0x20 + 8*entry_id  -> FighterEntry
entry + 0xF0              -> FighterInformation (embedded; vtable at +0)
info + 8 (= entry + 0xF8) -> information record (kind, CPU, stock, rebirth)
entry + 0x4160            -> fighter object
obj + 0x20                -> BattleObjectModuleAccessor

accessor + 0x38  -> PostureModule   pos    = module + 0x30   (x, y, z)
                                    facing = module + 0xAC
accessor + 0x40  -> StatusModule    action_state   = module + 0x98
                                    situation_kind = module + 0xDC
accessor + 0x48  -> ControlModule   input = *(module + 0x110)
                                    buttons = input + 0x7C
                                    stick   = input + 0x10 / 0x14
                                    C-stick = input + 0x40 / 0x44
                                    trigger = input + 0x88
accessor + 0x50  -> WorkModule      ints           = [[module+0x18][0] + 0x20]
                                    terms          = module + 0x28
accessor + 0x88  -> MotionModule    action_frame   = module + 0x70
accessor + 0xA8  -> DamageModule    damage = *(module + 0x18) + 0x1C0*index
accessor + 0x128 -> CancelModule    is_enable_cancel = module + 0x1C
```

`impl + 0xD2` is the game's own ready-go flag: 1 while a fight is running.

### ControlModule — resolved from the live vtable

The `ControlModule` getters are virtual. The lua_bind wrappers do
`ldr x0,[accessor,#0x48]`, load the vtable and branch to a fixed slot:

| field | slot | concrete loader (13.0.4) |
|---|---|---|
| `buttons` | `+0x238` | `ldr w0,[x8,#0x7c]` |
| `stick_x` | `+0x178` | `ldr s0,[x9,#0x10]`, then `fneg`/`fcsel` on `module+0x120` |
| `stick_y` | `+0x188` | `ldr s0,[x8,#0x14]` |
| `cstick_x` | `+0x200` | `ldr s0,[x9,#0x40]`, same sign flip |
| `cstick_y` | `+0x210` | `ldr s0,[x8,#0x44]` |
| `trigger` | `+0x220` | `ldr w0,[x8,#0x88]` |

`gamestate.py` reads the vtable target, reads the tiny loader and decodes its
`ldr` immediates at run time; the constants above are only the fallback for a
torn-down module. The raw button bitmask is decoded into the canonical order
`A B X Y L R ZL ZR PLUS MINUS DPAD_L DPAD_R` with
`A=0x1 B=0x2 X=0x4 Y=0x8 L=0x20 R=0x400 ZL=0x1000 ZR=0x2000
PLUS=0x4000 MINUS=0x10 DPAD_L=0x40 DPAD_R=0x80`. That mapping is the game's
own `check_button_trigger` table (guest `0x4538fec`), not a guess.

### Stage id

```
primary:  stage_id = u32(*(*(guest_base + dynsym(StageManager::instance_))) + 0x17C)
fallback: stage_id = u32(guest_base + 0x52c41b0 + 0x420)
```

The primary chain is exactly `app::sv_information::stage_id`; `0xffffffff`
means "no stage". On this build that field is 0 during a fight (it is the
select-screen information copy), so the reader falls back to the global match
block's in-match stage id — the same value the debug hook prints at match start
(`runtime/debug_mode.c`: `GA(0x71052c41b0)`, stage at `+0x420`). With
`stage=end` the hook prints `stage id 0x3 ... (end)`, so **Final Destination is
id 3 in this build**. `GameState.last_stage_source` says which chain was used.

### `is_actionable` — the game's own transition terms

The WorkModule keeps one byte per status transition term at `work + 0x28`.
The game's `is_enable_transition_term(term)` reads that byte and returns
`0 if (b & 0x38) else ((b >> 2) & 1) if (b & 2) else (b & 1)`. Counting the
enabled terms over the whole 256-byte table is a direct measure of "can this
fighter act":

| state | enabled terms |
|---|---|
| WAIT | ~49 |
| dash | ~44 |
| jump | ~32 |
| cliff catch | ~12 |
| hitstun | 2 |
| tumble / KO | 0 |

The reader publishes `1` when the count is `>= 4`. This replaced the earlier
empirical `ACTIONABLE_TABLE` (kept only as a fallback for the rare frame where
the term table cannot be read).

### `jumps_left`

The WorkModule instance ints are at `*(*(*(work + 0x18)) + 0x20)`, stride 4.
Index **6** is `jump_count_max`, index **4** is `jump_count`, so:

```
jumps_left = max(0, int[6] - int[4])
```

Live: Mario `2` on the ground, `1` after one jump / while on a cliff; DK `1`.

### Blast zones — live camera, not a constant

```
g       = guest_base + dynsym(BossManager::instance_) + 8
a       = u64(g)
manager = u64(a)
bounds  = f32x4 at manager + 0xC30   # left, right, top, bottom
```

This is exactly the chain `app::camera::get_dead_area` and
`sv_camera_manager::dead_range` read. Live values observed:
`(-240,240,180,-140)`, `(-280,280,180,-140)`, `(-250,170,185,-65)`.
`dist_to_blast = max(0, min(x-left, right-x, y-bottom, top-y))`.

---

## Field status (v7)

| field | source | verification |
|---|---|---|
| `entry_id` | player slot | slots 0/1 are P1/P2, fixed order |
| `kind` | `u8(record+0x30) - 1` | live: slot 0 = 0 (Mario), slot 1 = 1 (DK) |
| `stock` | `u32(record+0xD8)` | accessor proves width; 0 under the 0-stock debug rule |
| `damage` | `*(DamageModule+0x18)` | memory `36.347` vs HUD **36.3 %** |
| `x`, `y` | `PostureModule+0x30/+0x34` | `y=0.01` standing, `y≈31` mid-jump, `y≈137` on the revival platform, `y≈-145` falling to a KO |
| `facing` | `PostureModule+0xAC` | flips `+1`/`-1` exactly when the character turns |
| `action_state` | `StatusModule+0x98` | 0 wait, 1 walk, 6 dash, 11 jump squat, 12 jump, 18 fall, 54 damage, 73 cliff catch, 263 KO, 476 tumble |
| `action_frame` | `MotionModule+0x70` | increments within an action, resets on status change |
| `on_ground` | `StatusModule+0xDC == 0` | 0 on the floor, 2 on the revival platform, 5 while falling |
| `is_actionable` | enabled transition terms `>= 4` | WAIT 49, dash 44, cliff 12, hitstun 2, tumble 0 |
| `jumps_left` | `max(0, int[6] - int[4])` | Mario 2 on the ground, 1 after a jump / on a cliff; DK 1 |
| `vx`, `vy` | position delta / dt | derived at 60 Hz (the engine getter needs a guest call) |
| `dist_to_blast` | live camera bounds | tracks the stage camera |
| `dx`, `dy` | `opponent - self` | derived from verified `x`,`y` |
| `facing_rel` | `sign(dx) == sign(facing)` | +1 in front, -1 behind, 0 unknown |
| `is_cpu` | `u8(record+0x92)` | 1 for both debug CPUs; no longer used for reordering |
| `is_on_rebirth` | `u8(record+0x38C)` | proven by `FighterInformation__is_on_rebirth` |
| `shield` | derived: GUARD status kind | v1; live shield toggles at status kind **10** |
| `shield_size` | no source yet | always `0.0` (contract) |
| `is_enable_cancel` | `u8(CancelModule+0x1C)` | proven by `CancelModule__is_enable_cancel` |
| `hitstun` | derived: DAMAGE reaction kinds | 1 in states 54..62/476..479/505..511 |
| `buttons` | ControlModule input+0x7C | raw mask, decoded into 12 canonical buttons |
| `stick_x/y`, `cstick_x/y` | ControlModule getters | [-1, 1], sign flip from module+0x120 |
| `trigger` | ControlModule getter | [0, 1] |
| `stage_id` | header; StageManager+0x17C, fallback match block+0x420 | live 3 (Final Destination) |
| `game_frame` | reader sample counter | not the game's own frame tick yet |
| `match_state` | `ready_go` + entry allocation | 0=MENU, 1=IN_MATCH, 2=RESULTS |

`probe.py` and the GUI show a field in **red** if it is missing, implausible,
or older than 500 ms; **yellow** if older than 100 ms; **green** otherwise.

---

## Observation vector (shared memory, v7)

```
[ 0] entry_id        int32    [ 8] action_frame  float32  [16] dy             float32  [24] buttons        uint32
[ 1] kind            int32    [ 9] on_ground     int32    [17] facing_rel     int32    [25] stick_x        float32
[ 2] stock           int32    [10] is_actionable  int32    [18] is_cpu         int32    [26] stick_y        float32
[ 3] damage          float32  [11] jumps_left    int32    [19] is_on_rebirth  int32    [27] trigger        float32
[ 4] x               float32  [12] vx            float32  [20] shield         int32    [28] cstick_x       float32
[ 5] y               float32  [13] vy            float32  [21] shield_size    float32  [29] cstick_y       float32
[ 6] facing          float32  [14] dist_to_blast float32  [22] is_enable_cancel int32   [30] raw_flags      uint32
[ 7] action_state    int32    [15] dx            float32  [23] hitstun        int32    [31] pad            uint32
```

The block holds two slots in **player order**: slot 0 = entry 0 (P1), slot 1 =
entry 1 (P2). There is no human-first reorder. An absent player is an all-zero
slot with `raw_flags = 0`. `raw_flags` bit *i* means field *i* was readable
this frame (including the control fields 24–29).

---

## Shared memory layout

`/dev/shm/ssbu_state.bin`, **336 bytes**, defined in `schema.py`:

* 80-byte header — magic `SSBU`, version 7, fighter count, `seq`, monotonic
  timestamp (ns), measured reader fps, reader pid, fight flag, match_state,
  the live blast bounds (left, right, top, bottom as `float32`), `stage_id`,
  `game_frame`, a header `flags` word and three padding words.
* 2 fighter slots × 128 bytes — 32 `int32`/`uint32`/`float32` fields
  (CONTRACTS.md order). `raw_flags` is field 30; field 31 is padding.

The header `flags` word currently carries:

* bit 0 — `stage_id` was read this frame;
* bit 1 — the ControlModule virtual getters were resolved.

Single writer, many readers, no locks: the writer sets `seq` to an **odd**
value, writes the body, then sets it to the next **even** value. A reader that
sees an odd `seq` (or a changed `seq` across its read) retries, so a torn frame
is detectable and never silently wrong.

Readers mmap the block once; reading a frame is a `struct.unpack_from` over the
mapped bytes — zero copies.

---

## Measured performance

```
stopped: 600 frames in 10.00s = 60.00 fps
```

Requirement was ≥ 55 fps. Each frame is ~14 `pread` calls per fighter, well
inside the budget; the loop schedules against an absolute monotonic deadline so
it does not drift.

---

## Known issues

* The game only exposes fighter state while a match is loaded. Before the
  ready-go flag goes to 1, `read_fighters()` returns `([None, None],
  fight_on=False)` and both slots are zeroed.
* `SSBU_DEBUG_MATCH` applies the **saved rules and saved roster**. This save
  has player 1 as a CPU and a 0-stock rule, so (a) the keyboard controls
  nobody on the first panel and (b) a KO ends the match. `stock=` is only
  applied when the panel state's rule kind is already 1, which it is not on
  this save. On this build the debug harness's fighter-name selection
  (`p1=donkey`) does not reach the spawned fighters either; only `p2=human`
  (the panel type) takes effect. Set player 1 to Human, player 2 to CPU and
  the rule to 3 stocks once in the game's own character-select screen; the
  save then makes every debug match playable. Slots are always P1/P2 in
  player order; `is_cpu` says who is who.
* The keyboard only reaches the game while the SDL window has both X11 focus
  and SDL's own keyboard focus. A real click inside the window is required;
  `xdotool windowfocus` alone can leave SDL's flag NULL. `tools/mem/live.sh`
  and `tools/mem/focus_game.sh` do the click for you.
* If the host GNOME session locks (15 min idle by default), the game keeps
  running but the screen shows the lock screen and no input reaches it.
  `live.sh` unlocks the active X11 session with `loginctl unlock-session`
  (no root needed); set `SSBU_LIVE_UNLOCK=0` to skip that.
* The first time a new fighter is seen the game may show a tutorial popup
  ("Ryu: Canceling Heavy Ankle Kick…"). Press `Z` (the A button) to dismiss it;
  until it is dismissed the match sits on the results/tutorial screen.
* `ACTION_STATE_NAMES` is the common Smash status-kind enum. The values seen in
  play (0,1,6,11,12,18,54,73,263,476) are confirmed; character-specific
  specials may be named a little off. The game's 485 `FIGHTER_STATUS_KIND_*`
  strings are referenced only by PC-relative code, so there is no data table to
  read the exact mapping from.
* `stock` is read from the game's own record as a 32-bit word (the game's own
  accessor loads `w0`), but only a 0-stock rule has been exercised; a stock
  match would confirm the unit.
* `shield` and `hitstun` are **derived** from the status kind (GUARD / DAMAGE
  reactions) in v7. `shield_size` has no verified source yet and is always
  `0.0`.
* The expert control fields come from the ControlModule input struct. They are
  the *internal* mask, not the raw SDL/HID buttons; `schema.BUTTON_BITS` is the
  game's own id→mask table. The C-stick is the game's "sub stick".
* `game_frame` is the reader's monotonic sample counter, not the game's own
  frame tick (no verified game-frame source yet).
* `vx`/`vy` are derived from position deltas, not read from the engine.
* This Python has no PyQt6/PySide6, so `verify_gui.py` uses Tkinter.
* On this GNOME/mutter session, `xwd`/`ffmpeg` screenshots of a **Tk** window
  come out black in the client area (the title bar is captured). That is a
  compositor capture limitation, not a rendering bug: the widget tree was
  verified headlessly by instantiating `App`, calling `refresh()` and reading
  every label's text and colour.

---

## Verified commands used during discovery

```bash
# symbol offsets
python3 tools/mem/elfsym.py build-linux/binaries/13.0.4/main.elf

# disassemble an accessor (guest image is AArch64)
aarch64-linux-gnu-objdump -d --start-address=0x206ae10 --stop-address=0x206ae40 \
    build-linux/binaries/13.0.4/main.elf

# live match to probe against
./play.sh --env SSBU_SKIP_MOVIES=1 \
    --env "SSBU_DEBUG_MATCH=p1=mario,p2=donkey,stage=end,cpu=1"

# focus + input check (the click round-trip is required: XSetInputFocus alone
# leaves SDL's keyboard focus NULL)
tools/mem/focus_game.sh
DISPLAY=:0 xdotool key z
```

---

## Run summary

* **Verified (v7):** `x`, `y`, `damage`, `facing`, `action_state`,
  `action_frame`, `on_ground`, `is_actionable`, `jumps_left`,
  `dist_to_blast`, `dx`, `dy`, `facing_rel`, `kind`, `is_cpu`,
  `is_on_rebirth`, `is_enable_cancel`, `shield`, `hitstun`, `buttons`,
  `stick_x/y`, `cstick_x/y`, `trigger`, `stage_id`, `match_state`;
  `vx`/`vy` derived.
* **Placeholder:** `shield_size` (0.0), `game_frame` (reader counter).
* **Partial:** `stock` (accessor/width proven, 0-stock rule live only).
* **Sustained fps:** 60.00 over 10 s (600 frames), requirement ≥ 55 — pass.
* **Input:** the click round-trip in `live.sh` is what makes SDL take keyboard
  focus; with it, `Z` advanced the game's title screen and started a match.
  `xdotool windowfocus` alone does not (SDL's keyboard focus stays NULL). The
  remaining blocker to *playing as Mario* is the saved roster: player 1 is a
  CPU and the rules are 0-stock, so change those once in the game's own
  character-select screen (see "Known issues").
* **Permissions:** no root needed when the game is started by
  `reader.py --spawn`; attaching to an already-running game needs
  `sudo sysctl -w kernel.yama.ptrace_scope=0` (documented above).
* **Commands:**

  ```bash
  tools/mem/live.sh                        # game + reader + verifier + focus
  tools/mem/live.sh --attach               # keep the game, restart the rest
  python3 tools/mem/reader.py --spawn      # reader + game, no root
  python3 tools/mem/verify_gui.py          # second window
  python3 tools/mem/probe.py               # one-shot check
  ```

---

## Gym / behaviour cloning (v1)

Root venv: `.venv` (see `tools/mem/requirements.txt`; gymnasium, numpy,
CPU-only torch). All live work is serialized by `tools/mem/game_session.py`;
the contracts are in `tools/mem/CONTRACTS.md`, acceptance in `ACCEPTANCE.md`,
live findings in `FINDINGS.md` and `extra_map.json`.

### CPU vs CPU 1v1 (Mario vs DK, FD)

```bash
.venv/bin/python tools/mem/game_session.py run --owner me \
  --restart-game --restart-reader --keep-game \
  --match 'p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end,players=2' \
  -- .venv/bin/python tools/mem/probe.py --frames 3
```

`players=2` forces slots 2..7 to none; without it the saved roster adds CPUs.

### Record, train, evaluate

```bash
# 60 Hz raw observer-mode recording (reset loop uses R2 + the control block)
.venv/bin/python tools/mem/game_session.py run --owner rec \
  --restart-game --restart-reader --keep-game \
  --match 'p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end,players=2' \
  -- .venv/bin/python tools/mem/gym_record.py --episodes 30 \
     --out tools/mem/test_artifacts/accept/mario_vs_dk_1v1_30 --hz 60 --no-lock

# non-naive BC: GRU, 8-decision history, 4-action chunks, 9x9 stick heatmaps,
# mirror augmentation, best-val early stopping, status-derived shield labels
.venv/bin/python tools/mem/gym_bc.py \
  --data tools/mem/test_artifacts/accept/mario_vs_dk_1v1_30 \
  --epochs 30 --patience 5 --cstick --out data/accept/bc.pt --json

# live closed-loop eval (P1 human/agent, P2 DK CPU; focus is required for pads)
.venv/bin/python tools/mem/game_session.py run --owner ev \
  --restart-game --restart-reader --keep-game \
  --match 'p1=mario,p1=human,p2=donkey,p2=cpu,cpu=9,stage=end,players=2' \
  -- bash -lc 'tools/mem/focus_game.sh; .venv/bin/python tools/mem/wait_live.py; \
     .venv/bin/python tools/mem/gym_eval.py --model data/accept/bc.pt \
       --episodes 1 --max-steps 150 --json'
```

### Important v1 caveats

- The ControlModule `buttons` field is the **CPU AI's virtual button state**, not
  the physical controller (human P1 reads 0). BC therefore uses
  `--shield-from-status` to relabel GUARD frames as `ZL=1, X=0`; the virtual pad
  maps ZL to Pro Controller ZL. The raw mask stays in the dataset.
- The saved rule is 0-stock; stock reads 0. The env ignores an unusable stock
  signal and terminates on `fight_on` 1->0.
- The 27-episode 1v1 model is a **pipeline proof**, not a usable policy: test
  active F1 0.279, stick 0.714, and the closed-loop policy is still near
  neutral. Collect 300+ 1v1 episodes (or human demos) and retrain; consider
  DAgger/offline RL and a stronger stick head.
- R2 reset captures spawn at first ready-go; one capture left DK offstage
  ((70,0.01)/(-82,-4.7)). Prefer the game's own rematch spawn or capture only
  WAIT/on-ground frames.
- The extra memory block `/dev/shm/ssbu_extra.bin` is raw-only (553 records,
  534 verified): shield parts, damage/stop hitlag, ground, kinetic, motion,
  work terms, fighter info, battle objects, stage/camera/RNG.

### Tests

```bash
.venv/bin/python tools/mem/test_gym_env.py     # 58
.venv/bin/python tools/mem/test_gym_bc.py      # 38
.venv/bin/python tools/mem/test_gym_input.py   # 31
.venv/bin/python tools/mem/test_extra.py       # 22
```
