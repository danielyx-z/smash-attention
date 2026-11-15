# SSBU Gym build contracts (agents must follow)

This file is owned by the coordinator. Agents may read it; only the coordinator
edits it. Every agent works on its own files only.

## Fixed decisions

- P1 = Mario CPU level 9, P2 = Donkey Kong CPU level 9, Final Destination,
  1 stock. `SSBU_DEBUG_MATCH=p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end`.
- The saved roster may be the opposite (P1 DK, P2 Mario) and the stage random;
  force both fighter names, both types, the level and `stage=end` every launch.
- Decision interval: **5 reader frames** (12 Hz at the 60 Hz reader).
- Action: buttons + left stick + C-stick, sent through a uinput virtual pad.
- Policy observation: minimal (both fighters); no stock/dead/stage/match state/
  opponent id.
- Raw state: everything readable, kept in `info`/raw files, not observed.
- R2 direct runtime reset: no R1 fallback; 20/20 resets required.
- Expert demonstrations: CPU vs CPU, observer mode, 60 Hz raw, resampled to 12 Hz.

## File ownership (no agent edits another's files)

| owner | files |
|---|---|
| coordinator | `tools/mem/CONTRACTS.md`, `tools/mem/locks.py`, `tools/mem/game_session.py`, integration tests, final docs |
| A (runtime) | `runtime/debug_mode.c` only (backup first; build lock) |
| B (state bridge) | `tools/mem/schema.py`, `gamestate.py`, `reader.py`, `probe.py`, `verify_gui.py`, `memmap.json`, `README.md` |
| C (gym env) | `tools/mem/gym_env.py`, `gym_obs.py`, `gym_action.py`, `gym_reward.py`, `test_gym_env.py` |
| D (input/reset) | `tools/mem/gym_input.py`, `gym_reset.py`, `control_block.py`, `test_gym_input.py` |
| E (record/BC) | `tools/mem/gym_record.py`, `gym_bc.py`, `gym_eval.py`, `test_gym_bc.py` |
| F (extras) | `tools/mem/extra_state.py`, `extra_map.json`, `FINDINGS_EXTRA.md`, `test_extra.py` |

If a file must change owner, ask the coordinator first.

## Shared memory v7: `/dev/shm/ssbu_state.bin`

Header, 80 bytes:

```
magic 4s "SSBU"
version I (=7)
n_fighters I
seq I
ts_ns Q
fps f
reader_pid I
fight_on I
match_state I
blast_left f
blast_right f
blast_top f
blast_bottom f
stage_id i
game_frame I
flags I
pad I
pad I
pad I
```

2 fighter slots, 128 bytes each, in this order (index: name, type):

```
0  entry_id        i32
1  kind            i32   (spawned fighter kind; match record +8, not the
                          select-screen information record, which keeps the
                          saved roster when SSBU_DEBUG_MATCH forces a name)
2  stock           i32   (read as u32 from record+0xD8, stored signed)
3  damage          f32
4  x               f32
5  y               f32
6  facing          f32
7  action_state    i32
8  action_frame    f32
9  on_ground       i32
10 is_actionable   i32
11 jumps_left      i32
12 vx              f32
13 vy              f32
14 dist_to_blast   f32
15 dx              f32
16 dy              f32
17 facing_rel      i32
18 is_cpu          i32
19 is_on_rebirth   i32
20 shield          i32
21 shield_size     f32
22 is_enable_cancel i32
23 hitstun         i32
24 buttons         u32   (expert raw button bitmask)
25 stick_x         f32
26 stick_y         f32
27 trigger         f32
28 cstick_x       f32
29 cstick_y       f32
30 raw_flags       u32
31 pad             u32
```

Block size = 80 + 2 * 128 = 336 bytes.

Rules:
- Publish slots in player order: slot 0 = entry 0 (P1), slot 1 = entry 1 (P2).
  Do NOT reorder by human/CPU.
- `raw_flags` bit i = field i was readable this frame (planning bit 24-29 for
  control fields too).
- Empty slot = all zero bytes, raw_flags = 0.
- Odd/even `seq` protocol exactly as before.

## Reset + status control block: `/dev/shm/ssbu_control.bin` (SSBC v2)

Owner: D. Runtime polls it. The first 24 bytes are the v1 reset layout, so old
readers keep working; v2 appends per-frame status and a small command channel.

```
offset  type  field
0       4s    magic "SSBC"
4       I     version (2)
8       I     reset_seq      env increments to request a reset
12      I     ack_seq        runtime copies reset_seq when the reset is done
16      I     state          0 idle, 1 requested, 2 running, 3 done, 4 error
20      I     flags          bit0 fight_on, bit1 reset_busy, bit2 pad_active,
                             bit3 turbo, bit4 pad_from_ctl, bit5 fixed_seed,
                             bit6 hash_valid, bits 8..31 epoch (game pid low
                             24 bits; a block from another process is stale)
24      I     game_frame     game frame: player 1 pad poll index
                             (runtime-written on the game thread; one per step)
28      I     state_hash     FNV-1a of the fighter simulation (runtime)
32      I     seed           runtime echo of the last set_seed
36      I     cmd_seq        writer increments for every command
40      I     cmd            0 none, 1 set_seed, 2 set_pad, 3 clear_pad
44      I     arg0           seed (cmd 1) / target frame hint (cmd 2, 0 = next)
48      I     arg1           XInput button mask (cmd 2)
52      I     arg2           left stick: int16 x | int16 y << 16
56      I     arg3           C-stick: int16 x | int16 y << 16
60      I     cmd_ack        runtime echoes cmd_seq when the command ran
64      I     ack_frame      game_frame when it ran
68      I     speed_milli    runtime echo: 1000 = 1x, 0 = uncapped (SSBU_SPEED)
72      I     pad0           triggers, lt | rt << 8
76      I     pad1           match-start game_frame (turbo_check base)
```

Reset protocol unchanged: the env writes `reset_seq += 1`, the runtime
acknowledges, and the env waits for `ack_seq == reset_seq`, `state == 3` and
stable `fight_on == 1` frames. Commands: increment `cmd_seq`, write
`cmd`/args, wait for `cmd_ack == cmd_seq`; `ack_frame` says when it ran.
`state_hash`/`game_frame` are written every presented frame and only valid
when `flags & 64`; they are the basis for verifying `SSBU_SPEED` runs.
`set_pad` is used only with `SSBU_PAD_FROM_CTL=1` (the synthetic XInput pad on
slot 0 is then driven by the block instead of the sequence).

## Runtime switches that matter for training

- `SSBU_RL=1` — maps the control block (reset + v2 status/commands). Required
  for R2, `state_hash`, `game_frame` and `set_pad`.
- `SSBU_INPUT_NO_FOCUS=1` — pads and keyboard stay live without the game
  window having SDL keyboard focus.
- `SSBU_SPEED=<n>` — run the game at n × 60 Hz, one game step per presented
  frame; `0` = uncapped. The presented frame (`game_frame`) and `state_hash`
  in the control block let a run verify turbo correctness.
- `SSBU_FIXED_SEED=<n>` — constant clock reads / fixed RNG source; required
  for reproducible turbo comparisons.
- `SSBU_PAD_FROM_CTL=1` — synthetic XInput pad on player 1 driven by the v2
  `set_pad` command (no uinput, no window focus).
- `SSBU_DEBUG_MATCH=...,items=off,stock=N` — the `stock=N` override now writes
  the stock rule and both fighter records at match setup/ready-go; `items=off`
  writes `item appear 0`. No menu edit is needed.
- `SSBU_SEQ`/`SSBU_XINPUT_SYNTH` — focus-free scripted sequence (unchanged).

## Extra block: `/dev/shm/ssbu_extra.bin`

Owner: F. Self-describing format (magic/version/seq/ts + records with name,
type, value). It must not depend on the core layout version and must not modify
the core reader. F documents the format in `extra_map.json` and
`FINDINGS_EXTRA.md`.

## Canonical action vector

Buttons, in this order (12):

```
A B X Y L R ZL ZR PLUS MINUS DPAD_L DPAD_R
```

Stick: `(x, y)` in [-1, 1]; C-stick: `(x, y)`; trigger: [0, 1].

The raw `buttons` bitmask from ControlModule must be decoded into this order
(B/D discover the bit positions). The recorder stores both the raw bitmask and
the decoded vector.

## Policy observation profile v1

Per fighter (P1 then P2), flat float32, **56 values per fighter (112 total)**:

```
x/100, y/100, vx/50, vy/50, damage/100, facing, on_ground, is_actionable,
jumps_left/2, action_frame/60, shield, dist_to_blast/300, hitstun,
is_on_rebirth, action_state one-hot(32), dx/100, dy/100, distance/200,
facing_rel, is_escape, is_tech, is_ledge, is_catch, is_landing,
is_shieldbreak
```

The six trailing flags are 0/1 families over the raw status id, so options the
32-entry one-hot hides in OTHER stay distinct:

```
is_escape      78 ESCAPE_F, 79 ESCAPE_B, 80 ESCAPE_AIR        (roll, spot dodge, air dodge)
is_tech        71 PASSIVE, 72 PASSIVE_FB                      (tech)
is_ledge       73 CLIFF_CATCH, 74 CLIFF_WAIT, 75/76 CLIMB_*, 77 CLIFF_ATTACK
is_catch       43 THROW, 44-47 THROW_*, 48-53 CATCH*/THROW_CATCH   (grab family)
is_landing     84 LANDING, 85 LANDING_AIR, 86 LANDING_FALL_SPECIAL
is_shieldbreak 124-130 SHIELD_BREAK_*
```

No stock, dead count, stage id, match state, opponent id, control, projectiles.

## Dataset schema

Raw recording (60 Hz), one `.npz` per episode + a `manifest.json`:

```
seq            (N,) uint64
ts_ns          (N,) uint64
raw_fighters   (N, 2, 32) float32   # the core slot, field order above
buttons_raw    (N, 2) uint32
stick          (N, 2, 2) float32
cstick         (N, 2, 2) float32
trigger        (N, 2) float32
frame_skip     int (5)
meta           dict: p1_char, p2_char, stage, cpu_level, build
```

BC dataset builder resamples to 12 Hz (every 5th sample), windows T=8,
action chunk H=4.

## Locks and test serialization

- `/tmp/ssbu_game.lock`: every launch/attach/restart/reader/uinput/record/eval.
  Use `tools/mem/game_session.py` or `tools/mem/locks.py`.
- `/tmp/ssbu_build.lock`: only A/coordinator rebuild.
- `/tmp/ssbu_pad.lock`: one virtual pad user at a time.
- Never call `play.sh` or `pkill ssbu_native` directly. Use the session manager.
- Unit tests use mocks/fixtures and need no lock. Integration tests acquire the
  game lock and run serially.
- Per-agent artifacts: `data/<agent>/`, `tools/mem/test_artifacts/<agent>/`,
  `/tmp/opencode/agent_<agent>.log`.

## Build and backup rules

- Backups: `/tmp/opencode/ssbu_backups/<timestamp>/` (already created for the
  pre-build state).
- A edits `runtime/debug_mode.c` only; rebuild with
  `deps/venv/bin/python scripts/setup.py --target linux --dump /home/daniel/Projects/SSBU/dump`
  while holding the build lock.
- `SSBU_RL=0` disables the runtime hook.
- Python changes need no rebuild.

## Acceptance gates (coordinator runs these)

1. R2 20/20 resets: no trap, `ready_go=1`, full stocks, 0%, same stage.
2. 10/10 matches report `p2 kind 1 (donkey)` and Final Destination.
3. `--pad-test` sees the virtual pad; `--move-right` moves P1.
4. Shield toggles; expert buttons/stick change; extras documented.
5. 10 matches recorded with correct 5-frame alignment.
6. `check_env` + 100 steps + reset.
7. BC: tiny-batch overfit, held-out F1/stick metrics, mirror test, closed-loop.

## Documentation

- `tools/mem/FINDINGS.md`: coordinator-curated narrative.
- `tools/mem/memmap.json`: core evidence (B).
- `tools/mem/extra_map.json`: extra evidence (F).
- Append findings as you go; do not rewrite another agent's section.

### Two-player synthetic pads (`/dev/shm/ssbu_pads.bin`, SSBP v1)

For self-play the recomp reads two synthetic XInput pads from one block
(`SSBU_PAD_FROM_CTL=1`): slot 0 = player 1, slot 1 = player 2.  No uinput, no
SDL, no hot-plug, no per-command ack; both players are read from one snapshot
per game frame.

    0   char4  "SSBP"
    4   u32    version 1
    8   u32    seq      (odd while the writer updates; even when stable)
    12  u32    players 2
    16  ...    reserved
    24  Pad[2], 24 bytes each:
        +0   u32  buttons (XInput mask; console A = 0x2000, B = 0x1000,
                  X = 0x8000, Y = 0x4000, L = 0x0100, R = 0x0200,
                  PLUS = 0x0010, MINUS = 0x0020, dpad 0x1/0x2/0x4/0x8)
        +4   s16  lx      +6   s16 ly
        +8   s16  rx      +10  s16 ry
        +12  u8   lt      +13  u8  rt
        +14  u16  flags (bit 0 connected)
        +16  ...  reserved

`rl.input.pads_block.PadsBlock` writes it.  Player 1 is verified live.  Player
2 still needs the game's own controller assignment (the debug match sets the
player type, not the controller port); see FINDINGS.
