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
1  kind            i32
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

## Reset control block: `/dev/shm/ssbu_control.bin`

Owner: D. Runtime polls it. Proposed layout (D may refine, document it here):

```
magic 4s "SSBC"
version I
reset_seq I      # env increments to request a reset
ack_seq I        # runtime copies reset_seq when reset is done
state I          # 0 idle, 1 requested, 2 running, 3 done, 4 error
pad I
```

The env writes `reset_seq += 1`; the runtime acknowledges; the env waits for
`ack_seq == reset_seq` and `fight_on == 1` plus stable spawn frames.

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

Per fighter (P1 then P2), flat float32:

```
x/100, y/100, vx/50, vy/50, damage/100, facing, on_ground, is_actionable,
jumps_left/2, action_frame/60, shield, dist_to_blast/300, hitstun,
is_on_rebirth, action_state one-hot(32), dx/100, dy/100, distance/200,
facing_rel
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
