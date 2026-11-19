# Working findings (live log)

Append-only. Each agent writes only under its own heading. Coordinator curates.

## Coordinator

- Backups of the pre-build state: `/tmp/opencode/ssbu_backups/20261101_232013/`
  (`tools_mem/`, `debug_mode.c`, `work_debug_mode.c`, `controllers.ini`).
- Root `.venv`: gymnasium 1.4.0, numpy 2.5.4, torch 2.15.0+cpu.
- `tools/mem/CONTRACTS.md` is the interface contract.
- Locks: `/tmp/ssbu_game.lock`, `/tmp/ssbu_build.lock`, `/tmp/ssbu_pad.lock`.
- Live tests go through `tools/mem/game_session.py`.
- Acceptance gates and commands: `tools/mem/ACCEPTANCE.md`.
- Launched 6 background workstreams (disjoint files, shared locks):
  A runtime/R2 `ses_e651288a9ffe0dfWcsY5OJvpDa`,
  B state bridge `ses_e651288a8ffezTuJNg4hU6ew3l`,
  C gym env `ses_e651288a6ffeVFHp7qDdd681qL`,
  D input/reset `ses_e6512889affezywsiWpNFcMNBR`,
  E record/BC `ses_e65128899ffe0LF3J1rt0T23iS`,
  F extras `ses_e65128898ffeAtRfuRD7mqEdYo`.
- Live roster observed at 23:2x: P1 = DK CPU, P2 = Mario (human), random stage,
  stock 0. The saved roster is NOT the target. Force every launch with:
  `SSBU_DEBUG_MATCH=p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end`.
  Verified ui_chara hashes: Mario `c100000edaf3c863`, DK `c100000f5421de55`.
- DK gate: 10/10 P1 kind 0 (mario), P2 kind 1 (donkey), both type CPU, level 9,
  stage Final Destination.
- Agent C follow-up: env now handles stock=0 correctly (`require_positive_stock`,
  `stock_termination`, `stock_zero_frames`, `info["stock_usable*"]`), 58 tests
  pass, `check_env` OK. Combat reward still uses stock deltas; when stock is
  unusable a match ends via `fight_on` with no win bonus (fine for BC).
- Agent E delivered: 60 Hz v7 recorder (atomic npz/manifest, resume), GRU+chunk
  BC (9x9 stick heatmap, mirror, calibration, temporal ensemble), eval.
  32 tests pass; synthetic overfit F1 1.000; mock closed loop 3/3.
  Uses B's `schema.BUTTON_BITS`; observation builder matches C's 100-dim
  profile exactly. Live recorder/BC still pending B's v7 reader.
- Agent B delivered v7 live-tested (1000 frames, 2 matches, no read errors):
  - ControlModule layout resolved: input=0x110, button=0x7c, stick=(0x10,0x14),
    cstick=(0x40,0x44), trigger=0x88 (w), flip_x=0x120.
  - Live buttons observed: X=+0x4, A=+0x1, Y=+0x8, B=+0x2; all masks carry
    0x8000 (not a canonical button; decoder ignores it).
  - Shield: status 10 = GUARD (0->1 at 10, off at 11/12/34). C/E curated list
    used 9; coordinator changed GUARD to 10 and aliased 8/9 to the same one-hot.
    C/E tests still pass (58 + 32).
  - Stage: `StageManager::instance_ -> +0x17C` is 0 in-match on this build;
    reader falls back to match block `0x71052c41b0+0x420`; FD = 3.
  - `shield_size` = 0.0 (unverified source); `stock` nonzero unobserved under
    the saved 0-stock rule; `game_frame` = reader sample counter.
  - P1/P2 slots proven (`is_cpu` 0/1 in human-P1 match; kind 0/1 in CPU match).
  - `live.sh`/verifier show P1/P2; empty slots zero/raw_flags=0.
- Agent D delivered: control block (`/dev/shm/ssbu_control.bin`, 24 B
  `<4sIIIII>`), raw uinput Xbox 360 pad, reset client, 31 unit + 35 live tests.
  Pad: evdev `SSBU Virtual Pad` 045e:028e, SDL name `Xbox 360 Controller`, GUID
  `030049205e0400008e02000014010000`, player 1; canonical buttons/sticks/
  trigger verified one-to-one. `controllers.ini` has a backed-up
  `[device 030049205e0400008e02000014010000]` section.
  Focus gate: real pads only count when SDL keyboard focus is the game window;
  use `focus_game.sh` and pick the client window whose PID is `ssbu_native`.
  D's blocker (SSBU_RL not passed) fixed: `game_session.py` now launches with
  `--env SSBU_RL=1`. Live 20/20 reset waits on A's runtime hook.
- Coordinator: `test_integration.py --probe/--reset/--env-steps/--record/--all`
  added; `--probe` validates the live v7 block. 121 unit tests pass
  (58 env + 32 BC + 31 input), plus B's 90 schema/env/BC drift tests.
- D blocker fixed: `game_session.py` launch now includes `--env SSBU_RL=1`
  (matching the contract's kill switch `SSBU_RL=0`). Live reset still waits on
  A's runtime hook.
- Agent F delivered the extra block `/dev/shm/ssbu_extra.bin` (SSBX v1, 553
  records, 534 verified, ~41 KB): shield parts/bubbles, damage/reaction/stop
  hitlag, GroundModule, Kinetic velocity entries, Motion Hash40/kind, Work
  terms, FighterInformation, BattleObjectManager pools, stage/match/camera/RNG.
  22 tests pass. Raw-only; policy observation unchanged. Unknown: absolute
  shield HP, cliff id, true floor distance, several virtual predicates, battle
  object owner/category semantics, match timer/platforms/guest frame counter.
  This block is ready for projectiles later.
- Coordinator live acceptance so far:
  - Boot gate: live match P1 Mario CPU / P2 DK CPU / stage 3, 60 fps.
  - R2 + D's client: reset seq 39 ack 39 state 3, 3 stable frames, 2.098 s;
    after reset P1 kind 0/P2 kind 1, damage 0, stage 3.
  - Live env: 20 steps, 5 frames each, obs 100, reward 0, 60 fps.
  - Recorder: 10/10 episodes, 900-1000 frames each, 0 seq gaps, P1 Mario/P2 DK,
    FD, ~6 min; data in `tools/mem/test_artifacts/accept/mario_vs_dk`.
  - BC 20 epochs on 10 episodes overfits (val 1.60@ep2 -> 3.67@ep20; test
    active F1 0.122, stick_acc 0.54). E is adding best-val early stopping.
    Dataset caveat: only A/B/X/Y nonzero; trigger ~0.00017; cstick zero; the
    CPU rarely shields/C-sticks in these 10 matches. More episodes (or human
    demos) are needed for shield coverage.
  - **Action-label calibration finding**: with P1 human + the virtual pad,
    every physical button still read `buttons=0`; the ControlModule mask is the
    **CPU AI's virtual button state**, not the physical controller. GUARD frames
    carry raw mask `0x8004` (currently decoded as X). A naive BC policy would
    press X instead of shield. E added `--shield-from-status` (default on):
    GUARD frames force canonical ZL=1 and clear X; ZL maps to Pro Controller
    ZL (shield). Raw mask stays in the record. 38 BC tests pass; best-epoch
    test active F1 0.306, ZL F1 0.889 (10 episodes).
  - **Roster is not 1v1**: every launch has players 0,1,2,3 type 1 active; the
    saved roster carries 4 CPUs. The state bridge only reads slots 0/1, so the
    extra CPUs are invisible to the policy. The 10-episode dataset and the live
    eval are contaminated. A is adding `players=2` (force slots 2..7 type none).
    The current dataset `test_artifacts/accept/mario_vs_dk` is pipeline-only;
    re-record `mario_vs_dk_1v1` and retrain after A's gate. Keep eval
    `--max-steps 300` (not 1800) for bounded runs.
  - BC policy diagnostic on the 10-episode checkpoint: thresholds
    `[A .434, B 1.0, X .595, Y .289, ...]`; an expert X frame had prob 0.478 <
    0.595 (suppressed), and the stick head predicted 0.12/0.06 vs expert
    0.32/0.95. The model is data-limited (8 train episodes); the pipeline is
    correct, the policy is not usable yet. 300+ clean 1v1 episodes are needed.
  - 27-episode 1v1 BC (21/3/3): best epoch 2, test active F1 0.279, stick
    0.714, ZL F1 1.0 (test episode has no expert ZL); predicted button rates
    close to expert. Still too small for a usable policy.
  - **Neutral control** (P1 human idle, P2 DK CPU, no policy): P1 72.9%,
    P2 59.0%, KO after 179 steps (15 s). The game/1v1 CPU is healthy; the
    eval's 0 damage is the BC policy outputting near-neutral (all buttons 0,
    stick abs 0.067 vs expert 0.307).
  - **Reset-spawn caveat**: A's R2 reset captures spawn at first ready-go; one
    eval reset captured (70, 0.01) / (-82, -4.7), which can leave DK offstage.
    A 20-reset gate passed, but for eval/recording prefer the game's own
    rematch spawn or capture only frames with `on_ground==1` + WAIT.

## Agent A — runtime reset

### Task 1: `p1=cpu|human` in SSBU_DEBUG_MATCH (2026-11-01)
- `runtime/debug_mode.c` now has `g_dm_p1type` beside `g_dm_p2type`; `p1=cpu|human`
  and `p2=cpu|human` set the panel state's player type (`wr32(st + 0x78, type)`),
  independently of the fighter-name keys (`p1=mario`, `p2=donkey` still write
  `st + 0x88`).  The type is applied for **both** idx 0 and idx 1 in
  `match_player_hook`, even when no fighter name is given for that player
  (the fighter stays the screen's own choice), and `cpu=<n>` sets
  `st + 0x91` / `st + 0x224` whenever the resulting type is CPU.  All of it is
  under the existing `type_was != 3` (empty panel) and `SSBU_DEBUG_KEEP` guards.
- Rebuild for experiments: `runtime/debug_mode.c` is a host source, not part of
  the 580 MB game library.  A host-only rebuild (same `build.py` flags, compile
  only changed host files + link `ssbu_native` against the existing
  `ssbu_game_eh_o1.so`) takes <1 s: `/tmp/opencode/agent_A/host_link.py`.
  The canonical full command remains
  `deps/venv/bin/python scripts/setup.py --target linux --dump /home/daniel/Projects/SSBU/dump`.

### R2: direct reset (in progress)
- `SSBU_RL=1` (default off) maps `/dev/shm/ssbu_control.bin` (`magic "SSBC"`,
  `version`, `reset_seq`, `ack_seq`, `state`, `pad`; 24 bytes) and polls it once
  per presented frame in `debug_mode_frame` (the game thread).  On a new
  `reset_seq` it sets `state=2`, calls the entry in-process with `call_guest`
  on a fresh Ctx (1 MB dedicated stack; `tpidr` captured from the game thread at
  `match_player_hook`), then writes `ack_seq` and `state=3` (or `4` on error).
  Research overrides: `SSBU_RL_ADDR=<hex guest address>`, `SSBU_RL_A0..A3`.
- Linux build: the existing hardware-write-watch helper (`watch_set`,
  debug registers) is Windows-only.  Discovery therefore attaches `gdb` to the
  game (ptrace_scope=0) and sets a hardware watch on `impl + 0xd2`
  (`is_ready_go`) and on `impl + 0x20` (entry 0 pointer); the host `.so` keeps
  local symbols (`f_<guest address>`), so a watch hit gives the exact lifted
  function and its caller chain.  Script: `/tmp/opencode/agent_A/watch_d2.py`.
- Static false leads found while looking for the writer of `impl+0xd2`:
  `f_7101780440` (match controls set-up), `f_71022b6f60` (a 0xEE00-byte
  over-merged lifted region that initialises an unrelated struct at +0xd2),
  `f_710296c200` (same kind of struct initialiser).  A plain store search for
  `+210` is not enough: many objects share that offset.
- Scene machinery seen so far: the select screen's confirm handler is
  `0x710178e2e0` (record builder `0x7101788260` called at
  `0x710178f494..498`, stage block filled by `0x710178ab60`); the outer
  character-select scene step is `0x71023108b0`.  Scene names are hashed and
  looked up by `0x710235c6b0`/`0x710235cbf0` (scene factory switch on the id at
  `0x71053040f0`); the manager singleton block is `0x710532e000` (slot 1840 =
  a manager created by `0x710326adb0`, vtable `0x5180e98`).  `SceneResetMessage`
  (`0x42c36c6`) is returned by `0x71017f4bd0`, a virtual method of the scene
  base class (vtable slot `0x50a00c8`).

### R2 discovery: the battle-start / rematch entry (2026-11-02)
- gdb hardware write watch on `impl+0xd2` (impl from
  `[[0x71052b84f8]]`, read through `/proc/<pid>/mem`) caught the first
  `is_ready_go = 1` write of a fight at `f_7100679ae0 + 0x...806`:
  `679af8: strh w8,[x0,#210]` (writes `0xd2=1, 0xd3=0`) then
  `strb wzr,[x0,#213]`.
- Guest stack at that write (inner to outer):
  `0x7100679ae0` <- `0x71014f7da0` (call at `0x14f7e5c`) <-
  `0x7101342780` <- vtable `call_indirect` <- `0x710135b810` <-
  `0x710135f570` <- `0x7101850f60` <- `0x7101850d80` <- ... <-
  `0x7103724a80` <- `0x71022defb0` <- `u_7103747270` <-
  `0x71002c5620` <- `nnMain`.
- `0x7101342780` is a 32-byte wrapper:
  `x8 = *(0x71052c2000+0x760); x0 = *x8; b 0x71014f7da0`.
  (lifted: `q = rd64(GA(0x71052c2760)); call 0x71014f7da0(rd64(q))`).
- `0x71014f7da0(x0 = melee start-parameter object)` is the full match
  (re)start: `x19 = x0`, calls `0x14e49e0(x19+0x60, x19+8)`, writes the
  parameter's launch fields (`+9..+20`), then
  `mgr = *(0x71052b84f8); impl = *mgr; impl+0xd0 = 0; 0x679ae0(impl)`
  (ready_go=1), reads `impl+0xa0` (entry count) and iterates the fighter
  entries, allocating/initialising them (`0x39c20c0`, `0x3776360`,
  `0x15901b0`, ...).  It is the function the select screen's scene invokes
  through the scene/manager vtable.  (A rematch uses `0x71014d4a20` instead,
  see below.)
- First R2 attempt called **`0x7101342780` with no arguments** (it fetches the
  persistent start-parameter object itself).  Result: the call returns, no
  guest trap, `ack_seq` advances, `state=3` - **but it is NOT a rematch**.
  Trace (`game_20261102_000855.log`, trace_reset.py): requested after a live
  fight, Mario went `x=-29.1 st=32` -> `x=-17.3 st=7` -> `x=+7.1 st=10` across the
  reset (continuous motion), then took 11.6% damage 0.5 s later.  The function
  early-outs on the existing FighterManager entries: it only re-arms
  `impl+0xd0 = 0` / `ready_go = 1` and re-runs the per-entry preparation that is
  already done.  So the real rematch must go through the scene/sequence level
  (destroy/recreate the battle scene), not through this start method.
- The same negative test showed the natural pre-fight sequence for the spawn
  check: fighters appear at `x = -36 / +36`, `action_state` 263/473 (entry),
  then 0 (idle, actionable), then `ready_go` (fight_on) becomes 1 and the CPUs
  start moving immediately.  At `ready_go` the fighters are *not* at spawn
  anymore; a spawn check must sample right after the reset call, not 0.3 s later.
- Rematch discovery (`seq_rematch.txt`, run `game_20261102_001542.log`, gdb
  `watch_d2.gdb.log`): fight 1 ready_go at presented frame 1930, `fight off`
  (ready_go=0) at frame 10465 after 8535 fight frames; the seq's A presses then
  reached the results screen and a **second fight** started at frame 10651
  (stock 1/1, same Mario/DK, stage id 0x3).
  The second `is_ready_go=1` write has a **different caller**:
  `0x7100679ae0` <- **`0x71014d4a20`** <- vtable call_indirect <-
  `0x710135b810` <- `0x710135f570` <- `0x7101850f60` <- ... (same scene/sequence
  frames as below).  The first fight used `0x71014f7da0`; the **rematch uses
  `0x71014d4a20`**.
- `0x71014d4a20` (disassembly 0x14d4a20..):
  loads the same match object `x20 = *(*(0x71052c2760))`, gets a string/table
  from `*(0x71052c2910)`, calls `0x15435c0` three times (looks up the rematch
  action/name), sets `[x20+9]=0`, `[x20+18]=1`, `[x20+0xdf58]=-1`, then
  `mgr = *(0x71052b84f8)`, `impl = *mgr`, reads `impl+0xa0` (entry count) and
  iterates the entries at `impl+0x20` -- i.e. the scene-level restart that
  reuses the loaded battle scene.  It takes no useful arguments.
- **The rematch entry alone is still not enough**: calling `0x71014d4a20`
  mid-fight re-arms `ready_go` and clears scene flags, but the FighterEntry
  damage/positions are not reset (trace `game_20261102_002446.log`: damage
  13.12 persisted and Mario walked on).  The results-screen flow has already
  put the scene into its result state before it calls this entry.
- The match-end function is `0x7101585b20(x0 = battle scene object)`:
  `strb 1,[x0+24]`, `strh 0x100,[impl+210]` (so `impl+0xd2=0`, `impl+0xd3=1`),
  then clears per-entry flags.  Its only caller `0x7101581270(x0=scene)`
  (a huge battle-scene update) passes its own x0; the scene object is captured
  by hooking `0x7101581270` and remembering `c->x[0]`.
- R2's working sequence (all in-process, on the game thread):
  1. capture each fighter's spawn position at the match's first ready-go
     (`accessor = [[entry+0x4160]+0x20]`, posture `[accessor+0x38]`, floats at
     `+0x30`), and the scene object from the `0x7101581270` hook;
  2. `0x7101585b20(scene)` -- the game's own end-of-match path (`ready_go=0`);
  3. wait up to 120 frames (state stays `2`); then `0x71014d4a20()` -- the
     game's rematch entry (`ready_go=1`, same match block);
  4. restore the fighters with the game's own module APIs:
     `DamageModule__init_damage_impl(acc, 0.0f)` = `0x710200b6b0`,
     `KineticModule__clear_speed_all_impl(acc)` = `0x7102043990`,
     `PostureModule__set_pos_impl(acc, &spawn)` = `0x710206aea0`;
  5. `ack_seq = reset_seq`, `state=3`.
  Latency ~2.05 s per reset (the end phase is 120 frames).  Evidence: 5/5
  resets in `game_20261102_004451.log` and **20/20 in
  `game_20261102_005608.log`** (all resets: Mario kind 0 type 1, DK kind 1
  type 1, stage 3, damage 0, x = -36 / +36 matching the captured spawn,
  match records unchanged, 0 guest traps).  The same log has **21/21
  "fight 0 on" lines** with `p1 kind 0 (mario), p2 kind 1 (donkey) type 1,
  stage id 0x3`: the initial fight plus one per reset (ending clears
  ready_go, the rematch sets it again), so the 10/10 match gate is covered
  as well.
- Commands:
  `python3 /tmp/opencode/agent_A/live_run.py --seq /tmp/opencode/agent_A/seq_discover.txt --env SSBU_RL=1 --keep-game -- python3 /tmp/opencode/agent_A/test_r2.py --count 20 --timeout 200`
  (game launched as `play.sh --linux --seq seq_discover.txt --env
  SSBU_DEBUG_MATCH=p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end --env
  SSBU_SKIP_MOVIES=1 --env SSBU_KEYBOARD_PLAYER=0 --env SSBU_RL=1`).
  Match gate: `python3 /tmp/opencode/agent_A/gate10.py --count 10`
  (10 fresh launches, `seq_gate.txt`, `SSBU_DEBUG_END=30`, logs under
  `/tmp/opencode/agent_A/gate/`).
- Caveats / limits:
  - the reset is the game's own end+rematch, so it takes ~2 s (120 frames of
    end phase); the env sees `ready_go=0` during that window;
  - after the rematch the fighters are placed at the captured spawn with the
    damage module set to 0 and kinetic speed cleared; the CPU starts moving on
    the following frames (the test samples immediately after the ack, so it
    sees the spawn);
  - `stock` is 0 under the canonical rule (the record field reads 0 / 0), the
    test compares it rather than assuming a positive count;
  - if the `0x7101581270` hook fails to install, mode 2 falls back to the
    direct rematch entry (which re-arms `ready_go` but does not restore the
    fighters); `SSBU_RL_MODE=1` forces that direct path.

### `players=<n>` in SSBU_DEBUG_MATCH (2026-11-02, 1v1 fix)
- The saved roster carries four CPU panels, so the old target match was a
  3-4 player match even though only P1/P2 were overridden.  `SSBU_DEBUG_MATCH`
  now accepts `players=1..8` (default unchanged, 0 = no forcing).
- Implemented in two places (both in `runtime/debug_mode.c`):
  1. `panel_copy_hook` (`0x710179e440`, the select screen's saved-choice copy):
     for every `idx >= players` it writes panel `base+0x10+0x1c8*idx`:
     `+0x30 = 3`, `+0x34 = -1`, `+0x38/+0x40 = 0xc1ffff0000000000`,
     `+0x48/+0x49 = 0`.  The hook is installed when `players` is set (and the
     `SSBU_DEBUG_MATCH` parse moved ahead of the hook install so the value is
     known).
  2. `match_player_hook`: belt-and-braces `wr32(st + 0x78, 3)` for `idx >=
     players` just before the game's record builder; the p3/p4 override block
     is skipped for those indices.
- Single-launch proof (`game_20261102_014124.log`):
  `FighterManager entries: 2` (entries 2..7 null); records 0/1 = Mario kind 0
  type 1 level 100 / DK kind 1 type 1 level 100; records 2..7 =
  `c1ffff0000000000, kind -1, type 2`; panels 2..7 type 3; stage 0x3.  The log
  has `players=` on the panel-copy hook line, `panel 2..7 forced to none`, and
  `player 2..7: screen choice ... type 3`.
- Limits: `players=1` is accepted by the parser but the game does not start a
  one-panel match (the existing note: "with player 1 alone the match cannot be
  started"); the useful values are 2..8.
- **10/10 fresh-launch gate** (`gate_players2.py`, match
  `p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end,players=2`, logs
  `/tmp/opencode/agent_A/gate2/run_01..10.log`): every run
  `entries=2 records 2..7 none panels 2..7 type 3`, 42 s each, `RESULT 10/10`.
  Each log has `player 2..7: screen choice ... type 3`, `panel 2..7 forced to
  none`, records 2..7 `c1ffff0000000000 kind -1 type 2`, records 0/1
  Mario kind 0 type 1 level 100 / DK kind 1 type 1 level 100, stage 0x3.
- R2 with `players=2`: **3/3 resets** on the two-entry match
  (`game_20261102_015048.log`): acks ~2.1 s, stage 3, damage 0, x = +36 / -36,
  records 0/1 unchanged.  The reset loops only over `impl+0x20` entries 0/1,
  so a two-player match is unaffected.

### Final status (2026-11-02)
- Task 1: `p1=cpu`/`p1=mario` are independent and both applied; verified in
  `game_20261102_005608.log` (`p1 kind 0 (mario), p2 kind 1 (donkey) type 1`,
  cpu power 100, stage id 0x3 / end).
- Task 2: `SSBU_RL=1` R2 passes **20/20** in one process
  (`game_20261102_005608.log`, test_r2.py): no guest trap, `ack_seq` advances,
  `state=3`, `ready_go=1` again within the 2 s end phase, stage 3 on every
  reset, both fighters damage 0 and back at their spawn x (-36 / +36), the
  match records unchanged (Mario kind 0 type 1, DK kind 1 type 1).  The same
  log has 21/21 correct "fight 0 on" lines.  A separate fresh-launch gate
  (`gate10.py`, `/tmp/opencode/agent_A/gate/run_01..10.log`) is also **10/10**
  (0 guest traps).
- Build: the changed file is a host source only.  The tested `ssbu_native` was
  built by `/tmp/opencode/agent_A/host_link.py` (stage + `build.py`'s link step
  with `RECOMP_TARGET=linux`, `RECOMP_CLANG=/usr/bin/clang`, existing
  `ssbu_game_eh_o1.so` kept; 1 host file recompiled, 0.6 s).  The canonical
  `scripts/setup.py --target linux --dump ...` would rebuild the same host
  objects and relink; do NOT run only its link step while `obj_eh_o1/` holds
  one stray object from the interrupted full compile (it would relink the .so
  from that object): a full `setup.py` compile is required if the game library
  itself is ever to be relinked.
- Agent D's client against the runtime (reader live, `gym_reset.py --timeout 30`,
  `game_20261102_010048.log`):
  `reset seq 37: ack 37, state 3, 3 stable fight_on frames (4 seen) in 2.063s`
  (state block `fight_on=1, match=IN_MATCH, fighters=2, stage=3`); the reader
  sees a brief `RESULTS` window during the 2 s end phase and `IN_MATCH` after
  the ack, as intended.
- 10/10 fresh-launch match gate (`gate10.py`, logs
  `/tmp/opencode/agent_A/gate/run_01..10.log`, one launch each with
  `SSBU_DEBUG_MATCH=p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end`):
  every run logged `p1 kind 0 (mario), p2 kind 1 (donkey) type 1 cpu power 100
  | stage id 0x3`; `RESULT 10/10`, 0 guest traps.

## Agent B — state bridge

### v7 implementation (2026-11-02)

Implemented CONTRACTS.md v7 in `schema.py`, `gamestate.py`, `reader.py`,
`probe.py`, `verify_gui.py`, `memmap.json`, `README.md`:

* header 80 bytes, fighter 128 bytes (32 words), block 336; `stage_id`,
  `game_frame`, `flags` in the header; `raw_flags` is fighter word 30.
* `stock` is read as **u32** at record+0xD8 and stored as int32.
* new fields: `is_cpu` (record+0x92), `is_on_rebirth` (record+0x38C),
  `is_enable_cancel` (CancelModule+0x1C), `hitstun` (DAMAGE statuses),
  `shield` (GUARD status, live kind **10**), `shield_size` (0.0 placeholder),
  expert `buttons`/`stick_x,y`/`trigger`/`cstick_x,y`.
* P1/P2 are fixed: slot 0 = entry 0, slot 1 = entry 1. The human-first
  reorder is gone; `is_cpu` is published instead.

### ControlModule evidence (all from the game's own code)

lua_bind wrappers all start `ldr x0,[x0,#72]` (= accessor+0x48) and branch
through the vtable:

| getter | wrapper | vtable slot | concrete loader (13.0.4) |
|---|---|---|---|
| get_button | 0x20013e0 | +0x238 | `0x3f8640: ldr x8,[x0,#0x110]; ldr w0,[x8,#0x7c]; ret` |
| get_stick_x | 0x2001260 | +0x178 | `0x3f84a0: ldr x9,[x0,#0x110]; ldr s0,[x9,#0x10]; ldr w8,[x0,#0x120]; cmp; fneg; fcsel` |
| get_stick_y | 0x2001280 | +0x188 | `0x3f84d0: ldr x8,[x0,#0x110]; ldr s0,[x8,#0x14]; ret` |
| get_sub_stick_x | 0x2001370 | +0x200 | `0x3f85c0: ... ldr s0,[x9,#0x40] ...` |
| get_sub_stick_y | 0x2001390 | +0x210 | `0x3f85f0: ... ldr s0,[x8,#0x44] ...` |
| get_trigger | 0x20013b0 | +0x220 | `0x3f8610: ldr x8,[x0,#0x110]; ldr w0,[x8,#0x88]; ret` |

`gamestate.py` does not use these constants blindly: it reads the live vtable,
reads the target function, decodes the tiny `ldr`/`ldrb` instructions
(`decode_control_getter`) and uses the offsets it finds. Fallback constants are
only for a teardown frame. Live resolution on 2026-11-02:

```
ControlLayout(input=0x110, button=0x7c, stick=(0x10,0x14),
              cstick=(0x40,0x44), trigger=0x88/w, flip_x=0x120)
```

Button bits come from the game's own `check_button_trigger` table at guest
`0x4538fec` (u32[18]): ids 0..11 = `A=1 B=2 X=4 Y=8 L=0x20 R=0x400
ZL=0x1000 ZR=0x2000 PLUS=0x4000 MINUS=0x10 DPAD_L=0x40 DPAD_R=0x80`.
Live CPU masks exercised the face bits: `0x8004` (X), `0x8201` (A + 0x200),
`0x8802` (B + 0x800), `0x8008` (Y), `0x800c` (X+Y). `0x8000` is set almost
always for a CPU; it is outside the 12 canonical bits and is kept raw only.
CPUs never set ZL/ZR in the raw mask (they use AI commands), so the shield bit
mapping rests on the table above, not on a live press.

### Stage id

* Requested chain `StageManager::instance_ -> u64 -> u32 +0x17C` resolves, but
  reads **0 during a fight** on this build:
  `a=u64(global)`, `b=u64(a)`, `a+0x17C=0`, `b+0x17C=0`.
* `app::sv_information::stage_id` is the same chain (0x227ebb0); the value is
  the select-screen information copy, not the in-match stage.
* The in-match value is the match block (`runtime/debug_mode.c`
  `GA(0x71052c41b0)`) at `+0x420`; the debug hook prints the same word.
  With `stage=end` the hook says `stage id 0x3 ... (end)`, and our reader
  publishes `stage_id=3` — **Final Destination is id 3 in this build**.
  `read_stage_id` uses the match block as a documented fallback and records
  `last_stage_source` (`matchblock` live).

### Live evidence (CPU vs CPU, 45 s, 2661 frames)

* slots: entry 0 Mario (kind 0), entry 1 DK (kind 1); both `is_cpu=1`;
  `stock=0` (0-stock debug rule); `is_on_rebirth=0`; `is_enable_cancel` moves
  0/1; no NaN/Inf.
* shield: transitions `0 -> 1 (status 10) -> 0 (status 11/12/34)` for both
  slots (29 and 27 transitions); 79 sampled frames at status 10. **Exact guard
  status kind = 10.** `GUARD_STATUS_KINDS = {10}` (the old approximate name
  called 10 GUARD_OFF; `ACTION_STATE_NAMES[10]` now says GUARD).
* hitstun: damage-reaction frames were exactly statuses 54, 476, 478, 479,
  506 (451/78/58/5/4 frames) — all in `DAMAGE_STATUS_KINDS`.
* buttons: 7 distinct masks per slot, changing with actions.
* stick: x and y both traversed the full `[-1, +1]`.
* trigger: tiny non-zero fixed-point values appeared (`0.0001`, `0.0078`,
  `0.0313`), so the u32 getter is live; a full 0..1 sweep was not produced by
  the CPU.
* cstick stayed 0 for both CPUs (CPUs do not use it).
* shared memory: version 7, slot order `[0, 1]`, `raw_flags=0x7fffffff` for
  both live slots, no NaNs, header `flags=3` (stage + control layout).
* crash safety: the reader ran through the match teardowns at 60.0 fps with
  **0 `read error` lines** (`session_B.log`), and the probe returned
  `[None, None]` while the block had `n_fighters=0` / `raw_flags=[0,0]`.

A second run with `stage=end` confirms it: the debug hook prints
`stage id 0x3 ... (end)` and the reader publishes `stage_id=3`. **Final
Destination is id 3 in this build** (0x125 was a different chosen stage in an
earlier match).

### Live acceptance run (1000 live frames, 2 short 0-stock matches)

```
sampled=1000 skipped=949 matches_started=2
slot 0 mask values: 0x8000 x858, 0x8004 (X) x111, 0x8201 (A) x20,
                    0x8008 (Y) x3, 0x8209, 0x8802 (B) x1, 0x800c (X+Y) x1
slot 0 stick x [-1,+1], y [-1,+1];  cstick both 0
slot 0 trigger 0.0 x956, 0.0001 x33, 0.0078 x5, 0.0313 x1
slot 0 shield transitions 21, slot 1 7;  guard status kind = 10
hitstun status kinds = 54, 477, 478
missing-field frames: []          NaN/Inf: []
shm: v7, fight live, slots [0,1], stage_id=3, flags=3,
     raw_flags=0x7fffffff both slots, no NaNs, masks change (0x8000/0x8004)
empty slot: probe rows [None, None]; shm n_fighters=0, raw_flags [0, 0]
FAILS: []   WARNS: []
```

`tools/mem/test_gym_bc.py` (E's schema-drift test, 32 tests) and
`tools/mem/test_gym_env.py` (C, 58 tests) both pass against this schema.

### Commands run

```bash
python3 -m py_compile tools/mem/{schema,gamestate,reader,probe,verify_gui}.py
python3 /tmp/opencode/b_struct_test.py          # layout + publisher + verifier
python3 /tmp/opencode/b_gui_test.py             # Tk verifier renders all 32 fields
python3 tools/mem/probe.py --json
python3 tools/mem/game_session.py run --owner B --restart-game --restart-reader \
    --keep-game --match 'p1=cpu,p2=cpu,p2=donkey,cpu=9' \
    -- python3 /tmp/opencode/b_final.py 45
python3 tools/mem/game_session.py run --owner B --restart-game --restart-reader \
    --keep-game --match 'p1=cpu,p2=cpu,p2=donkey,cpu=9,stage=end' \
    -- python3 /tmp/opencode/b_final2.py 1000 90
cd tools/mem && ../../.venv/bin/python -m unittest -q test_gym_bc test_gym_env
```

Still open: `shield_size` has no verified source (published 0.0);
`game_frame` is the reader's sample counter, not the game's tick; `stock`'s
non-zero value has not been seen under the 0-stock saved rule (offset/width
proven by `FighterInformation__stock_count`, which loads `w0`).

## Agent C — gym environment

Delivered (owned files only):

- `tools/mem/gym_env.py` — v7 `RawSnapshot`/`RawSnapshotReader`, `SSBUGymEnv`,
  `ShmFixture` mock writer, input/reset adapters.
- `tools/mem/gym_obs.py` — policy observation profile v1.
- `tools/mem/gym_action.py` — canonical action space + decoder + macros.
- `tools/mem/gym_reward.py` — `RewardConfig`, none/combat rewards.
- `tools/mem/test_gym_env.py` — 58 unit tests, mock shm only (no game/locks).

### v7 shared-memory reader

- Layout implemented exactly from CONTRACTS.md: header 80 B, two 128 B slots
  (32 fields), block 336 B; `seq` at byte 12; confirmed by struct sizes
  (`HEADER_FMT="<4sIIIQfIIIffffiIIIII"`, `FIGHTER_FMT` built from the 32-field
  table). Slot order is player order; the env does **not** reorder.
- Odd/even protocol: copy body between two equal, even `seq` reads; odd or
  changed seq is retried. `wait_frames(n)` targets `start_seq + 2*n` (writer
  bumps seq twice per frame); a writer restart (seq backwards) is a new epoch.
- Strict version check: v≠7 raises `SchemaMismatch`; a short legacy v6 block
  (224 B) reports the version instead of just timing out. Missing/short/
  uninitialised blocks time out with `SnapshotTimeout` (a `TimeoutError`).
- Reopens automatically if the writer recreates the file (inode check) and
  re-checks `os.stat` each read.
- `RawSnapshot` exposes parsed slots plus a `(2,32)` float64 raw copy and
  `header_dict()`; all of it goes into `info`.

### Observation profile v1 (`gym_obs.py`)

- 50 floats/fighter × 2 = **100**, dtype float32, P1 then P2, contract order:
  14 scalars, `action_state` one-hot(32), then dx/dy/distance/facing_rel.
- `action_state` curated list = 31 game status kinds + `OTHER` (index 31).
  Curated ids (cross-referenced with B's `schema.ACTION_STATE_NAMES`):
  WAIT 0, WALK 1, RUN 2, TURN 4, DASH 6, GUARD 9, JUMP_SQUAT 11, JUMP 12,
  JUMP_AERIAL 13, FALL 18, FALL_AERIAL 19, FALL_SPECIAL 20, ATTACK 22,
  ATTACK_DASH 23, ATTACK_S3 24, ATTACK_HI3 25, ATTACK_LW3 26, ATTACK_S4 27,
  ATTACK_HI4 28, ATTACK_LW4 29, ATTACK_AIR 30, SPECIAL_S 35, SPECIAL_HI 36,
  SPECIAL_LW 37, SPECIAL_N 38, THROW 43, DAMAGE 54, DAMAGE_FALL 55,
  DAMAGE_FLY 56, DOWN 63, CLIFF_WAIT 74. Unknown ids → OTHER.
- `normalize=False` is the default and returns **raw physical values** in the
  same order (BC-friendly); `normalize=True` applies the contract divisors
  (x/100, vx/50, damage/100, action_frame/60, dist/300, distance/200, ...).
  Field divisors are documented in `SCALAR_FIELDS`/`RELATIVE_FIELDS` and
  `observation_metadata()`. Non-finite/empty slots → 0 (an empty slot is all
  zeros, not a fake WAIT one-hot).
- Excluded as required: stock, dead, stage, match_state, opponent id, control
  fields, projectiles. Box is `shape=(100,) float32`, bounds ±1e6 (finite).

### Action space (`gym_action.py`)

- Default space: `Dict(buttons=MultiBinary(12), stick=Box(-1,1,(2,)),
  cstick=Box(-1,1,(2,)))`. Button order is the contract order A B X Y L R ZL
  ZR PLUS MINUS DPAD_L DPAD_R. Optional `macro=True` → `Discrete(12)`:
  NOOP, LEFT, RIGHT, UP, DOWN, JUMP, ATTACK, SPECIAL, SHIELD, GRAB,
  SMASH_LEFT, SMASH_RIGHT.
- `decode_action()` returns the canonical dict: `buttons int8[12]`,
  `button_names`, `buttons_mask` (bit i = buttons[i]), `stick float32[2]`,
  `cstick float32[2]`, `trigger=0.0`, `macro=None`. Sticks clipped to [-1,1];
  bad shapes/non-finite values raise. `encode_action()` is the inverse for
  round-trips.

### Reward (`gym_reward.py`)

- `reward_mode="none"` (default) always returns 0.0 (BC/recording).
- `reward_mode="combat"`: `damage_scale * (damage_dealt*dealt -
  damage_taken*taken)` with `damage_scale=0.01` (100 % = 1.0), positive
  clamped damage deltas (KO damage reset is not a penalty), per-stock
  `ko_bonus=+1`, `stock_loss=-1`, `win_bonus=+1`, `loss_penalty=-1`,
  `draw_reward=0`. Stock reaching 0 counts as a win/loss in the 1-stock rules
  even before `fight_on` flips; double KO is a draw. `ego_slot` default 0 (P1).

### Environment (`SSBUGymEnv`)

- `frame_skip=5` (12 Hz at 60 Hz); action written **once** through the lazy
  `gym_input` adapter, then the env reads the first post-write frame and waits
  for 5 more fresh frames (`info["frames_advanced"] >= 5`, seq delta ≥ 10).
  `info` carries raw header + both raw fighter slots + `(2,32)` raw array,
  stocks/damages, stock usability, seq/age, action, reward mode, termination
  flags and decode/write/read/total timings.
- Stock signal robustness (`require_positive_stock=True`,
  `stock_termination=True`, `stock_zero_frames=10` by default):
  * per-slot state is exposed as `info["stock_usable"]` (True if any slot is
    usable, False if none, None while undecided) and
    `info["stock_usable_slots"]` (True/False/None per slot);
  * a slot that reports a positive stock is usable forever (a later 0 is a
    real KO); a slot that is 0 for `stock_zero_frames` consecutive live frames
    (`fight_on==1`, readable) without ever being positive is declared
    unavailable, emits one clear `RuntimeWarning`, and is ignored — reset may
    then become ready on `fight_on==1` + readable slots + stable frames, and
    termination relies on `fight_on` 1→0;
  * reset readiness still requires positive stock for slots that are usable or
    still undecided (`require_positive_stock=True`); a late positive stock
    revives a slot and stock termination works again;
  * mixed signals are handled per slot (usable slot still terminates on
    1→0, unusable slot never terminates); set `require_positive_stock=False`
    / `stock_termination=False` to opt out explicitly.
- Termination: `fight_on` 1→0, or a **usable** slot's stock ≤ 0. Truncation:
  `max_steps` (default 1800 = 150 s at 12 Hz). `terminated` wins over
  `truncated`. `render()` is a no-op; `close()` is idempotent.
- `reset()` calls `gym_reset` if importable, then waits (timeout) for
  **fresh** frames after the request with `fight_on==1`, two readable slots,
  and (when the stock signal is usable) positive stocks, for
  `spawn_stable_frames` consecutive frames. A stale pre-reset `fight_on=1`
  frame cannot satisfy a reset. Default timeout 15 s.
- `check_env` note: a live game is non-deterministic, so the test sets
  `env.spec = EnvSpec(..., nondeterministic=True)` before calling
  `check_env(..., skip_render_check=True, skip_close_check=True)`. That only
  skips the "same action ⇒ same observation/info" assertion; every structural
  check runs.

### Interface assumptions for D (input/reset)

- `gym_input` (or an injected object) must expose one of
  `apply_action`/`write_action`/`send_action`/`set_action`/`apply` accepting a
  **single canonical action dict** (keys above) and return immediately after
  updating the uinput report. `close()` is called by the env if present.
  Import failure or no recognised entry point → `NoopInputAdapter` + one
  `RuntimeWarning` (actions are no-ops). Tests inject a recorder so uinput is
  never touched.
- `gym_reset` (or `reset_fn=`) must expose a zero-argument
  `reset`/`request_reset`/`reset_game`/`do_reset`/`reset_match`; the env then
  waits for the v7 state itself. If absent, reset only waits for an existing
  match.
- Env assumes v7 slot 0 = P1/entry 0 (contract); current v6 `gamestate.py`
  sorts human-first, so integration must use the v7 writer before slot order
  matters. `ego_slot` selects the reward perspective.

### Tests / results (no game, no lock)

Commands and observed results (2026-11-01, root `.venv`, Python 3.12.3,
gymnasium 1.4.0, numpy 2.5.4):

```
.venv/bin/python tools/mem/test_gym_env.py -v
# Ran 58 tests ... OK   (~3.3 s; repeated 3×, all OK)
```

Coverage: layout sizes/offsets, reader round-trip, odd-seq retry, writer
starvation timeout, v6 mismatch, observation shape/order/raw+normalized
values/empty slots/one-hot, action space + round-trip + macros, reward modes
(including none ignoring damage changes), `check_env`, 100 steps,
action→adapter delivery, macro mode, info extras, reset
fn/timeout/fresh-frame checks, **stock signal matrix**: 1-stock positive,
positive→0 drop, always-zero fallback with warning, mixed per-slot signals,
late stock init/recovery, `require_positive_stock=False`,
`stock_termination=False`, strict reset once a real signal was seen; fight-off
and zero-stock termination, max-steps truncation, combat reward wiring,
normalized env, step timeout, render/close, lazy `gym_input`/`gym_reset`
discovery and the no-op fallback.

Standalone acceptance-style run (mock writer at 500 Hz):

```
check_env: OK (0 warnings)
reset: obs (100,) float32 seq=62 fight_on=1
100 steps: OK in 1.003s, frames_advanced=5, seq_delta=10,
           reward_mode=none, sum_reward=0.0

stock=0 reset: OK stock_usable=False slots=[False, False] warnings=2
stock=0 steps: terminated=False truncated=False stock_usable=False
stock=0 fight_off: terminated=True reason=fight_off
stock=1 reset: OK stock_usable=True slots=[True, True] warnings=0
stock 1->0: terminated=True reason=stock
```

Live-state note: `/dev/shm/ssbu_state.bin` on this box is still the B v6
224-byte block. The env fails fast with
`SchemaMismatch: shared-memory layout v6 (224 bytes) ... needs v7 (336 bytes)`.
Once B updates `schema.py`/`reader.py` to v7, no env change is needed.


## Agent D — input / reset client

Delivered (owned files only):

- `tools/mem/control_block.py` — 24-byte `/dev/shm/ssbu_control.bin`
  (`<4sIIIII`): magic `SSBC`, version 1, `reset_seq`, `ack_seq`, `state`
  (0 idle, 1 requested, 2 running, 3 done, 4 error), `pad`.  `ControlBlock`
  (open/read/write/request_reset/ack/wait_for_ack/wait_for_done), module
  helpers and a CLI (`--init --read --request --wait SEQ`).
- `tools/mem/gym_input.py` — raw uinput Xbox 360 pad (struct/fcntl/ioctl only,
  no python-evdev), pad lock, context manager, lazy create, `apply(Action)`,
  and `--test [--seconds N] [--demo]`.
- `tools/mem/gym_reset.py` — `ResetClient` (request → ack/done → `fight_on == 1`
  for N fresh state frames), `MockRuntime`/`ResetClient(mock=True)`, CLI.
- `tools/mem/test_gym_input.py` — 31 unit tests + live `--pad-test` test.
- Agent C integration: `apply()` accepts C's action dict (`buttons` as a
  12-slot numpy int8 vector, `stick`/`cstick` numpy arrays, `trigger`), a list
  of names, an `Action`, or the 17-value vector.  `VirtualPad` creates lazily
  on first `apply()` and exposes `close()` so `gym_env` can use it as its
  default input adapter; `ResetClient` is what `gym_env.default_reset_fn()`
  finds.

### Control block

Layout matches CONTRACTS.md and Agent A's `RlCtl` in `runtime/debug_mode.c`
(verified by reading the source: same 24 bytes/order/states).  A creates the
file 0666 and sets `ack_seq = reset_seq` on init.  Our `request_reset()` reads,
increments `reset_seq`, writes the whole struct with `state = 1`; the client
then waits for `ack_seq == reset_seq`, `state == done`, and fresh stable
`fight_on` frames.  `wait_for_ack` raises on `state == error`.

### Virtual pad

- evdev name `SSBU Virtual Pad`; bus USB 3; ids `045e:028e` version `0114`.
- SDL 3.2.6 overrides the name to **`Xbox 360 Controller`** and reports GUID
  **`030049205e0400008e02000014010000`** (`type Xbox 360`, `SDL mapping: yes`,
  14 buttons / 6 axes / 1 hat, player 1).  The GUID is stable for this SDL;
  `gym_input.SDL_GUID` records it.
- Descriptor: EV_KEY `BTN_SOUTH/EAST/NORTH/WEST`, `BTN_TL/TR`, `BTN_TL2/TR2`,
  `BTN_SELECT/START`, `BTN_DPAD_UP/DOWN/LEFT/RIGHT`; EV_ABS `ABS_X/Y/RX/RY`
  (-32768..32767), `ABS_Z/RZ` (0..255), and the standard 360 hat
  `ABS_HAT0X/Y` (-1..1).  D-pad is sent as both the dpad keys and the hat.
- Permissions: `/dev/uinput` is root:input 0660 with ACL `user:daniel:rw-`;
  daniel is **not** in the `input` group, so the ACL is what allows creation.
- During creation the process holds `/tmp/ssbu_pad.lock` until
  `UI_DEV_DESTROY` + close; `destroy()` always releases keys/axes first.
  No leftover `/dev/input` node or pad-lock holder after the tests.

Canonical action mapping (verified live through `--pad-test` sources):
A→south/A, B→east/B, X→west/X, Y→north/Y, L→leftshoulder/L, R→rightshoulder/R,
ZL→lefttrigger/ZL, ZR→righttrigger/ZR, PLUS→start/PLUS, MINUS→back/MINUS,
DPAD_L/R→dpleft/dpright/LEFT/RIGHT.  Left stick x right positive, y up positive
(ABS_Y is negated: evdev up is negative), C-stick same, trigger→ABS_RZ
(0..255) → ZR.  Live checks: stick right `L(32767,0)`, up `L(0,32767)`, left
`L(-32767,0)`, down `L(0,-32767)`, C right/up `R(32767,0)`/`R(0,32767)`.

`BTN_SELECT`/`BTN_TR2` are advertised but not driven: SDL's stock 360 map turns
BTN_SELECT into **Guide** (the runtime opens the settings menu on Guide) and
BTN_START into leftstick.  MINUS therefore uses BTN_TL2 ("back"), PLUS uses
BTN_START with `map.STICK_L = none`, and ZL/ZR are analog only.

### controllers.ini change

Backup: `/tmp/opencode/ssbu_backups/20261101_232917_D_input/controllers.ini`
(sha256 `3ebcbd7a371b8885f47095a9f28fc203f7502726ef406b465a6ef4b887561d00`).
One section appended at the end, `[device 030049205e0400008e02000014010000]`,
mapping raw button indices 0..13 explicitly and disabling STICK_L/STICK_R.
No other controller settings changed.  Reason: the stock 360 mapping makes
south = Pro B (positional), TL2/TR2 = Back/Start and expects a hat for the
D-pad, so it cannot produce the canonical Pro buttons as named.  The section
only affects this GUID.

### Results

- `python3 tools/mem/test_gym_input.py` → **31 tests OK** (1 live test skipped).
- Live: `game_session.py run --owner D --no-reader -- ... --live` → **35 tests
  OK**; output shows the GUID/name above and every canonical button on
  exactly its Pro button, no spurious STICK_L/MINUS/PLUS; sticks/C-stick/trigger
  as listed above.  Padding is 0 before the pad appears and after it disappears.
- In-game: one valid IN_MATCH run moved P1 from x 70.00 to 81.70 (+11.70) while
  holding right, and another moved x -2.10 → -38.23 (-36.13) while holding left
  over 96 frames.  Later runs were inconclusive because the saved match rule is
  **stock 0** and the fight goes to RESULTS ~500 frames after ready-go; the
  shared raw control fields (`buttons/stick_x/...`, bits 24-29) were all zero in
  those frames.  Re-run in-game motion once the coordinator fixes the saved
  roster/rule (stock 3, target CPU levels).

### Focus behavior (exact)

`runtime/nnhid.c`: `devices_live() = g_x_synth || g_pad_test ||
keyboard_ours()`.  `keyboard_ours()` is 0 for scripted runs (`SSBU_SEQ`/
`SSBU_PLAY` without `SSBU_SEQ_KEYBOARD`), otherwise it is 1 only when
`GetForegroundWindow()` is this process's window, which on Linux is
`plat_game_window && SDL_GetKeyboardFocus() == plat_game_window`
(`runtime/plat_posix.c:1721`).  So **real pads and the keyboard are ignored
unless the game's SDL window has keyboard focus**; both are zeroed, not just
the keyboard.  `--pad-test` sets `g_pad_test`, so it is always live.

To give focus use `tools/mem/focus_game.sh` inside the session command, e.g.
`python3 tools/mem/game_session.py run --owner D -- sh -c
'tools/mem/focus_game.sh; python3 tools/mem/test_gym_input.py --live'`.
Caveat found live: `xdotool search --name 'SSBU native'` returns both the game
client window and mutter's decoration frame (`mutter-x11-frames`, a different
PID, often larger); stale frames from closed windows also persist.  Picking
`tail -1` can focus the decoration, after which SDL loses focus and input is
gated.  Select the client window whose `xdotool getwindowpid` equals the
`ssbu_native` PID.  Plain `xdotool windowfocus` does not always deliver the
FocusOut SDL needs (a real click does), which is why focus_game.sh clicks the
HUD and then the game.  The external focus keeper (pid 187449) uses the same
`tail -1` search and may focus a decoration frame; the coordinator may want to
tighten it to the game PID.

### Reset live test / blocker

Mock mode works (`ResetClient(mock=True)` and `python3 tools/mem/gym_reset.py
--mock`).  Against the rebuilt runtime (binary 00:04, contains
`ssbu_control.bin`) our request reached the block
(`reset_seq 5, ack 4, state requested`) but was **not acknowledged within 30 s**.
Cause: `scripts/run.py:156` strips every parent `SSBU_*` variable and
`game_session.py` passes only `SSBU_SKIP_MOVIES`, `SSBU_KEYBOARD_PLAYER`,
`SSBU_DEBUG_MATCH` via `play.sh --env`; `SSBU_RL=1` never reaches the game, so
`rl_init()` is never enabled (A's code is off unless `SSBU_RL > 0`).  The
previous block state 4 was from A's own run without a reset entry.

**Coordinator action needed:** add `"--env", "SSBU_RL=1"` to the play.sh launch
in `tools/mem/game_session.py` (or have A default the hook on; CONTRACTS says
"`SSBU_RL=0` disables the runtime hook", which reads as default-on, while
`runtime/debug_mode.c` is currently default-off).  Then rerun:
`python3 tools/mem/gym_reset.py --timeout 20` and, once fight_on is stable,
`python3 tools/mem/game_session.py run --owner D -- .venv/bin/python
tools/mem/gym_reset.py --timeout 20`.  No live 20/20 reset result is claimed
yet; the client side is ready.


## Agent E — recording / behavior cloning

Delivered (owned files only):

- `tools/mem/gym_record.py` — observer-mode 60 Hz recorder (v7 reader, one
  `.npz` per episode + atomic `manifest.json`, crash-safe partials,
  resume/skip, optional reset through `gym_reset`/`control_block`).
- `tools/mem/gym_bc.py` — GRU + action-chunk BC (dataset builder, mirror
  augmentation, normalization, threshold calibration, temporal ensemble,
  training CLI, tiny-batch overfit).
- `tools/mem/gym_eval.py` — checkpoint → policy → `SSBUGymEnv` or mock; wins/
  losses/damage/lengths + action distribution vs expert.
- `tools/mem/test_gym_bc.py` — 31 synthetic unit tests (no game, no lock).

### Recorder

- Layout v7 is duplicated in `gym_record.py` (`HEADER_FMT_V7`,
  `FIGHTER_FMT_V7`, 80+2×128 = 336 B) and cross-checked against B's
  `schema.py` by a unit test (`test_layout_matches_agent_b_schema`). A v6
  224 B block is still readable best-effort (18 fields only, control arrays
  zero) so a recorder started before B's v7 launch does not crash.
- Rows are the reader's published frames, one per new even `seq`; the recorder
  polls at 2×`--hz` (120 Hz for `--hz 60`) and de-duplicates by `seq`, so it
  sees all 60 Hz frames without aliasing. `--hz` is the polling/rate hint, the
  recorded rate is the reader's.
- Episode lifecycle: `fight_on` 0→1 starts, 1→0 saves; each `.npz` gets
  `seq, ts_ns, raw_fighters (N,2,32) float32, buttons_raw (N,2) uint32,
  buttons (N,2,12) uint8` (decoded canonical order, per the contract's
  "raw mask + decoded vector"), `stick/cstick (N,2,2), trigger (N,2),
  frame_skip=5` plus `meta_json`/`meta` (JSON string — npz cannot hold a dict
  without pickle), `result_json`, `stage_id`, `game_frame`, `bounds`,
  `seq_gaps`. `manifest.json` has the same per-episode metadata plus frame
  counts/seq range/winner guess.
- `raw_fighters` stores each 32-bit word as float32 (all values used are
  exactly representable); `buttons_raw` preserves the exact mask.
- Crash safety: partial file `episode_NNNN.partial.npz` every
  `--flush-every` frames (default 300); final save is temp-file + `os.replace`;
  manifest writes are atomic. A restart counts existing finals toward
  `--episodes`, skips them, and either resumes a partial (fight still running,
  seq gap ≤ `--resume-window` 5 s) or finalises it as an episode.
- Reset is lazy and optional: `load_reset_fn()` tries `gym_reset` first, then
  `control_block`. With D's files present it selects
  `gym_reset.ResetClient().reset`, which blocks for ack + `fight_on` stable
  frames (best fit for the recorder); `control_block.request_reset` is the
  fallback; `--no-reset` disables. Failed resets are logged at most 3×/run.
- The recorder takes `/tmp/ssbu_game.lock` unless
  `SSBU_GAME_LOCK_HELD=1` (set by `game_session.py`) or `--no-lock`; if the
  lock is busy it exits 3 instead of recording unlocked.
- `raw_fighters` slot order is player order (no reordering). `stage` defaults
  to `final_destination` (`--stage`, `SSBU_STAGE`); `stage_id` is stored raw.

### BC pipeline

- Resampling is every 5th reader frame (12 Hz). Window at decision `t` =
  observations `t-T+1..t` (T=8) followed by the previous action `a[t-1]`;
  target chunk = `a[t..t+H-1]` (H=4). Windows never cross an episode
  (`window_samples` is per episode; a test tags episodes and checks it).
- `window_samples` is the single alignment primitive used by dataset building,
  training, metrics and the unit tests; it returns `(N,T,100)`, `(N,prev_dim)`
  and target buttons/stick-bin/cstick-bin/trigger arrays.
- Observations exactly match C's `gym_obs.PolicyObservation`: 14 scalars,
  curated 32-entry `action_state` one-hot (31 states + OTHER), dx/dy/distance/
  facing_rel; unreadable slot = all zeros. `gym_bc` imports
  `gym_obs.SCALAR_FIELDS`/`RELATIVE_FIELDS`/`CURATED_ACTION_STATES` when present
  (fallback copies the same table), and `test_matches_agent_c_profile` compares
  raw and divisor modes against C's builder bit-for-bit.
- `--obs-normalize` selects C's divisor mode; **default is C's default**
  (`normalize_obs=False`, raw physical values) and the checkpoint stores
  `config.obs_normalize`; `gym_eval` constructs `SSBUGymEnv` with the same
  value. Learned per-feature mean/std are in the checkpoint.
- Expert actions are slot `--player` (default 0 = P1), observations contain
  both fighters. Raw button masks decode with B's evidence
  (`A 0x1 B 0x2 X 0x4 Y 0x8 L 0x20 R 0x400 ZL 0x1000 ZR 0x2000 PLUS 0x4000
  MINUS 0x10 DPAD_L 0x40 DPAD_R 0x80`; imported from `schema.BUTTON_BITS`).
  Important: these are not `1<<i`, so a `1<<i` decoder would misread R/ZL/ZR/
  PLUS/MINUS/D-pad.
- Model: GRU(hidden 256) → MLP(hidden) → heads for `H×12` button logits
  (BCEWithLogits with clipped `neg/pos` pos_weight ≤ 20), `H×81` stick heatmap
  (cross-entropy; soft-argmax at inference), optional `H×81` C-stick and `H×1`
  trigger (BCE). Loss = button BCE + stick CE + 0.5·cstick CE + 0.5·trigger
  BCE, AdamW lr 1e-3, cosine schedule, grad clip 5.
- Mirror augmentation is applied per episode (obs and actions together), then
  windows/targets are rebuilt, so alignment is exact. Mirrored: x, vx, facing,
  dx, stick_x, C-stick_x negated; D-pad L/R swapped; everything else unchanged.
  An involution test checks double-mirror equals the original.
- Episode-level deterministic 80/10/10 split by `--seed` (n=3 → 1/1/1, n=2 →
  train 1 / val+test 1). Train stats only feed normalization; val chooses the
  best checkpoint; val probabilities calibrate one threshold per button
  (`quantile(1 - expert_rate)`) so predicted positive rates match the expert;
  test is untouched.
- Inference: `TemporalEnsemble(policy, execute=2, blend=0.5)` plans H=4,
  executes the first 2, then re-plans and averages the overlapping tail
  (temporal ensemble). `BCPolicy` keeps the T-observation history and the last
  executed action internally; call `reset()` between episodes. `--execute` and
  `--no-blend` control it.
- Env action form: `format_action("auto")` returns C's Dict sample
  (`buttons`/`stick`/`cstick`, `trigger` extra and ignored by
  `gym_action.decode_action`); the flat form is the 16-vector
  `[buttons(12), stick(2), cstick(2)]` (C's decoder rejects longer vectors).
- CPU threads: small batches (≤128) default to 1 intra-op thread because the
  container oversubscribes badly (3 epochs on 5 synthetic episodes: 41 s at 8
  threads vs 3 s at 1); `--threads N` overrides. The tiny overfit does the same.
- Checkpoint (`ssbu_bc_v1`): config (window/horizon/frame_skip/hidden/bins/
  cstick/trigger/player/obs_normalize), `state_dict`, observation mean/std,
  per-button thresholds, pos_weight, train/val/test metrics, expert action
  stats, split, alignment, data/build metadata. Round-trip and deterministic
  outputs are tested.

### Exact commands and results (synthetic; no game needed)

```
cd /home/daniel/Projects/SSBU/ssbu-recomp
.venv/bin/python tools/mem/test_gym_bc.py
# Ran 31 tests in 1.48 s — OK (window/chunk alignment, no cross-episode
# windows, mirror involution, normalization, action decode with B's real bits,
# gym_obs compatibility, recorder v7/v6/torn/incremental/resume, checkpoint
# round-trip, temporal ensemble, early stopping restore, shield-from-status,
# real SSBUGymEnv+ShmFixture closed loop)

.venv/bin/python tools/mem/gym_bc.py --overfit --overfit-steps 150 --json
# loss 0.00139, mean_f1 1.000, mean_f1_active 1.000, stick_acc 1.000,
# action_rate_divergence 0.000

.venv/bin/python tools/mem/gym_bc.py --data /tmp/opencode/agent_E/synth_data \
    --epochs 5 --batch 16 --out /tmp/opencode/agent_E/bc_synth.pt
# 5×420-frame synthetic episodes, 5 epochs in 7.4 s
# train/val/test F1 ≈ 0.997/0.996/0.996, alignment ok (seq_gaps 0)

.venv/bin/python tools/mem/gym_eval.py --model /tmp/opencode/agent_E/bc_synth.pt \
    --mock --episodes 3 --max-steps 60 --json
# 3/3 mock wins, mean length 16.7; button-rate divergence 0.057 (mock obs is
# deliberately out of the expert distribution; this is a plumbing smoke test)

# recorder against a synthetic v7 block written at 60 Hz (fake_writer.py):
.venv/bin/python tools/mem/gym_record.py --shm /tmp/opencode/agent_E/fake_shm.bin \
    --out /tmp/opencode/agent_E/rec_out --episodes 2 --no-reset --no-lock
# episode 0: 76 frames, seq_gaps 0; episode 1: 90 frames, seq_gaps 0;
# meta p1=mario p2=donkey_kong stage=final_destination; npz shapes
# (N,2,32)/(N,2)/(N,2,2), frame_skip=5
```

Live recorder/BC/eval runs are not in this section: `/dev/shm/ssbu_state.bin`
was still B's v6 224 B block while E worked, so the game-path acceptance run is
the coordinator's. The adapters above are exercised against C's real
`SSBUGymEnv` + `ShmFixture` (Dict action space, 5-tuple step, `stocks`/
`damages` info), not just a hand-written mock.

Interface assumptions worth checking in the live run:

1. The recorder assumes B publishes v7 slot 0 = P1/entry 0 (contract) and that
   `reader.py` bumps `seq` by 2 per frame. Its manifest `hz` is 60.
2. Reset uses `gym_reset.ResetClient().reset()` (blocking, waits for
   `fight_on`); if D changes that API, the fallback is
   `control_block.request_reset`, and `--no-reset` always works.
3. BC uses C's `normalize_obs=False` observation values by default. If the
   coordinator changes the env default, train with `--obs-normalize` (or update
   the checkpoint config) so both sides match.
4. The cloned player is slot 0 by default (`--player`); recordings include both
   slots, so P2 cloning only needs the flag.
5. Button bits are B's ControlModule positions, imported from
   `schema.BUTTON_BITS`; if that changes, `BUTTON_BITS` and checkpoints trained
   with the old bits must be regenerated.

### Follow-up: early stopping, regularization, shield-from-status (2026-11-02)

**Early stopping / regularization.** `gym_bc.py` now stops on validation loss:

- `--patience N` (default **5**; `0` disables) and `--min-epochs N`
  (default **1**). The model keeps the best-val state; on stop it restores that
  state, and the saved checkpoint is the best-val state (not the last epoch).
- `--weight-decay` (default `1e-4`) is now a CLI parameter; `--dropout`
  (default `0`) inserts dropout in the two trunk layers and is stored in the
  checkpoint config. Eval runs `net.eval()`, so inference stays deterministic.
- New checkpoint/metrics fields: `best_epoch`, `best_val_loss`, `epochs_run`,
  `stopped_early`, `patience`, `min_epochs`, `weight_decay`, `dropout`; the
  existing `train`/`val`/`test` metrics are computed with the restored
  best-val weights. The CLI summary prints all of them.
- `test_early_stopping_restores_best_weights` mocks `_eval_loss` with the
  rising curve `[1.0, 0.5, 0.8, 1.1, 1.4]`, patience 2: it asserts stop after
  epoch 4, `best_epoch == 2`, `best_val_loss == 0.5`, and that the saved
  `state_dict` equals the state captured at the best epoch.

**Shield-from-status (dataset builder, default ON).**

B's live evidence (their section above) agrees with the finding: guard frames
are status kind **10** (the contract family is 8/9/10), the CPU mask is
`0x8000`/`0x8004`, and `0x4` decodes as **X** under B's button table.  The CPU
mask is the AI's virtual button state, not the physical pad; `trigger` stays
~0.0001 (max 0.0313) and never carries shield.  So the training target, and
only the target, is corrected:

- For the cloned player only, when `action_state` is 8/9/10:
  canonical `ZL=1`, `X=0`; stick, C-stick, trigger and every other button are
  unchanged.  `--no-shield-from-status` disables the relabel.
- The raw mask (`buttons_raw`), the raw slot (`raw_fighters[...,24]`) and the
  npz's decoded `buttons` array remain exactly as recorded; the correction is
  applied when `EpisodeData.acts` is built and flows through previous-action
  vectors and action chunks.  `EpisodeData.shield_from_status` preserves the
  choice through resampling and mirroring.
- Checkpoint/JSON: `config.shield_from_status` plus a top-level
  `shield_from_status: true`, and `shield_labels` with per-split
  `frames`/`by_state`/`spurious_x_cleared`.
- On the 10-episode dataset (seed-0 split, 12 Hz decisions): **52 guard
  decisions** total — train 42, val 4, test 6 — all status **10**; 35 carried
  the spurious X bit (train 29).  Dataset button rates before -> after:
  X 0.1028 -> 0.0865, ZL 0.0000 -> 0.0233 (A/B/Y unchanged).  The remaining
  non-guard `0x8004` frames (176 in the resampled data) are left as X.

**`gym_eval.py`.** `BCPolicy` restores the checkpoint config (window, horizon,
frame_skip, obs_normalize, trigger, cstick, dropout, shield_from_status); the
env is built with that `frame_skip` and `normalize_obs`.  `--cstick` is
accepted (it verifies the checkpoint has the C-stick head; the head itself is
restored from the checkpoint), and the restored config is echoed in the JSON
`config` object.

**Command and result (10-episode Mario-vs-DK dataset):**

```
.venv/bin/python tools/mem/gym_bc.py \
    --data tools/mem/test_artifacts/accept/mario_vs_dk \
    --out data/E/bc_mario_vs_dk_cstick.pt \
    --cstick --patience 5 --epochs 30 --json

# 10 episodes: train 8, val 1, test 1; 3170/196/187 samples
# shield-from-status: 52 guard decisions relabeled ZL=1/X=0
#   (train 42, 29 spurious X)
# best_epoch 2, best_val_loss 1.6519, epochs_run 7, stopped_early true
# train: mean_f1 0.7045, mean_f1_active 0.2907, stick_acc 0.7164, div 0.0099
# val:   mean_f1 0.7470, mean_f1_active 0.2411, stick_acc 0.5867, div 0.0000
# test:  mean_f1 0.7685, mean_f1_active 0.3056, stick_acc 0.5187,
#        cstick_acc 1.0000 (no C-stick labels), mean_rate_divergence 0.0036
#        ZL F1 0.889, X F1 0.333, trigger_bce 0.0001
# shield labels: total 52 (train 42, val 4, test 6), all status 10,
#                35 spurious X cleared (train 29)
```

`best_epoch 2` is the same selection point as the pre-shield run on this split
(val 1.5834 then); the target distribution changed (ZL added, X reduced), so
the absolute val numbers are not directly comparable.  Early stopping prevents
the epoch-20 collapse (val 3.67 in the original report); held-out action F1 is
still limited by 8 training episodes (X/ZL are the only active buttons, stick
accuracy ~0.52).  Conservative follow-ups, not run here: `--dropout 0.1
--weight-decay 1e-3`, and/or more episodes.  `test_gym_bc.py` now runs **38
tests, OK (~1.5 s)** including the early-stop and shield tests.

## Agent F — extra memory

- Extras are a separate self-describing block: `/dev/shm/ssbu_extra.bin`
  (`SSBX` v1, 32-byte header + records: 48-byte name, type, verified, index,
  value; odd/even seq).  Format and every offset are in
  `tools/mem/extra_map.json`; narrative in `FINDINGS_EXTRA.md`; reader/writer
  and decoder in `tools/mem/extra_state.py`; tests in `test_extra.py`.
- New modules beyond the core bridge: `ShieldModule=*(accessor+0x100)`,
  `StopModule=*(accessor+0x90)`, `HitModule=*(accessor+0xB0)`, plus
  Damage/Ground/Kinetic/Motion/Work and FighterInformation extras.
- Live-verified: shield part array/count (5), part size/life/attack/speed,
  bubble node size, shield-active flag; damage amount/power_max/reaction and
  damage-lock; hitlag (`StopModule::get_hit_stop_real_frame`, observed
  19→12→6→0 and 4 on a shield hit); engine velocity (kinetic entry vectors
  summed, non-zero and tracking movement); motion kind = Hash40
  (`hash40("wait")`), frame/rate/prev/end; full 256-byte Work transition-term
  table + 256-bit mask, work int/float arrays and the command/cancel flag
  array; FighterInformation stock/dead/rebirth/category/gravity/damage-log
  pointers; BattleObjectManager pools (two active fighter slots, ids/entry/
  accessor/pos); match rule/stock/stage id, RNG state, stage object and camera
  pos/target/range/dead area/fov.
- Unknown/partial: absolute shield HP (no accessor; raw part/node blobs are
  published), numeric cliff id, true contact distance to floor, full
  battle-object owner/category layout, match timer/platform geometry, and a
  guest-side frame counter.  These are listed in `extra_map.json` →
  `unverified`.

## Coordinator — live state verification (2026-11-14)

New tool: `src/rl/scripts/verify_live_state.py` (`python -m
rl.scripts.verify_live_state`). It samples the v7 block at 60 Hz and checks
every field two ways: (a) field-to-field invariants that must hold on every
frame, (b) known input -> known state, in three modes:

* `physical` — P1 human + uinput pad; pad script (walk, jump, shield, attack,
  special, C-stick, trigger). Requires SDL keyboard focus.
* `cpu` — observe a CPU-vs-CPU fight; validates the expert control fields and
  the lifetime/rule header.
* `seq` — `SSBU_XINPUT_SYNTH=1` + `SSBU_SEQ=<text>`; the recomp's synthetic pad
  drives P1 from a scripted sequence, independent of window focus. This is the
  mode that actually worked in this environment.

### Verified (seq run: synthetic pad, known keys)

`@draws:206-450/60 280` boot part of `tests/seq_battle_play.txt`, then
`r/80, l/80, C, C, A, Z, X, V`. 1340 sampled frames, 900+ live:

* P1 walked right then left: x range `[-144.9, 76.2]`, facing flipped `+1/-1`.
* jump: y `0 -> 52.6`, `on_ground` 1->0, `jumps_left` 2->1, status 11/12.
* shield: status **10**, `shield=1`, 23 frames.
* attacks: statuses `{24, 27, 28, 29, 34, 45}`; special: `{41}`.
* expert fields follow the pad: masks `{0x8000, 0x8004, 0x8008, 0x8201,
  0x8802}` (0x1/0x2/0x4/0x8 = A/B/X/Y), `stick_x/y` reach +-1.
* `dx/dy` = opponent - self (max err 6e-6), `facing_rel` 100 % consistent,
  `dist_to_blast` recomputes from the live camera bounds, `vx/vy` = d(pos)/dt.
* `shield <=> status 10` and `hitstun <=> DAMAGE statuses` hold for both slots.
* no NaN/Inf; `raw_flags=0x7fffffff`.

### Verified (cpu run after an R2 reset)

Clean start (damage 0/0), Mario/DK, `is_cpu` 1/1, 1200 sampled frames:
expert buttons populated **every** frame, `|stick|` reaches 1.0, trigger max
0.0079, `stick_x` sign agrees with `vx` in 94 % of grounded frames,
`is_actionable=1` in 100 % of WAIT frames and 0 in 95 % of hitstun/tumble.

### Corrections to earlier notes

* **The ControlModule mask is not "AI-only".** A synthetic pad's face buttons
  and sticks appear in `buttons`/`stick_x/y` exactly as pressed. The earlier
  "P1 human + pad reads 0" was an **input-delivery** failure, not a bridge
  property: pads and keyboard are gated on `SDL_GetKeyboardFocus()` (see
  `runtime/nnhid.c:keyboard_ours`), and X focus + `_NET_ACTIVE_WINDOW` being
  the game is *not* enough. After a GNOME lock/unlock the SDL flag stayed
  unset even with a focus cycle and a real click; the keyboard failed
  identically. `--shield-from-status` is still right for **ZL/ZR**, because
  the virtual pad drives them as analog triggers (ABS_Z/RZ) and they do not
  appear as mask bits.
* **The virtual pad was not broken.** In the game's menus it dismissed an
  unlock popup and navigated the Challenges screen (A). The match failures
  were entirely the focus gate.
* **`rule_kind=1 (stock), stock=0`** confirmed live from the extra block
  (`g.match.rule_kind`, `g.match.stock`).
* **`SSBU_SEQ` is inline text, not a file.** `scripts/run.py --seq` reads the
  file, drops `#` comments and sets `SSBU_SEQ` to the joined text; the
  synthetic pad needs `SSBU_XINPUT_SYNTH=1`. Passing a path yields
  `input sequence: bad step <path>` and zero steps.

### New caveats found

* **Stale rows at `fight_on` 1.** `impl+0xd2` goes to 1 a few frames before
  the entries hold the new match; the first ~15 frames can still carry the
  previous match's `kind`/`damage` (observed 5/37, 51/50, 60/30 and kind 8).
  `wait_live` now requires the `(kind, entry_id)` tuple to be stable for 20
  new frames. The R2 reset zeroes damage (`DamageModule__init_damage_impl`),
  which is why the recorded datasets start at 0 %.
* **`action_frame` resets to 0** when a new action instance reuses the same
  status kind (observed in 54, 72, 119). This is a reset, not a torn read; the
  invariant now allows a reset to <=1 and only flags other decreases.
* **Sim rate under GPU load.** With the Isaac Sim job holding ~7.6/8 GB of the
  RTX 3070, the guest runs at ~15 fps (`action_frame` +0.25 per 60 Hz sample,
  long stalls), the renderer showed stale/white frames, and one launch died
  with `VK_ERROR_OUT_OF_DEVICE_MEMORY`. Wall-clock timing tests are invalid
  under that load; use `resolution_scale=0.5`, `shader_load=loading`,
  `shader_prepare=off` (backed up from `graphics.ini`) as a stopgap.
* **15 s episodes are still not explained.** The reset rematch ran >20 s wall
  (>300 sim frames at 15 fps) with no `fight_on` transition, so 900 frames is
  not a hard timer. The recorded 901-923-frame lengths are consistent with a
  first KO at 60 fps, but their last frames carry no KO/DEAD state and damage
  is not reset. Check on a healthy GPU (or decode the match block's time
  field) before treating them as natural terminal states.

### Still unverified live

`cstick_x/y` (CPUs never set it and the keyboard sequence has no C-stick),
`shield_size` (no source, always 0), `game_frame` (reader counter), and
`stock`'s non-zero value.

### Harness fixes in this pass

* `rl/memory/reader.py`: `REPO` resolved to `ssbu-rl/src` after the move, so
  `reader.py --spawn` could not find `play.sh`; it is now
  `$SSBU_RECOMP_DIR` or the sibling `ssbu-recomp`.
* `rl/utils/game_session.py`: `SSBU_KEYBOARD_PLAYER` is taken from the
  environment (was hardcoded 0) and a repeatable `--env KEY=VALUE` is added
  for the game (used for `SSBU_SEQ`/`SSBU_XINPUT_SYNTH`).

## Coordinator — live stage view (2026-11-14)

`src/rl/scripts/live_view.py` is the new second window: a data-only 2D view of
the match.  It replaces `rl.tests.verify_gui` in `src/rl/scripts/live.sh`
(`verify_gui.py` stays as the text-only fallback).

* Canvas: blast-zone box, a **Final Destination outline assumed** from the
  state (platform at world y=0, x -100..100), a 50-unit grid, one box per
  fighter (damage-tinted, facing wedge, shield bubble, position trail,
  `P1 MARIO 41%` + state badge like `AIRBORNE`/`HITSTUN`/`SHIELD`), and one
  marker per battle object from `bom.obj.*`.
* Side panel: every v7 field for both slots (decoded buttons, state names,
  raw hex toggle), match/rule header, per-fighter extras (stock/dead, shield
  parts, hitlag, kinetic speed, floor y), and a bounded event log
  (`P1 WAIT -> JUMP_SQUAT`, `fight_on 0 -> 1`, shield/hitstun on/off).
* Sources: `/dev/shm/ssbu_state.bin` at 60 Hz; `--extra auto` prefers
  `/dev/shm/ssbu_extra.bin` and otherwise samples `ExtraState` directly at
  12 Hz (so projectiles work with no extra reader); `--extra off` hides them.
* `--mock` animates two fighters and a projectile with no game;
  `--mock --dump-once --json` prints one model headlessly.
* Tests: `test_live_view.py` (15 tests: world map, state labels, extra object
  filtering, fighter-accessor exclusion, mock source, events, fake v7 block
  through `ShmSource`, CLI dump).  Full suite: 163 passed, 5 skipped.
* Not drawn: hitboxes (no accessor) and true platform geometry (FD assumed).
  `bom.obj` fighters are filtered by matching `bom.obj.accessor` against
  `f.accessor` slot 1/2.

## Build incident — never relink the game .so from a cold object cache (2026-11-14)

While adding a stock override to `runtime/debug_mode.c`, `build.py link` was
run to rebuild only the host binary.  Its first step, `build_game_dll()`,
checks whether the `.so` is newer than everything in `obj_eh_o1`.  That
directory held **no `.obj` files** (only 829 interrupted `.tmp` files and one
stray object), so the check failed and the ~553 MB
`ssbu_game_eh_o1.so` was relinked from the data objects alone: a 3 MB library
that overwrote the working one.  No `.so` backup existed (only `ssbu_native`
had been copied).  This is the trap Agent A warned about in the final-status
notes; the warning was accurate.

Recovery: the lifted sources are intact (`build-linux/work/gen_eh_o2`, 1.3 GB,
1161 files), so the canonical
`deps/venv/bin/python scripts/setup.py --target linux --dump /home/daniel/Projects/SSBU/dump`
recompiles all game objects and relinks (the slow files are first; tens of
minutes).  After a good build, **back up both**
`build-linux/work/ssbu_game_eh_o1.so` and `build-linux/work/ssbu_native`.

The host-side link also needed `-Wl,--allow-shlib-undefined` while the old
.object cache was missing; after the full rebuild the `.so` defines its lifted
symbols and the stock build.py line should work unchanged (the temporary
`build-linux/work/build.py` patch is backed up at `/tmp/opencode/build.py.bak`).

Pending verification: the `SSBU_DEBUG_MATCH stock=<n>` override now writes the
match block `+0x05` (rule kind = stock) / `+0x0A` (count) and both
FighterInformation records `+0xD8` at ready-go.  Whether the game then runs a
real N-stock match (instead of the saved 0-stock first-KO match) is unconfirmed
until the rebuilt binary boots.

Tool for the future: `src/rl/scripts/host_link.py` (`rl-host-link`).  It stages
`runtime/<file>.c` into `build-linux/work/`, backs up the current
`ssbu_native`, compiles only the stale host objects, and links the exe against
the **existing** `ssbu_game_eh_o1.so` with `build_game_dll()` monkeypatched
out.  It refuses when a full build is running or when the `.so` is <100 MB,
takes `/tmp/ssbu_build.lock`, and restores the old exe on failure.  Use it for
any `debug_mode.c` / `nnhid.c` / `display.c` / header change instead of
`build.py link`; see "Host-only rebuild" in `MIGRATION.md`.

## uinput virtual pad: event-node permissions (2026-11-14)

The virtual pad was being injected correctly but the game could not read it.
`VirtualPad` writes to `/dev/uinput`; the kernel creates the corresponding
`/dev/input/eventN` as **root:root 0600**.  SDL still enumerates the device and
the game logs `nn::pad: connected "Xbox 360 Controller" ... -> player 1`, but
every `read()` on the node fails, so the pad state is permanently zero.

`VirtualPad.create()` now calls `_grant_read()` on the node: it grants the
current uid a read ACL (`setfacl`, through passwordless `sudo -n` when the node
is root-owned; otherwise a clear warning with the exact command).  Verified:

* reading the node shows `EV_KEY BTN_SOUTH 1/0` and `EV_ABS ABS_X 32767`;
* `test_gym_input.py --live` is **35/35**: SDL 3.2.6 sees GUID
  `030049205e0400008e02000014010000`, every canonical button lands on its Pro
  button, sticks/C-stick/trigger all read.

Host rebuilds now refuse while the game is running (`ETXTBSY`: the running
executable cannot be replaced); stop the game first, link, restart.  That guard
is in `rl.scripts.host_link` and documented in MIGRATION.md.

Still open: the **in-game** P1 test (`p1=human`) still shows `dx=0` even with a
real `focus_game.sh` click.  The pad-test proves the SDL/uinput side, so the
remaining gate is the game's `player_connected`/`fill_pad` path or the debug
match's controller/save roster.  `SSBU_RL_INPUT_DEBUG=1` (nnhid.c) prints
`input dbg: id0 ...` with `connected`, `styles`, `idok`, `nnpad_present`, the
nnpad button/stick state and the final output state; it is in the current host
binary for the next clean boot.

## uinput in the game works: the earlier `dx=0` was the harness (2026-11-14)

The "P1 does not move" result was not an input-gate problem.  Two harness
mistakes produced it:

* **The game was still in its menus.**  `SSBU_DEBUG_MATCH` forces the select
  screens and the rules, but nothing walks from the title screen to the
  character select.  A run without a menu walk sits in `match=MENU`
  (`fight_on=0`, no fighters) forever; taps every second are not enough
  (`live_checks focus` used to do only that).
* The ad-hoc probe accepted the **first** `fight_on=1` block, which could be
  the previous run's last frame or a match already near its end, and then read
  `dx=0` after the match was over.

The working headless recipe (all inside one `game_session run`):

```sh
SEQ="$(grep -v '^#' seq_nav.txt | tr '\n' ' ')"   # menu walk, no battle inputs
.venv/bin/python -m rl.utils.game_session run --owner x --restart-game \
  --restart-reader --keep-game \
  --match 'p1=mario,p1=human,p2=donkey,p2=cpu,cpu=1,stage=end,players=2,items=off,stock=3' \
  --env SSBU_INPUT_NO_FOCUS=1 --env "SSBU_SEQ=$SEQ" -- \
  bash -lc '.venv/bin/python -m rl.scripts.wait_live 150 && \
            .venv/bin/python -m rl.scripts.live_checks focus --json --hold 2'
```

`seq_nav.txt` is `tests/seq_battle_play_fast.txt` cut after the "ready to
fight" press: it walks boot -> Smash -> rules -> stage -> character select with
the recomp's scripted pad and then leaves P1 alone.  With
`SSBU_INPUT_NO_FOCUS=1`, `devices_live()` is 1 even in a scripted run, so the
SDL/uinput pad is still merged into P1's state after the walk.

Verified on the current host binary:

* `live_checks focus` **passes**: 121 live frames, `dx=153.67`, the window was
  **not** focused (`X11 focus is on ... not on the game window`), stocks 3/3,
  `is_cpu=[0,1]`.
* The same hold through a bare `pad_verify.py` gave `dx=154.88`, action states
  0/4/7/14 (idle/walk/run), with `input dbg: out id=0 kind=1 b=40000
  lx=32767`.

One run failed with `dx=0` and no nnpad "connected" line at all: the game's
watchdog shows a **1-2.4 s stall at the ready-go** (shader pipelines), and the
fixed 2 s hold fell entirely inside it, so the game's input poll never ran and
the pad was destroyed before it resumed.  `hold_right_and_collect` now holds
until P1's state has shown the stick/buttons for `--hold` seconds, with
`DEFAULT_HOLD_GRACE` (4 s) of extra time for such a stall, and
`evaluate_focus_motion` fails with "the game received the pad input" when the
expert `buttons`/`stick_x` stay 0 the whole time.  (Repeated live runs on this
desktop later hit unrelated `VK_ERROR_OUT_OF_DEVICE_MEMORY` on llvmpipe and a
`BadRROutput` X error after a monitor/session change; those are environmental.)

`live_checks focus` now skips its A taps when `SSBU_SEQ`/`SSBU_PLAY` is set
(`scripted_menus()`): an extra A during the scripted walk skips a menu item and
loses the anchors, which is why the check previously timed out even with the
walk running.  It still requires a **live** match; start the walk first (the
recipe above) or start a match by hand.

`SSBU_RL_INPUT_DEBUG=1` (nnhid.c) now prints one throttled line per npad id
(`input dbg: poll id=... kind=... conn=... got=...`) and, when the state is
non-zero, the final output (`input dbg: out id=... b=... lx=...`); it is
environment-gated and stays in the host binary.

## Characters from code, and why the state view said Mario/DK (2026-11-14)

`SSBU_DEBUG_MATCH` already accepts `p1=<fighter>` / `p2=<fighter>` and forces
`ui_chara_<name>` in the select-screen panel at every match set-up
(`runtime/debug_mode.c`, `panel_copy_hook` / `match_player_hook`).  A live run
with `p1=samus,p1=human,p2=kirby,p2=cpu,cpu=9` really spawned **Samus (kind 3)
vs Kirby (kind 6)**; the runtime's fight-on line said so and the game window
showed them.

The state block still reported `kind 0` (Mario) and `kind 8` (Donkey Kong).
`gamestate.py` read the kind from the fighter **information record**
(`entry+0xF8`, `+0x30` = kind+1).  That record is the select-screen copy: when
the debug hook forces a character by name it still carries the saved roster.
The match's own player record (13.0.4 match block at guest `0x71052c4730`,
record n at `+0x260*n`, **+8 = fighter kind**) has the kind that actually
spawned.  `read_fighter` now prefers that byte and falls back to the
information record.  After the fix the reader/viewer show `kind 3 / 6`
(SAMUS / KIRBY) for the same match.

`fighter_kind_by_name()` (the numeric kind used only in the parse log) looks
for `..\reference\skyline-smash\src\lua_const.rs`, a Windows path that does not
exist here, so the launch log prints `p1 kind -1` even though the name is
applied.  The name is what matters; the log is cosmetic.

### Viewer "hang"

`live_view` was fine, but two things kept it on a frozen picture:

* `game_session --keep-game` left the game running and stopped the reader it
  had started, so `/dev/shm/ssbu_state.bin` never advanced.  `--keep-game` now
  keeps the reader too.
* `RawSnapshotReader.read_latest()` keeps returning the newest complete frame,
  even when it is hours old.  The viewer re-rendered that same `seq` at 60 Hz
  (a frozen picture, ~13% CPU) and never said anything was wrong.  `live_view`
  now skips repeated `seq`s and shows
  `STALE: last frame N.Ns old — is rl.memory.reader running?`.

Character selection from code is documented in `MIGRATION.md`; the helper is
`rl.utils.match_config` (`MatchSpec`, `write_match_list`, `append_match`).

## Turbo proof: the state hash works, and 4x is not simulation-identical yet (2026-11-14)

The v2 control block publishes `game_frame` + `state_hash` every presented
frame, and `rl.scripts.turbo_check` records/compares them.  The checker was
hardened for real runs:

* it records only `fight_on` frames (boot/menu frames have different relative
  positions in two launches and misaligned every comparison);
* the runtime now puts the game pid's low 24 bits in `flags` bits 8..31;
  `record` rejects a block left by another process and restarts if the epoch
  changes, so a stale block cannot poison a run;
* it accepts a sample only after `--stable-reads` agreeing reads (the v2 block
  has no seqlock; a torn read was the earlier "1 frame then fight_off");
* it fails if the first live frame has `seed=0` (no `SSBU_FIXED_SEED`) or if
  `--expect-speed` does not match the block (catches `SSBU_SPEED` not reaching
  the game);
* `--settle` skips the first frames after fight_on and `--end-grace` ignores
  short fight_on dropouts.

Verified with `SSBU_FIXED_SEED=12345`, the same nav sequence, Mario vs Kirby,
both human and idle:

* **1x vs 1x: 600/600 hashes match** -> the recorder and the hash are reliable,
  and two launches start from the same state in this setup.
* **1x vs 4x: first divergence at relative frame 80** (600 common frames).
  Since 1x vs 1x passes, this is caused by turbo, not launch noise: some part
  of the fighter simulation still follows wall-clock time rather than one step
  per presented frame.  `motion + 0x70` (action frame) is the first suspect.
* CPU-vs-CPU matches are **not** deterministic across launches even at 1x
  (divergence at relative frame 30): the CPU AI/script generators use a match
  seed that `SSBU_FIXED_SEED` (re-seeding the global generator at ready-go)
  does not fully control.  Use human/idle players, or a frame-synced pad, for
  turbo checks.

So the hash is in place and does its job; turbo is **not proven correct yet**.
The next step is to make `SSBU_SPEED` step the simulation identically (frame
delta, not wall-clock delta) and then rerun 1x vs 4x/8x.

## Fixed frame clock: 1x is deterministic, 4x still diverges (2026-11-14)

Why 1x vs 1x CPU-vs-CPU diverged: the CPU AI and the fight's timers read
`nn::os::GetSystemTick`, which returned the real monotonic clock.  Frame-time
jitter moved the AI's decisions to different frames between launches.  The
fixed frame clock (`SSBU_FIXED_CLOCK`, default with `SSBU_FIXED_SEED`/
`SSBU_SPEED`) makes every game frame exactly 1/60 s, and the v2 hash is now
sampled on the game thread (player 1's pad poll) instead of the presenting
thread.  With that, two 1x CPU-vs-CPU launches produce the **same 600-frame
hash sequence** (the recorder reports a one-frame phase anchor from the live
flag; the tool tolerates it with `--max-shift` and warns).

4x is **not yet simulation-identical**.  Human-vs-human idle, 1x vs 4x,
diverges at about relative frame 90, i.e. at the end of the countdown when the
battle becomes active, and stays different afterwards.  The divergence is
reproducible per speed (all 4x launches share the same 4x hashes), so it is a
speed-dependent input to the battle-start state, not launch noise.  The fixed
`GetSystemTick` and `GetCurrentTime` did not remove it; the remaining source is
another clock read at the battle start (or a value derived from the absolute
clock, whose origin still differs with boot time).  Next: make the frame
clock's origin fixed (not the real tick at the first poll) and/or trace the
remaining time reads at the countdown->battle transition.

To reproduce: human/idle match, `SSBU_FIXED_SEED=12345`, `SSBU_SPEED=1` and
`SSBU_SPEED=4`, `turbo_check record --settle 30 --expect-speed ...`, then
`turbo_check compare ... --max-shift 2`.

## 4x divergence is only the action frame at the GO transition (2026-11-14)

`SSBU_RL_HASHDUMP=<path>` now records every value that enters `rl_state_hash`
at the exact sampling point (live, entry presence, stock/cpu/rebirth, pos
x/y/z/facing, status kind, situation, motion frame, damage, RNG words) plus
the hash.  Comparing 1x and 4x human/idle at the first differing frame:

```
rel 89  mf1=0x423c0000 (47.0)  both, hashes equal
rel 90  mf1=0x42400000 (48.0)  1x
        mf1=0x00000000 ( 0.0)  4x   <- only differing field
rel 91  mf1=49.0 vs 1.0         <- still the only differing field
```

So at the battle-start/GO transition the 4x run **restarts P1's action**
(frame 0) while 1x continues the running idle action (frame 48).  Positions,
facing, status kind/situation, damage, stock, CPU/rebirth and all four RNG
words are identical on every frame; the hash cascade is entirely the action
frame.  This is not the frame clock (the scene-hook advance and the virtual
time-of-day did not change it); it is the GO transition's action reset, whose
timing differs between the speeds.  Next: find the GO timer that restarts the
action and make it frame-based, or decide explicitly whether the action frame
is presentation state for turbo purposes (the RL observation still contains
it, so an offset is not free).

Confirmed with the in-hash dump: the 4x action frame restarts every **48**
game frames (42, 90, 138, 186, 234) while the 1x loop is **192** frames
(43, 235).  The ratio is exactly the speed (192/4 = 48), so the action/animation
loop length is still derived from the real frame delta (16.7 ms at 1x,
4.17 ms at 4x), not from the fixed `GetSystemTick`/`GetCurrentTime` clock.
Positions, facing, status kind/situation, damage, stock, CPU/rebirth and the
four RNG words are identical on every frame; the hash cascade is entirely the
animation frame.  So the fighter/physics/AI simulation is turbo-correct; the
remaining speed-dependent input is the animation frame delta (a presentation
clock, most likely the nvn/display frame interval).  The next fix is to feed
that delta 1/60 per game frame, or to exclude the animation frame from a
gameplay-only turbo hash (the RL observation still includes it).

Result after the gameplay hash (2026-11-14):

* human/idle, 1x vs 4x, 300 frames: **300/300 gameplay hashes match**.  The
  fighter physics, status, damage, stock and RNG are turbo-correct.
* `state_hash` now excludes `motion+0x70` (the presentation action frame) by
  default; `SSBU_RL_HASH_FULL=1` includes it for animation research.  The
  action frame is the only field that differs (the 4x idle action loops every
  48 frames vs 192 at 1x); its source is still an unidentified presentation
  clock (not GetSystemTick/GetCurrentTime, the clock origin, the match seed,
  the keyboard, or the battle-scene hook).
* CPU-vs-CPU, 1x vs 4x: **0/300 hashes match even with the action frame
  excluded**, at every phase shift.  1x vs 1x CPU matches after the recorded
  phase anchor, so the CPU AI is deterministic at 1x but its decision timing
  changes with `SSBU_SPEED`.  The AI therefore uses yet another wall-clock
  dependent timer.  For training this is a stochastic-opponent difference, not
  a physics error; a bit-exact CPU opponent needs that timer virtualized too.

### Observation profile v1.1 — action-state family flags (IL step)

The 32-entry curated one-hot put roll/spot-dodge/air-dodge, tech, ledge
options, grabs/throws, landing lag and shield break into `OTHER`, so a policy
could not distinguish the defensive options that matter (especially for DK).
Six 0/1 families over the raw status id follow the relative block:

```
is_escape       78 ESCAPE_F, 79 ESCAPE_B, 80 ESCAPE_AIR
is_tech         71 PASSIVE, 72 PASSIVE_FB
is_ledge        73 CLIFF_CATCH, 74 CLIFF_WAIT, 75/76 CLIMB_*, 77 CLIFF_ATTACK
is_catch        43 THROW, 44-47 THROW_*, 48-53 CATCH*/THROW_CATCH
is_landing      84 LANDING, 85 LANDING_AIR, 86 LANDING_FALL_SPECIAL
is_shieldbreak  124-130 SHIELD_BREAK_*
```

Per fighter 50 -> 56 fields, observation 100 -> 112. The flags are derived
from `action_state`, so `gym_bc.obs_from_raw` rebuilds them from the existing
`raw_fighters` recordings — no re-record needed for this change. No character
embedding yet: with one character in the dataset it would be a constant input.

Recording protocol for the first DK model: DK (CPU 9) vs DK (CPU 9), Final
Destination, 3 stocks / 5 minutes, items off, 1x, fixed seed, observer mode.
Episode 0: 6418 frames (~107 s), 0 seq gaps.

### IL recording: the slow reset had two causes

1. The recorder's reset client built `ControlBlock()` but never opened it, so
   every reset raised `control block is not open` and the match sat on the
   results screen. `load_reset_fn` now prefers the module-level
   `control_block.request_reset` (it opens/closes the block itself) and opens a
   class instance when it has to use one.
2. The runtime only resets a **live fight** (`rl_live()` = the FighterManager's
   ready-go). After the last KO the results screen has no live fight, so a
   reset is refused. The recorder now supports `--episode-frames N`: it cuts
   the episode at N reader frames and asks for a rematch while the fight is
   still live (SSBU_RL_MODE=1, direct rematch entry). The results screen is
   never shown.

Measured with DK (CPU 9) vs DK, 99 stocks / 99 min, `--episode-frames 5400`:
episodes are exactly 5400 frames (90 s), 0 sequence gaps, ~100 s of wall time
each including the in-fight reset (6 episodes in 10 minutes). The debug
match's `stock=` and `time=` are re-applied at every ready-go, so the loop can
run for as long as needed.

### Live eval: who owns player 1, and starting a match from code

Getting a policy into a fight used to need a person: the match has to be
started from the Smash menus, and a pad plugged in mid-session went to the
wrong player. Both are now code paths.

**The pad goes to the first free player slot, and the keyboard holds player 1.**
`nn::pad` logs `connected "Xbox 360 Controller" -> player 2` when a uinput pad
appears while the keyboard owns player 1 (`nn::hid: keyboard = player 1`).
The CSS then shows P1 idle and nothing the policy writes reaches a fighter.
`SSBU_KEYBOARD_PLAYER=0` (`live.sh --keyboard 0`) makes it `keyboard = nobody`,
and the pad takes player 1 — the CSS shows the "Joined!" panel before the match
starts. This is the single cause of "P1 is not moving at all".

**A human slot without its controller is dropped to the title.** With
`p1=human` and no pad attached, the match starts (the debug harness forces the
panel types), then after ~900 fight frames the ready-go clears and the game
sits at "PRESS ANY BUTTON" (`fight off ... after 897 fight frames`). The pad
must exist *before* the fight starts and stay connected for the whole episode;
that is why the menu walk lives in the eval process, not in a helper that exits.

**The menus are one A per screen.** From the title: A (main menu), A (Smash
submenu), A (rules), A (stage select), A (stage), A (CSS roster), stick up
(hand into the roster), stick right (onto the highlighted fighter), A (pick →
"READY TO FIGHT"), A (start). `gym_eval --start-menus` replays that with the
policy pad and then polls the state block until `fight_on` is 1
(`drive_menus`, unit-tested with a fake pad). It is timing-tolerant: the macro
runs once, then A is repeated every second until the fight is on (harmless on
any remaining menu screen), and the walk stops the moment `fight_on` is seen.

**Two more switches matter.**
* `SSBU_INPUT_NO_FOCUS=1` keeps pads live when the game window does not have
  the focus, so no focus keeper and no clicking (`keyboard_ours()` returns 1
  early).
* `SSBU_RL_MODE=1` must be set explicitly: the installed binary still defaults
  to the old mode 2 (results-screen path), even though the source defaults to 1.

One command for the whole loop:

```
src/rl/scripts/live_eval.sh --model data/models/bc_dk9_dir200.pt --episodes 5
```

It reboots the game with `SSBU_KEYBOARD_PLAYER=0`, `SSBU_INPUT_NO_FOCUS=1`,
`SSBU_RL_MODE=1` and the eval match spec, then runs `gym_eval --start-menus`
(policy pad = P1, P2 = DK CPU 9). `--attach` skips the reboot.

### Closed loop: the checkpoint's thresholds are silent on its own states

First closed-loop numbers (DK policy vs DK CPU 9, 400 decisions x 5 frames per
episode, pad on P1):

| decoding | A rate | damage dealt (P2) | damage taken (P1) |
|---|---|---|---|
| checkpoint thresholds | 0.002 | 0.0, 0.0 | 80.7, 44.1 |
| all thresholds 0.5, execute 1, no blend | 0.019 | 18.9, 0.0 | 23.9, 24.7 |
| all thresholds 0.2 | 0.056 | 36.5, 20.2 | 83.4, 107.0 |

The expert presses A on 3.9 % of decision points and the policy reproduces
that offline (predicted rate 0.039 at the tuned 0.92 threshold, F1 0.85), but
in the live fight it presses A on 0.2 % of them: covariant shift. The policy
drives P1 into states the CPU-vs-CPU expert never visited (standing still,
getting hit), and in those states its probabilities collapse to neutral, so it
never re-engages. Lowering the thresholds buys activity and some damage, not
competence (it also takes more damage at 0.2 than at 0.5).

`gym_eval --button-threshold` exists for this diagnosis; it is not a fix. The
candidate fixes, in the order they are being tried:

1. **30 Hz decisions.** The recordings hold every 60 Hz frame, but training
   only sampled every fifth one, so the policy can only change its stick or
   press a button 12 times a second; Smash inputs are timed at 60. A
   `frame_skip=2, window=12, horizon=8` model is training on the same 200
   episodes.
2. **DAgger with the CPU as the expert is not available**: the CPU only acts in
   its own slot, so its actions cannot be labelled on the policy's states. What
   is available is the reverse (policy vs CPU recordings) for offline
   evaluation, and history dropout as a robustness augmentation.

### Closed loop: decoding decides, decision rate does not

Three configs, 2 episodes each, DK policy vs DK CPU 9, 1800 game frames per
episode (360 decisions at 12 Hz, 900 at 30 Hz), `--execute 1 --no-blend`:

| model | decoding | damage dealt (P2) | damage taken (P1) | A rate |
|---|---|---|---|---|
| 12 Hz | argmax at tuned thresholds | 0.0, 0.0 | 85.9, 151.8 | 0.003 |
| 30 Hz | argmax at tuned thresholds | 0.0, 0.0 | 116.9, 101.4 | 0.006 |
| 30 Hz | `--sample` | 18.9, 37.5 | 101.4, 73.1 | 0.054 |
| 12 Hz | `--sample` | 26.5, 18.9 | 63.8, 101.1 | 0.046 |

Two conclusions:

1. **The argmax stands still.** Whatever the decision rate, the policy's
   argmax is neutral in the states it actually reaches, so P1 deals no damage
   at all. Sampling the same distributions gets it moving (A 0.003 -> 0.054)
   and dealing damage. The decision rate (12 vs 30 Hz) barely matters next to
   the decoding.
2. **The policy is still far from the expert** — it loses both stocks while
   dealing 20-40 %. This is a data/learning problem, not a plumbing one: the
   plumbing (pad owns P1, menus walked by the pad, in-fight rematch) now runs
   three consecutive evaluations with no manual step, reaching the fight in
   30 s each time.

The offline numbers are not the problem: test `f1_change` 0.695-0.711,
`f1_active` 0.61-0.70, stick direction 0.85. They are measured on the expert's
own state distribution, which is exactly the distribution the closed loop
leaves.

### Conservative return weighting (damage differential) beats plain BC closed loop

Reward: `0.01 * (damage dealt - damage taken)` per decision, the offline twin of
`gym_reward.compute_reward` with the stock/KO/win terms off.
Weight: `clip(exp(4 * (G - median G)), 0.1, 10)`, mean-normalised, `G` the
discounted (gamma 0.995) return to the end of the episode; measured distribution
mean 1.000, p10 0.385, p90 1.804, max 6.836.

30 Hz, `--sample --execute 1 --no-blend`, 900 decisions = 1800 frames per
episode, DK vs DK CPU 9:

| model | data | damage dealt | damage taken | differential | stick neutral |
|---|---|---|---|---|---|
| plain BC | 200 eps | 31.5, 17.1, 30.2 (26.3) | 35.3, 148.9, 110.3 (98.2) | -71.9 | 0.616 |
| conservative return-weighted | 100 eps | 25.1, 0.0, 20.2 (15.1) | 74.3, 12.9, 45.4 (44.2) | -29.1 | 0.518 |

The weighted model loses far less damage and moves more (stick neutral 0.52 vs
0.62) — and it did that with **half the episodes**, because the 1.7M-sample run
was OOM-killed on a machine that was also running 6.8 GB of other work. Offline
it is also better on its own split (test f1_active 0.780, f1_change 0.743,
stick MAE 0.076).

Caveats: 3 episodes each, and episode-to-episode variance is large (one plain
episode took 149 %), so this is a direction, not a measurement. The two models
also saw different data volumes.

### Joint-action vocab, hold labels, damage buckets (30 Hz, 100 eps)

Four dataset/head changes, trained on the same 100 episodes and with the same
80/10/10 split and return weights as the conservative run above:

* **Hold labels** (`--label-mode hold`): a decision is labelled with the union
  of the buttons over the `frame_skip` frames the pad actually holds it for
  (stick/C-stick take the largest deflection). Labelling one frame lost 23 % of
  A and 56 % of B taps at 30 Hz. Raises the expert rates to A 4.36 %, B 0.75 %,
  Y 4.86 %, X 8.49 % (from 3.51/0.48/4.27/6.06).
* **Damage buckets** (`--damage-mode bucket`): damage in the observation is a
  10 % bucket index (21 values, 200 %+ is one bucket) instead of the raw
  percent. 83 % and 87 % are the same input; the signal stays causal but stops
  being a near-unique sample id. The data maxes out at 234 %, and only 0.4 % of
  frames are at 190 %+, so the last bucket loses almost nothing. The live env takes the same mode from the checkpoint
  (`PolicyObservation(damage_mode=...)`).
* **Content dedupe** (on by default): episodes whose `raw_fighters` hash equal
  are dropped before the limit, so identical recordings cannot straddle the
  train/val split.
* **One token head** (`--action-mode vocab`): one softmax over the 63 observed
  joint action tuples plus a repeat token, instead of 12 independent button
  sigmoids and separate stick heads. A decision whose tuple equals the previous
  one is labelled repeat; 78 % of expert decisions at 30 Hz are repeats, so
  "hold" becomes one class instead of hundreds of near-equal tuples. At
  inference the repeat token re-emits the previously executed action, and
  temporal-ensemble blending is disabled for token checkpoints (averaging two
  actions can invent a tuple the head never chose). With a softmax the argmax
  is unchanged by the collapse; the gain is parameter sharing.

Offline (test): token accuracy 0.794, f1_active 0.719, f1_change 0.629, stick
direction 0.870, repeat rate predicted 0.767 vs expert 0.779. The token loss
converges in 2 epochs and then overfits; early stopping on val loss selected
epoch 2/9 (the old BCE model selected 19/26). It is a weaker offline model than
CWBC, which is the usual warning that offline F1 does not predict the loop.

Closed loop, fresh boot for each, `--sample --execute 1 --no-blend`, 3 x 900
decisions, DK vs DK CPU 9:

| model | damage dealt | damage taken | differential | A | X | stick neutral |
|---|---|---|---|---|---|---|
| CWBC (this session) | 7.6, 18.8, 0.0 (8.8) | 86.9, 50.4, 48.8 (62.0) | -53.2 | 2.0 | 20.1 | 0.502 |
| vocab + hold + buckets | 46.6, 32.3, 27.7 (35.5) | 99.2, 104.0, 70.5 (91.2) | -55.7 | 3.7 | 8.4 | 0.453 |

The token model deals **4x** the damage of CWBC and shields at the expert rate
(8.4 % vs 20.1 %), but takes 47 % more; the net differential is a wash. The
historical CWBC -29.1 did not reproduce in this session (the same checkpoint
scored -53.2 on a fresh boot), so the CPU/session variance is larger than the
difference between the two policies. For an RL initialisation the token model
is the more active starting point, but 3 episodes cannot rank them.

### Regularized vocab model: best BC closed-loop result so far

The first all-200-episode run exposed two problems: it overfit after epoch 6
(train loss fell while validation loss rose), and a single corrupt raw frame
(`action_frame = 3.4e38`) overflowed the float16 window storage. The overflow
made the normalization statistics `inf`/`NaN`, so the run trained to `NaN` and
saved a repeat-only model. This is now fixed at four layers: absurd/non-finite
observations become missing values, the float16 cast is bounded, `norm_stats`
refuses non-finite input, and training aborts instead of saving a non-finite
model. Two related bugs were also fixed: ego-swap augmentation was applied
after resampling (losing the hold-label OR on half the augmented data), and
`--player 1` labelled P2 actions while leaving P1 first in the observation.

The regularized retrain used all 200 unique episodes, `--ego-swap`, hold
labels, 10 % damage buckets, aux dense losses (0.5), label smoothing 0.05,
hidden 192, dropout 0.15 on the GRU summary and MLP, AdamW weight decay 1e-2,
lr 8e-4, and `--select-metric f1_change` with patience 4. The vocabulary kept
all **136 observed tuples + repeat = 137 classes** (the old 64-class cap was
silently snapping rare actions). Training early-stopped at epoch 13, selecting
epoch 9; unlike the earlier run, train and validation loss both fell
(2.16 -> 1.75 and 1.84 -> 1.70) and the predicted repeat rate matched the
expert instead of overshooting it.

Offline (test): token accuracy 0.844, f1_active 0.761, f1_change 0.684, stick
direction 0.902, stick MAE 0.065, repeat predicted 0.774 vs expert 0.788.
Previous vocab test: 0.794 / 0.719 / 0.629 / 0.870 / 0.085 / 0.767 vs 0.779.

Closed loop, fresh boot, `--sample --execute 1 --no-blend`, 8 x 900 decisions,
DK vs DK CPU 9:

| model | damage dealt | damage taken | differential | A | X | ZL | stick neutral |
|---|---|---|---|---|---|---|---|
| CWBC (same session) | 8.8 | 62.0 | -53.2 | 2.0 | 20.1 | - | 0.502 |
| old vocab (100 eps) | 35.5 | 91.2 | -55.7 | 3.7 | 8.4 | - | 0.453 |
| **regularized vocab (200 eps, ego-swap)** | **39.3** | **57.9** | **-18.5** | 6.2 | 12.6 | 5.7 | 0.458 |

Per-episode differentials: -69.3, +21.8, -37.2, -65.8, -31.0, +58.2, +0.3,
-25.1. The policy deals 49 % of the teacher's 79.6 %/30 s, takes 80 % of the
teacher's 72.5 %/30 s output, and is the first BC checkpoint to finish clearly
ahead of the historical CWBC -29.1. Two of eight episodes were positive, so
the ranking is still noisy; 900 decisions per episode ends at the step limit,
so `winner` is `None` and the damage differential is the metric.

### PPO pilot: 20k decisions is too little; clip-only drifted

`rl/rl/ppo.py` adds a scalar value head to the BC trunk and runs token-level
PPO against the live 30 Hz env, with `0.01 * (dealt - taken)` and every
stock/KO/win reward term set to zero. The first pilot used 10 iterations x
2048 decisions = **20,480 decisions** (about 20 episodes, 11 minutes), two
episodes per rollout, clip 0.2, no KL term. It completed without errors but
produced a worse policy:

| policy | dealt | taken | differential |
|---|---|---|---|
| BC (regularized) | 39.3 | 57.9 | -18.5 |
| PPO pilot, 20k decisions | 33.3 | 96.1 | -62.8 |

One of the pilot's ten logged rollouts reported 0/0 damage. The episode row
used only the final `info` frame, which is 0/0 during a respawn/damage reset;
the per-step reward was not affected. The logger now keeps the episode's peak
P1/P2 damage, so a future 0/0 means the episode really was dead/paused rather
than a post-KO reset. Checkpoints are now written per iteration
(`rl_dk9_ppo_iterNNN.pt`) as well as to the latest path, so the next run can
select the best iteration instead of keeping the last one. A serious run needs
more episodes per rollout (8-16), 200k+ decisions, and a BC-anchored KL term
(or BC auxiliary loss) to prevent this kind of drift.

### CPU3 PPO: dead episodes were a pad/rematch stall, not policy collapse

The CPU3 PPO run initially looked like it collapsed: after three positive
rollouts (+22.8, +7.0, +28.1) it logged four rollouts with **zero damage dealt**
while still taking 40-64 damage per episode. A fresh eval of the same
`iter008` checkpoint scored **75.0 dealt / 35.7 taken (+39.2 over 3)** and its
offline action rates were still BC-like, so the zeros were dead episodes: P1's
pad/game input had stalled while CPU3 kept attacking. The same failure appeared
in eval after 4-5 rematches and in an attached eval as a fully paused 0/0.

Fixes: a dead-episode guard stops PPO before it updates on any zero-dealt
episode; the reset client now has a 20 s timeout; and long runs restart the
game every four episodes. The stock reward is detected from `is_on_rebirth`
0->1, since the live stock field is unreadable; `--stock-value 1.0` means one
stock = 100 % damage.

Retrain with the KL anchor on BC (`kl=0.1`), entropy 0.002, lr 1e-4, and two
epochs, warm-started from `iter008`:

| run | episodes | dealt | taken | differential |
|---|---:|---:|---:|---:|
| retrain iterations 1-3 | 29 | 62.1 | 33.0 | **+29.0** |
| 5 rebooted chunks of 4 | 20 | 81.6 | 28.6 | **+53.0** |
| fresh eval `retrain_iter002` | 5 | 63.8 | 31.9 | +31.9 (one dead ep.) |
| fresh eval `chunk_01` | 5 | 54.5 | 29.9 | +24.6 |

All five chunks were clean and positive. The retrain then hit the old 5 s reset
timeout; the timeout fix and chunking address that. Best checkpoints are kept
as `rl_dk9_cpu3_best.pt` (retrain iter002) and `rl_dk9_cpu3_iter008.pt`.

### Damage resets on a KO: what the reward does and does not count

Measured over 30 episodes of `data/il/dk9` (255-stock rule):

* **every** episode has at least one stock loss: 72 in total in 30 episodes (~2.4
  per episode), so KOs are common, not absent — the earlier "255 stocks means no
  KOs" note in this file was wrong.  The `stock` field reads 1..9 (a display
  counter), which is why it never looked like 255; the reliable death signal is
  the damage trace dropping to 0 (54 drops for P1, 48 for P2 in the same 30
  episodes).
* the damage delta is clamped at 0 (`max(0, curr - prev)`), so the reset after a
  KO contributes **0, not negative reward**: the policy is never paid for
  "healing" the opponent.
* the reward on the decision where a stock changes is exactly 0.000 in all 72
  cases: the damage that led to the KO was already paid out decision by
  decision, and the KO itself carries no bonus and dying carries no penalty.

That is consistent ("damage is the currency"), but it is a modelling choice worth
stating: a KO is worth no more than the damage that preceded it, and losing a
stock costs nothing beyond the damage taken.  `gym_reward.RewardConfig` already
has `ko_bonus` / `stock_loss` for the alternative; the trainer passes 0 for both.

### Building the recomp: never interrupt the link

Killing `setup.py --from compile` mid-way left `ssbu_game_eh_o1.so` truncated
(5.8 MB instead of 580 MB), because the compile step links the game library
in place.  A relink of the host runtime against the Nov-14 backup `.so` produced
an exe that booted, joined the pad, and then ignored all input — the host runtime
and the game library have to come from one build.

Two practical notes:

* `setup.py --only link` rebuilds the host runtime and relinks in about 90 s
  (deps, stage, link): that is the edit-link-run loop for `runtime/*.c`.
* the compile step recompiles the whole game (1161 files, ~80 min with 6 jobs)
  unless its done marker is present; 17 files crash LLVM 19 in
  `MemorySSAUpdater::recursePhi`, and `--no-inline` builds those (objects built
  earlier keep their own flag, so the result is valid).
* keep a copy of a known-good `ssbu_native` + `ssbu_game_eh_o1.so` pair outside
  the tree (`/tmp/opencode/ssbu_backups/`), because there is no atomic replace.

### Death credit assignment: the -5 is real but arrives too late

Checked the overnight CPU7 logs: 594 P1 deaths over 595 episodes, and every
logged `death_rew` was exactly `-5.0` (or `+5.0` for a P2 death).  The reward
plumbing is correct.

The problem is the credit horizon.  A death is credited when `is_on_rebirth`
turns on, i.e. at respawn.  Across `data/il/dk9`, the median fall from below the
stage to respawn is **61 decisions** at 30 Hz (~2 s).  With `gamma=0.995,
lam=0.95`, the GAE factor per decision is `0.945`, so only `0.945**61 = 0.032`
of the `-5.0` reaches the edge decision (~`-0.16`).  The critic agrees: on
recorded fall states `rl_cpu7_best.pt` predicts `V=-0.25` at fall start and
`V=-0.59` deep offstage, not anything near `-5`.  Rare catastrophic states are
too small a fraction of a 3600-step rollout for the value MSE to fit them
(observed `value_loss ~0.2`).

`gym_eval.py` had a separate reporting bug: it built the env with
`reward_mode="none"`, discarded the returned reward, and reported only peak
damage.  It now uses the checkpoint's stored reward config and counts rebirth
deaths, so eval and training measure the same objective.

Retraining from `rl_cpu7_best.pt` uses `gamma=0.997, lam=0.99`
(`0.987**61 = 0.45`, ~14x more death credit at the edge) plus dense recovery
shaping (`recovery_weight=0.05`, `offstage_penalty=0.02`).

`opponent_offstage_reward` mirrors `offstage_penalty`: the ego is paid while
the opponent is below the stage (same ramp, no horizontal-edge term), so
edgeguarding has value even when the KO does not land.  Eval restores the term
from the checkpoint meta.

The first retry with `recovery_weight=0.05` and plain MSE on the value head
collapsed after iteration 1: the sparse `-5` returns pushed `value_loss` from
0.08 to 6.7 and `kl_bc` from 0.06 to 0.84, because the value gradient flows
through the shared trunk.  The value loss now uses `smooth_l1_loss` (Huber),
which keeps large stock returns from dominating; the relaunch uses
`recovery_weight=0.02`, `offstage_penalty=0.01`,
`opponent_offstage_reward=0.02`, `gamma=0.997`, `lam=0.99`.

### Edgeguard run (20 iterations): deaths down, damage down, no net win

From `rl_cpu7_best.pt` with the credit fix plus
`opponent_offstage_reward=0.02`, `offstage_penalty=0.01`,
`recovery_weight=0.02`, `gamma=0.997`, `lam=0.99`, Huber value loss:

* training: 43 P1 deaths in 68 episodes (0.63/episode), down from ~1.0;
* best rollout: chunk 3 iteration 4, 0 deaths, reward +0.15, damage -12.8;
* death-aware eval of `rl_cpu7e_chunk_03_iter003.pt` (5 episodes):
  3 P1 deaths, reward -4.94, cumulative damage -55.1,
  stock-adjusted -355.1.

The old `rl_cpu7_best.pt` remains better: 2 deaths, reward -2.25, damage +2.4,
stock-adjusted -197.6.  The state-based opponent reward pays whenever the CPU
happens to be offstage (it is below the stage ~11% of frames), which is largely
independent of the ego's actions and may have traded away damage.  A
potential-based opponent term (pay when their danger increases, charge when
they recover) is the causal version to try next.

`opponent_offstage_potential` is the causal version: reward
`w * (danger_P2(curr) - danger_P2(prev))`, so the ego is paid when it pushes
P2 below the stage and charged when P2 recovers, with no free reward for the
CPU's own offstage time.  The relaunch from `rl_cpu7_best.pt` uses
`opponent_offstage_potential=0.1`, `opponent_offstage_reward=0`,
`recovery_weight=0.02`, `offstage_penalty=0.01`, `gamma=0.997`, `lam=0.99`.

### Potential-based edgeguard run: also no eval win

From `rl_cpu7_best.pt`, 20 iterations with
`opponent_offstage_potential=0.1`, `opponent_offstage_reward=0`,
`recovery_weight=0.02`, `offstage_penalty=0.01`, `gamma=0.997`, `lam=0.99`:

* training: 50 P1 deaths in 72 episodes (0.69/episode);
* best rollout: chunk 2 iteration 3, 1 death, reward -1.94, damage +9.5;
* death-aware eval of `rl_cpu7p_chunk_02_iter002.pt` (5 episodes):
  4 P1 deaths, reward -6.24, damage -38.4, stock-adjusted -338.4.

The old `rl_cpu7_best.pt` still wins: 2 deaths, reward -2.25, damage +2.4,
stock-adjusted -197.6.  Both edgeguard variants reduced the training death
rate slightly but lost too much damage.  Keep `rl_cpu7_best.pt` as the best
live checkpoint until a new approach is evaluated on more than 5 episodes.

### Dense recovery experiment

Three changes, trained from `rl_cpu7_best.pt` with `gamma=0.995, lam=0.95`:

* `--recovery-ledge --recovery-weight 0.05`: the recovery potential is 2-D
  distance to the nearest ledge, so climbing and moving toward the ledge both
  pay, not just gaining height.
* `--doomed-penalty 0.05`: below the stage, falling, and out of jumps ramps a
  penalty toward the blast bottom, so a doomed fall is paid before respawn.
* `--offstage-loss-weight 3.0`: offstage transitions are 3x in the policy and
  Huber value losses, so the critic does not fit only common on-stage states.

The edgeguard terms are off in this run.  The aim is to separate recovery
shaping from the edgeguard reward that regressed the previous checkpoints.

### Recovery order: jump before Up Special

The doomed-fall penalty now requires the **helpless** action state, not merely
"no jumps".  Action states 20/21 (`FALL_SPECIAL`, `FALL_SPECIAL_AERIAL`) are
special fall after Up Special; states 36/40 (`SPECIAL_HI*`) are the Up Special
itself; 18/19 are a normal fall.

So:

* no jumps but `FALL` -> Up Special is still available, not doomed;
* currently using Up Special -> not doomed yet;
* `FALL_SPECIAL` below the stage, falling -> doomed; the ramp applies.

This matches the recovery rule: use the midair jump before Up Special, because
Up Special ends in helpless special fall.  The next run (`cpu7s`) uses the
corrected penalty.

### Recovery drill and mirror wrapper

`gym_drill.RecoveryDrillEnv` manufactures recoverable ledge states without a
runtime teleport:

1. reset the match;
2. wait out the revival platform (the fast rematch can return a fighter still
   on rebirth);
3. run P1 toward the nearest ledge with the stick only, so the midair jump and
   Up Special stay available;
4. hand control to the policy mid-fall (setup observations seed its window and
   `setup_action` seeds `prev`);
5. end on recovery (small bonus) or death.

Drill training uses **CPU3**, not CPU7: the point is to learn recovery under
low pressure, not to survive a strong edgeguard while learning.

`rl.rl.mirror.MirrorWrapper` mirrors whole episodes left/right: policy
observation in, mirrored action to the game, mirrored observation back.  It is
applied outside the drill wrapper so setup observations and `setup_action` are
mirrored too.  Token-level mirroring is intentionally avoided; the BC action
head is an absolute vocabulary.

First live drill episodes: setup ~24-138 decisions, offstage P1 at
`x≈-98, y≈-10`, several successful recoveries (+1.0 reward, 0 deaths) and
several deaths (-13 reward).  The viewer now starts beside the game window
(`[view] window 1440x860+1390+310`) instead of on top of it.
