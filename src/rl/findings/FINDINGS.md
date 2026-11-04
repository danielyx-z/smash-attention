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
