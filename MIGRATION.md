# ssbu-rl migration notes

`ssbu-rl/` is a sibling of `ssbu-recomp/` and is **not** its own git repo.
The recomp tree keeps only the runtime hook (`runtime/debug_mode.c`); all RL,
memory-reading, recording, BC and live-test tooling now lives here.

## Layout

```
ssbu-rl/
  .venv/                     editable install of this package
  data/                      checkpoints (accept/, E/)
  test_artifacts/            recorded episodes + manifests
  pyproject.toml
  requirements.txt
  README.md                  pre-move tools/mem README (paths still say tools/mem)
  src/rl/
    env/       gym_env, gym_obs, gym_action, gym_reward
    bc/        gym_bc, gym_eval
    record/    gym_record
    input/     gym_input, gym_reset, control_block
    memory/    gamestate, schema, memapi, reader, probe, elfsym, memmap.json
    extra/     extra_state, extra_map.json
    utils/     locks, game_session
    scripts/   wait_live, live.sh, focus_game.sh, verify_live_state, live_view
    tests/     test_gym_env, test_gym_bc, test_gym_input, test_extra,
               test_integration, verify_gui
    contracts/ CONTRACTS.md, ACCEPTANCE.md
    findings/  FINDINGS.md, FINDINGS_EXTRA.md
```

Internal package name is `rl`; distribution name is `ssbu-rl`.

## Import mapping

Moved modules keep their code; only import paths changed.

| old (flat, tools/mem) | new |
|---|---|
| `import gym_env` | `from rl.env import gym_env` / `rl.env.gym_env` |
| `import gym_action/obs/reward` | `from rl.env import ...` |
| `import gym_bc` / `gym_eval` | `from rl.bc import ...` |
| `import gym_record` | `from rl.record import gym_record` |
| `import gym_input` / `gym_reset` / `control_block` | `from rl.input import ...` |
| `import schema` / `gamestate` / `memapi` / `reader` / `probe` / `elfsym` | `from rl.memory import ...` |
| `import extra_state` | `from rl.extra import extra_state` |
| `import locks` / `game_session` | `from rl.utils import ...` |

Lazy/dynamic imports were updated in the same way:
`gym_env` uses `importlib.import_module("rl.input.gym_input"|"...gym_reset")`;
`gym_record.load_reset_fn` uses `rl.input.gym_reset` / `rl.input.control_block`.

## Path fixes required by the sibling move

The old code resolved `build-linux/` relative to `ssbu-recomp/tools/mem/`.
After the move those relative paths landed inside `ssbu-rl/src/`, so three
places were corrected (nothing else in those files changed):

* `rl/memory/gamestate.py:default_elf()` and `rl/extra/extra_state.py:DEFAULT_ELF`
  now resolve `../ssbu-recomp/build-linux/binaries/13.0.4/main.elf`
  (override with `SSBU_RECOMP_DIR`).
* `rl/memory/reader.py:REPO` now resolves the sibling `ssbu-recomp` tree, so
  `reader.py --spawn` finds `play.sh` (it used to resolve inside `src/`).
* `rl/utils/game_session.py`: `REPO` is the sibling recomp tree, `RLROOT` is
  this repo, the reader is launched as `python -m rl.memory.reader`, and the
  command runs with `cwd=RLROOT`. `DEFAULT_MATCH` now includes `players=2`.
* `rl/tests/test_gym_input.py`: `REPO` points at the sibling recomp tree.

`src/rl/scripts/live.sh` was updated to launch the game from the sibling recomp
tree and the reader/stage view from this package.  The stage view
(`rl.scripts.live_view`, a data-only 2D view of the stage, fighters, and
battle objects) replaces the old text-only `rl.tests.verify_gui` in
`live.sh`; `verify_gui.py` is kept as a fallback.

## Verification

* All 25 moved modules were restored from the pre-move sources and compared
  against the original Nov 2 `__pycache__` bytecode; every module matched
  (same code objects, constants, names and line table).
* Unit suite: `163 passed, 5 skipped` (skips are the live-gated tests):
  148 before this pass + 15 new live-view tests.
* Live smoke through the lock-safe harness:
  `python -m rl.utils.game_session run --owner smoke --restart-game --restart-reader --keep-game -- ...`
  * reader: 60 fps, `fight_on=True`, 2 fighters, stage 3
  * `SSBUGymEnv.reset()`: R2 reset acknowledged and stable in 2.25 s
  * 20 steps at `frame_skip=5` (12 Hz), no exceptions
  * uinput pad enumerated as `Xbox 360 Controller` on player 1
* `rl-eval --mock` and a 1-epoch `rl-train` run both complete.
* `rl-record`, `rl-probe`, `rl-train`, `rl-eval`, `rl-wait-live` entry points
  resolve; `load_reset_fn()` returns `gym_reset.ControlBlock().request_reset`.

## Running

```sh
cd ssbu-rl
.venv/bin/python -m pytest src/rl/tests -q

# live: always through the session harness
.venv/bin/python -m rl.utils.game_session run --owner me --restart-game \
    --restart-reader --keep-game -- bash -c \
    '.venv/bin/python -m rl.scripts.wait_live 300 && <command>'

# training / evaluation
.venv/bin/rl-train --data test_artifacts/accept/mario_vs_dk_1v1_30 --epochs 10 --out data/E/bc.pt
.venv/bin/rl-eval --model data/accept/bc_mario_vs_dk_1v1_27.pt --mock --episodes 3

# live stage view: 2D boxes for fighters/projectiles + the full field panel
src/rl/scripts/live.sh                      # game + reader + view
.venv/bin/python -m rl.scripts.live_view    # attach to a running reader
.venv/bin/python -m rl.scripts.live_view --mock        # fake data, no game
.venv/bin/python -m rl.scripts.live_view --mock --dump-once --json
```

### Host-only rebuild (runtime C changes)

A change to `runtime/debug_mode.c`, `runtime/nnhid.c`, `runtime/display.c`,
`runtime/gfx_options.c`, etc. changes **only the host executable**
(`build-linux/work/ssbu_native`).  Do **not** run `build.py link` or
`scripts/setup.py` for that: `build.py link` first calls `build_game_dll()`,
and with a cold `obj_eh_o1/` it relinks the 553 MB
`ssbu_game_eh_o1.so` from the data objects alone (3 MB).  Use the hot-rebuild
tool:

```bash
# from ssbu-rl; pass every runtime source you changed (and any header)
.venv/bin/rl-host-link debug_mode.c
# or: .venv/bin/python -m rl.scripts.host_link debug_mode.c nnhid.c
```

It copies `runtime/<file>` into `build-linux/work/`, marks it stale, backs up
the current `ssbu_native`, recompiles the stale host objects, and links
`ssbu_native` against the **existing** `.so` with `build_game_dll()` disabled.
It refuses to run while a full build is running or when the `.so` is smaller
than 100 MB, takes `/tmp/ssbu_build.lock`, and restores the previous exe if the
link fails.  Restart the game afterwards
(`src/rl/scripts/live.sh ...`).

A **full** rebuild (needed only when the game library itself is rebuilt) is the
canonical `deps/venv/bin/python scripts/setup.py --target linux --dump <dump>`.
After a successful full build, back up **both**
`build-linux/work/ssbu_game_eh_o1.so` and `build-linux/work/ssbu_native`.

### Runtime switches for training (see CONTRACTS.md for the v2 block)

- `SSBU_RL=1` — maps the 80-byte control block (reset + per-frame
  `game_frame`/`state_hash`/`flags` + commands). `live.sh` and `game_session`
  pass it.
- `SSBU_INPUT_NO_FOCUS=1` — pads/keyboard stay live without the game window
  holding SDL keyboard focus (headless/locked sessions).
- `SSBU_SPEED=<n>` — run at n × 60 Hz, one game step per presented frame;
  `0` = uncapped. Use `rl.scripts.turbo_check` with the same
  `SSBU_FIXED_SEED` to prove a turbo run matches `SSBU_SPEED=1`.
- `SSBU_FIXED_SEED=<n>` — constant RNG-seed clock reads, a fixed RNG source,
  and (with `SSBU_SPEED`) the frame-locked game clock; the runtime also
  re-seeds the fight at every ready-go so turbo comparisons start from the
  same state.
- `SSBU_FIXED_CLOCK=1|0` — force the frame-locked `nn::os::GetSystemTick`
  clock on/off.  Default: on when `SSBU_FIXED_SEED` or `SSBU_SPEED` is set.
  The game sees exactly 1/60 s per game frame at any speed; host pacing stays
  on the real clock.  `SSBU_FIXED_CLOCK=0` restores wall time.
- `SSBU_PAD_FROM_CTL=1` — player 1's synthetic XInput pad is driven by the v2
  `set_pad` command instead of uinput/the sequence.
- `SSBU_DEBUG_MATCH=...,items=off,stock=N` — the stock count is forced at
  match setup/ready-go; no menu edit.

Characters and player types from code (no character-select visit):

```python
from rl.utils.match_config import MatchSpec, write_match_list

spec = MatchSpec(p1="samus", p1_type="human", p2="kirby", p2_type="cpu",
                 cpu=9, stage="end", stock=3, items="off")
print(spec.to_string())          # -> SSBU_DEBUG_MATCH=...
# or, one set-up per match in a rematch loop (no relaunch):
write_match_list("/tmp/matches.txt", [spec, MatchSpec(p1="mario", p2="link")])
#   --env SSBU_DEBUG_MATCH_LIST=/tmp/matches.txt
# append_match("/tmp/matches.txt", later_spec) queues one more.
```

The runtime forces the character by name (`ui_chara_<name>`) at the
select-screen panel, so any fighter name works; `p1=cpu|human` is independent
of the name. `python -m rl.utils.match_config --p1 samus --p2 kirby` prints the
same string. Verified live: Samus (kind 3) vs Kirby (kind 6), P1 human, P2 CPU.

Turbo check (two separate launches, same match/seed):

```bash
# speed 1
.venv/bin/python -m rl.utils.game_session run --owner t1 --restart-game \
  --restart-reader --keep-game \
  --match 'p1=mario,p2=donkey,p1=cpu,p2=cpu,cpu=9,stage=end,players=2,items=off,stock=3' \
  --env SSBU_FIXED_SEED=12345 --env SSBU_SPEED=1 -- \
  .venv/bin/python -m rl.scripts.turbo_check record --out /tmp/ref.json \
      --frames 600 --expect-speed 1000 --settle 10 --stable-reads 2
# then speed N (SSBU_SPEED=4, --out /tmp/turbo.json, --expect-speed 4000)
# record only starts at fight_on (menu frames are skipped); a block from an
# earlier process is rejected via the epoch in flags bits 8..31.
# Use human/idle players: the CPU AI is not deterministic across launches.
.venv/bin/python -m rl.scripts.turbo_check compare /tmp/ref.json /tmp/turbo.json
```

### Live runtime checks

`rl.scripts.live_checks` (observe-only) verifies the runtime changes on a
running game:

```bash
# with the game launched with SSBU_RL=1 and --env SSBU_INPUT_NO_FOCUS=1
.venv/bin/python -m rl.scripts.live_checks rules --expected-stock 3 --seconds 25 --json
.venv/bin/python -m rl.scripts.live_checks focus --json
.venv/bin/python -m rl.scripts.live_checks all --expected-stock 3 --json
```

It reads `stock` (and the stock drops inside the match), `g.match.item_appear`
from the extra block (run `rl.extra.extra_state --duration 120 --if-live` for
that), and holds the uinput pad right while the game has no focus (focus mode).

`live_checks focus` needs a **live match**. `SSBU_DEBUG_MATCH` forces the
select screens, not the walk to them. For a headless run, pair it with a
menu-walk sequence (`SSBU_SEQ`, e.g. `tests/seq_battle_play_fast.txt` cut after
the "ready to fight" press) and wait for the match first:

```bash
SEQ="$(grep -v '^#' seq_nav.txt | tr '\n' ' ')"
.venv/bin/python -m rl.utils.game_session run --owner lc --restart-game \
  --restart-reader --keep-game \
  --match 'p1=mario,p1=human,p2=donkey,p2=cpu,cpu=1,stage=end,players=2,items=off,stock=3' \
  --env SSBU_INPUT_NO_FOCUS=1 --env "SSBU_SEQ=$SEQ" -- \
  bash -lc '.venv/bin/python -m rl.scripts.wait_live 150 && \
            .venv/bin/python -m rl.scripts.live_checks focus --json --hold 2'
```

With `SSBU_SEQ`/`SSBU_PLAY` set the check does not tap A (the walk owns the
menus); it assumes the match is live when it starts. Verified pass: 121 live
frames, `dx=153.67`, game window unfocused.

## Known follow-ups

* `README.md`, module docstrings and CLI help still show `tools/mem/...` paths;
  they are documentation only. Update them in a separate docs pass.
* The 27/30-episode dataset and checkpoint are pipeline proof, not a strong
  policy. A useful BC policy still needs a few hundred clean 1v1 episodes.
* The v7 stock field reads 0 for some live 1-stock frames; the env already
  falls back to `fight_on` transitions (warning at reset).
* `ssbu-recomp/tools/mem/` was left as an empty directory by the move and was
  removed. The recomp runtime change is still only `runtime/debug_mode.c`.
