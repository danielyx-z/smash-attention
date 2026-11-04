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
    scripts/   wait_live, live.sh, focus_game.sh
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
* `rl/utils/game_session.py`: `REPO` is the sibling recomp tree, `RLROOT` is
  this repo, the reader is launched as `python -m rl.memory.reader`, and the
  command runs with `cwd=RLROOT`. `DEFAULT_MATCH` now includes `players=2`.
* `rl/tests/test_gym_input.py`: `REPO` points at the sibling recomp tree.

`src/rl/scripts/live.sh` was updated to launch the game from the sibling recomp
tree and the reader/verifier from this package.

## Verification

* All 25 moved modules were restored from the pre-move sources and compared
  against the original Nov 2 `__pycache__` bytecode; every module matched
  (same code objects, constants, names and line table).
* Unit suite: `149 passed, 4 skipped` (skips are the live-gated tests).
  Breakdown: env 58, BC 38, input 31, extra 22.
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
```

## Known follow-ups

* `README.md`, module docstrings and CLI help still show `tools/mem/...` paths;
  they are documentation only. Update them in a separate docs pass.
* The 27/30-episode dataset and checkpoint are pipeline proof, not a strong
  policy. A useful BC policy still needs a few hundred clean 1v1 episodes.
* The v7 stock field reads 0 for some live 1-stock frames; the env already
  falls back to `fight_on` transitions (warning at reset).
* `ssbu-recomp/tools/mem/` was left as an empty directory by the move and was
  removed. The recomp runtime change is still only `runtime/debug_mode.c`.
