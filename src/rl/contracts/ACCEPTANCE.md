# Acceptance suite (coordinator only)

All live steps are serialized through `tools/mem/game_session.py`
(`/tmp/ssbu_game.lock`). Unit tests use mocks and need no game.

## 0. Static

```bash
cd /home/daniel/Projects/SSBU/ssbu-recomp
.venv/bin/python -m py_compile tools/mem/*.py
.venv/bin/python -m json.tool tools/mem/memmap.json >/dev/null
.venv/bin/python -m json.tool tools/mem/extra_map.json >/dev/null  # when present
```

## 1. R2 direct reset (20/20)

Agent A's script under `/tmp/opencode/agent_A/`; run under the game lock:

```bash
.venv/bin/python tools/mem/game_session.py run --owner accept-A --restart-game \
    -- /tmp/opencode/agent_A/test_r2.py --resets 20
```

Pass: no guest trap, `ready_go=1` after each reset, full stocks, 0 damage,
same stage, spawn state, 20/20 lines.

## 2. CPU vs CPU DK gate (10/10)

```bash
.venv/bin/python tools/mem/game_session.py run --owner accept-DK --restart-game \
    -- python3 tools/mem/probe.py --frames 1
```

Read `~/smash.log`: 10/10 `debug match: fight ... p2 kind 1 (donkey)`, stage
Final Destination (`stage_id == 3` on this build), levels 9/9.

## 3. uinput virtual pad + reset client

```bash
.venv/bin/python tools/mem/game_session.py run --owner accept-pad \
    -- .venv/bin/python tools/mem/test_gym_input.py --live
.venv/bin/python tools/mem/game_session.py run --owner accept-reset \
    --restart-game --restart-reader --keep-game \
    -- .venv/bin/python tools/mem/test_integration.py --reset
```

Pass: evdev `SSBU Virtual Pad` 045e:028e, SDL `Xbox 360 Controller` GUID
`030049205e0400008e02000014010000`, player 1; canonical buttons/sticks/
trigger verified; device destroyed; reset request acked and a fresh match
reaches `fight_on=1`.

## 4. State bridge

```bash
.venv/bin/python tools/mem/game_session.py run --owner accept-state \
    --restart-reader -- .venv/bin/python tools/mem/probe.py --frames 5
.venv/bin/python tools/mem/game_session.py run --owner accept-shm \
    --restart-reader -- .venv/bin/python tools/mem/test_gym_env.py
```

Pass: all v7 fields present; shield toggles; expert buttons/stick change;
stock reads as u32; P1/P2 slots; empty slots zero.

## 5. Data recording alignment

```bash
.venv/bin/python tools/mem/game_session.py run --owner accept-rec \
    --restart-game -- .venv/bin/python tools/mem/gym_record.py \
    --episodes 10 --out tools/mem/test_artifacts/accept/mario_vs_dk
```

Pass: 10 episodes, each with contiguous seq, correct 60 Hz rows, `frame_skip=5`,
P1 Mario/P2 DK metadata, no NaNs; resampler produces 12 Hz windows.

## 6. Gym env

```bash
.venv/bin/python tools/mem/test_gym_env.py
```

Pass: `check_env`, 100 mock steps, reset, action/observation shapes, reward
modes.

## 7. Behavior cloning

```bash
.venv/bin/python tools/mem/test_gym_bc.py
.venv/bin/python tools/mem/gym_bc.py --data tools/mem/test_artifacts/accept/mario_vs_dk \
    --epochs 5 --out tools/mem/test_artifacts/accept/bc.pt
.venv/bin/python tools/mem/gym_eval.py --model tools/mem/test_artifacts/accept/bc.pt \
    --episodes 5
```

Pass: window/chunk alignment, mirror correctness, tiny-batch overfit, held-out
F1/stick metrics, checkpoint round-trip, closed-loop episodes run end-to-end.

## Rollback

- Restore `/tmp/opencode/ssbu_backups/20261101_232013/debug_mode.c`.
- Runtime hook off: `SSBU_RL=0`.
- Rebuild with the documented setup command.
