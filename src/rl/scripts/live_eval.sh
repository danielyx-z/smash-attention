#!/usr/bin/env bash
# Closed-loop eval of a BC checkpoint against the CPU, in one command.
#
#   src/rl/scripts/live_eval.sh --episodes 5 --max-steps 600
#   src/rl/scripts/live_eval.sh --model data/models/bc_dk9_dir200.pt --episodes 3
#
# What it fixes compared with doing the steps by hand:
#
#  * the keyboard is bound to nobody (SSBU_KEYBOARD_PLAYER=0), so the uinput pad
#    the policy uses takes PLAYER 1.  With the keyboard on player 1 the pad is
#    given player 2 and the fighter on player 1 never moves;
#  * SSBU_INPUT_NO_FOCUS=1: pads stay live without the window having the focus,
#    so no focus keeper and no clicking is needed;
#  * the pad itself walks boot -> match (gym_eval --start-menus): one A per
#    menu screen, a stick flick into the roster, A to pick, A on "READY TO
#    FIGHT".  The pad stays open for the whole run, because a human player left
#    without a controller is dropped back to the title by the game;
#  * SSBU_RL_MODE=1: the runtime's direct in-fight rematch (the build's default
#    is still the old results-screen path, mode 2).
set -uo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
RLROOT="$(cd "$HERE/../../.." && pwd)"
PY="$RLROOT/.venv/bin/python"

MODEL="$RLROOT/data/models/bc_dk9_dir200.pt"
MATCH="${SSBU_LIVE_MATCH:-p1=donkey,p1=human,p2=donkey,p2=cpu,cpu=9,stage=end,players=2,stock=255,time=999,items=off,hazards=off}"
BOOT=1
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --match)  MATCH="$2"; shift 2 ;;
        --attach) BOOT=0; shift ;;
        --help|-h) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)        ARGS+=("$1"); shift ;;
    esac
done

export SSBU_KEYBOARD_PLAYER=0          # free player 1 for the policy pad
export SSBU_INPUT_NO_FOCUS=1           # no focus keeper, no clicking
export SSBU_RL_MODE=1                  # direct in-fight rematch
unset SSBU_SEQ SSBU_PLAY SSBU_PRESS SSBU_SCRIPT   # the pad drives the menus

if [[ $BOOT -eq 1 ]]; then
    "$HERE/live.sh" --no-focus --keyboard 0 --match "$MATCH" || exit 1
    # No wait_live here: that waits for a fight, and the menu walk in gym_eval
    # is what starts it.  gym_eval waits for the state block itself.
fi

exec "$PY" -m rl.bc.gym_eval --model "$MODEL" --start-menus "${ARGS[@]}"
