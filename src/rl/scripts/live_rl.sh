#!/usr/bin/env bash
# PPO fine-tuning of a BC checkpoint in the live game, in one command.
#
#   src/rl/scripts/live_rl.sh --iterations 10
#   src/rl/scripts/live_rl.sh --model data/models/bc_dk9_30hz_vocab_egoswap.pt \
#       --out data/models/rl_dk9_ppo.pt --iterations 20
#
# Same live assumptions as live_eval.sh:
#  * keyboard is bound to nobody, so the virtual pad takes player 1;
#  * the pad walks boot -> match and stays open;
#  * SSBU_RL_MODE=1 selects the direct in-fight rematch;
#  * the reward is damage differential only (stock/KO/win weights are zero).
set -uo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
RLROOT="$(cd "$HERE/../../.." && pwd)"
PY="$RLROOT/.venv/bin/python"

MODEL="$RLROOT/data/models/bc_dk9_30hz_vocab_egoswap.pt"
OUT="$RLROOT/data/models/rl_dk9_ppo.pt"
MATCH="${SSBU_LIVE_MATCH:-p1=donkey,p1=human,p2=donkey,p2=cpu,cpu=9,stage=end,players=2,stock=255,time=999,items=off,hazards=off}"
BOOT=1
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  MODEL="$2"; shift 2 ;;
        --out)    OUT="$2"; shift 2 ;;
        --match)  MATCH="$2"; shift 2 ;;
        --attach) BOOT=0; shift ;;
        --help|-h) sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)        ARGS+=("$1"); shift ;;
    esac
done

export SSBU_KEYBOARD_PLAYER=0
export SSBU_INPUT_NO_FOCUS=1
export SSBU_RL_MODE=1
unset SSBU_SEQ SSBU_PLAY SSBU_PRESS SSBU_SCRIPT

CPU_LEVEL="$(printf '%s' "$MATCH" | sed -n 's/.*cpu=\([0-9][0-9]*\).*/\1/p' | tail -1)"

if [[ $BOOT -eq 1 ]]; then
    "$HERE/live.sh" --no-focus --keyboard 0 --match "$MATCH" || exit 1
fi

exec "$PY" -m rl.rl.ppo --model "$MODEL" --out "$OUT" --start-menus \
    --cpu-level "${CPU_LEVEL:-0}" "${ARGS[@]}"
