#!/usr/bin/env bash
# One-command live setup for the external memory layer.
#
#   src/rl/scripts/live.sh            game + 60 fps reader + stage view + focus
#   src/rl/scripts/live.sh --attach   keep the running game, restart the rest
#   src/rl/scripts/live.sh --no-focus do not install the focus keeper
#   src/rl/scripts/live.sh --match '...'  SSBU_DEBUG_MATCH string
#   src/rl/scripts/live.sh --keyboard N   bind the keyboard to player N (1..8)
#
# Focus: the game reads the keyboard only when GetForegroundWindow() returns
# its SDL window, which needs SDL_GetKeyboardFocus() == that window.  Setting
# X input focus (xdotool windowfocus) to a window that already has it produces
# no FocusIn, so SDL keeps ignoring keys.  live.sh moves focus to the HUD with
# a real click and then back to the game; focus_game.sh does the same by hand.
# The keeper only pulls focus back when it is on neither the game nor the HUD.
#
# One-time game setup: the saved roster makes player 1 a CPU and the rule
# 0-stock, so set player 1 to Human and the rule to 3 stocks in the game's own
# character-select screen (see README).  Then Mario is manually controllable.
set -uo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
RLROOT="$(cd "$HERE/../../.." && pwd)"
REPO="${SSBU_RECOMP_DIR:-$(cd "$RLROOT/../ssbu-recomp" && pwd)}"
PY="$RLROOT/.venv/bin/python"
# Some login shells here export a dead DISPLAY=:10; the desktop is :0.
export DISPLAY="${SSBU_LIVE_DISPLAY:-:0}"

MATCH="${SSBU_LIVE_MATCH:-p1=mario,p2=donkey,cpu=0}"
#: The saved select-screen roster decides who is human; the debug harness can
#: only force the second panel's type.  Set player 1 to human once in the
#: game's own character-select screen (see README), then the keyboard controls
#: Mario.  KB_PLAYER follows that choice.
KB_PLAYER="${SSBU_LIVE_KEYBOARD_PLAYER:-1}"
ATTACH=0
KEEP_FOCUS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --attach)    ATTACH=1; shift ;;
        --no-focus)  KEEP_FOCUS=0; shift ;;
        --match)     MATCH="$2"; shift 2 ;;
        --keyboard)  KB_PLAYER="$2"; shift 2 ;;
        -h|--help)   sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

command -v xdotool >/dev/null 2>&1 || {
    echo "warning: xdotool not found; the game may not receive the keyboard" >&2
}

# -- stop the previous copy -------------------------------------------------
# Anchored on the full python command line so a shell that merely mentions the
# path (e.g. this script's own caller) is never matched and killed.
stop_pat() {
    local pids
    pids="$(pgrep -f "$1" 2>/dev/null || true)"
    [[ -n "$pids" ]] && kill $pids 2>/dev/null
}
if [[ $ATTACH -eq 0 ]]; then
    pkill -x ssbu_native 2>/dev/null
fi
stop_pat 'python3? .*rl\.memory\.reader'
stop_pat 'python3? .*rl\.scripts\.live_view'
[[ -f "$HOME/.ssbu_focus.pid" ]] && kill "$(cat "$HOME/.ssbu_focus.pid")" 2>/dev/null
rm -f "$HOME/.ssbu_focus.pid"
sleep 1

# -- start the game ---------------------------------------------------------
# setsid + nohup + full redirection: the child is in its own session and does
# not keep this script (or its caller's pipe) alive.
if [[ $ATTACH -eq 0 ]]; then
    echo "starting game (log: ~/smash.log)"
    # play.sh drops every SSBU_* variable of its own environment (run.py), so
    # any runtime switch this script was started with has to be passed --env.
    EXTRA_ENV=()
    while IFS= read -r kv; do
        [[ -n "$kv" ]] && EXTRA_ENV+=(--env "$kv")
    done < <(env | grep -E '^SSBU_' | grep -vE '^SSBU_LIVE_|^SSBU_READER_FPS=')
    setsid nohup bash -c 'cd "$1" && shift && exec "$@"' _ "$REPO" \
        ./play.sh \
        --env SSBU_SKIP_MOVIES=1 \
        --env "SSBU_KEYBOARD_PLAYER=$KB_PLAYER" \
        --env SSBU_RL=1 \
        --env "SSBU_DEBUG_MATCH=$MATCH" \
        "${EXTRA_ENV[@]}" \
        > "$HOME/smash.log" 2>&1 < /dev/null &
    disown
fi

for _ in $(seq 1 120); do
    pgrep -x ssbu_native >/dev/null 2>&1 && break
    sleep 0.5
done
if ! pgrep -x ssbu_native >/dev/null 2>&1; then
    echo "error: the game did not start (see ~/smash.log)" >&2
    exit 1
fi

# -- expert recorder (optional, on from the start) --------------------------
# SSBU_LIVE_RECORD=<dir>: start gym_record with the game, so every fight is
# collected from its first frame (the recorder polls the state block and starts
# an episode on fight_on 0->1; it must be up before the match, which is why it
# belongs here and not in a later step).
if [[ -n "${SSBU_LIVE_RECORD:-}" ]]; then
    echo "starting recorder (log: ~/ssbu_record.log) -> $SSBU_LIVE_RECORD"
    setsid nohup "$PY" -m rl.record.gym_record \
        --episodes "${SSBU_LIVE_RECORD_EPISODES:-200}" \
        --out "$SSBU_LIVE_RECORD" \
        --episode-frames "${SSBU_LIVE_EPISODE_FRAMES:-5400}" \
        --cpu-level "${SSBU_LIVE_RECORD_CPU:-9}" --stage "${SSBU_LIVE_RECORD_STAGE:-end}" \
        --no-lock > "$HOME/ssbu_record.log" 2>&1 < /dev/null &
    disown
fi

# -- reader + verifier ------------------------------------------------------
# At SSBU_SPEED=n the game presents n*60 frames/s.  The reader must sample at
# that rate too, otherwise env frame_skip counts skipped game frames and the
# policy sees a coarser, shifted action interval.
SPEED="${SSBU_SPEED:-1}"
READER_FPS="${SSBU_READER_FPS:-$(awk -v s="$SPEED" 'BEGIN { print (s > 0 ? 60*s : 60) }')}"
echo "starting reader at ${READER_FPS} fps (log: ~/ssbu_reader.log)"
setsid nohup "$PY" -m rl.memory.reader --fps "$READER_FPS" \
    > "$HOME/ssbu_reader.log" 2>&1 < /dev/null &
disown

echo "starting stage view (log: ~/ssbu_view.log)"
setsid nohup "$PY" -m rl.scripts.live_view \
    > "$HOME/ssbu_view.log" 2>&1 < /dev/null &
disown

# -- focus the game ---------------------------------------------------------
# A locked GNOME session is the most common reason the game "does not take
# input": the game keeps running behind the lock screen and no keystroke ever
# reaches it.  Unlocking our own session needs no password and no root.
if [[ "${SSBU_LIVE_UNLOCK:-1}" == "1" ]] && command -v loginctl >/dev/null 2>&1; then
    for s in $(loginctl list-sessions --no-legend 2>/dev/null | awk '{print $1}'); do
        t="$(loginctl show-session "$s" -p Type --value 2>/dev/null)"
        a="$(loginctl show-session "$s" -p Active --value 2>/dev/null)"
        l="$(loginctl show-session "$s" -p LockedHint --value 2>/dev/null)"
        if [[ "$t" == "x11" && "$a" == "yes" && "$l" == "yes" ]]; then
            if loginctl unlock-session "$s" 2>/dev/null; then
                echo "unlocked the X11 session $s"
            else
                echo "warning: session $s is locked; unlock it to see the game" >&2
            fi
        fi
    done
fi

sleep 2
GAME_WIN="$(xdotool search --class ssbu_native 2>/dev/null | tail -1)"
[[ -z "$GAME_WIN" ]] && GAME_WIN="$(xdotool search --name 'SSBU native' 2>/dev/null | tail -1)"
if [[ -n "$GAME_WIN" ]]; then
    xdotool windowraise "$GAME_WIN" 2>/dev/null
    xdotool windowactivate --sync "$GAME_WIN" 2>/dev/null
    # SDL only updates SDL_GetKeyboardFocus on a real FocusIn event.  Setting
    # X input focus to a window that already has it generates no event, so the
    # game keeps ignoring keys.  Move focus to the HUD with a real click, then
    # back to the game with another click; that produces FocusOut/FocusIn.
    HUD_WIN="$(xdotool search --name 'SSBU live view' 2>/dev/null | tail -1)"
    if [[ -n "$HUD_WIN" ]]; then
        xdotool mousemove --window "$HUD_WIN" 60 60 2>/dev/null
        xdotool click 1 2>/dev/null
        sleep 0.4
    fi
    xdotool windowraise "$GAME_WIN" 2>/dev/null
    xdotool windowfocus "$GAME_WIN" 2>/dev/null
    xdotool mousemove --window "$GAME_WIN" 640 360
    sleep 0.1
    xdotool click 1
    sleep 0.2
    xdotool windowfocus "$GAME_WIN" 2>/dev/null
    echo "focused game window $GAME_WIN"
else
    echo "warning: game window not found yet; run src/rl/scripts/focus_game.sh" >&2
fi

if [[ $KEEP_FOCUS -eq 1 ]]; then
    setsid nohup bash -c '
        while true; do
            g="$(xdotool search --class ssbu_native 2>/dev/null | tail -1)"
            [ -z "$g" ] && g="$(xdotool search --name "SSBU native" 2>/dev/null | tail -1)"
            if [ -n "$g" ]; then
                f="$(xdotool getwindowfocus getwindowname 2>/dev/null)"
                case "$f" in
                    *"SSBU native"*|*"SSBU live view"*) ;;
                    *) xdotool windowfocus "$g" 2>/dev/null ;;
                esac
            fi
            sleep 0.5
        done
    ' > /dev/null 2>&1 < /dev/null &
    disown
    echo $! > "$HOME/.ssbu_focus.pid"
    echo "focus keeper on (pid $(cat "$HOME/.ssbu_focus.pid"))"
fi

echo
echo "live.  controls: click the game window, then use the keyboard."
echo "  default keys: J/L = left/right, I/K = up/down, Z = A (jump), X = B,"
echo "                C = X, V = Y, Q/E = L/R, A/S = ZL/ZR, Enter = +"
echo "  stage view: P = pause, T = trails, H = raw hex, S = snapshot, Q = quit"
echo "  a tutorial popup in the game is dismissed with Z (the A button)"
echo "  logs: ~/smash.log  ~/ssbu_reader.log  ~/ssbu_view.log"
