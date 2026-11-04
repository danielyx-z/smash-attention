#!/usr/bin/env bash
# Give the game window the X11 keyboard focus.
#
# The game only reads the keyboard while its SDL window has focus, and
# `xdotool windowactivate` (EWMH) does not work under this mutter setup.
# `xdotool windowfocus` calls XSetInputFocus directly and does work.
set -uo pipefail
export DISPLAY="${SSBU_LIVE_DISPLAY:-:0}"

win="$(xdotool search --class ssbu_native 2>/dev/null | tail -1)"
[[ -z "$win" ]] && win="$(xdotool search --name 'SSBU native' 2>/dev/null | tail -1)"
if [[ -z "$win" ]]; then
    echo "no SSBU window found on $DISPLAY" >&2
    exit 1
fi
xdotool windowraise "$win" 2>/dev/null
xdotool windowactivate --sync "$win" 2>/dev/null
# SDL only raises its keyboard-focus flag on a real FocusIn.  Click the HUD
# first, then the game, so the WM really moves focus away and back.
hud="$(xdotool search --name 'SSBU state verifier' 2>/dev/null | tail -1)"
if [[ -n "$hud" && "$hud" != "$win" ]]; then
    xdotool mousemove --window "$hud" 60 60 2>/dev/null
    xdotool click 1 2>/dev/null
    sleep 0.4
fi
xdotool windowraise "$win" 2>/dev/null
xdotool windowfocus "$win" 2>/dev/null
xdotool mousemove --window "$win" 640 360 2>/dev/null
xdotool click 1 2>/dev/null
xdotool windowfocus "$win" 2>/dev/null
echo "focused $win ($(xdotool getwindowname "$win" 2>/dev/null))"
