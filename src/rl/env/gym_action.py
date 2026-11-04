#!/usr/bin/env python3
"""Canonical SSBU action space and decoders.

The canonical action vector (CONTRACTS.md) is::

    buttons: A B X Y L R ZL ZR PLUS MINUS DPAD_L DPAD_R   (12, 0/1)
    stick:   (x, y) in [-1, 1]
    cstick:  (x, y) in [-1, 1]
    trigger: float in [0, 1]                (always 0 in the gym spaces)

``make_action_space()`` returns the policy-facing space:

    gymnasium.spaces.Dict({
        "buttons": MultiBinary(12),
        "stick":   Box(-1, 1, (2,), float32),
        "cstick":  Box(-1, 1, (2,), float32),
    })

or, with ``macro=True``, a single ``Discrete(len(MACROS))``.  The macro list is
deliberately small and documented below; macro mode is a convenience for
scripted exploration, *not* the BC action space (recordings use the raw
buttons bitmask and sticks).

``decode_action`` turns either native space sample into the *canonical action
dict* consumed by the uinput writer (``gym_input``)::

    {
      "buttons":     int8 ndarray (12,), contract order, 0/1
      "button_names": tuple of pressed names, e.g. ("A", "DPAD_R")
      "buttons_mask": int, bit i == buttons[i]
      "stick":       float32 ndarray (2,)
      "cstick":      float32 ndarray (2,)
      "trigger":     float, always 0.0 for the gym spaces
      "macro":       macro name or None
    }

This module has no dependency on the shared-memory layout and can be unit
tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from gymnasium import spaces

#: Button names in canonical order.  The order is part of the contract and is
#: used by the recorder/BC dataset, so never reorder this tuple.
BUTTONS: tuple[str, ...] = (
    "A", "B", "X", "Y", "L", "R", "ZL", "ZR", "PLUS", "MINUS", "DPAD_L", "DPAD_R",
)
N_BUTTONS = len(BUTTONS)
BUTTON_INDEX = {name: i for i, name in enumerate(BUTTONS)}

STICK_SHAPE = (2,)
STICK_LOW = -1.0
STICK_HIGH = 1.0
TRIGGER_DEFAULT = 0.0

#: Observation/action arrays use float32, matching the dataset schema.
DTYPE = np.float32
BUTTON_DTYPE = np.int8


@dataclass(frozen=True)
class Macro:
    """One entry of the optional discrete macro action space."""

    name: str
    buttons: tuple[str, ...] = ()
    stick: tuple[float, float] = (0.0, 0.0)
    cstick: tuple[float, float] = (0.0, 0.0)


#: Documented macro list.  Index in this tuple == Discrete action value.
#: These are the human-readable, minimal actions a scripted policy needs:
#: neutral, the four stick directions (full tilt -> walk/dash), jump, attack,
#: special, shield, grab, and the two forward smash directions.
MACROS: tuple[Macro, ...] = (
    Macro("NOOP"),
    Macro("LEFT", stick=(-1.0, 0.0)),
    Macro("RIGHT", stick=(1.0, 0.0)),
    Macro("UP", stick=(0.0, 1.0)),
    Macro("DOWN", stick=(0.0, -1.0)),
    Macro("JUMP", buttons=("X",)),
    Macro("ATTACK", buttons=("A",)),
    Macro("SPECIAL", buttons=("B",)),
    Macro("SHIELD", buttons=("L",)),
    Macro("GRAB", buttons=("ZR",)),
    Macro("SMASH_LEFT", buttons=("A",), stick=(-1.0, 0.0)),
    Macro("SMASH_RIGHT", buttons=("A",), stick=(1.0, 0.0)),
)
MACRO_INDEX = {macro.name: i for i, macro in enumerate(MACROS)}


# --------------------------------------------------------------------------
# spaces
# --------------------------------------------------------------------------
def make_action_space(macro: bool = False) -> spaces.Space:
    """Build a fresh action space (``Dict`` by default, ``Discrete`` macros)."""
    if macro:
        return spaces.Discrete(len(MACROS))
    return spaces.Dict({
        "buttons": spaces.MultiBinary(N_BUTTONS),
        "stick": spaces.Box(low=STICK_LOW, high=STICK_HIGH, shape=STICK_SHAPE,
                            dtype=DTYPE),
        "cstick": spaces.Box(low=STICK_LOW, high=STICK_HIGH, shape=STICK_SHAPE,
                             dtype=DTYPE),
    })


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _clamp_stick(value, name: str = "stick") -> np.ndarray:
    arr = np.asarray(value, dtype=DTYPE)
    if arr.shape != (2,):
        raise ValueError(f"{name} must have shape (2,), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    return np.clip(arr, STICK_LOW, STICK_HIGH).astype(DTYPE, copy=False)


def buttons_array(names: Iterable[str]) -> np.ndarray:
    """0/1 float-free ``int8[12]`` for an iterable of canonical button names."""
    out = np.zeros(N_BUTTONS, dtype=BUTTON_DTYPE)
    for name in names:
        try:
            out[BUTTON_INDEX[name]] = 1
        except KeyError as exc:
            raise ValueError(f"unknown button {name!r}; expected one of {BUTTONS}") from exc
    return out


def buttons_mask(buttons: Sequence[int]) -> int:
    """Bitmask with bit i set when ``buttons[i]`` is truthy (canonical order)."""
    arr = np.asarray(buttons)
    if arr.shape != (N_BUTTONS,):
        raise ValueError(f"buttons must have shape ({N_BUTTONS},), got {arr.shape}")
    mask = 0
    for i, bit in enumerate(arr):
        if bit:
            mask |= 1 << i
    return mask


def pressed_buttons(buttons: Sequence[int]) -> tuple[str, ...]:
    arr = np.asarray(buttons)
    return tuple(name for name, bit in zip(BUTTONS, arr) if bit)


def _canonical(buttons, stick, cstick, macro: str | None = None,
               trigger: float = TRIGGER_DEFAULT) -> dict:
    b = np.asarray(buttons, dtype=BUTTON_DTYPE)
    if b.shape != (N_BUTTONS,):
        raise ValueError(f"buttons must have shape ({N_BUTTONS},), got {b.shape}")
    b = (b != 0).astype(BUTTON_DTYPE)
    return {
        "buttons": b,
        "button_names": pressed_buttons(b),
        "buttons_mask": buttons_mask(b),
        "stick": _clamp_stick(stick, "stick"),
        "cstick": _clamp_stick(cstick, "cstick"),
        "trigger": float(np.clip(trigger, 0.0, 1.0)),
        "macro": macro,
    }


def neutral_action() -> dict:
    """A fresh canonical no-op action dict."""
    return _canonical(np.zeros(N_BUTTONS, dtype=BUTTON_DTYPE),
                      np.zeros(2, dtype=DTYPE), np.zeros(2, dtype=DTYPE))


def macro_action(index: int) -> dict:
    """Decode one discrete macro index into a canonical action dict."""
    try:
        index = int(index)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"macro action must be an integer, got {index!r}") from exc
    if not (0 <= index < len(MACROS)):
        raise ValueError(f"macro index {index} out of range [0, {len(MACROS)})")
    macro = MACROS[index]
    return _canonical(buttons_array(macro.buttons), macro.stick, macro.cstick,
                      macro=macro.name)


# --------------------------------------------------------------------------
# decode / encode
# --------------------------------------------------------------------------
def decode_action(action, macro: bool = False) -> dict:
    """Decode a gym action into the canonical action dict.

    Accepted inputs:
      * ``Dict`` sample from :func:`make_action_space` (dict with
        ``buttons``/``stick``/``cstick``), or an already-canonical dict;
      * a flat length-16 vector ``[buttons(12), stick(2), cstick(2)]``;
      * an integer macro index when ``macro=True``.

    Sticks are clipped to [-1, 1].  Buttons are coerced to 0/1 ``int8``.
    """
    if macro:
        if isinstance(action, (int, np.integer)):
            return macro_action(action)
        # A native Dict action is also accepted in macro mode for convenience;
        # it cannot be represented as a macro, so ``macro`` stays None.
        if isinstance(action, dict) or np.ndim(action) == 1:
            return decode_action(action, macro=False)
        raise ValueError(f"unsupported macro action {action!r}")

    if isinstance(action, dict):
        if not all(k in action for k in ("buttons", "stick", "cstick")):
            raise ValueError(
                "action dict must contain 'buttons', 'stick' and 'cstick'; "
                f"got keys {sorted(action)}")
        return _canonical(action["buttons"], action["stick"], action["cstick"],
                          macro=action.get("macro"))

    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.size == N_BUTTONS + 4:
        return _canonical(arr[:N_BUTTONS], arr[N_BUTTONS:N_BUTTONS + 2],
                          arr[N_BUTTONS + 2:])
    if arr.size == N_BUTTONS:
        return _canonical(arr, np.zeros(2), np.zeros(2))
    raise ValueError(
        f"cannot decode action of shape {np.shape(action)}; expected a Dict, "
        f"a length-{N_BUTTONS + 4} vector, or a macro index")


def encode_action(canonical: dict) -> dict:
    """Convert a canonical action dict back into the native Dict space sample.

    This is the inverse of :func:`decode_action` for the button/stick/cstick
    part (trigger and macro metadata are not part of the space).
    """
    can = decode_action(canonical)
    return {
        "buttons": can["buttons"].copy(),
        "stick": can["stick"].copy(),
        "cstick": can["cstick"].copy(),
    }


def action_metadata() -> dict:
    """Small JSON-able description of the action space for docs/records."""
    return {
        "buttons": list(BUTTONS),
        "stick": {"shape": list(STICK_SHAPE), "low": STICK_LOW, "high": STICK_HIGH},
        "cstick": {"shape": list(STICK_SHAPE), "low": STICK_LOW, "high": STICK_HIGH},
        "trigger": {"low": 0.0, "high": 1.0, "gym_default": TRIGGER_DEFAULT},
        "macros": [m.name for m in MACROS],
    }
