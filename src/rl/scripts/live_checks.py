#!/usr/bin/env python3
"""Live checks for the runtime debug-match patch (observe only, no writes).

Three properties of the rebuilt game are checked here:

``rules``   ``SSBU_DEBUG_MATCH stock=N`` is honoured: both slots report N
            stocks (not the saved 0-stock rule) and the match does not end at
            the first KO.  Also checks ``items=off``: the extra block's match
            rule says ``g.match.item_appear == 0`` and no non-fighter battle
            objects appear in the first seconds.

``focus``   ``SSBU_INPUT_NO_FOCUS=1`` makes the uinput pad work while the game
            window does *not* hold X11 focus.  The game window is required to
            be unfocused (it is moved out of focus if it happens to hold it),
            the virtual pad is created, and P1 is asked to walk right for ~1.5 s
            with no focus.  P1's x must increase by more than ``--min-dx``.
            The expert ``buttons``/``stick_x`` fields are recorded as info.

``all``     ``rules`` followed by ``focus``.

The tool only observes the game; it never writes game memory, never runs the
game and never resets a match.  It does create the uinput pad in ``focus`` mode.

Exit codes: ``0`` pass, ``1`` check failed, ``2`` setup problem or skipped
(no live match, no extra block for a mode that needs one, missing
``SSBU_INPUT_NO_FOCUS``, no pad, no X display, ...).

Usage::

    .venv/bin/python -m rl.scripts.live_checks rules --json
    .venv/bin/python -m rl.scripts.live_checks rules --seconds 25 --expected-stock 3
    .venv/bin/python -m rl.scripts.live_checks focus --json
    .venv/bin/python -m rl.scripts.live_checks all --timeout 60

The stock expectation defaults to ``SSBU_DEBUG_MATCH stock=N`` when that
variable is set in this shell, otherwise to 3 (``--expected-stock`` wins).
``items`` is always checked against 0, which is the point of this mode; the
debug string's ``items=`` value is only recorded in the JSON summary.

Notes for the live run
----------------------
* ``rules`` counts *stock drops* to decide whether a match end is legitimate.
  With ``stock=3`` the match must only end after three KOs, even if the results
  screen zeroes the stock values.  Ending after one KO (the saved 0-stock rule)
  is a failure.
* The extra block does not publish a battle-object class, so every non-fighter
  ``bom.obj.*`` object is reported as an item candidate.  Run ``rules`` with no
  inputs (for example ``p1=human,p2=donkey,cpu=0``) so a Fireball cannot be
  mistaken for an item.  That is conservative on purpose: it can flag a
  projectile, but it never silently passes an item.
* ``focus`` must run with the game launched with ``SSBU_INPUT_NO_FOCUS=1``.
  The flag is read from this shell and from ``/proc/<pid>/environ`` of the
  running ``ssbu_native`` process.  A missing flag is a skip (exit 2), not a
  traceback.
* ``focus`` needs a **live match**.  ``SSBU_DEBUG_MATCH`` only forces the
  select screens; something must walk from the title screen to the character
  select.  With ``SSBU_SEQ``/``SSBU_PLAY`` set, this check no longer taps A
  (the walk owns the menus) but it still waits for the match: start the walk
  first, e.g. ``wait_live 150 && live_checks focus``.  Without a script the A
  taps can dismiss the title screen, but they cannot navigate the menus.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from rl.env.gym_env import (
    SHM_PATH,
    RawSnapshotReader,
    SnapshotError,
    SnapshotTimeout,
)
from rl.extra.extra_state import SHM_PATH as EXTRA_SHM_PATH
from rl.extra.extra_state import read_block
from rl.input.gym_input import VirtualPad

TOOL = "live_checks"
DEFAULT_TIMEOUT = 60.0
DEFAULT_SECONDS = 25.0
DEFAULT_EXPECTED_STOCK = 3
DEFAULT_HOLD = 1.5
DEFAULT_HOLD_GRACE = 4.0
DEFAULT_MIN_DX = 3.0
DEFAULT_DISPLAY = ":0"

NO_FOCUS_ENV = "SSBU_INPUT_NO_FOCUS"
DEBUG_MATCH_ENV = "SSBU_DEBUG_MATCH"
NEUTRAL = {"buttons": [], "stick": (0.0, 0.0), "cstick": (0.0, 0.0)}

#: Known free/poison BattleObjectManager slot id (filtered by extra_state too).
FREE_OBJECT_ID = 0x50000000
#: The two fighter object ids seen live (pool 0).
FIGHTER_OBJECT_IDS = (0, 0x10000)

_LEVELS = {True: "PASS", False: "FAIL", None: "INFO"}


# ---------------------------------------------------------------------------
# result model
# ---------------------------------------------------------------------------
@dataclass
class Check:
    """One pass/fail observation.  ``ok=None`` means info only."""

    name: str
    ok: bool | None
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class Report:
    """One mode's checks, summary and exit status."""

    mode: str
    checks: list[Check] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None
    children: list["Report"] = field(default_factory=list)

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.checks.append(Check(name, ok, detail))

    def skip(self, reason: str) -> None:
        """Mark the run inconclusive (exit 2) with a readable reason."""
        self.skipped = reason
        self.add("setup", None, reason)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.checks if c.ok is False)

    @property
    def passes(self) -> int:
        return sum(1 for c in self.checks if c.ok is True)

    @property
    def infos(self) -> int:
        return sum(1 for c in self.checks if c.ok is None)

    def exit_code(self) -> int:
        if self.children:
            codes = [child.exit_code() for child in self.children]
            if 2 in codes:
                return 2
            if 1 in codes:
                return 1
            return 0
        if self.skipped is not None:
            return 2
        if self.failures:
            return 1
        return 0

    @property
    def status(self) -> str:
        code = self.exit_code()
        return "pass" if code == 0 else "fail" if code == 1 else "skip"

    def as_dict(self) -> dict:
        out = {
            "tool": TOOL,
            "mode": self.mode,
            "status": self.status,
            "exit_code": self.exit_code(),
            "checks": [c.as_dict() for c in self.checks],
            "summary": self.summary,
        }
        if self.skipped is not None:
            out["skipped"] = self.skipped
        if self.children:
            out["reports"] = [child.as_dict() for child in self.children]
        return out


def format_report(report: Report) -> str:
    """Human-readable report text (no stdout side effect; tests use this)."""
    lines = [f"{TOOL} {report.mode}: {report.status.upper()}"]
    for check in report.checks:
        lines.append(f"  [{_LEVELS[check.ok]}] {check.name:<52s} {check.detail}")
    if report.children:
        for child in report.children:
            lines.append("")
            lines.extend(format_report(child).splitlines())
    if report.skipped is not None:
        lines.append(f"  skipped: {report.skipped}")
    lines.append(f"  {report.passes} pass, {report.failures} fail, "
                 f"{report.infos} info")
    return "\n".join(lines)


def print_report(report: Report, stream=None) -> None:
    print(format_report(report), file=stream or sys.stdout)


def _progress(args: argparse.Namespace, message: str) -> None:
    """Progress goes to stderr in --json mode so stdout stays one document."""
    stream = sys.stderr if getattr(args, "json", False) else sys.stdout
    print(message, file=stream, flush=True)


def _neutral(pad) -> None:
    """Release every control, tolerating a vanished uinput device."""
    try:
        pad.apply(NEUTRAL)
    except Exception:                        # pragma: no cover - teardown
        pass


# ---------------------------------------------------------------------------
# SSBU_DEBUG_MATCH parsing
# ---------------------------------------------------------------------------
def parse_debug_match(value: str | None) -> dict[str, str]:
    """Parse ``p1=mario,p2=donkey,items=off,stock=3`` into a dict.

    Keys are lower-cased; a bare token is kept with an empty value.  Empty
    tokens (``,,``) are ignored and the last duplicate wins.
    """
    out: dict[str, str] = {}
    for token in str(value or "").split(","):
        token = token.strip()
        if not token:
            continue
        key, sep, val = token.partition("=")
        key = key.strip().lower()
        if not key:
            continue
        out[key] = val.strip() if sep else ""
    return out


@dataclass(frozen=True)
class DebugMatchExpectations:
    """What ``SSBU_DEBUG_MATCH`` asked the runtime to force."""

    options: dict[str, str] = field(default_factory=dict)
    stock: int | None = None
    items_off: bool | None = None


def _parse_int(value: Any) -> int | None:
    try:
        return int(str(value).strip(), 0)
    except (TypeError, ValueError):
        return None


def _parse_switch(value: str | None) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in ("off", "0", "false", "no", "none", "disable", "disabled"):
        return True
    if text in ("on", "1", "true", "yes", "enable", "enabled"):
        return False
    return None


def debug_match_expectations(value: str | None) -> DebugMatchExpectations:
    """Extract the stock and item expectations from a debug-match string."""
    options = parse_debug_match(value)
    stock = _parse_int(options.get("stock"))
    if stock is not None and stock < 0:
        stock = None
    items_off = _parse_switch(options.get("items"))
    return DebugMatchExpectations(options=options, stock=stock,
                                  items_off=items_off)


def resolve_expected_stock(cli_value: int | None,
                           debug_env: str | None = None) -> int:
    """CLI ``--expected-stock`` > ``SSBU_DEBUG_MATCH stock=`` > 3."""
    if cli_value is not None:
        return int(cli_value)
    stock = debug_match_expectations(debug_env).stock
    return DEFAULT_EXPECTED_STOCK if stock is None else int(stock)


# ---------------------------------------------------------------------------
# snapshots
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FrameSample:
    """The few core-state fields the checks need (plain, testable)."""

    seq: int = 0
    ts_ns: int = 0
    fight_on: int = 0
    stocks: tuple[int, int] = (0, 0)
    x: tuple[float, float] = (0.0, 0.0)
    buttons: tuple[int, int] = (0, 0)
    stick_x: tuple[float, float] = (0.0, 0.0)
    is_cpu: tuple[int, int] = (0, 0)

    @property
    def live(self) -> bool:
        return int(self.fight_on) == 1


def _field(fighters: Sequence[Mapping], slot: int, name: str, default):
    if slot < len(fighters):
        value = fighters[slot].get(name, default)
        return default if value is None else value
    return default


def sample_frame(snapshot) -> FrameSample:
    """Normalise a ``RawSnapshot`` (or a compatible object) to a FrameSample."""
    fighters = tuple(getattr(snapshot, "fighters", ()) or ())
    return FrameSample(
        seq=int(getattr(snapshot, "seq", 0) or 0),
        ts_ns=int(getattr(snapshot, "ts_ns", 0) or 0),
        fight_on=int(getattr(snapshot, "fight_on", 0) or 0),
        stocks=tuple(int(_field(fighters, slot, "stock", 0))
                     for slot in (0, 1)),
        x=tuple(float(_field(fighters, slot, "x", 0.0))
                for slot in (0, 1)),
        buttons=tuple(int(_field(fighters, slot, "buttons", 0))
                      for slot in (0, 1)),
        stick_x=tuple(float(_field(fighters, slot, "stick_x", 0.0))
                      for slot in (0, 1)),
        is_cpu=tuple(int(_field(fighters, slot, "is_cpu", 0))
                     for slot in (0, 1)),
    )


def wait_for_live(reader: RawSnapshotReader, timeout: float,
                  settle: int = 20) -> Any:
    """Wait for a live match whose fighter rows have stopped changing identity.

    ``impl+0xd2`` goes to 1 a few frames before the entries hold the new match,
    so the first ~15 frames can still carry the previous match.  Require the
    ``(kind, entry_id)`` tuple to stay put for ``settle`` new frames.
    """
    settle = max(1, int(settle))
    deadline = time.monotonic() + max(0.0, float(timeout))
    last = None
    last_seq = -1
    key = None
    stable = 0
    while time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            snap = reader.read_latest(timeout=min(0.5, remaining))
        except SnapshotTimeout:
            continue
        last = snap
        live = (int(snap.fight_on) == 1 and int(snap.n_fighters) >= 2 and
                all(int(f.get("raw_flags", 0)) for f in snap.fighters))
        if not live:
            key = None
            stable = 0
            last_seq = -1
            continue
        if snap.seq == last_seq:
            continue
        last_seq = snap.seq
        now_key = tuple((int(f.get("kind", -1)), int(f.get("entry_id", -1)))
                        for f in snap.fighters)
        stable = stable + 1 if now_key == key else 1
        key = now_key
        if stable >= settle:
            return snap
    raise TimeoutError(
        "no stable live match within "
        f"{float(timeout):.0f}s (last fight_on="
        f"{getattr(last, 'fight_on', None)}, "
        f"n_fighters={getattr(last, 'n_fighters', None)}, "
        f"stable={stable}/{settle}); start a match or raise --timeout")


# ---------------------------------------------------------------------------
# rules: stock evaluation
# ---------------------------------------------------------------------------
def _span_seconds(frames: Sequence[FrameSample]) -> float:
    if len(frames) < 2:
        return 0.0
    return max(0.0, (frames[-1].ts_ns - frames[0].ts_ns) / 1e9)


def evaluate_stock_run(frames: Sequence[FrameSample], expected_stock: int,
                       ) -> tuple[list[Check], dict]:
    """Evaluate initial stocks and match lifetime from sampled frames.

    With ``expected_stock == N > 1`` a match end is legitimate only after N
    observed stock drops for one slot.  The results screen can zero the stock
    values, so a zero alone is not enough evidence; ending after one KO is the
    saved 0-stock rule and must fail.  With one expected stock a zero in the
    last live frame is sufficient.
    """
    checks: list[Check] = []
    frames = list(frames)
    live = [f for f in frames if f.live]
    summary: dict[str, Any] = {
        "expected_stock": int(expected_stock),
        "frames": len(frames),
        "live_frames": len(live),
        "observed_s": round(_span_seconds(frames), 2),
    }
    if not live:
        checks.append(Check("live match observed", False,
                            "no fight_on=1 frames were captured"))
        return checks, summary

    first = live[0]
    summary["stocks_initial"] = list(first.stocks)
    checks.append(Check("fight_on == 1 at the start", True,
                        f"seq={first.seq} stocks={list(first.stocks)}"))
    for slot in (0, 1):
        value = int(first.stocks[slot])
        if int(expected_stock) > 0:
            ok = value == int(expected_stock) and value > 0
            name = f"slot {slot} stock reads {int(expected_stock)}"
        else:
            ok = value > 0
            name = f"slot {slot} stock reads > 0"
        checks.append(Check(name, ok, f"stock={value}"))

    drops = [0, 0]
    for prev, curr in zip(frames, frames[1:]):
        if not prev.live or not curr.live:
            # Only count KOs observed inside the match.  The transition to the
            # results screen can zero the stock values; that is not a KO.
            continue
        for slot in (0, 1):
            delta = int(prev.stocks[slot]) - int(curr.stocks[slot])
            if delta > 0:
                drops[slot] += delta
    summary["stock_drops"] = list(drops)
    summary["stocks_final"] = list(live[-1].stocks)

    end_index = None
    for i in range(1, len(frames)):
        if frames[i - 1].live and not frames[i].live:
            end_index = i
            break

    if end_index is None:
        checks.append(Check(
            "match did not end during the observation window", True,
            f"{_span_seconds(frames):.1f}s, stocks "
            f"{list(live[0].stocks)} -> {list(live[-1].stocks)}"))
        summary["ended"] = False
    else:
        last_live = next((frames[j] for j in range(end_index - 1, -1, -1)
                          if frames[j].live), first)
        end_stocks = last_live.stocks
        ko_counts = (max(drops) if drops else 0)
        zeroed = any(int(stock) <= 0 for stock in end_stocks)
        enough_kos = (int(expected_stock) > 0 and
                      max(drops) >= int(expected_stock))
        if int(expected_stock) > 1:
            # Starting at N stocks, only N observed KOs make an end legitimate.
            # Do not accept a zeroed results screen alone: the saved 0-stock
            # rule can also zero the values after the first KO.
            ok = bool(enough_kos)
        elif int(expected_stock) > 0:
            ok = bool(zeroed or enough_kos)
        else:
            ok = bool(zeroed)
        summary["ended"] = True
        summary["end_stocks"] = list(end_stocks)
        summary["end_after_s"] = round(
            _span_seconds(frames[:end_index]), 2)
        summary["ko_counts"] = list(drops)
        if ok:
            name = "match ended only after the expected stocks were lost"
        else:
            name = "match did not end at the first KO"
        checks.append(Check(
            name, ok,
            f"ended after {ko_counts} KO(s), stocks at end {list(end_stocks)}, "
            f"expected {int(expected_stock)}"))

    if drops[0] or drops[1]:
        detail = (f"slot0={drops[0]} slot1={drops[1]} KO(s); "
                  f"match {'ended' if end_index is not None else 'continued'}")
        checks.append(Check("KO observed during the observation window",
                            None, detail))
    else:
        checks.append(Check("KO observed during the observation window", None,
                            "no stock decreased; no KO happened yet"))
    return checks, summary


# ---------------------------------------------------------------------------
# rules: items / extra block
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BattleObject:
    """One non-fighter ``bom.obj.*`` record group from the extra block."""

    pool: int
    oid: int
    category: int
    accessor: int | None = None

    @property
    def label(self) -> str:
        return f"pool{self.pool}/0x{self.oid:x}"


def extra_pairs(block: Mapping) -> dict[tuple[str, int], Any]:
    """Flatten an unpacked extra block into ``(name, index) -> value``."""
    pairs: dict[tuple[str, int], Any] = {}
    for rec in block.get("records", ()) or ():
        try:
            value = rec.value_python()
        except AttributeError:                  # an already-plain value
            value = getattr(rec, "value", None)
        pairs[(str(rec.name), int(getattr(rec, "index", 0)))] = value
    return pairs


def extra_get(pairs: Mapping[tuple[str, int], Any], name: str,
              index: int = 0, default=None):
    return pairs.get((name, int(index)), default)


def battle_object_candidates(
        pairs: Mapping[tuple[str, int], Any]) -> list[BattleObject]:
    """Non-fighter battle objects, i.e. conservative item candidates.

    Fighters are recognised by matching ``bom.obj.accessor`` against the
    ``f.accessor`` records, or (when those are missing) by the two live
    fighter object ids in pool 0.  The extra block does not expose a battle
    object class, so everything else is reported rather than guessed away.
    """
    accessors = {
        int(value)
        for (name, _index), value in pairs.items()
        if name == "f.accessor" and isinstance(value, (int, float))
        and int(value) > 0x1000
    }
    indices = sorted({index for (name, index) in pairs
                      if name == "bom.obj.id"})
    out: list[BattleObject] = []
    for index in indices:
        oid = extra_get(pairs, "bom.obj.id", index)
        if not isinstance(oid, (int, float)):
            continue
        oid = int(oid)
        pool = int(extra_get(pairs, "bom.obj.pool", index, 0) or 0)
        category = int(extra_get(pairs, "bom.obj.category", index, 0) or 0)
        accessor = extra_get(pairs, "bom.obj.accessor", index)
        if isinstance(accessor, (int, float)):
            accessor = int(accessor)
        else:
            accessor = None
        is_fighter = bool(
            (accessor is not None and accessor in accessors) or
            (pool == 0 and oid in FIGHTER_OBJECT_IDS))
        if is_fighter or oid == FREE_OBJECT_ID:
            continue
        out.append(BattleObject(pool=pool, oid=oid, category=category,
                                accessor=accessor))
    return out


def evaluate_items(item_appear_values: Sequence[int],
                   object_snapshots: Sequence[Sequence[BattleObject]],
                   extra_available: bool,
                   extra_error: str | None = None,
                   extra_samples: int = 0) -> tuple[list[Check], dict]:
    """Evaluate ``g.match.item_appear == 0`` and item-object presence."""
    checks: list[Check] = []
    values = [int(v) for v in item_appear_values]
    objects: list[BattleObject] = []
    seen: set[str] = set()
    for snapshot in object_snapshots:
        for obj in snapshot:
            if obj.label not in seen:
                seen.add(obj.label)
                objects.append(obj)
    summary: dict[str, Any] = {
        "extra_available": bool(extra_available),
        "extra_samples": int(extra_samples),
        "item_appear_values": sorted(set(values)),
        "item_objects": [{"pool": o.pool, "id": o.oid,
                          "category": o.category} for o in objects],
    }
    if not extra_available:
        detail = "extra unavailable"
        if extra_error:
            detail += f" ({extra_error})"
        detail += f"; start the extras publisher on {EXTRA_SHM_PATH}"
        checks.append(Check("item rule: g.match.item_appear readable",
                            None, detail))
        checks.append(Check("no item battle objects in the first seconds",
                            None, detail))
        return checks, summary

    if not values:
        checks.append(Check("item rule: g.match.item_appear readable", None,
                            "extra block has no g.match.item_appear record"))
    else:
        ok = all(value == 0 for value in values)
        detail = f"values={sorted(set(values))} ({len(values)} samples)"
        checks.append(Check("g.match.item_appear == 0", ok, detail))

    if objects:
        labels = ", ".join(obj.label for obj in objects[:8])
        if len(objects) > 8:
            labels += f", ... (+{len(objects) - 8})"
        # The extra block has no battle-object class.  With the game's own
        # item rule at 0, a remaining non-fighter object is a fighter
        # projectile/effect, not an item; report it without failing the item
        # gate (the item_appear check above is authoritative).
        items_on = bool(values) and any(value != 0 for value in values)
        checks.append(Check(
            "no item battle objects in the first seconds",
            False if items_on else None,
            "non-fighter battle objects (unclassified): "
            f"{labels}" + ("" if items_on else
                           "; g.match.item_appear is 0, so these are not items")))
    else:
        checks.append(Check(
            "no item battle objects in the first seconds", True,
            f"0 non-fighter battle objects in {int(extra_samples)} extra "
            "sample(s)"))
    return checks, summary


def read_extra(path: str) -> tuple[dict | None, str | None]:
    """Read one extra block; return ``(block, error_or_None)``."""
    try:
        return read_block(path), None
    except Exception as exc:                   # missing file, torn block, ...
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# focus: environment and X11 helpers
# ---------------------------------------------------------------------------
def env_truthy(value: Any) -> bool:
    return str(value if value is not None else "").strip().lower() in (
        "1", "true", "yes", "on")


def parse_environ(data: bytes) -> dict[str, str]:
    """Parse ``/proc/<pid>/environ`` bytes into a dict."""
    out: dict[str, str] = {}
    for item in data.split(b"\0"):
        if not item or b"=" not in item:
            continue
        key, _, value = item.partition(b"=")
        try:
            out[key.decode("utf-8", "replace")] = value.decode(
                "utf-8", "replace")
        except (UnicodeError, ValueError):      # pragma: no cover - replaced
            continue
    return out


def game_env_value(pid: int | None, name: str) -> str | None:
    """Read one variable from a running process's environment."""
    if not pid:
        return None
    try:
        with open(f"/proc/{int(pid)}/environ", "rb") as fh:
            return parse_environ(fh.read()).get(name)
    except OSError:
        return None


def focus_env_enabled(environ: Mapping[str, str] | None = None,
                      pid: int | None = None) -> tuple[bool, str]:
    """Is ``SSBU_INPUT_NO_FOCUS=1`` set for this shell or the game process?

    ``pid=0`` disables the ``/proc`` lookup (used by the unit tests).
    """
    environ = os.environ if environ is None else environ
    own = environ.get(NO_FOCUS_ENV)
    if env_truthy(own):
        return True, f"{NO_FOCUS_ENV}={own} in this shell"
    if pid is None:
        try:
            from rl.memory.memapi import find_game_pid
            pid = find_game_pid()
        except Exception:                   # pragma: no cover - optional
            pid = None
    if pid:
        value = game_env_value(pid, NO_FOCUS_ENV)
        if env_truthy(value):
            return True, f"{NO_FOCUS_ENV}={value} in game pid {pid}"
        if value is not None:
            return False, (f"{NO_FOCUS_ENV}={value} in game pid {pid} "
                           "(not enabled)")
    if own is not None:
        return False, f"{NO_FOCUS_ENV}={own} in this shell (not enabled)"
    return False, (f"{NO_FOCUS_ENV} is not set in this shell or the "
                   "ssbu_native process environment")


def scripted_menus(environ: Mapping[str, str] | None = None) -> bool:
    """True when an input sequence (``SSBU_SEQ`` / ``SSBU_PLAY``) drives the menus.

    Such a run already walks from the title screen to the select screens; an
    extra A tap from this check would skip a menu item and lose that walk.
    """
    environ = os.environ if environ is None else environ
    return bool(environ.get("SSBU_SEQ") or environ.get("SSBU_PLAY"))


@dataclass(frozen=True)
class FocusDecision:
    """What to do about the X11 focus before the pad test."""

    state: str            # not-focused / game-focused / no-window / unknown
    action: str           # proceed / defocus / skip
    game_window: str | None
    focused_window: str | None
    detail: str

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "action": self.action,
            "game_window": self.game_window,
            "focused_window": self.focused_window,
            "detail": self.detail,
        }


def decide_focus(game_windows: Sequence[str], focused_window: str | None,
                 xdotool_available: bool = True) -> FocusDecision:
    """Decide from two window ids whether the pad test can run unfocused."""
    game = tuple(str(w) for w in game_windows if str(w).strip())
    if not game:
        return FocusDecision(
            "no-window", "skip", None, focused_window,
            "no ssbu_native window found (is the game running and DISPLAY set?)")
    if not xdotool_available:
        return FocusDecision(
            "unknown", "skip", game[0], focused_window,
            "xdotool is not installed; cannot read or move X11 focus")
    if focused_window is None:
        return FocusDecision(
            "unknown", "skip", game[0], None,
            "xdotool getwindowfocus returned no window")
    if str(focused_window) in game:
        return FocusDecision(
            "game-focused", "defocus", game[0], str(focused_window),
            f"the game window {game[0]} currently holds the X11 focus")
    return FocusDecision(
        "not-focused", "proceed", game[0], str(focused_window),
        f"X11 focus is on {focused_window}, not on the game window {game[0]}")


def _display(display: str | None = None) -> str:
    return (display or os.environ.get("SSBU_LIVE_DISPLAY")
            or os.environ.get("DISPLAY") or DEFAULT_DISPLAY)


def _xdotool(*args: str, display: str | None = None,
             timeout: float = 5.0) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(
            ["xdotool", *args], capture_output=True, text=True,
            timeout=timeout, env={**os.environ, "DISPLAY": _display(display)})
    except (OSError, subprocess.SubprocessError):
        return None


def xdotool_available() -> bool:
    return shutil.which("xdotool") is not None


def game_windows(display: str | None = None) -> tuple[list[str], str | None]:
    """Ids of the game's X11 client windows (``--class ssbu_native``)."""
    if not xdotool_available():
        return [], "xdotool is not installed"
    proc = _xdotool("search", "--class", "ssbu_native", display=display)
    if proc is None:
        return [], "xdotool search failed"
    ids = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not ids:
        proc = _xdotool("search", "--name", "SSBU native", display=display)
        if proc is not None:
            ids = [line.strip() for line in proc.stdout.splitlines()
                   if line.strip()]
    return ids, None


def focused_window(display: str | None = None) -> tuple[str | None, str | None]:
    """The X11 focus window id, or ``(None, reason)``."""
    if not xdotool_available():
        return None, "xdotool is not installed"
    proc = _xdotool("getwindowfocus", display=display)
    if proc is None:
        return None, "xdotool getwindowfocus failed"
    if proc.returncode != 0:
        return None, (proc.stderr.strip() or
                      f"xdotool getwindowfocus exited {proc.returncode}")
    return (proc.stdout.strip() or None), None


def _root_window_id(display: str | None = None,
                    timeout: float = 5.0) -> str | None:
    try:
        proc = subprocess.run(["xwininfo", "-root"], capture_output=True,
                              text=True, timeout=timeout,
                              env={**os.environ, "DISPLAY": _display(display)})
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Window id:\s*(0x[0-9a-fA-F]+)", proc.stdout)
    return match.group(1) if match else None


def defocus_game(game_ids: Sequence[str], display: str | None = None,
                 ) -> tuple[bool, str]:
    """Move X11 focus off the game window (root, then any other window)."""
    game = {str(w) for w in game_ids}
    if xdotool_available():
        root = _root_window_id(display)
        if root:
            _xdotool("windowfocus", root, display=display)
            current, err = focused_window(display)
            if current and current not in game:
                return True, f"focused the root window {root}"
            if err:
                return False, err
    proc = _xdotool("search", "--onlyvisible", "--name", ".*",
                    display=display)
    if proc is None:
        return False, "xdotool could not enumerate windows"
    candidates = [line.strip() for line in proc.stdout.splitlines()
                  if line.strip() and line.strip() not in game]
    for candidate in reversed(candidates):
        _xdotool("windowfocus", candidate, display=display)
        current, _err = focused_window(display)
        if current and current not in game:
            return True, f"focused another window {candidate}"
    return False, "could not move X11 focus away from the game window"


# ---------------------------------------------------------------------------
# focus: motion evaluation
# ---------------------------------------------------------------------------
def evaluate_focus_motion(frames: Sequence[FrameSample], slot: int = 0,
                          min_dx: float = DEFAULT_MIN_DX,
                          ) -> tuple[list[Check], dict]:
    """Check that holding right moved P1 by more than ``min_dx``."""
    checks: list[Check] = []
    frames = list(frames)
    live = [f for f in frames if f.live]
    summary: dict[str, Any] = {
        "slot": int(slot),
        "frames": len(frames),
        "live_frames": len(live),
        "held_s": round(_span_seconds(frames), 2),
        "ended": any(not f.live for f in frames),
    }
    if not live:
        checks.append(Check("P1 moved right with the pad while unfocused",
                            False, "no live frames were captured during the "
                            "hold"))
        return checks, summary

    start = float(live[0].x[slot])
    end = float(live[-1].x[slot])
    peak = max(float(f.x[slot]) for f in live)
    dx = peak - start
    summary.update({"start_x": round(start, 3), "peak_x": round(peak, 3),
                    "end_x": round(end, 3), "dx": round(dx, 3),
                    "min_dx": float(min_dx)})
    checks.append(Check(
        f"P1 moved right with the pad while unfocused (dx > {min_dx:g})",
        dx > float(min_dx),
        f"dx={dx:.2f} (start={start:.2f}, peak={peak:.2f}, "
        f"end={end:.2f}, {len(live)} frames)"))

    max_buttons = max(int(f.buttons[slot]) for f in live)
    max_stick = max(abs(float(f.stick_x[slot])) for f in live)
    input_frames = sum(1 for f in live
                       if int(f.buttons[slot]) or float(f.stick_x[slot]))
    summary.update({"max_buttons": max_buttons,
                    "max_abs_stick_x": round(max_stick, 4),
                    "input_frames": input_frames})
    checks.append(Check(
        "expert buttons/stick_x moved (recorded, not scored)", None,
        f"max buttons=0x{max_buttons:x}, max |stick_x|={max_stick:.3f}"))
    if not (max_stick > 0 or max_buttons):
        checks.append(Check(
            "the game received the pad input", False,
            "expert buttons/stick_x stayed 0 for the whole hold: the "
            "SDL/uinput pad never reached the game (hotplug, event-node "
            "read permission, or the game was stalled)"))
    if summary["ended"]:
        checks.append(Check("match stayed live during the hold", None,
                            "fight_on reached 0 during the hold"))
    return checks, summary


def hold_right_and_collect(reader: RawSnapshotReader, pad,
                           hold: float = DEFAULT_HOLD,
                           poll: float = 0.01,
                           grace: float = DEFAULT_HOLD_GRACE,
                           ) -> list[FrameSample]:
    """Hold the left stick right for ``hold`` seconds of *received* play.

    A match can stall for a second or two as it starts (shader pipelines are
    made then), and a fixed-length hold can fall entirely inside the stall:
    the pad is alive, but the game's input poll does not run, so the expert
    stick stays 0 and the motion check fails for the wrong reason.  Keep
    holding until P1's state has shown the stick/buttons for ``hold`` seconds,
    with ``grace`` extra seconds to ride out such a stall.
    """
    frames: list[FrameSample] = []
    last_seq = -1
    last_ts: int | None = None
    received = 0.0
    limit = time.monotonic() + max(0.0, float(hold)) + max(0.0, float(grace))
    try:
        pad.apply({"buttons": [], "stick": (1.0, 0.0)})
        while time.monotonic() < limit:
            remaining = max(0.0, limit - time.monotonic())
            try:
                snap = reader.read_latest(timeout=min(0.2, remaining))
            except SnapshotTimeout:
                continue
            if snap.seq == last_seq:
                time.sleep(min(max(0.0, float(poll)), remaining))
                continue
            last_seq = snap.seq
            sample = sample_frame(snap)
            frames.append(sample)
            dt = 0.0
            if last_ts is not None and sample.ts_ns > last_ts:
                dt = (sample.ts_ns - last_ts) / 1e9
            last_ts = sample.ts_ns
            if sample.live and (sample.stick_x[0] or sample.buttons[0]):
                received += dt
            if received >= float(hold):
                break
            time.sleep(min(max(0.0, float(poll)), remaining))
    finally:
        _neutral(pad)
    return frames


# ---------------------------------------------------------------------------
# runners
# ---------------------------------------------------------------------------
def _sampling_loop(reader: RawSnapshotReader, seconds: float,
                   extra_path: str, extra_every: float = 0.5,
                   ) -> tuple[list[FrameSample], list[int],
                              list[list[BattleObject]], int,
                              str | None]:
    """Sample core frames and the extra block until the match ends or time up."""
    frames: list[FrameSample] = []
    item_values: list[int] = []
    object_snapshots: list[list[BattleObject]] = []
    extra_errors: list[str] = []
    extra_seen = False
    extra_samples = 0
    last_extra = -1e9
    last_seq = -1
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        now = time.monotonic()
        remaining = max(0.0, deadline - now)
        new_frame = False
        try:
            snap = reader.read_latest(timeout=min(0.2, remaining))
        except SnapshotTimeout:
            snap = None
        if snap is not None and snap.seq != last_seq:
            last_seq = snap.seq
            frames.append(sample_frame(snap))
            new_frame = True
        now = time.monotonic()
        if now - last_extra >= extra_every:
            last_extra = now
            block, err = read_extra(extra_path)
            extra_samples += 1
            if block is not None:
                extra_seen = True
                pairs = extra_pairs(block)
                value = extra_get(pairs, "g.match.item_appear")
                if isinstance(value, (int, float)):
                    item_values.append(int(value))
                object_snapshots.append(battle_object_candidates(pairs))
            elif err:
                extra_errors.append(err)
        ended = (len(frames) >= 2 and frames[-2].live and
                 not frames[-1].live)
        if ended or time.monotonic() >= deadline:
            break
        if not new_frame:
            time.sleep(min(0.002, max(0.0, deadline - time.monotonic())))
    return (frames, item_values, object_snapshots, extra_samples,
            extra_errors[-1] if (extra_errors and not extra_seen) else None)


def run_rules(args: argparse.Namespace,
              reader_factory=RawSnapshotReader) -> Report:
    """Check stock=N, match lifetime and items=off."""
    report = Report("rules")
    expected = resolve_expected_stock(getattr(args, "expected_stock", None),
                                      os.environ.get(DEBUG_MATCH_ENV))
    report.summary.update({
        "state_path": args.path,
        "extra_path": args.extra_path,
        "expected_stock": expected,
        "seconds": float(args.seconds),
        "debug_match": os.environ.get(DEBUG_MATCH_ENV),
    })
    debug = debug_match_expectations(os.environ.get(DEBUG_MATCH_ENV))
    if debug.items_off is not None:
        report.summary["debug_items"] = "off" if debug.items_off else "on"

    reader = reader_factory(args.path, timeout=1.0)
    try:
        _progress(args,
                  f"waiting for a live match (timeout {args.timeout:g}s) ...")
        snap = wait_for_live(reader, args.timeout)
        _progress(args,
                  f"live: seq={snap.seq} stage={snap.stage_id} "
                  f"fighters={snap.n_fighters} "
                  f"stocks={[int(f.get('stock', 0)) for f in snap.fighters]}")
        (frames, item_values, object_snapshots,
         extra_samples, extra_error) = _sampling_loop(
            reader, args.seconds, args.extra_path)
    except TimeoutError as exc:
        report.skip(str(exc))
        return report
    except (SnapshotError, OSError) as exc:
        report.skip(f"{type(exc).__name__}: {exc}")
        return report
    finally:
        reader.close()

    stock_checks, stock_summary = evaluate_stock_run(frames, expected)
    for check in stock_checks:
        report.checks.append(check)
    report.summary["stock"] = stock_summary

    item_checks, item_summary = evaluate_items(
        item_values, object_snapshots,
        extra_available=(extra_samples > 0 and extra_error is None),
        extra_error=extra_error, extra_samples=extra_samples)
    for check in item_checks:
        report.checks.append(check)
    report.summary["items"] = item_summary
    report.summary["extra_error"] = extra_error
    return report


def _make_pad(pad_factory, lock_timeout: float = 10.0):
    """Create the pad without the default 120 s pad-lock wait."""
    try:
        return pad_factory(lock_timeout=max(1.0, float(lock_timeout)))
    except TypeError:                        # a test double without kwargs
        return pad_factory()


def run_focus(args: argparse.Namespace, pad_factory=VirtualPad,
              reader_factory=RawSnapshotReader) -> Report:
    """Check that the uinput pad moves P1 while the game window is unfocused."""
    report = Report("focus")
    display = _display(getattr(args, "display", None))
    report.summary["display"] = display
    enabled, why = focus_env_enabled()
    report.summary["no_focus"] = {"enabled": bool(enabled), "detail": why}
    if not enabled:
        report.skip(f"{NO_FOCUS_ENV} is not enabled: {why}. "
                    "Launch the game with "
                    f"{NO_FOCUS_ENV}=1 before running this check.")
        return report
    report.add(f"{NO_FOCUS_ENV} is enabled", True, why)

    ids, window_error = game_windows(display)
    focused, focus_error = focused_window(display)
    decision = decide_focus(ids, focused, xdotool_available())
    report.summary["focus"] = decision.as_dict()
    if window_error:
        report.add("game window found", False, window_error)
    else:
        report.add("game window found", bool(ids),
                   f"ids={ids}" if ids else "xdotool found no ssbu_native "
                   "window")
    if focus_error:
        report.add("X11 focus readable", False, focus_error)

    if decision.action == "skip":
        report.skip(decision.detail)
        return report
    if decision.action == "defocus":
        ok, detail = defocus_game(ids, display)
        report.add("moved focus away from the game window", ok, detail)
        if not ok:
            report.skip(f"the game window held the focus and could not be "
                        f"defocused: {detail}")
            return report
        focused, focus_error = focused_window(display)
        decision = decide_focus(ids, focused, xdotool_available())
        report.summary["focus_after_defocus"] = decision.as_dict()
    ok = decision.state == "not-focused"
    report.add("game window does not hold the X11 focus", ok, decision.detail)
    if not ok:
        report.skip(f"cannot prove the no-focus path while the game window "
                    f"has the focus ({decision.state})")
        return report

    pad = None
    reader = None
    try:
        try:
            pad = _make_pad(pad_factory,
                            min(10.0, max(1.0, float(args.timeout))))
            pad.create()
        except (OSError, RuntimeError) as exc:
            report.skip(f"cannot create the uinput pad: "
                        f"{type(exc).__name__}: {exc}")
            return report
        try:
            info = pad.describe()
            event = info.get("event") if isinstance(info, dict) else None
        except Exception:                    # pragma: no cover - optional
            event = None
        report.add("virtual pad created", True,
                   f"event={event}" if event else "uinput device created")

        # The title screen ("PRESS ANY BUTTON") waits for an input; the debug
        # match only takes over after it.  Tap A while waiting, so the tap
        # lands whenever the title/select screen appears (boot takes ~20 s).
        # A scripted menu walk (SSBU_SEQ / SSBU_PLAY) drives those screens
        # already: an extra A would skip an item and lose the walk.
        tap = not scripted_menus()
        if tap:
            for _ in range(2):
                pad.apply({"buttons": ["A"], "stick": (0.0, 0.0),
                           "cstick": (0.0, 0.0)})
                time.sleep(0.15)
                _neutral(pad)
                time.sleep(0.25)

        reader = reader_factory(args.path, timeout=1.0)
        _progress(args,
                  f"waiting for a live match (timeout {args.timeout:g}s) ...")
        snap = None
        deadline = time.monotonic() + float(args.timeout)
        last_tap = time.monotonic()
        while time.monotonic() < deadline:
            try:
                snap = wait_for_live(reader, min(0.6, max(0.1,
                                    deadline - time.monotonic())))
                break
            except TimeoutError:
                pass
            except (SnapshotError, OSError) as exc:
                report.skip(f"{type(exc).__name__}: {exc}")
                return report
            now = time.monotonic()
            if tap and now - last_tap >= 0.9:
                pad.apply({"buttons": ["A"], "stick": (0.0, 0.0),
                           "cstick": (0.0, 0.0)})
                time.sleep(0.08)
                _neutral(pad)
                last_tap = now
        if snap is None:
            report.skip(f"no stable live match within {args.timeout:g}s; "
                        "start a match or raise --timeout")
            return report

        initial = sample_frame(snap)
        report.summary["initial"] = {
            "seq": initial.seq,
            "stocks": list(initial.stocks),
            "x": list(initial.x),
            "is_cpu": list(initial.is_cpu),
        }
        if initial.is_cpu[0]:
            report.skip("P1 is a CPU in this match; start with p1=human so "
                        "pad motion is attributable")
            return report

        focused, _err = focused_window(display)
        if focused and focused in set(ids):
            report.skip("the game window gained the X11 focus before the "
                        "hold; cannot test the no-focus path")
            return report

        frames = [initial]
        try:
            frames.extend(hold_right_and_collect(reader, pad, args.hold))
        except (SnapshotError, OSError) as exc:
            report.skip(f"state read failed during the hold: "
                        f"{type(exc).__name__}: {exc}")
            return report
        finally:
            _neutral(pad)

        focused_after, _err = focused_window(display)
        still_unfocused = not (focused_after and focused_after in set(ids))
        report.add("game window still unfocused after the hold",
                   still_unfocused,
                   f"focus={focused_after}" if focused_after else "focus=none")

        motion_checks, motion_summary = evaluate_focus_motion(
            frames, slot=0, min_dx=args.min_dx)
        for check in motion_checks:
            report.checks.append(check)
        report.summary["motion"] = motion_summary
        return report
    finally:
        if pad is not None:
            try:
                pad.destroy()
            except Exception:                # pragma: no cover - teardown
                pass
        if reader is not None:
            reader.close()


def run_all(args: argparse.Namespace) -> Report:
    """Run ``rules`` then ``focus`` and report each separately."""
    report = Report("all")
    report.children = [run_rules(args), run_focus(args)]
    report.summary["modes"] = [child.mode for child in report.children]
    report.summary["results"] = {
        child.mode: child.status for child in report.children}
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common_options(ap: argparse.ArgumentParser,
                        suppress_defaults: bool = False) -> None:
    """Add shared options; sub-parsers suppress defaults so flags may precede
    the subcommand (``--json rules`` and ``rules --json`` both work)."""

    def default(value):
        return argparse.SUPPRESS if suppress_defaults else value

    ap.add_argument("--json", action="store_true", default=default(False),
                    help="print one JSON document instead of the text report")
    ap.add_argument("--expected-stock", type=int, default=default(None),
                    help="expected stock count (default: SSBU_DEBUG_MATCH "
                         f"stock=, else {DEFAULT_EXPECTED_STOCK})")
    ap.add_argument("--timeout", type=float, default=default(DEFAULT_TIMEOUT),
                    help=f"seconds to wait for a live match "
                         f"(default {DEFAULT_TIMEOUT:g})")
    ap.add_argument("--path", default=default(SHM_PATH),
                    help=f"core state block (default {SHM_PATH})")
    ap.add_argument("--extra-path", default=default(EXTRA_SHM_PATH),
                    help=f"extra state block (default {EXTRA_SHM_PATH})")
    ap.add_argument("--seconds", type=float, default=default(DEFAULT_SECONDS),
                    help=f"rules: seconds to observe "
                         f"(default {DEFAULT_SECONDS:g})")
    ap.add_argument("--hold", type=float, default=default(DEFAULT_HOLD),
                    help=f"focus: seconds of received play to hold right "
                         f"(default {DEFAULT_HOLD:g}; a start stall adds up "
                         f"to {DEFAULT_HOLD_GRACE:g} s)")
    ap.add_argument("--min-dx", type=float, default=default(DEFAULT_MIN_DX),
                    help=f"focus: required P1 x increase "
                         f"(default {DEFAULT_MIN_DX:g})")
    ap.add_argument("--display", default=default(None),
                    help="X11 display for xdotool (default "
                         "$SSBU_LIVE_DISPLAY or $DISPLAY)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_common_options(ap)
    sub = ap.add_subparsers(dest="mode", required=True, metavar="MODE")

    rules = sub.add_parser(
        "rules", help="stock=N and items=off are honoured",
        description="Check stock=N (both slots), the match lifetime, and "
                    "items=off (g.match.item_appear == 0, no item objects).")
    _add_common_options(rules, suppress_defaults=True)

    focus = sub.add_parser(
        "focus", help="the pad moves P1 with the game window unfocused",
        description="Check SSBU_INPUT_NO_FOCUS=1: with the game window not "
                    "focused, holding the VirtualPad right moves P1.")
    _add_common_options(focus, suppress_defaults=True)

    every = sub.add_parser(
        "all", help="run rules then focus",
        description="Run the rules check followed by the focus check.")
    _add_common_options(every, suppress_defaults=True)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runners = {"rules": run_rules, "focus": run_focus, "all": run_all}
    report = runners[args.mode](args)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    else:
        print_report(report)
    return report.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
