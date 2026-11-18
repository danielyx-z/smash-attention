#!/usr/bin/env python3
"""Data-only stage view of the live SSBU state.

Draws a crude Final Destination (always assumed) from the state bridge: the
blast-zone box, one box per fighter with a state badge, and a box/circle per
battle object (projectiles/items) from the extra block.  Everything that has no
position is printed in a side panel (all v7 fields, decoded buttons, extras,
recent transitions).

    .venv/bin/python -m rl.scripts.live_view            # attach to the reader
    .venv/bin/python -m rl.scripts.live_view --mock     # animated fake data
    .venv/bin/python -m rl.scripts.live_view --mock --dump-once --json

Data sources
    /dev/shm/ssbu_state.bin   v7 core block (reader.py)      60 Hz
    /dev/shm/ssbu_extra.bin   SSBX v1 block (extra_state.py) 10-15 Hz
    --extra direct            sample the game with ExtraState itself
    --extra off               fighters only, no projectiles/extras

The window is read-only and topmost but never calls ``focus_set``: taking the
keyboard focus away from the game makes the game ignore its pad/keyboard.

Keys (the viewer must have the focus for these)
    P  pause / resume        T  trails on/off
    H  raw hex of both rows  S  save a JSON snapshot
    Space  static/mock zoom  Q  quit
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from rl.memory import schema
from rl.memory.memapi import find_game_pid

# ---------------------------------------------------------------------------
# palette / geometry
# ---------------------------------------------------------------------------
BG = "#0e1116"
PANEL_BG = "#151a21"
PANEL_W = 560
DEFAULT_STATS_FILE = "/tmp/ssbu_training_stats.json"
FG = "#e6edf3"
DIM = "#7d8590"
OK = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"
CYAN = "#39c5cf"
SLOT = ("#4ea1ff", "#ff9f43")
SLOT_DARK = ("#1f4f80", "#7a4a12")

#: Final Destination (assumed): a single flat platform; world y is up, the
#: platform surface is y = 0 (fighters stand at y ~ 0.01).
FD_LEFT, FD_RIGHT, FD_TOP = -100.0, 100.0, 0.0
FIGHTER_W, FIGHTER_H = 17.0, 33.0   # cosmetic character box, world units
SHIELD_R = 21.0

FIELD_NAMES = list(schema.FIELD_NAMES)
FIELD_INDEX = dict(schema.FIELD_INDEX)
NICE = {
    "entry_id": "entry", "kind": "kind", "stock": "stock",
    "damage": "damage%", "x": "x", "y": "y", "facing": "facing",
    "action_state": "state", "action_frame": "frame", "on_ground": "ground",
    "is_actionable": "action", "jumps_left": "jumps", "vx": "vx", "vy": "vy",
    "dist_to_blast": "blast", "dx": "dx", "dy": "dy",
    "facing_rel": "facing_rel", "is_cpu": "cpu", "is_on_rebirth": "rebirth",
    "shield": "shield", "shield_size": "shield_sz",
    "is_enable_cancel": "cancel", "hitstun": "hitstun", "buttons": "buttons",
    "stick_x": "stick_x", "stick_y": "stick_y", "trigger": "trigger",
    "cstick_x": "cstick_x", "cstick_y": "cstick_y", "raw_flags": "raw_flags",
    "pad": "pad",
}
PLAUSIBLE = {
    "entry_id": lambda v: 0 <= v <= 7,
    "kind": lambda v: 0 <= v < 200,
    "stock": lambda v: -1 <= v <= 99,
    "damage": lambda v: 0.0 <= v <= 999.0,
    "x": lambda v: -2000.0 <= v <= 2000.0,
    "y": lambda v: -2000.0 <= v <= 2000.0,
    "vx": lambda v: -5000.0 <= v <= 5000.0,
    "vy": lambda v: -5000.0 <= v <= 5000.0,
    "facing": lambda v: -1.0 <= v <= 1.0,
    "action_state": lambda v: 0 <= v <= 2000,
    "action_frame": lambda v: -1.0 <= v <= 100000.0,
    "on_ground": lambda v: v in (0, 1),
    "is_actionable": lambda v: v in (0, 1),
    "jumps_left": lambda v: 0 <= v <= 9,
    "dist_to_blast": lambda v: 0.0 <= v <= 4000.0,
    "dx": lambda v: -4000.0 <= v <= 4000.0,
    "dy": lambda v: -4000.0 <= v <= 4000.0,
    "facing_rel": lambda v: v in (-1, 0, 1),
    "is_cpu": lambda v: v in (0, 1),
    "is_on_rebirth": lambda v: v in (0, 1),
    "shield": lambda v: v in (0, 1),
    "shield_size": lambda v: 0.0 <= v <= 100.0,
    "is_enable_cancel": lambda v: v in (0, 1),
    "hitstun": lambda v: 0 <= v <= 999,
    "buttons": lambda v: 0 <= v <= 0xFFFFFFFF,
    "stick_x": lambda v: -1.0 <= v <= 1.0,
    "stick_y": lambda v: -1.0 <= v <= 1.0,
    "trigger": lambda v: 0.0 <= v <= 1.0,
    "cstick_x": lambda v: -1.0 <= v <= 1.0,
    "cstick_y": lambda v: -1.0 <= v <= 1.0,
    "raw_flags": lambda v: 0 <= v <= 0xFFFFFFFF,
    "pad": lambda v: True,
}
MATCH_STATE = {0: "MENU", 1: "IN_MATCH", 2: "RESULTS"}


def action_name(kind: Any) -> str:
    try:
        k = int(kind)
    except (TypeError, ValueError):
        return "?"
    return schema.ACTION_STATE_NAMES.get(k, f"ST{k}")


def state_badge(f: dict) -> str:
    """A short human label for a fighter's current condition."""
    if int(f.get("is_on_rebirth", 0)):
        return "REBIRTH"
    if int(f.get("shield", 0)):
        return "SHIELD"
    if int(f.get("hitstun", 0)):
        return "HITSTUN"
    if not int(f.get("on_ground", 1)):
        name = action_name(f.get("action_state", 0))
        if name.startswith("ATTACK"):
            return "AIR ATK"
        if name.startswith("SPECIAL"):
            return "SPECIAL"
        return "AIRBORNE"
    name = action_name(f.get("action_state", 0))
    return name if name else "?"


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
@dataclass
class BattleObject:
    pool: int
    oid: int
    x: float
    y: float
    status: int | None = None
    motion: int | None = None
    category: int | None = None

    @property
    def key(self) -> tuple[int, int]:
        return (int(self.pool), int(self.oid))

    @property
    def label(self) -> str:
        bits = [f"p{self.pool}", f"{self.oid:x}"]
        if self.status is not None:
            bits.append(f"s{self.status}")
        return " ".join(bits)


class ExtraView:
    """A flattened view of one extra block (or mock records)."""

    def __init__(self, pairs: dict[tuple[str, int], Any] | None = None,
                 seq: int = 0, ts_ns: int = 0, age_ms: float | None = None):
        self.by_name: dict[str, dict[int, Any]] = defaultdict(dict)
        for (name, index), value in (pairs or {}).items():
            self.by_name[str(name)][int(index)] = value
        self.seq = int(seq)
        self.ts_ns = int(ts_ns)
        self.age_ms = age_ms

    @classmethod
    def from_records(cls, records, seq: int, ts_ns: int,
                     age_ms: float | None = None) -> "ExtraView":
        pairs = {}
        for r in records:
            value = getattr(r, "value", None)
            if value is None and hasattr(r, "value_python"):
                value = r.value_python()
            pairs[(r.name, r.index)] = value
        return cls(pairs, seq=seq, ts_ns=ts_ns, age_ms=age_ms)

    def get(self, name: str, index: int = 0, default=None):
        return self.by_name.get(name, {}).get(int(index), default)

    def has(self, name: str) -> bool:
        return bool(self.by_name.get(name))

    def fighter_accessors(self) -> set[int]:
        out = set()
        for slot in (0, 1):
            acc = self.get("f.accessor", slot + 1)
            if isinstance(acc, (int, float)) and int(acc) > 0x1000:
                out.add(int(acc))
        return out

    def objects(self) -> list[BattleObject]:
        pos = self.by_name.get("bom.obj.pos", {})
        if not pos:
            return []
        fighters = self.fighter_accessors()
        out: list[BattleObject] = []
        for idx in sorted(pos):
            acc = self.get("bom.obj.accessor", idx)
            if fighters and isinstance(acc, (int, float)) and int(acc) in fighters:
                continue
            p = pos[idx]
            if not isinstance(p, (list, tuple)) or len(p) < 2:
                continue
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            pool = int(self.get("bom.obj.pool", idx, 0) or 0)
            oid = int(self.get("bom.obj.id", idx, 0) or 0)
            # without f.accessor, never draw the two fighter objects
            if not fighters and pool == 0 and oid in (0, 0x10000):
                continue
            out.append(BattleObject(
                pool=pool, oid=oid, x=x, y=y,
                status=self.get("bom.obj.status_kind", idx),
                motion=self.get("bom.obj.motion_kind", idx),
                category=self.get("bom.obj.category", idx),
            ))
        return out


@dataclass
class FrameModel:
    header: dict
    fighters: list[dict]
    objects: list[BattleObject] = field(default_factory=list)
    extras: ExtraView | None = None
    age_ms: float = 0.0
    reader_fps: float = 0.0
    source: str = "shm"

    def to_json(self) -> dict:
        h = self.header
        return {
            "source": self.source,
            "seq": h.get("seq"),
            "fight_on": h.get("fight_on"),
            "match_state": h.get("match_state"),
            "stage_id": h.get("stage_id"),
            "fps": h.get("fps"),
            "game_frame": h.get("game_frame"),
            "blast": list(h.get("blast", (0, 0, 0, 0))),
            "age_ms": round(self.age_ms, 2),
            "fighters": [
                {name: f.get(name) for name in FIELD_NAMES}
                for f in self.fighters
            ],
            "objects": [
                {"pool": o.pool, "id": o.oid, "x": o.x, "y": o.y,
                 "status": o.status, "motion": o.motion}
                for o in self.objects
            ],
            "extra": None if self.extras is None else {
                "seq": self.extras.seq,
                "age_ms": None if self.extras.age_ms is None
                else round(self.extras.age_ms, 2),
                "records": len(self.extras.by_name),
            },
        }


class EventTracker:
    """A bounded human-readable log of state changes."""

    def __init__(self, maxlen: int = 12):
        self.history: deque[str] = deque(maxlen=maxlen)
        self.last: dict[Any, Any] = {}

    def update(self, model: FrameModel) -> None:
        h = model.header
        fo = int(h.get("fight_on", 0) or 0)
        if "fight" in self.last and self.last["fight"] != fo:
            self.history.append(f"fight_on {self.last['fight']} -> {fo}")
        self.last["fight"] = fo
        for i, f in enumerate(model.fighters):
            st = int(f.get("action_state", 0) or 0)
            prev = self.last.get(("st", i))
            if prev is not None and prev != st:
                self.history.append(
                    f"P{i + 1} {action_name(prev)} -> {action_name(st)}")
            self.last[("st", i)] = st
            for flag, label in (("shield", "shield"), ("hitstun", "hitstun")):
                v = int(f.get(flag, 0) or 0)
                pv = self.last.get((flag, i))
                if pv is not None and pv != v:
                    self.history.append(f"P{i + 1} {label} {'on' if v else 'off'}")
                self.last[(flag, i)] = v

    def text(self) -> list[str]:
        return list(self.history)


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
class ShmSource:
    def __init__(self, shm: str, extra: str = "auto",
                 extra_path: str = "/dev/shm/ssbu_extra.bin",
                 extra_fps: float = 12.0):
        from rl.env.gym_env import RawSnapshotReader
        self.reader = RawSnapshotReader(path=shm, timeout=0.05)
        self.extra_mode = extra
        self.extra_path = extra_path
        self.extra_period = 1.0 / max(1.0, extra_fps)
        self._extra_stamp = 0.0
        self._extra: ExtraView | None = None
        self._direct: _DirectExtras | None = None
        self._last_error: str | None = None

    def close(self) -> None:
        if self._direct is not None:
            self._direct.stop()
        try:
            self.reader.close()
        except Exception:  # noqa: BLE001
            pass

    def _read_extra(self) -> ExtraView | None:
        now = time.monotonic()
        if now - self._extra_stamp < self.extra_period:
            return self._extra
        self._extra_stamp = now
        if self.extra_mode == "off":
            self._extra = None
            return None
        # Prefer the published block (cheap); fall back to sampling the game.
        if self.extra_mode in ("auto", "shm"):
            try:
                from rl.extra.extra_state import read_block
                block = read_block(self.extra_path)
                age = (time.monotonic_ns() - int(block.get("ts_ns", 0))) / 1e6
                if age < 1000.0:
                    self._extra = ExtraView.from_records(
                        block["records"], block.get("seq", 0),
                        block.get("ts_ns", 0), age)
                    return self._extra
            except Exception as exc:  # noqa: BLE001
                self._last_error = f"{type(exc).__name__}: {exc}"
        if self.extra_mode in ("auto", "direct"):
            if self._direct is None:
                self._direct = _DirectExtras()
                self._direct.start()
            snap = self._direct.snapshot()
            if snap is not None:
                self._extra = snap
                return snap
        return self._extra

    def read(self) -> FrameModel | None:
        from rl.env.gym_env import SnapshotTimeout
        try:
            snap = self.reader.read_latest(timeout=0.05)
        except SnapshotTimeout:
            return None
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            return None
        h = snap.header_dict()
        h["blast"] = tuple(snap.blast)
        extras = self._read_extra()
        objects = extras.objects() if extras is not None else []
        age = max(0.0, (time.monotonic_ns() - int(snap.ts_ns)) / 1e6)
        return FrameModel(
            header=h, fighters=[dict(f) for f in snap.fighters],
            objects=objects, extras=extras, age_ms=age,
            reader_fps=float(snap.fps), source="shm")


class _DirectExtras(threading.Thread):
    """Sample the game with ExtraState in the background (no extra reader)."""

    def __init__(self, period: float = 0.08):
        super().__init__(daemon=True)
        self.period = period
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._snap: ExtraView | None = None
        self._error: str | None = None

    def run(self) -> None:
        from rl.extra.extra_state import ExtraState
        ex = None
        pid = None
        while not self._stop.is_set():
            now_pid = find_game_pid()
            if now_pid is None:
                if ex is not None:
                    ex.close()
                    ex = None
                self._stop.wait(0.5)
                continue
            try:
                if ex is None or pid != now_pid:
                    if ex is not None:
                        ex.close()
                    ex = ExtraState(now_pid)
                    pid = now_pid
                recs = ex.read()
                snap = ExtraView.from_records(recs, seq=0,
                                              ts_ns=time.monotonic_ns(),
                                              age_ms=0.0)
                with self._lock:
                    self._snap = snap
                    self._error = None
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._error = f"{type(exc).__name__}: {exc}"
                if ex is not None:
                    try:
                        ex.close()
                    except Exception:  # noqa: BLE001
                        pass
                ex = None
                pid = None
            self._stop.wait(self.period)
        if ex is not None:
            ex.close()

    def snapshot(self) -> ExtraView | None:
        with self._lock:
            return self._snap

    def stop(self) -> None:
        self._stop.set()


class MockSource:
    """Animated fake state: verifies the whole viewer without a game."""

    def __init__(self, hz: float = 60.0):
        self.t0 = time.monotonic()
        self.hz = hz
        self.frame = 0
        self.objects_trail: dict[tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=8))

    def close(self) -> None:
        return None

    def read(self) -> FrameModel:
        t = self.frame / self.hz
        self.frame += 1
        jump1 = max(0.0, 34.0 * math.sin(t * 1.7))
        jump2 = max(0.0, 26.0 * math.sin(t * 1.1 + 1.9))
        x1 = 55.0 * math.sin(t * 0.55)
        x2 = 55.0 * math.sin(t * 0.55 + math.pi)
        shield = (math.sin(t * 0.9) > 0.75)
        hitstun = (math.sin(t * 0.8 + 0.7) > 0.88)
        state1 = 54 if hitstun else (12 if jump1 > 1 else (10 if shield else 0))
        state2 = 30 if (math.sin(t * 1.3) > 0.7) else (12 if jump2 > 1 else 0)
        objx = 90.0 * math.cos(t * 0.9)
        objy = 25.0 + 42.0 * abs(math.sin(t * 1.6))
        self.objects_trail[(1, 0x30001)].append((objx, objy))

        def fighter(slot: int, **over) -> dict:
            base = {name: 0 for name in FIELD_NAMES}
            base.update(over)
            return base

        spike = 20.0 + 18.0 * math.sin(t * 0.7)
        f1 = fighter(
            0, entry_id=0, kind=0, stock=0, damage=spike, x=x1,
            y=jump1, facing=1.0 if math.cos(t * 0.55) >= 0 else -1.0,
            action_state=state1, action_frame=(t * 60.0) % 90.0,
            on_ground=0 if jump1 > 1 else 1,
            is_actionable=1 if state1 in (0, 12) else 0,
            jumps_left=1 if jump1 > 1 else 2,
            vx=30.0 * math.cos(t * 0.55), vy=50.0 * math.cos(t * 1.7),
            dist_to_blast=150.0 - abs(x1), dx=x2 - x1, dy=jump2 - jump1,
            facing_rel=1, is_cpu=0, is_on_rebirth=0,
            shield=1 if shield else 0, shield_size=0.0,
            is_enable_cancel=0, hitstun=1 if hitstun else 0,
            buttons=0x8004 if state1 in (12, 54) else 0x8000,
            stick_x=0.7 if state1 == 12 else 0.0, stick_y=0.0,
            trigger=0.0, cstick_x=0.0, cstick_y=0.0,
            raw_flags=0x7FFFFFFF, pad=0)
        f2 = fighter(
            1, entry_id=1, kind=1, stock=0, damage=15.0 + 12.0 * math.cos(t * 0.5),
            x=x2, y=jump2,
            facing=1.0 if math.cos(t * 0.55 + math.pi) >= 0 else -1.0,
            action_state=state2, action_frame=(t * 60.0) % 70.0,
            on_ground=0 if jump2 > 1 else 1,
            is_actionable=1 if state2 in (0, 12) else 0,
            jumps_left=1 if jump2 > 1 else 2,
            vx=-30.0 * math.cos(t * 0.55), vy=40.0 * math.cos(t * 1.1 + 1.9),
            dist_to_blast=150.0 - abs(x2), dx=x1 - x2, dy=jump1 - jump2,
            facing_rel=-1, is_cpu=1, is_on_rebirth=0,
            shield=0, shield_size=0.0, is_enable_cancel=0, hitstun=0,
            buttons=0x8201 if state2 == 30 else 0x8000,
            stick_x=-0.4 if state2 == 30 else 0.0, stick_y=0.0,
            trigger=0.0, cstick_x=0.0, cstick_y=0.0,
            raw_flags=0x7FFFFFFF, pad=0)
        extras = ExtraView({
            ("g.match.rule_kind", 0): 1,
            ("g.match.stock", 0): 0,
            ("f.accessor", 1): 0x11110000,
            ("f.accessor", 2): 0x22220000,
            ("f.info.stock", 1): 0, ("f.info.stock", 2): 0,
            ("f.info.dead_count", 1): 0, ("f.info.dead_count", 2): 0,
            ("f.damage.amount", 1): spike,
            ("f.damage.amount", 2): 15.0,
            ("f.ground.floor_y", 1): 0.0, ("f.ground.floor_y", 2): 0.0,
            ("f.shield.part_count", 1): 0, ("f.shield.part_count", 2): 0,
            ("f.shield.likely", 1): 1 if shield else 0,
            ("f.stop.hit_stop_frame", 1): 0, ("f.stop.hit_stop_frame", 2): 0,
            ("f.kinetic.engine_speed", 1): abs(f1["vx"]),
            ("f.kinetic.engine_speed", 2): abs(f2["vx"]),
            ("f.motion.frame", 1): f1["action_frame"],
            ("f.motion.frame", 2): f2["action_frame"],
            ("bom.obj.pool", 0): 1, ("bom.obj.id", 0): 0x30001,
            ("bom.obj.pos", 0): (objx, objy, 0.0),
            ("bom.obj.status_kind", 0): 0,
            ("bom.obj.accessor", 0): 0x33330000,
        }, seq=self.frame, ts_ns=time.monotonic_ns(), age_ms=0.0)
        header = {
            "seq": self.frame * 2, "version": 7, "n_fighters": 2,
            "ts_ns": time.monotonic_ns(), "fps": self.hz, "reader_pid": 0,
            "fight_on": 1, "match_state": 1, "stage_id": 3,
            "game_frame": self.frame,
            "flags": 3,
            "blast": (-240.0, 240.0, 180.0, -140.0),
        }
        return FrameModel(header=header, fighters=[f1, f2],
                          objects=extras.objects(), extras=extras,
                          age_ms=0.0, reader_fps=self.hz, source="mock")


# ---------------------------------------------------------------------------
# world -> screen
# ---------------------------------------------------------------------------
class WorldMap:
    def __init__(self, x0: float, x1: float, y0: float, y1: float,
                 width: int, height: int, margin: float = 24.0):
        self.x0, self.x1 = float(x0), float(x1)
        self.y0, self.y1 = float(y0), float(y1)
        self.w = max(1, int(width))
        self.h = max(1, int(height))
        self.margin = margin
        sx = (self.w - 2 * margin) / max(1e-6, self.x1 - self.x0)
        sy = (self.h - 2 * margin) / max(1e-6, self.y1 - self.y0)
        self.scale = max(0.1, min(sx, sy))
        self.cx = (self.x0 + self.x1) / 2
        self.cy = (self.y0 + self.y1) / 2

    def px(self, x: float, y: float) -> tuple[float, float]:
        return (self.w / 2 + (x - self.cx) * self.scale,
                self.h / 2 - (y - self.cy) * self.scale)

    def rect(self, x: float, y: float, wu: float, hu: float):
        a = self.px(x - wu / 2, y + hu)
        b = self.px(x + wu / 2, y)
        return a[0], a[1], b[0], b[1]


def blast_box(header: dict) -> tuple[float, float, float, float]:
    b = header.get("blast") or (-240.0, 240.0, 180.0, -140.0)
    left, right, top, bottom = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    if not all(math.isfinite(v) for v in (left, right, top, bottom)) or \
            left >= right or bottom >= top:
        return (-240.0, 240.0, 180.0, -140.0)
    return left, right, top, bottom


# ---------------------------------------------------------------------------
# Tk application
# ---------------------------------------------------------------------------
class LiveView:
    def __init__(self, root, source, *, topmost: bool = True,
                 width: int = 1440, height: int = 860, fps: float = 60.0,
                 panel_hz: float = 12.0, trails: int = 48,
                 max_objects: int = 32,
                 stats_file: str = DEFAULT_STATS_FILE):
        import tkinter as tk
        self.tk = tk
        self.root = root
        self.source = source
        self.fps = max(5.0, fps)
        self.panel_every = max(1, int(round(self.fps / max(1.0, panel_hz))))
        self.trails_on = True
        self.trail_len = max(2, int(trails))
        self.max_objects = max(1, int(max_objects))
        self.paused = False
        self.show_hex = False
        self.events = EventTracker()
        self.map: WorldMap | None = None
        self.last: FrameModel | None = None
        self.last_ok = time.monotonic()
        self.last_seq = -1
        self.tick_no = 0
        self.fighter_trails: list[deque] = [deque(maxlen=self.trail_len),
                                            deque(maxlen=self.trail_len)]
        self.obj_trails: dict[tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=10))
        self.items: dict[str, Any] = {}
        self.stats_file = os.path.expanduser(stats_file)
        self.stats_data: dict | None = None
        self.stats_next = 0.0

        root.title("SSBU live view")
        root.geometry(f"{int(width)}x{int(height)}")
        root.configure(bg=BG)
        if topmost:
            try:
                root.attributes("-topmost", True)
            except tk.TclError:
                pass
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0,
                                width=max(200, width - PANEL_W), height=height)
        self.canvas.pack(side="left", fill="both", expand=True)
        side = tk.Frame(root, bg=PANEL_BG, width=PANEL_W)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        self.panel = tk.Text(side, bg=PANEL_BG, fg=FG, insertbackground=FG,
                             font=("TkFixedFont", 9), wrap="none",
                             borderwidth=0, highlightthickness=0,
                             padx=8, pady=6)
        scroll = tk.Scrollbar(side, command=self.panel.yview)
        self.panel.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.panel.pack(side="left", fill="both", expand=True)
        self.panel.tag_configure("bad", foreground=BAD)
        self.panel.tag_configure("warn", foreground=WARN)
        self.panel.tag_configure("dim", foreground=DIM)
        self.panel.tag_configure("head", foreground=CYAN)
        self.panel.tag_configure("ok", foreground=FG)
        self.panel.tag_configure("slot0", foreground=SLOT[0])
        self.panel.tag_configure("slot1", foreground=SLOT[1])

        root.bind("<KeyPress-p>", lambda _e: self.toggle_pause())
        root.bind("<KeyPress-t>", lambda _e: self.toggle_trails())
        root.bind("<KeyPress-h>", lambda _e: self.toggle_hex())
        root.bind("<KeyPress-s>", lambda _e: self.save_snapshot())
        root.bind("<KeyPress-q>", lambda _e: self.quit())
        self.canvas.bind("<Configure>", lambda _e: self.rebuild())

        self.hint = self.canvas.create_text(
            12, 4, anchor="nw", fill=DIM, font=("TkFixedFont", 8),
            text="P pause  T trails  H hex  S snapshot  Q quit   |   "
                 "Final Destination assumed")
        self.status = self.canvas.create_text(
            12, 26, anchor="nw", fill=FG, font=("TkFixedFont", 8),
            text="waiting for /dev/shm/ssbu_state.bin ...")
        self.stats_bg = self.canvas.create_rectangle(
            0, 0, 0, 0, fill="#0b0f14", outline="#39424e", width=1,
            state="hidden")
        self.stats_text = self.canvas.create_text(
            16, max(200, height) - 16, anchor="sw", fill=DIM,
            font=("TkFixedFont", 9),
            text="TRAINING  waiting for stats ...", justify="left")
        self.rebuild()
        self.update_stats()
        self.root.after(0, self.tick)

    # -- controls ------------------------------------------------------
    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def toggle_trails(self) -> None:
        self.trails_on = not self.trails_on

    def toggle_hex(self) -> None:
        self.show_hex = not self.show_hex

    def save_snapshot(self) -> None:
        if self.last is None:
            return
        path = os.path.join(os.getcwd(),
                            f"live_view_{int(time.time())}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"model": self.last.to_json(),
                       "events": self.events.text()}, fh, indent=2)
        self.status_set(f"snapshot saved: {path}")

    def quit(self) -> None:
        try:
            self.source.close()
        except Exception:  # noqa: BLE001
            pass
        self.root.destroy()

    # -- canvas --------------------------------------------------------
    def status_set(self, text: str) -> None:
        try:
            self.canvas.itemconfigure(self.status, text=text)
        except self.tk.TclError:
            pass

    def _place_stats(self) -> None:
        c = self.canvas
        h = max(200, c.winfo_height())
        try:
            c.coords(self.stats_text, 16, h - 16)
            bbox = c.bbox(self.stats_text)
            if bbox:
                c.coords(self.stats_bg, bbox[0] - 8, bbox[1] - 6,
                         bbox[2] + 8, bbox[3] + 6)
                c.itemconfigure(self.stats_bg, state="normal")
                c.tag_raise(self.stats_bg)
                c.tag_raise(self.stats_text)
            else:
                c.itemconfigure(self.stats_bg, state="hidden")
        except self.tk.TclError:
            pass

    def update_stats(self) -> None:
        data = None
        try:
            with open(self.stats_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError, TypeError):
            data = None
        self.stats_data = data
        now = time.time()
        if not data:
            text = "TRAINING  no stats file yet"
            colour = DIM
        else:
            active = bool(data.get("active"))
            updated = float(data.get("updated", 0.0) or 0.0)
            age = max(0.0, now - updated) if updated else 999.0
            if not active:
                state = "idle"
                colour = DIM
            elif age > 45.0:
                state = f"stalled {age:2.0f}s"
                colour = WARN
            else:
                state = "training"
                mean_diff = float(data.get("mean_diff", 0.0) or 0.0)
                colour = OK if mean_diff > 0 else BAD if mean_diff < 0 else FG
            cpu = data.get("cpu_level") or "?"
            eps = int(data.get("episodes", 0) or 0)
            pos = 100.0 * float(data.get("positive_rate", 0.0) or 0.0)
            wins = int(data.get("wins", 0) or 0)
            losses = int(data.get("losses", 0) or 0)
            p1_deaths = int(data.get("p1_deaths", 0) or 0)
            p2_deaths = int(data.get("p2_deaths", 0) or 0)
            cur_ep = int(data.get("current_episode", 0) or 0)
            cur_steps = int(data.get("current_steps", 0) or 0)
            cur_max = int(data.get("current_max_steps", 0) or 0)
            cur_dealt = float(data.get("current_dealt", 0.0) or 0.0)
            cur_taken = float(data.get("current_taken", 0.0) or 0.0)
            cur_diff = float(data.get("current_diff", 0.0) or 0.0)
            cur_p1d = int(data.get("current_p1_deaths", 0) or 0)
            cur_p2d = int(data.get("current_p2_deaths", 0) or 0)
            mean_d = float(data.get("mean_dealt", 0.0) or 0.0)
            mean_t = float(data.get("mean_taken", 0.0) or 0.0)
            mean_diff = float(data.get("mean_diff", 0.0) or 0.0)
            last_diff = float(data.get("last_diff", 0.0) or 0.0)
            it = int(data.get("iteration", 0) or 0)
            its = int(data.get("iterations", 0) or 0)
            text = "\n".join([
                f"TRAINING {state}   CPU{cpu}   iter {it}/{its}   eps {eps}",
                f"positive {pos:5.1f}%   wins {wins} / losses {losses}   "
                f"deaths P1 {p1_deaths} / P2 {p2_deaths}",
                f"current ep {cur_ep}   {cur_steps}/{cur_max or '?'} steps   "
                f"dealt {cur_dealt:6.1f}   taken {cur_taken:6.1f}   "
                f"diff {cur_diff:+6.1f}   deaths P1 {cur_p1d} / P2 {cur_p2d}",
                f"mean dealt {mean_d:6.1f}   mean taken {mean_t:6.1f}   "
                f"mean diff {mean_diff:+6.1f}   last {last_diff:+6.1f}",
                f"PPO loss {float(data.get('policy_loss', 0.0) or 0.0):+.4f}   "
                f"value {float(data.get('value_loss', 0.0) or 0.0):.4f}   "
                f"kl_bc {float(data.get('kl_bc', 0.0) or 0.0):.4f}   "
                f"ent {float(data.get('entropy', 0.0) or 0.0):.3f}",
            ])
        try:
            self.canvas.itemconfigure(self.stats_text, text=text, fill=colour)
            self._place_stats()
        except self.tk.TclError:
            pass

    def rebuild(self) -> None:
        c = self.canvas
        keep = (self.hint, self.status, self.stats_bg, self.stats_text)
        for item in c.find_all():
            if item in keep:
                continue
            c.delete(item)
        self.items.clear()
        w = max(200, c.winfo_width())
        h = max(200, c.winfo_height())
        if self.last is not None:
            left, right, top, bottom = blast_box(self.last.header)
        else:
            left, right, top, bottom = -240.0, 240.0, 180.0, -140.0
        self.map = WorldMap(left - 10, right + 10, bottom - 10, top + 10, w, h)
        m = self.map
        # blast zone + grid
        x0, y0 = m.px(left, top)
        x1, y1 = m.px(right, bottom)
        c.create_rectangle(x0, y0, x1, y1, outline="#803a3a", dash=(5, 4),
                           width=1)
        for gx in range(int(left // 50) * 50, int(right) + 1, 50):
            a = m.px(gx, top)
            b = m.px(gx, bottom)
            c.create_line(a[0], a[1], b[0], b[1], fill="#1c2330")
        for gy in range(int(bottom // 50) * 50, int(top) + 1, 50):
            a = m.px(left, gy)
            b = m.px(right, gy)
            c.create_line(a[0], a[1], b[0], b[1], fill="#1c2330")
        # Final Destination (assumed): platform slab + base
        if m is not None:
            base = [m.px(FD_LEFT, FD_TOP), m.px(FD_RIGHT, FD_TOP),
                    m.px(FD_RIGHT - 18, FD_TOP - 30),
                    m.px(FD_LEFT + 18, FD_TOP - 30)]
            c.create_polygon([v for p in base for v in p], fill="#20262f",
                             outline="#39424e")
            top = m.px(FD_LEFT, FD_TOP)
            bot = m.px(FD_RIGHT, FD_TOP)
            c.create_line(top[0], top[1], bot[0], bot[1], fill="#6b7787",
                          width=4)
            c.create_text(top[0] + 6, m.px(0, FD_TOP - 34)[1],
                          anchor="w", fill=DIM, font=("TkFixedFont", 9),
                          text="FINAL DESTINATION (assumed)")
        c.tag_raise(self.hint)
        c.tag_raise(self.status)
        self._place_stats()

    # -- main loop -----------------------------------------------------
    def tick(self) -> None:
        now = time.monotonic()
        if now >= self.stats_next:
            self.stats_next = now + 0.5
            self.update_stats()
        if not self.paused:
            try:
                model = self.source.read()
            except Exception as exc:  # noqa: BLE001
                model = None
                self.status_set(f"source error: {type(exc).__name__}: {exc}")
            seq = -1 if model is None else int(model.header.get("seq", -1))
            if model is not None and seq != self.last_seq:
                self.last_seq = seq
                self.last = model
                self.last_ok = time.monotonic()
                self.events.update(model)
                self.render(model)
                self.tick_no += 1
                if self.tick_no % self.panel_every == 0:
                    self.render_panel(model)
            elif model is not None:
                # The reader's last frame stays readable after the reader (or
                # the game) is gone; read_latest keeps returning it.  Do not
                # repaint the same frame at 60 Hz: say that it is stale.
                age = float(model.age_ms) / 1000.0
                if age >= 1.0:
                    self.status_set(
                        f"STALE: last frame {age:4.1f}s old — "
                        "is rl.memory.reader running? (the game can still be live)")
            else:
                stale = time.monotonic() - self.last_ok
                if stale >= 1.0:
                    self.status_set(
                        f"WAITING: no state block read for {stale:4.1f}s — "
                        "is rl.memory.reader running?")
        self.root.after(max(1, int(1000 / self.fps)), self.tick)

    def render(self, model: FrameModel) -> None:
        w = max(200, self.canvas.winfo_width())
        h = max(200, self.canvas.winfo_height())
        if self.map is None or abs(self.map.w - w) > 2 or abs(self.map.h - h) > 2:
            self.rebuild()
        m = self.map
        if m is None:
            return
        c = self.canvas
        self.status_set(
            f"{model.source}  seq {model.header.get('seq')}  "
            f"{model.reader_fps:4.0f} fps  age {model.age_ms:4.0f} ms  "
            f"fight {model.header.get('fight_on')}  "
            f"{MATCH_STATE.get(int(model.header.get('match_state', 0) or 0), '?')}  "
            f"obj {len(model.objects)}")

        for slot in (0, 1):
            f = model.fighters[slot] if slot < len(model.fighters) else {}
            self.render_fighter(slot, f, m)

        for i in range(self.max_objects):
            if i < len(model.objects):
                self.render_object(i, model.objects[i], m)
            else:
                for k in (f"obj{i}_dot", f"obj{i}_lbl", f"obj{i}_trail"):
                    if k in self.items:
                        c.itemconfigure(self.items[k], state="hidden")

        if self.trails_on:
            for slot in (0, 1):
                f = model.fighters[slot]
                if int(f.get("raw_flags", 0) or 0) == 0:
                    self.fighter_trails[slot].clear()
                    continue
                self.fighter_trails[slot].append(
                    (float(f.get("x", 0.0)), float(f.get("y", 0.0))))
                self.update_trail(f"ft{slot}", self.fighter_trails[slot],
                                  SLOT[slot], width=1)
        else:
            for slot in (0, 1):
                if f"ft{slot}" in self.items:
                    c.itemconfigure(self.items[f"ft{slot}"], state="hidden")

    def render_fighter(self, slot: int, f: dict, m: WorldMap) -> None:
        c = self.canvas
        prefix = f"f{slot}"
        x = float(f.get("x", 0.0) or 0.0)
        y = float(f.get("y", 0.0) or 0.0)
        raw = int(f.get("raw_flags", 0) or 0)
        alive = raw != 0
        dmg = float(f.get("damage", 0.0) or 0.0)
        hitstun = int(f.get("hitstun", 0) or 0)
        shield = int(f.get("shield", 0) or 0)
        airborne = not int(f.get("on_ground", 1) or 0)
        facing = float(f.get("facing", 1.0) or 1.0)

        if not alive:
            for key in (prefix + "_body", prefix + "_face", prefix + "_shield",
                        prefix + "_name", prefix + "_state", prefix + "_shadow"):
                if key in self.items:
                    c.itemconfigure(self.items[key], state="hidden")
            return

        # body box (world y is the feet)
        bx0, by0, bx1, by1 = m.rect(x, y, FIGHTER_W, FIGHTER_H)
        fill = damage_color(dmg)
        body = self.items.get(prefix + "_body")
        if body is None:
            body = c.create_rectangle(bx0, by0, bx1, by1, outline=SLOT[slot],
                                      width=2, fill=fill)
            self.items[prefix + "_body"] = body
        else:
            c.coords(body, bx0, by0, bx1, by1)
            c.itemconfigure(body, fill=fill, outline=SLOT[slot],
                            state="normal")
        # facing wedge
        wy = (by0 + by1) / 2
        tip = bx1 + (bx1 - bx0) * 0.35 if facing >= 0 else bx0 - (bx1 - bx0) * 0.35
        face = self.items.get(prefix + "_face")
        pts = (bx1, wy - 5, tip, wy, bx1, wy + 5) if facing >= 0 else \
              (bx0, wy - 5, tip, wy, bx0, wy + 5)
        if face is None:
            face = c.create_polygon(*pts, fill=SLOT[slot])
            self.items[prefix + "_face"] = face
        else:
            c.coords(face, *pts)
            c.itemconfigure(face, state="normal")
        # shield bubble
        cx, cy = m.px(x, y + FIGHTER_H / 2)
        r = SHIELD_R * m.scale
        sh = self.items.get(prefix + "_shield")
        if shield:
            if sh is None:
                sh = c.create_oval(cx - r, cy - r, cx + r, cy + r,
                                   outline=CYAN, width=2, dash=(3, 3))
                self.items[prefix + "_shield"] = sh
            else:
                c.coords(sh, cx - r, cy - r, cx + r, cy + r)
                c.itemconfigure(sh, state="normal")
        elif sh is not None:
            c.itemconfigure(sh, state="hidden")
        # labels: stagger the two slots so boxes that meet do not collide
        name = f"P{slot + 1} {char_name(f.get('kind'))} {dmg:5.1f}%"
        if int(f.get("is_cpu", 0) or 0):
            name += "  CPU"
        label = self.items.get(prefix + "_name")
        label_y = m.px(x, y + FIGHTER_H + 30 + slot * 26)[1]
        if label is None:
            label = c.create_text(cx, label_y, text=name, fill=SLOT[slot],
                                  font=("TkFixedFont", 9, "bold"))
            self.items[prefix + "_name"] = label
        else:
            c.coords(label, cx, label_y)
            c.itemconfigure(label, text=name, state="normal")
        badge = state_badge(f)
        extra = []
        jumps = int(f.get("jumps_left", 0) or 0)
        if jumps:
            extra.append(f"j{jumps}")
        if airborne:
            extra.append("air")
        if int(f.get("is_on_rebirth", 0) or 0):
            extra.append("rebirth")
        if int(f.get("is_actionable", 0) or 0):
            extra.append("act")
        st = badge + ("  " + " ".join(extra) if extra else "")
        slabel = self.items.get(prefix + "_state")
        sy = label_y + 14
        colour = BAD if hitstun else (CYAN if shield else
                                      (WARN if airborne else FG))
        if slabel is None:
            slabel = c.create_text(cx, sy, text=st, fill=colour,
                                   font=("TkFixedFont", 8))
            self.items[prefix + "_state"] = slabel
        else:
            c.coords(slabel, cx, sy)
            c.itemconfigure(slabel, text=st, fill=colour, state="normal")
        c.tag_raise(prefix + "_name")
        c.tag_raise(prefix + "_state")

    def render_object(self, i: int, o: BattleObject, m: WorldMap) -> None:
        c = self.canvas
        cx, cy = m.px(o.x, o.y)
        r = max(3.0, 5.0 * m.scale)
        dot_key, lbl_key, trail_key = (f"obj{i}_dot", f"obj{i}_lbl",
                                       f"obj{i}_trail")
        dot = self.items.get(dot_key)
        colour = "#c792ea" if o.pool == 1 else "#8bd5ca"
        if dot is None:
            dot = c.create_oval(cx - r, cy - r, cx + r, cy + r,
                                fill=colour, outline="")
            self.items[dot_key] = dot
        else:
            c.coords(dot, cx - r, cy - r, cx + r, cy + r)
            c.itemconfigure(dot, fill=colour, state="normal")
        lbl = self.items.get(lbl_key)
        if lbl is None:
            lbl = c.create_text(cx + r + 3, cy, anchor="w", text=o.label,
                                fill=DIM, font=("TkFixedFont", 7))
            self.items[lbl_key] = lbl
        else:
            c.coords(lbl, cx + r + 3, cy)
            c.itemconfigure(lbl, text=o.label, state="normal")
        if self.trails_on:
            tr = self.obj_trails[o.key]
            tr.append((o.x, o.y))
            self.update_trail(trail_key, tr, colour, width=1)
        elif trail_key in self.items:
            c.itemconfigure(self.items[trail_key], state="hidden")

    def update_trail(self, key: str, pts, colour: str, width: int = 1) -> None:
        c = self.canvas
        m = self.map
        if m is None or len(pts) < 2:
            if key in self.items:
                c.itemconfigure(self.items[key], state="hidden")
            return
        coords = []
        for x, y in pts:
            p = m.px(x, y)
            coords.extend(p)
        item = self.items.get(key)
        if item is None:
            item = c.create_line(*coords, fill=colour, width=width,
                                 smooth=True)
            self.items[key] = item
        else:
            c.coords(item, *coords)
            c.itemconfigure(item, fill=colour, width=width, state="normal")

    # -- side panel ----------------------------------------------------
    def render_panel(self, model: FrameModel) -> None:
        t = self.panel
        t.configure(state="normal")
        t.delete("1.0", "end")
        h = model.header
        age = model.age_ms
        t.insert("end", "SSBU live view\n", "head")
        t.insert("end", f"seq {h.get('seq')}   reader {model.reader_fps:5.1f} fps"
                        f"   age {age:6.1f} ms\n", self.age_tag(age))
        t.insert("end", f"fight {h.get('fight_on')}   "
                        f"match {MATCH_STATE.get(int(h.get('match_state', 0) or 0), '?')}"
                        f"   stage {h.get('stage_id')} FD\n")
        blast = blast_box(h)
        t.insert("end", f"blast {blast[0]:.0f} {blast[1]:.0f} {blast[2]:.0f} "
                        f"{blast[3]:.0f}   flags 0x{int(h.get('flags', 0) or 0):x}\n")
        if model.extras is not None:
            e = model.extras
            t.insert("end", f"extra seq {e.seq}  age "
                            f"{0 if e.age_ms is None else e.age_ms:5.1f} ms  "
                            f"objects {len(model.objects)}\n", "dim")
        else:
            t.insert("end", "extra: off / unavailable   objects 0\n", "dim")
        t.insert("end", "\n")

        head = f"{'field':13s} {'P1':>20s}  {'P2':>20s}\n"
        t.insert("end", head, "head")
        for name in FIELD_NAMES:
            v1 = model.fighters[0].get(name) if model.fighters else None
            v2 = model.fighters[1].get(name) if len(model.fighters) > 1 else None
            t.insert("end", f"{NICE.get(name, name):13s} ")
            self.insert_value(t, model, 0, name, v1)
            t.insert("end", "  ")
            self.insert_value(t, model, 1, name, v2)
            t.insert("end", "\n")
        if self.show_hex:
            t.insert("end", "\nraw rows (32 x u32, little endian)\n", "head")
            for slot in (0, 1):
                pairs = []
                for name in FIELD_NAMES:
                    v = model.fighters[slot].get(name, 0)
                    try:
                        if name in ("buttons", "raw_flags", "pad"):
                            w = int(v) & 0xFFFFFFFF
                        else:
                            import struct as _s
                            w = _s.unpack("<I", _s.pack("<f",
                                         float(v)))[0] if isinstance(v, float) \
                                else int(v) & 0xFFFFFFFF
                    except Exception:  # noqa: BLE001
                        w = 0
                    pairs.append(f"{w:08x}")
                t.insert("end", f"P{slot + 1} " + " ".join(pairs) + "\n",
                         "dim")

        e = model.extras
        t.insert("end", "\nEXTRAS\n", "head")
        if e is None:
            t.insert("end", "not available (run with --extra direct)\n", "dim")
        else:
            t.insert("end", f"rule kind {e.get('g.match.rule_kind', 0)}   "
                            f"stock {e.get('g.match.stock', 0)}   "
                            f"stage_raw {e.get('g.match.stage_id_raw', 0)}\n",
                     "dim")
            for slot in (0, 1):
                i = slot + 1
                t.insert("end", f"P{slot + 1}  ", f"slot{slot}")
                t.insert("end",
                         f"stock {e.get('f.info.stock', i, '-')}  "
                         f"dead {e.get('f.info.dead_count', i, '-')}  "
                         f"shield_parts {e.get('f.shield.part_count', i, '-')}  "
                         f"likely {e.get('f.shield.likely', i, '-')}  "
                         f"hitlag {e.get('f.stop.hit_stop_frame', i, '-')}  "
                         f"speed {e.get('f.kinetic.engine_speed', i, '-')}  "
                         f"floor {e.get('f.ground.floor_y', i, '-')}\n",
                         "dim")
        t.insert("end", "\nEVENTS\n", "head")
        for line in self.events.text():
            t.insert("end", line + "\n", "dim")
        t.configure(state="disabled")
        t.yview_moveto(0.0)

    def age_tag(self, age_ms: float) -> str:
        if age_ms > 500.0:
            return "bad"
        if age_ms > 100.0:
            return "warn"
        return "ok"

    def insert_value(self, t, model: FrameModel, slot: int, name: str,
                     value) -> None:
        text, tag = format_value(name, value)
        flags = int(model.fighters[slot].get("raw_flags", 0) or 0) if \
            slot < len(model.fighters) else 0
        bit = FIELD_INDEX.get(name, -1)
        if bit >= 0 and name not in ("pad",) and not ((flags >> bit) & 1):
            tag = "bad"
        elif tag == "ok" and model.age_ms > 500:
            tag = "bad"
        elif tag == "ok" and model.age_ms > 100:
            tag = "warn"
        t.insert("end", f"{text:>20s}", tag)


def char_name(kind) -> str:
    try:
        k = int(kind)
    except (TypeError, ValueError):
        return "?"
    return {0: "MARIO", 1: "DONKEY KONG"}.get(k, f"KIND {k}")


def damage_color(dmg: float) -> str:
    """Green -> yellow -> red as damage grows (cosmetic)."""
    x = max(0.0, min(1.0, dmg / 150.0))
    if x < 0.5:
        a, b, f = (63, 185, 80), (210, 153, 34), x / 0.5
    else:
        a, b, f = (210, 153, 34), (248, 81, 73), (x - 0.5) / 0.5
    rgb = tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))
    return "#%02x%02x%02x" % rgb


def format_value(name: str, value) -> tuple[str, str]:
    if value is None:
        return "-", "dim"
    if name == "buttons":
        mask = int(value) & 0xFFFFFFFF
        names = schema.buttons_pressed(mask)
        return f"0x{mask:x} " + (",".join(names) if names else "-"), "ok"
    if name == "raw_flags":
        return f"0x{int(value):08x}", "ok"
    if name == "action_state":
        return f"{int(value)} {action_name(value)}", "ok"
    if name == "kind":
        return f"{int(value)} {char_name(value)}", "ok"
    if isinstance(value, float):
        if name in ("x", "y", "vx", "vy", "dx", "dy", "dist_to_blast"):
            text = f"{value:8.2f}"
        elif name in ("damage", "shield_size"):
            text = f"{value:7.2f}"
        elif name == "action_frame":
            text = f"{value:8.2f}"
        elif name in ("stick_x", "stick_y", "cstick_x", "cstick_y", "trigger"):
            text = f"{value:+.2f}"
        else:
            text = f"{value:g}"
    else:
        text = str(value)
    ok = PLAUSIBLE.get(name)
    try:
        good = bool(ok(float(value))) if ok else True
    except (TypeError, ValueError):
        good = True
    return text, ("ok" if good else "bad")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def dump_model(model: FrameModel, as_json: bool) -> None:
    if as_json:
        print(json.dumps(model.to_json(), indent=2, default=str))
        return
    h = model.header
    print(f"source={model.source} seq={h.get('seq')} fight_on={h.get('fight_on')} "
          f"stage={h.get('stage_id')} age={model.age_ms:.1f}ms")
    for i, f in enumerate(model.fighters):
        print(f"P{i + 1} {char_name(f.get('kind'))} damage={f.get('damage')} "
              f"x={f.get('x')} y={f.get('y')} state={f.get('action_state')} "
              f"({action_name(f.get('action_state'))}) "
              f"badge={state_badge(f)}")
    print(f"objects={len(model.objects)}")
    for o in model.objects:
        print(f"  {o.label} at ({o.x:.1f},{o.y:.1f})")
    if model.extras is not None:
        print(f"extra seq={model.extras.seq} names={len(model.extras.by_name)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shm", default=schema.SHM_PATH)
    ap.add_argument("--extra", choices=("auto", "shm", "direct", "off"),
                    default="auto",
                    help="extra block source (auto = published block, else "
                         "direct sampling; direct shows projectiles even "
                         "without extra_state.py; off hides extras)")
    ap.add_argument("--extra-path", default="/dev/shm/ssbu_extra.bin")
    ap.add_argument("--extra-fps", type=float, default=12.0)
    ap.add_argument("--mock", action="store_true",
                    help="animated fake data (no game, no shm)")
    ap.add_argument("--dump-once", action="store_true",
                    help="print one model and exit (no window)")
    ap.add_argument("--json", action="store_true", help="with --dump-once")
    ap.add_argument("--no-topmost", action="store_true")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=860)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--panel-hz", type=float, default=12.0)
    ap.add_argument("--trails", type=int, default=48)
    ap.add_argument("--max-objects", type=int, default=32)
    ap.add_argument("--stats-file", default=DEFAULT_STATS_FILE,
                    help="PPO training status JSON for the bottom-left overlay")
    args = ap.parse_args(argv)

    if args.mock:
        source = MockSource(hz=args.fps)
        source.label = "mock"
    else:
        source = ShmSource(args.shm, extra=args.extra,
                           extra_path=args.extra_path,
                           extra_fps=args.extra_fps)
        source.label = "shm"
    if args.dump_once:
        for _ in range(3):
            model = source.read()
            if model is not None:
                break
            time.sleep(0.05)
        if model is None:
            print("no frame", file=sys.stderr)
            source.close()
            return 2
        dump_model(model, args.json)
        source.close()
        return 0

    import tkinter as tk
    root = tk.Tk()
    view = LiveView(root, source, topmost=not args.no_topmost,
                    width=args.width, height=args.height, fps=args.fps,
                    panel_hz=args.panel_hz, trails=args.trails,
                    max_objects=args.max_objects,
                    stats_file=args.stats_file)
    try:
        root.mainloop()
    finally:
        try:
            source.close()
        except Exception:  # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
