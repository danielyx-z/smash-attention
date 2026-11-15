#!/usr/bin/env python3
"""Live verification of every v7 state field against the running game.

Three modes, because the game exposes different truths:

``physical``  P1 is human and driven by the uinput pad.  This proves the
              pad -> game -> memory path for position, facing, action state,
              action frame, on_ground, is_actionable, jumps_left, shield and
              velocity.  The expert control fields follow the pad too; if they
              stay zero, the pad input never reached the game (the pads and the
              keyboard are gated on ``SDL_GetKeyboardFocus()``).

``cpu``       Observe a CPU-vs-CPU fight.  The expert fields are live here, so
              this validates buttons/stick/cstick/trigger plus the derived
              shield/hitstun/is_actionable flags against the game's own status
              kinds, and reads the match rule header from the extra block.

``seq``       Drive P1 with the recomp's synthetic pad (``SSBU_XINPUT_SYNTH=1``
              plus ``SSBU_SEQ=<inline sequence text>``).  This is independent of
              window focus and is the mode that works in a locked/unfocused
              desktop session.  The expected inputs are baked into the sequence;
              this mode asserts walk/jump/shield/attack/special signatures.

All modes run the same field-invariant checks: dx/dy, facing_rel,
dist_to_blast, vx/vy, raw_flags, slot identity, fps and (when available)
damage/stock/velocity against the independent extra block.

Usage (always under the session lock)::

    .venv/bin/python -m rl.scripts.verify_live_state --mode physical --json
    .venv/bin/python -m rl.scripts.verify_live_state --mode cpu --json
    .venv/bin/python -m rl.scripts.verify_live_state --mode seq --json

``physical`` needs the game window to hold SDL's keyboard focus; ``seq`` does
not.  Every run writes the captured frames to ``--out`` so it can be audited
offline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

from rl.env.gym_env import RawSnapshotReader, SnapshotTimeout
from rl.memory import schema
from rl.memory.gamestate import DAMAGE_STATUS_KINDS, GUARD_STATUS_KINDS
from rl.memory.memapi import find_game_pid

FIELD = schema.FIELD_INDEX
NEUTRAL = {"buttons": [], "stick": (0.0, 0.0), "cstick": (0.0, 0.0)}

# Status-kind families (common Smash enum; only used to classify an observed
# state, never to derive a field).
ATTACKS = set(range(22, 35)) | set(range(43, 50)) | set(range(59, 62))
SPECIALS = set(range(35, 43)) | {183, 184, 185, 186, 187, 188, 189}
JUMPS = {11, 12, 13, 14, 15, 16, 17}
DEAD = {165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 263}
FINAL_SCENE = {197, 199, 214}


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
class Sampler(threading.Thread):
    """Poll the published block at ~60 Hz and tag every new frame."""

    def __init__(self, reader: RawSnapshotReader, poll: float = 0.002):
        super().__init__(daemon=True)
        self.reader = reader
        self.poll = poll
        self.segment = "boot"
        self.frames: list[tuple[str, float, object]] = []
        self.stopped = False
        self.error: BaseException | None = None
        self._last_seq = -1

    def run(self) -> None:
        try:
            while not self.stopped:
                snap = self.reader.read_latest(timeout=0.5)
                if snap.seq != self._last_seq:
                    self._last_seq = snap.seq
                    self.frames.append((self.segment, time.monotonic(), snap))
                else:
                    time.sleep(self.poll)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the caller
            self.error = exc

    def stop(self) -> None:
        self.stopped = True
        self.join(timeout=2.0)


def wait_live(reader: RawSnapshotReader, timeout: float, settle: int = 20):
    """Wait for a fight whose fighter rows have stopped changing identity.

    ``impl+0xd2`` (fight_on) goes to 1 a few frames before the entries hold the
    new match: the first frames can still carry the previous match's kind /
    damage.  Require the (kind, entry_id) tuple to stay put for ``settle`` new
    frames so callers never start on that stale window.
    """
    deadline = time.monotonic() + timeout
    last = None
    last_seq = -1
    key = None
    stable = 0
    while time.monotonic() < deadline:
        try:
            snap = reader.read_latest(timeout=1.0)
        except SnapshotTimeout:
            time.sleep(0.05)
            continue
        last = snap
        live = (int(snap.fight_on) == 1 and int(snap.n_fighters) >= 2 and
                all(int(f.get("raw_flags", 0)) for f in snap.fighters))
        if not live:
            key = None
            stable = 0
            last_seq = snap.seq
            time.sleep(0.02)
            continue
        if snap.seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = snap.seq
        now_key = tuple((int(f.get("kind", -1)), int(f.get("entry_id", -1)))
                        for f in snap.fighters)
        if now_key == key:
            stable += 1
        else:
            key = now_key
            stable = 1
        if stable >= settle:
            return snap
        time.sleep(0.005)
    raise TimeoutError(
        f"no stable live match within {timeout}s (last fight_on="
        f"{getattr(last, 'fight_on', None)}, n={getattr(last, 'n_fighters', None)}, "
        f"stable={stable}/{settle})")


def frames_gone_quiet(sampler: "Sampler", quiet_s: float = 1.0) -> bool:
    """True once every sampled frame has been fight_on=0 for a while."""
    if not sampler.frames:
        return False
    last = sampler.frames[-1][2]
    if int(last.fight_on) != 0:
        return False
    last_live = max((t for _l, t, s in sampler.frames if int(s.fight_on) == 1),
                    default=None)
    return last_live is not None and time.monotonic() - last_live > quiet_s


def wait_slot(reader: RawSnapshotReader, slot: int, pred, timeout: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            snap = reader.read_latest(timeout=0.2)
        except SnapshotTimeout:
            continue
        if int(snap.fight_on) != 1:
            return False
        if pred(snap.fighters[slot]):
            return True
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, name: str, ok: bool | None, detail: str = "") -> None:
        self.rows.append({"check": name, "ok": ok, "detail": detail})

    @property
    def failures(self) -> int:
        return sum(1 for r in self.rows if r["ok"] is False)

    def print(self) -> None:
        for r in self.rows:
            tag = "INFO" if r["ok"] is None else ("PASS" if r["ok"] else "FAIL")
            print(f"  [{tag}] {r['check']:38s} {r['detail']}")
        print(f"  {sum(1 for r in self.rows if r['ok'] is True)} pass, "
              f"{self.failures} fail, "
              f"{sum(1 for r in self.rows if r['ok'] is None)} info")


def _f(fighter: dict, name: str, default=float("nan")) -> float:
    v = fighter.get(name, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _series(frames, slot: int, name: str) -> np.ndarray:
    return np.array([_f(f, name) for f in (s.fighters[slot] for _, _, s in frames)],
                    dtype=float)


def _rows(frames, slot: int) -> list[dict]:
    return [s.fighters[slot] for _, _, s in frames]


# ---------------------------------------------------------------------------
# shared invariant checks
# ---------------------------------------------------------------------------
def check_snapshot_header(rep: Report, snap) -> None:
    rep.add("shm version == 7", int(snap.version) == 7, f"version={snap.version}")
    rep.add("stage_id == 3 (FD)", int(snap.stage_id) == 3, f"stage_id={snap.stage_id}")
    rep.add("reader fps ~ 60", abs(float(snap.fps) - 60.0) <= 2.0,
            f"fps={snap.fps:.2f}")
    rep.add("header flags: stage+control bits", int(snap.flags) & 3 == 3,
            f"flags=0x{int(snap.flags):x}")


def check_slots(rep: Report, snap, expect_cpu: bool,
                expect_kinds: bool = True) -> None:
    f0, f1 = snap.fighters[0], snap.fighters[1]
    kinds = (int(f0.get("kind", -1)), int(f1.get("kind", -1)))
    if expect_kinds:
        rep.add("slot 0 = Mario (kind 0)", kinds[0] == 0, f"kind={kinds[0]}")
        rep.add("slot 1 = DK (kind 1)", kinds[1] == 1, f"kind={kinds[1]}")
    else:
        rep.add("fighter kinds (seq match selects its own)", None, str(kinds))
    rep.add("entry_id 0 / 1", (int(f0.get("entry_id", -1)),
                               int(f1.get("entry_id", -1))) == (0, 1),
            f"{f0.get('entry_id')}/{f1.get('entry_id')}")
    if expect_cpu:
        rep.add("both slots is_cpu == 1",
                int(f0.get("is_cpu", 0)) == 1 and int(f1.get("is_cpu", 0)) == 1,
                f"{f0.get('is_cpu')}/{f1.get('is_cpu')}")
    rep.add("raw_flags == 0x7fffffff (all fields)",
            int(f0.get("raw_flags", 0)) == 0x7FFFFFFF and
            int(f1.get("raw_flags", 0)) == 0x7FFFFFFF,
            f"{int(f0.get('raw_flags', 0)):#x}/{int(f1.get('raw_flags', 0)):#x}")


def check_invariants(rep: Report, frames) -> None:
    """Field-to-field invariants that hold on every frame, for both slots."""
    frames = [(l, t, s) for l, t, s in frames if int(s.fight_on) == 1]
    if not frames:
        rep.add("invariants: live frames captured", False, "no live frames")
        return
    snaps = [s for _, _, s in frames]
    raw = np.stack([s.raw for s in snaps])          # (N,2,32) float64

    x = raw[:, :, FIELD["x"]]
    y = raw[:, :, FIELD["y"]]
    dx = raw[:, :, FIELD["dx"]]
    dy = raw[:, :, FIELD["dy"]]
    fr = raw[:, :, FIELD["facing_rel"]]
    facing = raw[:, :, FIELD["facing"]]
    expect_dx = x[:, ::-1] - x
    expect_dy = y[:, ::-1] - y
    rep.add("dx == opponent_x - self_x", bool(np.allclose(dx, expect_dx, atol=1e-4)),
            f"maxerr={np.max(np.abs(dx - expect_dx)):.2e}")
    rep.add("dy == opponent_y - self_y", bool(np.allclose(dy, expect_dy, atol=1e-4)),
            f"maxerr={np.max(np.abs(dy - expect_dy)):.2e}")
    with np.errstate(invalid="ignore"):
        expect_fr = np.where(np.sign(expect_dx) == np.sign(facing), 1,
                             np.where(np.sign(expect_dx) == 0, 0, -1))
    rep.add("facing_rel follows sign(dx)/facing",
            bool(np.all(fr == expect_fr)), f"mismatch={int(np.sum(fr != expect_fr))}")

    blast = snaps[0].blast
    expect_dist = np.maximum(0.0, np.minimum(
        np.minimum(x - blast[0], blast[1] - x),
        np.minimum(y - blast[3], blast[2] - y)))
    dist = raw[:, :, FIELD["dist_to_blast"]]
    rep.add("dist_to_blast == live camera bounds",
            bool(np.allclose(dist, expect_dist, atol=0.05)),
            f"maxerr={np.max(np.abs(dist - expect_dist)):.3f}")

    ts = np.array([s.ts_ns for s in snaps], dtype=np.int64)
    for slot in (0, 1):
        dt = np.diff(ts) / 1e9
        ok = dt > 0
        vx = raw[1:, slot, FIELD["vx"]]
        vy = raw[1:, slot, FIELD["vy"]]
        ex = np.diff(x[:, slot]) / np.where(ok, dt, 1.0)
        ey = np.diff(y[:, slot]) / np.where(ok, dt, 1.0)
        # velocity is derived from the reader's own monotonic clock, so it is
        # identical only up to the timestamp granularity; 5 % + 1 unit is fair.
        errx = np.abs(vx - ex) - (0.05 * np.abs(ex) + 1.0)
        erry = np.abs(vy - ey) - (0.05 * np.abs(ey) + 1.0)
        bad = int(np.sum((errx > 0) | (erry > 0)))
        rep.add(f"slot{slot} vx/vy == d(position)/dt", bad <= 2,
                f"bad={bad}/{len(dt)} of frames")

    for slot in (0, 1):
        st = raw[:, slot, FIELD["action_state"]].astype(int)
        shield = raw[:, slot, FIELD["shield"]].astype(int)
        hitstun = raw[:, slot, FIELD["hitstun"]].astype(int)
        guard = np.isin(st, list(GUARD_STATUS_KINDS))
        dmg = np.isin(st, list(DAMAGE_STATUS_KINDS))
        rep.add(f"slot{slot} shield <=> status in {sorted(GUARD_STATUS_KINDS)}",
                bool(np.all(shield == guard)), f"frames={int(guard.sum())}")
        rep.add(f"slot{slot} hitstun <=> DAMAGE statuses",
                bool(np.all(hitstun == dmg)), f"frames={int(dmg.sum())}")

        # action_frame: never runs backwards inside one status, never jumps
        # more than two frames; in an active status it advances roughly one
        # frame per step (WAIT/GUARD/knockdown may hold it).
        same = st[1:] == st[:-1]
        af = raw[:, slot, FIELD["action_frame"]]
        daf = np.diff(af)
        # A new action instance reuses the same status kind and restarts the
        # frame at 0; that is a reset, not a torn field.
        reset = (daf < -0.51) & (af[1:] <= 1.01)
        bad_dec = int(np.sum(same & (daf < -0.51) & ~reset))
        bad_jump = int(np.sum(same & (daf > 2.51)))
        rep.add(f"slot{slot} action_frame monotonic within a status",
                bad_dec == 0 and bad_jump == 0,
                f"decreases={bad_dec} resets={int(np.sum(same & reset))} "
                f"jumps>2={bad_jump} of {int(same.sum())}")
        active = same & ~np.isin(st[:-1], [0, 10, 64])
        if int(active.sum()):
            ok = (np.abs(daf - 1.0) <= 0.51) | reset
            frac = float(np.mean(ok[active]))
            rep.add(f"slot{slot} action_frame advances in active statuses",
                    frac >= 0.80,
                    f"{frac * 100:.0f}% of {int(active.sum())} steps")

    # no non-finite value in any field
    finite = np.isfinite(raw[:, :, [FIELD[n] for n in schema.FIELD_NAMES
                                    if n not in ("buttons", "raw_flags", "pad")]])
    rep.add("no NaN/Inf in any scalar field", bool(np.all(finite)))


# ---------------------------------------------------------------------------
# extra block cross-check
# ---------------------------------------------------------------------------
def extra_snapshot() -> dict:
    try:
        from rl.extra.extra_state import ExtraState
    except Exception:
        return {}
    pid = find_game_pid()
    if not pid:
        return {}
    try:
        with ExtraState(pid) as es:
            return {rec.name: rec.value_python() for rec in es.read()}
    except Exception:
        return {}


def check_extra(rep: Report, snap, extra: dict) -> None:
    if not extra:
        rep.add("extra block cross-check", None, "unavailable")
        return
    if snap is None:
        rep.add("extra block cross-check", None, "no live frame to compare")
        return
    f0, f1 = snap.fighters[0], snap.fighters[1]
    pairs = (
        ("damage", f0, "f0.damage.amount"),
        ("damage", f1, "f1.damage.amount"),
    )
    for name, fighter, key in pairs:
        if key in extra:
            a = _f(fighter, name)
            b = float(extra[key])
            rep.add(f"extra: {key.split('.')[0]} damage matches core",
                    abs(a - b) < 0.05, f"core={a:.3f} extra={b:.3f}")
    for idx, fighter in ((0, f0), (1, f1)):
        key = f"f{idx}.info.stock"
        if key in extra:
            rep.add(f"extra: stock slot{idx} matches core",
                    int(fighter.get("stock", -1)) == int(extra[key]),
                    f"core={fighter.get('stock')} extra={extra[key]}")
        key = f"f{idx}.kinetic.engine_vel"
        if key in extra:
            v = extra[key]
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                a = np.array([_f(fighter, "vx"), _f(fighter, "vy")])
                b = np.array([float(v[0]), float(v[1])])
                # engine velocity is units/frame; core is units/second
                rep.add(f"extra: engine velocity direction slot{idx}",
                        float(np.dot(a, b)) >= -1e-3 or float(np.linalg.norm(b)) < 1e-6,
                        f"core=({a[0]:.1f},{a[1]:.1f}) engine=({b[0]:.2f},{b[1]:.2f})")
    rule = {k: extra.get(k) for k in ("g.match.rule_kind", "g.match.stock",
                                     "g.match.item_appear", "g.match.melee_mode")
            if k in extra}
    if rule:
        rep.add("extra: match rule header", None, json.dumps(rule))


# ---------------------------------------------------------------------------
# physical mode
# ---------------------------------------------------------------------------
class MatchEnded(Exception):
    """The fight went to results while the script was running."""


def physical_script(pad, sampler: Sampler, reader: RawSnapshotReader, slot: int,
                    gap: float = 0.15) -> list[str]:
    done: list[str] = []

    def run(label: str, action: dict, hold: float, wait: str = "actionable") -> None:
        try:
            snap = reader.read_latest(timeout=0.3)
            if int(snap.fight_on) != 1:
                raise MatchEnded(label)
        except SnapshotTimeout:
            pass
        if wait == "actionable":
            wait_slot(reader, slot, lambda f: int(f.get("is_actionable", 0)) == 1, 0.9)
        elif wait == "grounded":
            wait_slot(reader, slot, lambda f: int(f.get("on_ground", 0)) == 1, 1.2)
        elif wait == "airborne":
            wait_slot(reader, slot, lambda f: int(f.get("on_ground", 1)) == 0, 0.6)
        sampler.segment = label
        pad.apply(action)
        time.sleep(hold)
        pad.apply(NEUTRAL)
        sampler.segment = "gap"
        time.sleep(gap)
        done.append(label)

    run("idle", dict(NEUTRAL), 0.50)
    run("stick_right", {"stick": (1.0, 0.0)}, 0.70)
    run("stick_left", {"stick": (-1.0, 0.0)}, 0.70)
    run("stick_down", {"stick": (0.0, -1.0)}, 0.45)
    run("jump_X", {"buttons": ["X"]}, 0.50)
    run("jump_X_again", {"buttons": ["X"]}, 0.50, wait="airborne")
    run("shield_ZL", {"buttons": ["ZL"]}, 0.60, wait="grounded")
    run("attack_A", {"buttons": ["A"]}, 0.50)
    run("special_B", {"buttons": ["B"]}, 0.50)
    run("cstick_right", {"cstick": (1.0, 0.0)}, 0.50)
    run("cstick_up", {"cstick": (0.0, 1.0)}, 0.50)
    run("trigger_ZR", {"trigger": 1.0}, 0.45)
    run("dpad_left", {"buttons": ["DPAD_L"]}, 0.40)
    run("idle_end", dict(NEUTRAL), 0.40)
    return done


def check_physical(rep: Report, frames, slot: int) -> None:
    by: dict[str, list] = {}
    for label, _t, snap in frames:
        by.setdefault(label, []).append(snap)

    def seg(label: str):
        return [(t, s) for label2, t, s in frames if label2 == label]

    def stats(label: str):
        rows = [s.fighters[slot] for _t, s in seg(label)]
        if not rows:
            return None
        return {
            "x": np.array([_f(f, "x") for f in rows]),
            "y": np.array([_f(f, "y") for f in rows]),
            "state": np.array([int(f.get("action_state", -1)) for f in rows]),
            "af": np.array([_f(f, "action_frame") for f in rows]),
            "og": np.array([int(f.get("on_ground", -1)) for f in rows]),
            "act": np.array([int(f.get("is_actionable", -1)) for f in rows]),
            "jumps": np.array([int(f.get("jumps_left", -1)) for f in rows]),
            "facing": np.array([_f(f, "facing") for f in rows]),
            "shield": np.array([int(f.get("shield", -1)) for f in rows]),
            "buttons": np.array([int(f.get("buttons", 0)) for f in rows]),
            "stick_x": np.array([_f(f, "stick_x") for f in rows]),
            "cstick_x": np.array([_f(f, "cstick_x") for f in rows]),
            "trigger": np.array([_f(f, "trigger") for f in rows]),
        }

    base = stats("idle")
    if base is None:
        rep.add("physical: idle captured", False, "no idle frames")
        return
    rep.add("pad mode: idle keeps the fighter still",
            float(np.nanmax(base["x"]) - np.nanmin(base["x"])) < 1.0 and
            float(np.nanmean(base["og"])) > 0.8,
            f"dx={np.nanmax(base['x']) - np.nanmin(base['x']):.2f} "
            f"onground={np.nanmean(base['og']):.2f}")
    rep.add("pad mode: idle is WAIT/actionable",
            int(base["state"][0]) in (0, 1) and bool(np.all(base["act"] == 1)),
            f"states={sorted(set(base['state'].tolist()))}")

    def delta(label: str, axis: str):
        st = stats(label)
        return None if st is None else float(st[axis][-1] - st[axis][0])

    d = delta("stick_right", "x")
    st = stats("stick_right")
    rep.add("stick right moves +x", d is not None and d > 3.0,
            f"dx={d if d is None else round(d, 2)} "
            f"states={sorted(set(st['state'].tolist())) if st else '-'}")
    if st is not None and "facing" in st:
        rep.add("stick right faces +1", float(np.nanmean(st["facing"])) > 0.5,
                f"facing={np.nanmean(st['facing']):.2f}")

    d = delta("stick_left", "x")
    st = stats("stick_left")
    rep.add("stick left moves -x", d is not None and d < -3.0,
            f"dx={d if d is None else round(d, 2)}")
    if st is not None:
        rep.add("stick left faces -1", float(np.nanmean(st["facing"])) < -0.5,
                f"facing={np.nanmean(st['facing']):.2f}")

    st = stats("stick_down")
    if st is not None:
        rep.add("stick down: grounded and crouch-ish", bool(np.all(st["og"] == 1)),
                f"states={sorted(set(st['state'].tolist()))}")

    j0, j1 = stats("jump_X"), stats("jump_X_again")
    if j0 is not None:
        rep.add("X jumps: leaves the ground",
                float(np.nanmax(j0["y"]) - np.nanmin(j0["y"])) > 8.0 and
                int(j0["og"].min()) == 0,
                f"dy={np.nanmax(j0['y']) - np.nanmin(j0['y']):.1f} "
                f"states={sorted(set(j0['state'].tolist()))}")
        rep.add("X jump consumes a jump", int(j0["jumps"].min()) < int(base["jumps"][0]),
                f"jumps {base['jumps'][0]}->{int(j0['jumps'].min())}")
    if j1 is not None and j0 is not None:
        rep.add("second X jump consumes another jump",
                int(j1["jumps"].min()) < int(j0["jumps"].min()),
                f"jumps {int(j0['jumps'].min())}->{int(j1['jumps'].min())}")

    sh = stats("shield_ZL")
    if sh is not None:
        mask = np.isin(sh["state"], list(GUARD_STATUS_KINDS))
        rep.add("ZL shield reaches GUARD status 10", bool(mask.any()),
                f"states={sorted(set(sh['state'].tolist()))}")
        if mask.any():
            rep.add("ZL shield raises the shield flag", bool(np.all(sh["shield"][mask] == 1)),
                    f"frames={int(mask.sum())}")

    for label, family, family_name in (("attack_A", ATTACKS, "attack"),
                                       ("special_B", SPECIALS, "special")):
        st = stats(label)
        if st is not None:
            got = sorted(set(int(v) for v in st["state"] if int(v) in family))
            rep.add(f"{label} reaches a {family_name} status", bool(got),
                    f"states={sorted(set(st['state'].tolist()))}")

    for label in ("cstick_right", "cstick_up"):
        st = stats(label)
        if st is not None:
            got = sorted(set(int(v) for v in st["state"] if int(v) in ATTACKS))
            rep.add(f"{label} reaches an attack status", bool(got),
                    f"states={sorted(set(st['state'].tolist()))}")
            rep.add(f"{label}: expert cstick stays 0 (pad is invisible)",
                    float(np.nanmax(np.abs(st["cstick_x"]))) < 1e-6,
                    f"|cstick_x|max={float(np.nanmax(np.abs(st['cstick_x']))):.3g}")

    # the known caveat, stated explicitly rather than scored as a bug
    all_btn = np.concatenate([stats(k)["buttons"] for k in by
                              if stats(k) is not None])
    all_stick = np.concatenate([stats(k)["stick_x"] for k in by
                                if stats(k) is not None])
    all_trig = np.concatenate([stats(k)["trigger"] for k in by
                               if stats(k) is not None])
    rep.add("physical: expert fields (0 = input never reached the game)",
            None,
            f"buttons nonzero={int(np.sum(all_btn != 0))}, "
            f"|stick|max={float(np.nanmax(np.abs(all_stick))):.3g}, "
            f"trigger max={float(np.nanmax(all_trig)):.3g}")


# ---------------------------------------------------------------------------
# cpu mode
# ---------------------------------------------------------------------------
def check_cpu(rep: Report, frames) -> dict:
    if not frames:
        rep.add("cpu: frames captured", False, "no frames")
        return {}
    rows0 = _rows(frames, 0)
    rows1 = _rows(frames, 1)
    rows = rows0 + rows1

    btn = np.array([int(f.get("buttons", 0)) for f in rows], dtype=np.uint32)
    st = np.array([int(f.get("action_state", -1)) for f in rows])
    stick = np.array([[float(f.get("stick_x", 0.0)), float(f.get("stick_y", 0.0))]
                      for f in rows])
    cstick = np.array([[float(f.get("cstick_x", 0.0)), float(f.get("cstick_y", 0.0))]
                       for f in rows])
    trig = np.array([float(f.get("trigger", 0.0)) for f in rows])
    rep.add("CPU expert buttons populated", int(np.sum(btn != 0)) > 0,
            f"{int(np.sum(btn != 0))}/{len(btn)} frames, "
            f"masks={[hex(int(v)) for v in sorted(set(btn.tolist()))][:8]}")
    rep.add("CPU expert stick populated", float(np.nanmax(np.abs(stick))) > 0.2,
            f"max|stick|={float(np.nanmax(np.abs(stick))):.2f}")
    rep.add("CPU expert cstick populated", None,
            f"max|cstick|={float(np.nanmax(np.abs(cstick))):.3g} "
            "(CPUs do not use the C-stick on this build)")
    rep.add("CPU expert trigger populated", float(np.nanmax(trig)) > 0.0,
            f"max trigger={float(np.nanmax(trig)):.4g}")

    # buttons should be pressed while doing something attack/jump-like
    active = np.isin(st, list(ATTACKS | SPECIALS | JUMPS)) | \
        np.isin(st, list(DAMAGE_STATUS_KINDS))
    pressed = btn != 0
    if pressed.any():
        rep.add("CPU buttons align with action states",
                float(np.mean(active[pressed])) > 0.35,
                f"{100 * float(np.mean(active[pressed])):.0f}% of pressed frames")

    # stick direction versus horizontal motion while grounded
    ground = np.array([int(f.get("on_ground", 0)) for f in rows]) == 1
    vx = np.array([float(f.get("vx", 0.0)) for f in rows])
    m = ground & (np.abs(stick[:, 0]) > 0.3) & (np.abs(vx) > 1.0)
    if int(m.sum()) >= 10:
        agree = float(np.mean(np.sign(stick[m, 0]) == np.sign(vx[m])))
        rep.add("CPU stick_x sign agrees with vx", agree > 0.8,
                f"{agree * 100:.0f}% of {int(m.sum())} grounded frames")

    # is_actionable: true in WAIT, false in hitstun/tumble
    act = np.array([int(f.get("is_actionable", -1)) for f in rows])
    wait = st == 0
    hurt = np.isin(st, list(DAMAGE_STATUS_KINDS)) | np.isin(st, (476, 477, 478, 479))
    if int(wait.sum()):
        rep.add("is_actionable is 1 in WAIT", float(np.mean(act[wait] == 1)) > 0.9,
                f"{100 * float(np.mean(act[wait] == 1)):.0f}%")
    if int(hurt.sum()):
        rep.add("is_actionable is 0 in hitstun/tumble",
                float(np.mean(act[hurt] == 0)) > 0.9,
                f"{100 * float(np.mean(act[hurt] == 0)):.0f}%")

    # match lifetime
    fight = [int(s.fight_on) for _, _, s in frames]
    live = [1 for v in fight if v == 1]
    ts = [t for _, t, _s in frames]
    duration = ts[-1] - ts[0] if len(ts) > 1 else 0.0
    ends = [i for i in range(1, len(fight)) if fight[i - 1] == 1 and fight[i] == 0]
    last_states = []
    last_damage = []
    if ends:
        i = ends[0]
        last_states = [int(frames[i - 1][2].fighters[k].get("action_state", -1))
                       for k in (0, 1)]
        last_damage = [round(_f(frames[i - 1][2].fighters[k], "damage"), 1)
                       for k in (0, 1)]
    states = set(int(f.get("action_state", -1)) for f in rows)
    summary = {
        "fight_frames": len(live),
        "observed_s": round(duration, 2),
        "fight_on_transitions": ends,
        "length_s_at_end": round(ts[ends[0]] - ts[0], 2) if ends else None,
        "last_states": last_states,
        "last_damage": last_damage,
        "has_KO_state": bool(states & DEAD),
        "has_FINAL_SCENE": bool(states & FINAL_SCENE),
    }
    rep.add("CPU fight lifetime observed", None, json.dumps(summary))
    return summary


# ---------------------------------------------------------------------------
# rule header / lifetime question
# ---------------------------------------------------------------------------
def check_rule(rep: Report, extra: dict) -> None:
    if not extra:
        rep.add("match rule header", None, "extra block unavailable")
        return
    keys = ("g.match.rule_kind", "g.match.stock", "g.match.time",
            "g.match.item_appear", "g.match.melee_mode")
    rule = {k: extra[k] for k in keys if k in extra}
    rep.add("match rule header", None, json.dumps(rule) if rule else "no rule keys")
    raw = extra.get("g.match.raw0x100")
    if isinstance(raw, (bytes, bytearray)):
        head = bytes(raw[:0x60])
        hits = [f"+0x{i:02x}=0x{b:02x}" for i, b in enumerate(head)
                if b in (15, 60, 180, 240, 300)]
        rep.add("rule header plausible time fields", None,
                ", ".join(hits[:12]) if hits else "no 15/60/180/240/300 bytes")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def save_capture(path: str, frames, slot: int, meta: dict) -> None:
    if not frames:
        return
    raw = np.stack([s.raw for _, _, s in frames]).astype(np.float32)
    labels = np.array([label for label, _, _ in frames])
    np.savez_compressed(
        path,
        seq=np.array([s.seq for _, _, s in frames], dtype=np.uint64),
        ts_ns=np.array([s.ts_ns for _, _, s in frames], dtype=np.uint64),
        fight_on=np.array([s.fight_on for _, _, s in frames], dtype=np.uint8),
        match_state=np.array([s.match_state for _, _, s in frames], dtype=np.uint8),
        blast=np.array(frames[0][2].blast, dtype=np.float32),
        stage_id=np.int64(frames[0][2].stage_id),
        labels=labels,
        raw_fighters=raw,
        meta_json=np.array(json.dumps(meta)),
        slot=np.int64(slot),
    )
    print(f"capture: {path} ({len(frames)} frames)")


def game_window() -> str | None:
    """The game's X11 client window id, or None."""
    for probe in (["--class", "ssbu_native"], ["--name", "SSBU native"]):
        try:
            p = subprocess.run(["xdotool", "search", *probe], capture_output=True,
                               text=True, timeout=5,
                               env={**os.environ, "DISPLAY": os.environ.get("SSBU_LIVE_DISPLAY", ":0")})
        except Exception:  # noqa: BLE001
            return None
        ids = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        if ids:
            return ids[-1]
    return None


def focus_game() -> str:
    """Give the game window focus (the pad and keyboard are gated on it)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "focus_game.sh")
    try:
        p = subprocess.run(["bash", path], capture_output=True, text=True, timeout=30)
        return (p.stdout + p.stderr).strip()
    except Exception as exc:  # noqa: BLE001
        return f"focus failed: {type(exc).__name__}: {exc}"


def focus_game_if_needed() -> str:
    """Make the game window hold SDL's keyboard focus.

    The pad is gated on ``SDL_GetKeyboardFocus()``.  X focus alone is not
    enough; SDL raises its flag on a real FocusIn, which a click inside the
    client area produces.  Raise/activate only when another window has the X
    focus (touching a live window more than necessary has wedged this build).
    """
    disp = os.environ.get("SSBU_LIVE_DISPLAY", ":0")
    env = {**os.environ, "DISPLAY": disp}
    try:
        win = game_window()
        if win is None:
            return "no game window yet"
        cur = subprocess.run(["xdotool", "getwindowfocus"], capture_output=True,
                             text=True, timeout=5, env=env).stdout.strip()
        if cur != win:
            msg = focus_game()
        else:
            msg = f"x focus already on {win}"
        # The click is what sets SDL's own keyboard-focus flag.
        subprocess.run(["xdotool", "mousemove", "--window", win, "640", "360"],
                       capture_output=True, timeout=5, env=env)
        subprocess.run(["xdotool", "click", "1"], capture_output=True,
                       timeout=5, env=env)
        return f"{msg}; clicked client area"
    except Exception as exc:  # noqa: BLE001
        return f"focus failed: {type(exc).__name__}: {exc}"


def check_seq(rep: Report, frames, slot: int) -> None:
    """Validate slot 0 against the known SSBU_SEQ battle inputs."""
    frames = [f for f in frames if int(f[2].fight_on) == 1]
    if not frames:
        rep.add("seq: live frames captured", False, "no live frames")
        return
    raw = np.stack([s.raw for _, _, s in frames])
    x = raw[:, slot, FIELD["x"]]
    y = raw[:, slot, FIELD["y"]]
    st = raw[:, slot, FIELD["action_state"]].astype(int)
    facing = raw[:, slot, FIELD["facing"]]
    states = set(st.tolist())

    rep.add("seq: script moved P1 horizontally", float(x.max() - x.min()) > 10.0,
            f"x range [{x.min():.1f},{x.max():.1f}]")
    imax = int(np.argmax(x))
    tail = x[imax:]
    rep.add("seq: walked right, then left",
            imax < len(x) - 2 and float(tail.min()) < float(x[imax]) - 5.0,
            f"xmax={x[imax]:.1f} then {tail.min():.1f}")
    rep.add("seq: jump raised y", float(y.max() - y[0]) > 8.0,
            f"y start {y[0]:.1f} max {y.max():.1f}")
    rep.add("seq: facing flipped both ways",
            bool(np.any(facing > 0.5) and np.any(facing < -0.5)),
            f"facing values {sorted(set(np.round(facing, 1).tolist()))}")
    guard = int(np.sum(np.isin(st, list(GUARD_STATUS_KINDS))))
    rep.add("seq: shield reached status 10", guard >= 3, f"{guard} frames")
    atk = sorted(v for v in states if v in ATTACKS)
    rep.add("seq: attack status seen", bool(atk), f"states={atk}")
    spc = sorted(v for v in states if v in SPECIALS)
    rep.add("seq: special status seen", bool(spc), f"states={spc}")
    rep.add("seq: expert control fields follow the synthetic pad",
            bool(np.any(raw[:, slot, FIELD["buttons"]] != 0)) and
            float(np.max(np.abs(raw[:, slot, FIELD["stick_x"]]))) > 0.5,
            f"buttons nonzero={int(np.sum(raw[:, slot, FIELD['buttons']] != 0))}, "
            f"|stick|max={float(np.max(np.abs(raw[:, slot, FIELD['stick_x']]))):.3g}")
    rep.add("seq: status kinds observed", None, str(sorted(states)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("physical", "cpu", "seq", "both"),
                    default="physical")
    ap.add_argument("--slot", type=int, default=0, help="slot controlled/observed")
    ap.add_argument("--observe", type=float, default=25.0,
                    help="seconds to observe in cpu mode")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="seconds to wait for a live match")
    ap.add_argument("--out", default="", help="write the capture to this .npz")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--reset-first", dest="reset_first", action="store_true",
                    default=True,
                    help="start from an R2 reset so the spawn/damage state is clean")
    ap.add_argument("--no-reset-first", dest="reset_first", action="store_false")
    ap.add_argument("--reset-timeout", type=float, default=25.0)
    ap.add_argument("--no-pad", action="store_true",
                    help="do not create the uinput pad")
    args = ap.parse_args(argv)

    reader = RawSnapshotReader()
    rep = Report()
    summary: dict = {"mode": args.mode, "slot": args.slot}
    pad = None

    try:
        # The pad must exist BEFORE the match is set up: a controller
        # hot-plugged into a running match is not assigned to player 1.
        if args.mode in ("physical", "both") and not args.no_pad:
            from rl.input.gym_input import VirtualPad
            pad = VirtualPad()
            pad.create()
            pad.apply(NEUTRAL)
            print("pad ready:", json.dumps(pad.describe()))
            summary["pad"] = pad.describe()

        print("waiting for a live match ...")
        snap = wait_live(reader, args.timeout)
        print(f"live: seq={snap.seq} stage={snap.stage_id} "
              f"fighters={snap.n_fighters} match_state={snap.match_state}")
        if args.reset_first and args.mode in ("physical", "cpu", "both"):
            try:
                from rl.input.gym_reset import ResetClient
                with ResetClient(timeout=args.reset_timeout) as client:
                    res = client.reset(timeout=args.reset_timeout)
                summary["reset"] = {
                    "seq": res.seq, "ack_seq": res.ack_seq, "state": res.state,
                    "stable_frames": res.stable_frames,
                    "elapsed": round(float(res.elapsed), 3),
                }
                print("reset:", json.dumps(summary["reset"]))
                snap = wait_live(reader, 15.0)
                print(f"after reset: seq={snap.seq} stage={snap.stage_id} "
                      f"damage=" + "/".join(f"{_f(f, 'damage'):.1f}"
                                            for f in snap.fighters))
            except Exception as exc:  # noqa: BLE001
                rep.add("R2 reset before the script", None,
                        f"skipped: {type(exc).__name__}: {exc}")
        if args.mode in ("physical", "both"):
            focus = focus_game_if_needed()
            print("focus:", focus)
            summary["focus"] = focus
            time.sleep(0.4)
        check_snapshot_header(rep, snap)
        check_slots(rep, snap, expect_cpu=(args.mode == "cpu"),
                    expect_kinds=(args.mode != "seq"))
        clean = all(abs(_f(f, "damage")) < 1.0 for f in snap.fighters) and \
            all(abs(_f(f, "x")) < 200.0 for f in snap.fighters)
        rep.add("match started cleanly (0% near spawn)", clean,
                "damage=" + "/".join(f"{_f(f, 'damage'):.1f}" for f in snap.fighters))
        summary["initial"] = {
            "seq": snap.seq, "fps": round(float(snap.fps), 2),
            "fighters": [{k: snap.fighters[i].get(k) for k in
                          ("entry_id", "kind", "is_cpu", "stock", "damage",
                           "x", "y", "action_state", "raw_flags")}
                         for i in (0, 1)],
            "blast": snap.blast,
        }

        sampler = Sampler(reader)
        sampler.start()
        try:
            if args.mode in ("physical", "both"):
                sampler.segment = "gap"
                try:
                    labels = physical_script(pad, sampler, reader, args.slot)
                except MatchEnded as exc:
                    labels = []
                    rep.add("physical: script reached the end of the match", None,
                            f"stopped before segment {exc}")
                summary["segments"] = labels
                sampler.segment = "post"
                time.sleep(0.4)
            if args.mode == "cpu" or args.mode == "both":
                sampler.segment = "cpu"
                t0 = time.monotonic()
                while time.monotonic() - t0 < args.observe:
                    time.sleep(0.2)
                    if sampler.error is not None:
                        break
                sampler.segment = "post"
                time.sleep(0.3)
            if args.mode == "seq":
                sampler.segment = "seq"
                t0 = time.monotonic()
                while time.monotonic() - t0 < args.observe:
                    time.sleep(0.2)
                    if sampler.error is not None:
                        break
                    if frames_gone_quiet(sampler):
                        break
                sampler.segment = "post"
                time.sleep(0.3)
        finally:
            sampler.stop()
    finally:
        if pad is not None:
            pad.destroy()
        reader.close()

    frames = list(sampler.frames)
    if sampler.error is not None:
        rep.add("sampler", False, f"{type(sampler.error).__name__}: {sampler.error}")

    # classify
    phys = [(l, t, s) for l, t, s in frames if l not in
            ("boot", "gap", "post", "cpu")]
    cpu_frames = [(l, t, s) for l, t, s in frames if l == "cpu"]

    if args.mode in ("physical", "both"):
        check_physical(rep, frames, args.slot)
    if args.mode in ("cpu", "both"):
        summary["cpu"] = check_cpu(rep, cpu_frames)
    if args.mode == "seq":
        check_seq(rep, [(l, t, s) for l, t, s in frames if l == "seq"], args.slot)

    check_invariants(rep, phys or frames)
    last_live = None
    for _label, _t, _s in reversed(frames):
        if int(_s.fight_on) == 1:
            last_live = _s
            break
    extra = extra_snapshot()
    check_extra(rep, last_live, extra)
    check_rule(rep, extra)

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        save_capture(args.out, frames, args.slot, summary)

    print("\n=== results ===")
    rep.print()
    if args.json:
        print(json.dumps({
            "summary": summary,
            "checks": rep.rows,
            "failures": rep.failures,
        }, indent=2, default=str))
    return 1 if rep.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
