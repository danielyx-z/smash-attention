#!/usr/bin/env python3
"""Scripted live scenarios that verify the status-kind (action_state) ids.

Every scenario drives P1 through the virtual pad, records the ego's status id
and physical signals (ground, actionable, jumps, position, velocity) per
decision, and prints the run-length state sequence.  Checks then compare what
the game actually did with the constants in ``rl.memory.status_kinds``: they
discover an id from *behaviour* (e.g. "not actionable, falling at terminal
speed right after Up-B") and assert it is the id the code uses.

Output: a report on stdout and ``status_ids_verified.json`` (id -> scenarios
that produced it, with mean signals) next to ``status_kinds.py``.

    # game running with the pad free (see probe_upb.py), then:
    python -m rl.scripts.status_scenarios --out src/rl/memory/status_ids_verified.json
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Callable

import numpy as np

from rl.bc import gym_eval as ge
from rl.env.gym_env import SSBUGymEnv
from rl.input.gym_input import VirtualPad
from rl.input.gym_reset import ResetClient
from rl.memory import status_kinds as sk

BTN = ["A", "B", "X", "Y", "L", "R", "ZL", "ZR", "PLUS", "MINUS",
       "DPAD_L", "DPAD_R"]
FIELDS = ("x", "y", "vx", "vy", "action_state", "action_frame", "jumps_left",
          "on_ground", "is_actionable", "hitstun", "is_on_rebirth", "damage",
          "shield")


def act(buttons=(), sx=0.0, sy=0.0):
    b = np.zeros(12, np.float32)
    for name in buttons:
        b[BTN.index(name)] = 1.0
    return {"buttons": b, "stick": np.array([sx, sy], np.float32),
            "cstick": np.zeros(2, np.float32), "trigger": 0.0}


def ego(info):
    f = (info.get("raw_fighters") or [{}])[0]
    out = {}
    for k in FIELDS:
        try:
            out[k] = float(f.get(k, 0) or 0)
        except (TypeError, ValueError):
            out[k] = 0.0
    out["state"] = int(out["action_state"])
    return out


class Rec:
    """Steps the env and records the ego per decision."""

    def __init__(self, env):
        self.env = env
        self.rows: list[dict] = []
        self.info = {}

    def reset(self):
        _o, self.info = self.env.reset()
        self.rows = []

    def step(self, a, label="", keep=True):
        _o, _r, term, trunc, self.info = self.env.step(a)
        f = ego(self.info)
        f["label"] = label
        if keep:
            self.rows.append(f)
        return f

    def run(self, label, n, stop_on_rebirth=True, **kw):
        last = None
        for _ in range(n):
            last = self.step(act(**kw) if kw else act(), label)
            if stop_on_rebirth and last["is_on_rebirth"]:
                break
        return last

    def settle(self, maxn=150):
        f = None
        for _ in range(maxn):
            f = self.step(act(), "settle", keep=False)
            if f["on_ground"] and not f["is_on_rebirth"] and f["is_actionable"]:
                return f
        return f


def runs(rows):
    """Run-length encode the ids: [(id, first, last, label, last_row)]."""
    out = []
    for i, f in enumerate(rows):
        if out and out[-1][0] == f["state"] and out[-1][3] == f["label"]:
            out[-1][2] = i
            out[-1][4] = f
        else:
            out.append([f["state"], i, i, f["label"], f])
    return out


# ---------------------------------------------------------------------------
# scenario helpers
# ---------------------------------------------------------------------------
def leave_stage(r: Rec, deep: float = -10.0):
    """Walk off the nearest ledge; returns the outward direction."""
    f = r.settle()
    d = -1.0 if f["x"] <= 0 else 1.0
    for _ in range(220):
        f = r.step(act(sx=d), "runoff", keep=False)
        if f["y"] < deep and not f["on_ground"]:
            return d, f
    return d, None


def at_hang_point(f) -> bool:
    """Position test only (no status ids), so the checks stay independent."""
    return (abs(abs(f["x"]) - sk.LEDGE_X) < 1.5
            and abs(f["y"] - sk.LEDGE_Y) < 1.5 and abs(f["vy"]) < 1.0)


def catch_ledge(r: Rec):
    """Fall off, Up-B back toward the stage and hang on the ledge.  Returns the
    stage-ward direction, or None if the hang point was not reached."""
    d, f = leave_stage(r)
    if f is None:
        return None
    toward = -d
    for _ in range(30):                      # fall a little first
        f = r.step(act(), "fall", keep=False)
        if f["y"] <= -35:
            break
    for _ in range(4):
        r.step(act(["B"], sx=toward * 0.3, sy=1.0), "upb", keep=False)
    still = 0
    for _ in range(120):
        f = r.step(act(sx=toward), "recover_hold", keep=False)
        still = still + 1 if at_hang_point(f) else 0
        if still >= 3:
            return toward
        if f["is_on_rebirth"]:
            return None
    return None


Scenario = Callable[[Rec], None]


def ground(label, buttons=(), sx=0.0, sy=0.0, press=2, after=30, pre=None):
    def run(r: Rec):
        r.reset()
        r.settle()
        r.run("idle", 6)
        if pre:
            pre(r)
        r.run(label, press, buttons=buttons, sx=sx, sy=sy)
        r.run("after", after)
    return run


def air(label, buttons=(), sx=0.0, sy=0.0, press=2, wait=14, after=30):
    def run(r: Rec):
        r.reset()
        r.settle()
        r.run("jump", 2, buttons=("X",))
        r.run("rise", wait)
        r.run(label, press, buttons=buttons, sx=sx, sy=sy)
        r.run("after", after)
    return run


def ledge(label, buttons=(), sx=0.0, sy=0.0, press=2, after=40, rel=False):
    def run(r: Rec):
        r.reset()
        toward = catch_ledge(r)
        if toward is None:
            raise RuntimeError("could not reach the ledge hang")
        r.rows = []
        r.run("hang", 40)
        r.run(label, press, buttons=buttons,
              sx=sx * (toward if rel else 1.0), sy=sy)
        r.run("after", after)
    return run


def upb(label, jump_first, steer_out, deep):
    def run(r: Rec):
        r.reset()
        d, f = leave_stage(r)
        if f is None:
            raise RuntimeError("did not leave the stage")
        toward = -d
        steer = d if steer_out else toward
        if deep:
            for _ in range(120):
                f = r.step(act(), "fall", keep=False)
                if f["y"] <= -90 or f["is_on_rebirth"]:
                    break
        if jump_first:
            r.run("jump", 1, buttons=("X",), sx=steer)
            r.run("rise", 12, sx=steer)
        r.run(label, 4, buttons=("B",), sx=steer * 0.3, sy=1.0)
        r.run("after", 140, sx=steer)
    return run


def ko(r: Rec):
    r.reset()
    leave_stage(r)
    for _ in range(40):
        r.step(act(), "fall")
    for _ in range(400):
        f = r.step(act(), "ko")
        if f["is_on_rebirth"]:
            break
    r.run("platform", 40, stop_on_rebirth=False)


def build() -> dict[str, Scenario]:
    s: dict[str, Scenario] = {}
    s["wait"] = ground("wait", press=1, after=10)
    s["walk"] = ground("walk", sx=0.35, press=20, after=10)
    s["dash"] = ground("dash", sx=1.0, press=14, after=20)
    s["run_then_stop"] = ground("run", sx=1.0, press=40, after=25)
    s["turn"] = ground("turnaround", sx=-1.0, press=6, after=10,
                       pre=lambda r: r.run("go", 20, sx=1.0))
    s["jump"] = ground("jump", buttons=("X",), press=2, after=50)
    s["jump_fwd"] = ground("jump_fwd", buttons=("X",), sx=1.0, press=2,
                           after=50)
    s["shield"] = ground("shield", buttons=("ZL",), press=25, after=15)
    s["grab"] = ground("grab", buttons=("ZR",), press=2, after=40)
    s["jab"] = ground("jab", buttons=("A",), press=2, after=30)
    s["ftilt"] = ground("ftilt", buttons=("A",), sx=0.5, press=2, after=30)
    s["utilt"] = ground("utilt", buttons=("A",), sy=0.6, press=2, after=30)
    s["dtilt"] = ground("dtilt", buttons=("A",), sy=-0.6, press=2, after=30)
    s["fsmash"] = ground("fsmash", buttons=("A",), sx=1.0, press=2, after=60)
    s["usmash"] = ground("usmash", buttons=("A",), sy=1.0, press=2, after=60)
    s["dsmash"] = ground("dsmash", buttons=("A",), sy=-1.0, press=2, after=60)
    s["spec_n"] = ground("spec_n", buttons=("B",), press=2, after=60)
    s["spec_s"] = ground("spec_s", buttons=("B",), sx=1.0, press=2, after=60)
    s["spec_hi"] = ground("spec_hi", buttons=("B",), sy=1.0, press=2, after=60)
    s["spec_lw"] = ground("spec_lw", buttons=("B",), sy=-1.0, press=2, after=60)
    s["nair"] = air("nair", buttons=("A",))
    s["fair_r"] = air("aerial_right", buttons=("A",), sx=1.0)
    s["bair_l"] = air("aerial_left", buttons=("A",), sx=-1.0)
    s["uair"] = air("uair", buttons=("A",), sy=1.0)
    s["dair"] = air("dair", buttons=("A",), sy=-1.0)
    s["air_spec_n"] = air("air_spec_n", buttons=("B",))
    s["air_spec_s"] = air("air_spec_s", buttons=("B",), sx=1.0)
    s["air_spec_hi"] = air("air_spec_hi", buttons=("B",), sy=1.0)
    s["air_spec_lw"] = air("air_spec_lw", buttons=("B",), sy=-1.0)
    s["airdodge"] = air("airdodge", buttons=("ZL",))
    s["midair_jump"] = air("midair_jump", buttons=("X",))
    s["fall_free"] = air("fall", press=1, wait=2, after=90)
    s["ledge_wait"] = ledge("wait", press=1, after=60)
    s["ledge_getup"] = ledge("getup", sy=1.0, press=3)
    s["ledge_roll"] = ledge("roll_in", buttons=("ZL",), press=3)
    s["ledge_attack"] = ledge("attack", buttons=("A",), press=3)
    s["ledge_jump"] = ledge("jump", buttons=("X",), press=3)
    s["ledge_drop"] = ledge("drop", sy=-1.0, press=3)
    s["upb_jump_toward"] = upb("upb", True, False, False)
    s["upb_only_toward"] = upb("upb", False, False, False)
    s["upb_helpless"] = upb("upb", True, True, True)
    s["ko_sequence"] = ko
    return s


# ---------------------------------------------------------------------------
# checks: discover an id from behaviour, assert the code's constant
# ---------------------------------------------------------------------------
def find(rows, pred, min_len=1):
    """Ids whose consecutive run (same id) satisfies pred, at least min_len."""
    found = collections.Counter()
    for sid, a, b, _lab, last in runs(rows):
        seg = rows[a:b + 1]
        if len(seg) >= min_len and all(pred(f) for f in seg):
            found[sid] += len(seg)
    return found


def runs_by_id(rows):
    """Run-length encode by id only (labels ignored): [(id, first, last)]."""
    out = []
    for i, f in enumerate(rows):
        if out and out[-1][0] == f["state"]:
            out[-1][2] = i
        else:
            out.append([f["state"], i, i])
    return [tuple(x) for x in out]


def seq_ids(rows, labels=None):
    return [r_[0] for r_ in runs([f for f in rows
                                  if labels is None or f["label"] in labels])]


def checks(res: dict[str, list[dict]], recordings: str | None = None):
    out = []

    def check(name, cond, detail):
        out.append({"name": name, "ok": bool(cond), "detail": detail})

    # --- Up-B: frames that rise, airborne and not actionable, after B+up
    for key in ("upb_jump_toward", "upb_only_toward"):
        rows = res.get(key)
        if rows:
            ids = collections.Counter(
                f["state"] for f in rows
                if f["label"] in ("upb", "after") and f["vy"] > 5
                and not f["is_actionable"] and not f["on_ground"])
            # the rising frames may also be a ledge catch; require 478 to
            # dominate and nothing else rising + non-actionable but 478
            check(f"{key}: Up-B id == SPECIAL_HI",
                  ids and ids.most_common(1)[0][0] == sk.SPECIAL_HI, dict(ids))
    # --- helpless: the id that follows Up-B while falling and not actionable
    found = collections.Counter()
    for key in ("upb_jump_toward", "upb_only_toward", "upb_helpless"):
        rows = res.get(key) or []
        rs = runs_by_id(rows)
        for i, (sid, a, b) in enumerate(rs[:-1]):
            if sid != sk.SPECIAL_HI:
                continue
            nid, na, nb = rs[i + 1]
            seg = rows[na:nb + 1]
            if (len(seg) >= 2 and all(not f["is_actionable"]
                                      and not f["on_ground"] and f["vy"] < -20
                                      for f in seg)):
                found[nid] += 1
    check("helpless fall id == HELPLESS_FALL",
          found and set(found) == sk.HELPLESS_KINDS, dict(found))
    # --- ledge: the hang point is held by exactly LEDGE_HANG
    rows = res.get("ledge_wait")
    if rows:
        hang = find([f for f in rows if f["label"] == "hang"], at_hang_point,
                    min_len=10)
        check("ledge hang id == LEDGE_HANG", hang and set(hang) == {sk.LEDGE_HANG},
              dict(hang))
        near = find([f for f in rows if f["label"] == "hang"],
                    lambda f: abs(abs(f["x"]) - sk.LEDGE_X) < 6.0 and f["y"] < -15)
        check("ids near the ledge are LEDGE_KINDS", near and set(near) <= sk.LEDGE_KINDS,
              dict(near))
    for key, want in (("ledge_getup", "exit"), ("ledge_roll", "exit"),
                      ("ledge_attack", "exit"), ("ledge_jump", "exit"),
                      ("ledge_drop", "fall")):
        rows = res.get(key)
        if not rows:
            continue
        label = key.split("_", 1)[1]
        seq = seq_ids(rows, {label, "after"})
        if want == "exit":
            exits = [i for i in seq if i in sk.LEDGE_KINDS
                     and i not in (sk.LEDGE_CATCH_MOVE, sk.LEDGE_CATCH,
                                   sk.LEDGE_HANG)]
            check(f"{key}: leaves through a ledge-exit id (120-123)", exits, seq[:8])
        else:
            check(f"{key}: falls (state 14) after the drop", 14 in seq, seq[:8])
    # --- KO: dead entry, KO flight, then the platform
    rows = res.get("ko_sequence")
    if rows:
        seq = seq_ids(rows)
        # 470 (KO flight) always appears; 181 is a one-decision entry that is
        # present only for some blast zones and is always followed by 470.
        after_entry = [seq[i + 1] for i, v in enumerate(seq[:-1])
                       if v == sk.DEAD_ENTRY]
        okseq = sk.KO in seq and all(v == sk.KO for v in after_entry)
        check("KO: 470 appears; every 181 is followed by 470", okseq, seq[-8:])
        plat = find([f for f in rows if f["label"] == "platform"],
                    lambda f: f["is_on_rebirth"] > 0, min_len=3)
        check("rebirth platform id == REBIRTH_PLATFORM",
              plat and set(plat) <= sk.REBIRTH_KINDS, dict(plat))
    # --- shield: ZL raises (27), holds (28), releases (29); field is derived
    rows = res.get("shield")
    if rows:
        held = seq_ids(rows, {"shield"})
        check("shield ids are 27 then 28", held[:2] == [sk.GUARD_ON, sk.GUARD], held)
        check("release enters 29", sk.GUARD_OFF in seq_ids(rows), seq_ids(rows))
    # --- jump-squat: pressing X on the ground enters JUMP_SQUAT then JUMP
    rows = res.get("jump")
    if rows:
        seq = seq_ids(rows, {"jump", "after"})
        pairs = [seq[i + 1] for i, v in enumerate(seq[:-1])
                 if v == sk.JUMP_SQUAT]
        check("jump press: JUMP_SQUAT (10) always followed by JUMP (11)",
              len(pairs) >= 2 and all(v == 11 for v in pairs), seq[:10])
    # --- every aerial attack is ATTACK_AIR
    for key, label in (("nair", "nair"), ("fair_r", "aerial_right"),
                       ("bair_l", "aerial_left"), ("uair", "uair"),
                       ("dair", "dair")):
        rows = res.get(key)
        if rows:
            seq = seq_ids(rows, {label})
            check(f"{key}: aerial attack id == ATTACK_AIR (54)",
                  seq and set(seq) == {sk.ATTACK_AIR}, seq)
    # --- specials ground and air
    for key in ("spec_n", "spec_s", "spec_hi", "spec_lw", "air_spec_n",
                "air_spec_s", "air_spec_hi", "air_spec_lw"):
        rows = res.get(key)
        if rows:
            lab = seq_ids(rows, {key})
            check(f"{key}: special id in 476-479", bool(set(lab) & sk.SPECIAL_KINDS),
                  lab[:6])
    # --- damage ids (offline, from the recordings)
    if recordings:
        frac, top = damage_ids_from_recordings(recordings)
        check("states entered after taking damage are DAMAGE_KINDS (>=90 %)",
              frac >= 0.90, {"fraction": round(frac, 3), "top": top})
    return out


def damage_ids_from_recordings(path: str, limit: int = 120):
    """State on the decision after the ego's damage rises (any fighter)."""
    import glob
    first: collections.Counter = collections.Counter()
    for p in sorted(glob.glob(os.path.join(path, "episode_*.npz")))[:limit]:
        rf = np.load(p, allow_pickle=False)["raw_fighters"]
        for k in (0, 1):
            st = rf[:, k, 7].astype(int)
            dm = rf[:, k, 3]
            up = np.where((dm[1:] > dm[:-1] + 0.5) & (dm[:-1] < 500))[0] + 1
            for i in up:
                first[int(st[min(i + 1, len(st) - 1)])] += 1
    total = sum(first.values()) or 1
    inside = sum(c for s_, c in first.items() if s_ in sk.DAMAGE_KINDS)
    return inside / total, first.most_common(6)


def table(res):
    """id -> {scenarios, mean signals} over everything observed."""
    acc: dict[int, dict] = {}
    for key, rows in res.items():
        for f in rows:
            a = acc.setdefault(f["state"], {"n": 0, "scenarios": set(),
                                            "ground": 0.0, "actionable": 0.0,
                                            "vy": 0.0})
            a["n"] += 1
            a["scenarios"].add(f"{key}/{f['label']}")
            a["ground"] += f["on_ground"]
            a["actionable"] += f["is_actionable"]
            a["vy"] += f["vy"]
    out = {}
    for sid, a in sorted(acc.items()):
        n = a["n"]
        out[str(sid)] = {"frames": n,
                         "ground": round(a["ground"] / n, 2),
                         "actionable": round(a["actionable"] / n, 2),
                         "mean_vy": round(a["vy"] / n, 1),
                         "scenarios": sorted(a["scenarios"])[:12]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma list of scenario names")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--game-log", default="~/smash.log")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rows", default=None,
                    help="write the raw recorded decisions here; with "
                         "--from-rows, re-evaluate the checks offline")
    ap.add_argument("--from-rows", action="store_true")
    ap.add_argument("--recordings", default=None,
                    help="expert recordings dir for the damage-id check")
    args = ap.parse_args()
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    scenarios = build()
    if args.from_rows:
        with open(args.rows, encoding="utf-8") as fh:
            res = json.load(fh)
        return report(res, args, log)
    if args.only:
        scenarios = {k: v for k, v in scenarios.items()
                     if k in args.only.split(",")}
    pad = VirtualPad().create()
    res: dict[str, list[dict]] = {}
    try:
        env = SSBUGymEnv(frame_skip=2, max_steps=100000, reward_mode="none",
                         stock_termination=False, require_positive_stock=False,
                         input_adapter=pad,
                         reset_fn=ResetClient(timeout=20.0).reset)
        ge.wait_for_pad(args.game_log, 15.0, log=log)
        if not ge.drive_menus(env, pad, 120.0, log=log):
            log("no fight")
            return 2
        rec = Rec(env)
        for name, fn in scenarios.items():
            for t in range(args.trials):
                try:
                    fn(rec)
                except Exception as exc:   # noqa: BLE001
                    log(f"[{name}#{t + 1}] FAILED: {exc}")
                    continue
                res.setdefault(name, []).extend(rec.rows)
                seq = [(sid, a, b, lab) for sid, a, b, lab, _ in
                       runs(rec.rows)]
                comp = " ".join(f"{sid}[{lab}]x{b - a + 1}"
                                for sid, a, b, lab in seq[:14])
                log(f"[{name}#{t + 1}] {comp}")
        env.close()
    finally:
        pad.destroy()

    if args.rows:
        with open(args.rows, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
    return report(res, args, log)


def report(res, args, log):
    log("\n=== checks")
    results = checks(res, args.recordings)
    failed = 0
    for c in results:
        failed += 0 if c["ok"] else 1
        log(f"  {'PASS' if c['ok'] else 'FAIL'}  {c['name']}: {c['detail']}")
    tab = table(res)
    log(f"\nobserved {len(tab)} distinct ids across {len(res)} scenarios; "
        f"{failed} check(s) failed")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"build": "13.0.4", "ledge": [sk.LEDGE_X, sk.LEDGE_Y],
                       "checks": results, "ids": tab}, fh, indent=1,
                      default=str)
        log(f"wrote {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
