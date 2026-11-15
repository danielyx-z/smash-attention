#!/usr/bin/env python3
"""Match set-up from code: characters, player types and rules (no menus).

The recomp reads ``SSBU_DEBUG_MATCH`` before the hooks are installed and
forces the select-screen panels at every match set-up (``runtime/debug_mode.c``,
``match_player_hook``).  The character is written by name as
``ui_chara_<name>``, so any fighter name the game knows works; the Windows-only
numeric kind table in the recomp is not needed for the spawn.

Two ways to use it from a program:

* one match, decided at launch::

      env SSBU_DEBUG_MATCH = MatchSpec(p1="samus", p2="kirby").to_string()

* one match per set-up (a rematch loop), no relaunch::

      write_match_list("/tmp/matches.txt", [spec1, spec2, ...])
      env SSBU_DEBUG_MATCH_LIST = "/tmp/matches.txt"

  The runtime consumes one line at each new match set-up and keeps the last,
  so ``append_match()`` can add later set-ups while the game runs.

The characters are forced even if the select screen picks something else; the
match itself still has to be started once (menu walk or by hand).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

TYPES = ("human", "cpu")
ITEMS = ("off", "on")
FORMS = ("normal", "battlefield", "omega")


def _clean_type(value: str, what: str) -> str:
    value = str(value).strip().lower()
    if value not in TYPES:
        raise ValueError(f"{what} must be one of {TYPES}, got {value!r}")
    return value


@dataclass
class MatchSpec:
    """One ``SSBU_DEBUG_MATCH`` set-up.

    ``p1``/``p2`` are fighter names (``mario``, ``samus``, ``kirby``, ...) or
    ``None`` for "whatever the select screen leaves there".  The type is
    independent: a named fighter can be CPU or human.
    """

    p1: str | None = None
    p2: str | None = None
    p1_type: str = "human"
    p2_type: str = "cpu"
    cpu: int = 9
    stage: str = "end"
    form: str | None = None
    players: int = 2
    stock: int | None = 3
    time: int | None = 5          # minutes; the recomp defaults to 5 too
    items: str = "off"
    hazards: bool = False

    def __post_init__(self) -> None:
        self.p1_type = _clean_type(self.p1_type, "p1_type")
        self.p2_type = _clean_type(self.p2_type, "p2_type")
        self.items = str(self.items).strip().lower()
        if self.items not in ITEMS:
            raise ValueError(f"items must be one of {ITEMS}, got {self.items!r}")
        if not 1 <= int(self.cpu) <= 9:
            raise ValueError(f"cpu level must be 1..9, got {self.cpu!r}")
        if not 1 <= int(self.players) <= 8:
            raise ValueError(f"players must be 1..8, got {self.players!r}")
        if self.stock is not None and int(self.stock) < 1:
            raise ValueError(f"stock must be >= 1, got {self.stock!r}")
        if self.time is not None and int(self.time) < 1:
            raise ValueError(f"time must be >= 1 minute, got {self.time!r}")
        if self.form is not None:
            form = str(self.form).strip().lower()
            if form not in FORMS:
                raise ValueError(f"form must be one of {FORMS}, got {form!r}")
            self.form = form
        for name in (self.p1, self.p2):
            if name is not None and (not str(name) or "," in str(name)):
                raise ValueError(f"bad fighter name {name!r}")

    def to_string(self) -> str:
        """The ``SSBU_DEBUG_MATCH`` string (parser-compatible key order)."""
        parts: list[str] = []
        if self.p1:
            parts.append(f"p1={self.p1}")
        parts.append(f"p1={self.p1_type}")
        if self.p2:
            parts.append(f"p2={self.p2}")
        parts.append(f"p2={self.p2_type}")
        parts.append(f"cpu={int(self.cpu)}")
        if self.stage:
            parts.append(f"stage={self.stage}")
        if self.form:
            parts.append(f"form={self.form}")
        parts.append(f"players={int(self.players)}")
        if self.stock is not None:
            parts.append(f"stock={int(self.stock)}")
        if self.time is not None:
            parts.append(f"time={int(self.time)}")
        parts.append(f"items={self.items}")
        parts.append(f"hazards={'on' if self.hazards else 'off'}")
        return ",".join(parts)

    __str__ = to_string

    @classmethod
    def from_string(cls, text: str) -> "MatchSpec":
        """Parse an ``SSBU_DEBUG_MATCH`` string back into a MatchSpec."""
        values: dict[str, object] = {}
        for token in str(text).split(","):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            key, value = key.strip().lower(), value.strip()
            if key in ("p1", "p2") and value.lower() in TYPES:
                values[f"{key}_type"] = value.lower()
            elif key in ("p1", "p2"):
                values[key] = value
            elif key in ("cpu", "players", "stock", "time"):
                values[key] = int(value)
            elif key == "hazards":
                values[key] = value.lower() in ("on", "1", "true", "yes")
            elif key in ("stage", "form", "items"):
                values[key] = value.lower() if key != "stage" else value
        return cls(**values)  # type: ignore[arg-type]


def write_match_list(path: str, specs: list[MatchSpec] | tuple[MatchSpec, ...],
                     append: bool = False) -> str:
    """Write ``SSBU_DEBUG_MATCH_LIST`` (one set-up per line); return the path.

    The file must exist when the game starts.  Use ``append=True`` (or
    :func:`append_match`) to queue later set-ups while the game runs.
    """
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for spec in specs:
            fh.write(spec.to_string() + "\n")
        fh.flush()
    return os.path.abspath(path)


def append_match(path: str, spec: MatchSpec) -> str:
    """Queue one more set-up for the running game (one line per match)."""
    return write_match_list(path, [spec], append=True)


def match_env(spec: MatchSpec) -> dict[str, str]:
    """The environment for ``play.sh`` / ``game_session --env``."""
    return {"SSBU_DEBUG_MATCH": spec.to_string()}


def match_list_env(path: str) -> dict[str, str]:
    return {"SSBU_DEBUG_MATCH_LIST": os.path.abspath(path)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p1", help="player 1 fighter name")
    ap.add_argument("--p2", help="player 2 fighter name")
    ap.add_argument("--p1-type", default="human", choices=TYPES)
    ap.add_argument("--p2-type", default="cpu", choices=TYPES)
    ap.add_argument("--cpu", type=int, default=9)
    ap.add_argument("--stage", default="end")
    ap.add_argument("--form", choices=FORMS)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--stock", type=int, default=3)
    ap.add_argument("--time", type=int)
    ap.add_argument("--items", choices=ITEMS, default="off")
    ap.add_argument("--hazards", action="store_true")
    ap.add_argument("--list", metavar="FILE",
                    help="write SSBU_DEBUG_MATCH_LIST to FILE instead of "
                         "printing SSBU_DEBUG_MATCH")
    ap.add_argument("--parse", metavar="STRING",
                    help="parse an existing SSBU_DEBUG_MATCH and print it back")
    args = ap.parse_args(argv)
    if args.parse:
        print(MatchSpec.from_string(args.parse).to_string())
        return 0
    spec = MatchSpec(
        p1=args.p1, p2=args.p2, p1_type=args.p1_type, p2_type=args.p2_type,
        cpu=args.cpu, stage=args.stage, form=args.form, players=args.players,
        stock=args.stock, time=args.time, items=args.items,
        hazards=args.hazards)
    if args.list:
        print(f"SSBU_DEBUG_MATCH_LIST={write_match_list(args.list, [spec])}")
    else:
        print(spec.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
