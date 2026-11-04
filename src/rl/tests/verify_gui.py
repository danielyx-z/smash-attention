#!/usr/bin/env python3
"""Second-window verifier for the SSBU state bridge.

Reads the same shared-memory block as reader.py and shows every v7 field for
P1 (slot 0) and P2 (slot 1) so a human can check it against the game.

    python3 tools/mem/reader.py      # in one terminal
    python3 tools/mem/verify_gui.py  # in another

Keys
    H   toggle the raw little-endian hex bytes of every row
    P   pause / resume refresh (for screenshots)
    Q   quit

Colours
    green   value is plausible and fresh
    yellow  value is plausible but the frame is older than 100 ms
    red     value is implausible, unreadable, or the frame is older than 500 ms

PyQt6 / PySide6 are used when available; this build of Python has neither, so
the Tkinter implementation below is the one that runs.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from rl.memory import schema

def action_state_name(state: int) -> str:
    return getattr(schema, "ACTION_STATE_NAMES", {}).get(state, f"UNKNOWN({state})")

STALE_NS = 100_000_000      # 100 ms
DEAD_NS = 500_000_000       # 500 ms

GREEN = "#1b7f3a"
YELLOW = "#b8860b"
RED = "#b00020"
DIM = "#666666"

#: fields that only exist once there is something to compare against; a missing
#: value here is "not applicable" rather than a broken read
DERIVED = {"vx", "vy", "dx", "dy", "facing_rel"}

#: plausible range per field, used for the colour coding
PLAUSIBLE = {
    "entry_id":      lambda v: 0 <= v <= 7,
    "kind":          lambda v: 0 <= v < 200,
    "stock":         lambda v: -1 <= v <= 99,
    "damage":        lambda v: 0.0 <= v <= 999.0,
    "x":             lambda v: -2000.0 <= v <= 2000.0,
    "y":             lambda v: -2000.0 <= v <= 2000.0,
    "vx":            lambda v: -500.0 <= v <= 500.0,
    "vy":            lambda v: -500.0 <= v <= 500.0,
    "facing":        lambda v: v in (-1.0, 1.0) or (v != 0 and abs(v) <= 1.0),
    "action_state":  lambda v: 0 <= v <= 2000,
    "action_frame":  lambda v: -1.0 <= v <= 100000.0,
    "on_ground":     lambda v: v in (0, 1),
    "is_actionable": lambda v: v in (0, 1),
    "jumps_left":    lambda v: 0 <= v <= 9,
    "dist_to_blast": lambda v: 0.0 <= v <= 400.0,
    "dx":            lambda v: -2000.0 <= v <= 2000.0,
    "dy":            lambda v: -2000.0 <= v <= 2000.0,
    "facing_rel":    lambda v: v in (-1, 0, 1),
    "is_cpu":        lambda v: v in (0, 1),
    "is_on_rebirth": lambda v: v in (0, 1),
    "shield":        lambda v: v in (0, 1),
    "shield_size":   lambda v: 0.0 <= v <= 100.0,
    "is_enable_cancel": lambda v: v in (0, 1),
    "hitstun":       lambda v: 0 <= v <= 999,
    "buttons":       lambda v: 0 <= v <= 0xFFFFFFFF,
    "stick_x":       lambda v: -1.0 <= v <= 1.0,
    "stick_y":       lambda v: -1.0 <= v <= 1.0,
    "trigger":       lambda v: 0.0 <= v <= 1.0,
    "cstick_x":      lambda v: -1.0 <= v <= 1.0,
    "cstick_y":      lambda v: -1.0 <= v <= 1.0,
    "raw_flags":     lambda v: 0 <= v <= 0xFFFFFFFF,
    "pad":           lambda v: True,
}


def read_frame(buf) -> dict | None:
    """Torn-read-safe snapshot of the shared block."""
    seq1 = struct.unpack_from("<I", buf, schema.SEQ_OFFSET)[0]
    if seq1 & 1:
        return None
    header = schema.unpack_header(buf)
    rows = [schema.unpack_fighter(buf, i) for i in range(schema.MAX_FIGHTERS)]
    seq2 = struct.unpack_from("<I", buf, schema.SEQ_OFFSET)[0]
    if seq1 != seq2:
        return None
    header["rows"] = rows
    return header


def fmt_value(name: str, value) -> str:
    if isinstance(value, float):
        if name in ("x", "y", "vx", "vy", "dx", "dy"):
            return f"{value:8.2f}"
        if name == "damage":
            return f"{value:6.2f}"
        if name == "action_frame":
            return f"{value:7.2f}"
        if name == "dist_to_blast":
            return f"{value:6.2f}"
        if name in ("stick_x", "stick_y", "cstick_x", "cstick_y", "trigger"):
            return f"{value:+.2f}"
        return f"{value:g}"
    if name == "buttons":
        names = schema.buttons_pressed(int(value))
        return f"0x{int(value):x} " + (",".join(names) if names else "-")
    if name == "raw_flags":
        return f"0x{int(value):08x}"
    return str(value)


def hex_bytes(row: dict) -> str:
    body = schema.pack_fighter(row, row.get("raw_flags", 0))
    return " ".join(f"{b:02x}" for b in body)


class App:
    def __init__(self, root, shm_path: str, topmost: bool = True):
        import tkinter as tk

        self.tk = tk
        self.root = root
        self.paused = False
        self.show_hex = False
        self.buf = None
        self.shm_path = shm_path
        self.last_frame = None
        self.missing_since = None

        root.title("SSBU state verifier")
        root.configure(bg="white")
        # This is a HUD, not a focus target.  The game's keyboard input only
        # counts while its own window is focused, so this window never calls
        # focus_set/focus_force and the launcher re-focuses the game after it
        # appears.  H/P/Q still work when the user clicks this window.
        root.attributes('-topmost', topmost)

        top = tk.Frame(root, bg="white")
        top.pack(fill="x", padx=8, pady=(8, 2))

        line1 = tk.Frame(top, bg="white")
        line1.pack(fill="x")
        self.status = tk.Label(line1, text="WAITING", font=("TkDefaultFont", 14, "bold"),
                               bg="white", fg=DIM, anchor="w")
        self.status.pack(side="left")
        self.detail = tk.Label(line1, text="", font=("TkFixedFont", 10), bg="white",
                               fg=DIM, anchor="w", justify="left")
        self.detail.pack(side="left", padx=12)

        self.detail2 = tk.Label(top, text="", font=("TkFixedFont", 10), bg="white",
                                fg=DIM, anchor="w", justify="left")
        self.detail2.pack(fill="x")

        self.hint = tk.Label(root, text="H: raw hex   P: pause   Q: quit",
                             font=("TkFixedFont", 9), bg="white", fg=DIM, anchor="w")
        self.hint.pack(fill="x", padx=8)

        table = tk.Frame(root, bg="white")
        table.pack(fill="both", expand=True, padx=8, pady=6)

        headers = ["field", "P1 (slot 0)", "P2 (slot 1)"]
        if self.show_hex:
            headers = ["field", "P1 (slot 0)", "P2 (slot 1)", "raw (P1)"]
        self.header_labels = []
        for c, text in enumerate(headers):
            lab = tk.Label(table, text=text, font=("TkDefaultFont", 10, "bold"),
                           bg="white", fg="#222", anchor="w")
            lab.grid(row=0, column=c, sticky="ew", padx=(0, 10), pady=(0, 4))
            self.header_labels.append(lab)

        self.cells: list[list] = []
        for r, (name, dtype, _desc) in enumerate(schema.FIELDS, start=1):
            name_lab = tk.Label(table, text=name, font=("TkFixedFont", 10),
                                bg="white", fg="#222", anchor="w")
            name_lab.grid(row=r, column=0, sticky="ew", padx=(0, 10))
            row_cells = []
            for c in (1, 2):
                lab = tk.Label(table, text="-", font=("TkFixedFont", 10, "bold"),
                               bg="white", fg=DIM, anchor="w", width=19)
                lab.grid(row=r, column=c, sticky="ew", padx=(0, 10))
                row_cells.append(lab)
            self.cells.append(row_cells)
        self.hex_labels = []

        for c in range(4):
            table.grid_columnconfigure(c, weight=1 if c else 0)

        root.bind("<KeyPress-h>", lambda _e: self.toggle_hex())
        root.bind("<KeyPress-H>", lambda _e: self.toggle_hex())
        root.bind("<KeyPress-p>", lambda _e: self.toggle_pause())
        root.bind("<KeyPress-P>", lambda _e: self.toggle_pause())
        root.bind("<KeyPress-q>", lambda _e: root.destroy())
        root.bind("<KeyPress-Q>", lambda _e: root.destroy())

        self.tick()

    # -- actions ----------------------------------------------------------
    def toggle_hex(self):
        self.show_hex = not self.show_hex
        self.rebuild_hex_column()

    def rebuild_hex_column(self):
        tk = self.tk
        if self.show_hex:
            lab = tk.Label(self.cells[0][0].master, text="raw",
                           font=("TkDefaultFont", 10, "bold"), bg="white", fg="#222", anchor="w")
            lab.grid(row=0, column=3, sticky="ew", padx=(0, 10), pady=(0, 4))
            self.hex_labels = []
            for r in range(len(self.cells)):
                hl = tk.Label(self.cells[0][0].master, text="", font=("TkFixedFont", 9),
                              bg="white", fg=DIM, anchor="w")
                hl.grid(row=r + 1, column=3, sticky="ew", padx=(0, 10))
                self.hex_labels.append(hl)
        else:
            for hl in self.hex_labels:
                hl.destroy()
            self.hex_labels = []

    def toggle_pause(self):
        self.paused = not self.paused

    # -- data -------------------------------------------------------------
    def ensure_buf(self) -> bool:
        if self.buf is not None:
            return True
        try:
            self.buf = schema.open_reader(self.shm_path)
            self.missing_since = None
            return True
        except (FileNotFoundError, ValueError, OSError):
            if self.missing_since is None:
                self.missing_since = time.monotonic()
            return False

    def tick(self):
        if not self.paused:
            self.refresh()
        self.root.after(33, self.tick)      # 30 fps

    def refresh(self):
        if not self.ensure_buf():
            self.status.config(text="WAITING", fg=DIM)
            self.detail.config(text=f"no shared memory at {self.shm_path} - is reader.py running?")
            return

        frame = read_frame(self.buf)
        if frame is None:
            return                          # raced; keep the previous values

        self.last_frame = frame
        now = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        age = now - frame["ts_ns"]

        if frame["magic"] != schema.MAGIC or frame["version"] != schema.VERSION:
            self.status.config(text="BAD LAYOUT", fg=RED)
            self.detail.config(text=f"magic={frame['magic']!r} version={frame['version']} "
                                    f"(expected {schema.MAGIC!r} / {schema.VERSION})")
            return

        ms_name = {0: "MENU", 1: "IN_MATCH", 2: "RESULTS"}.get(frame["match_state"], str(frame["match_state"]))

        if age > DEAD_NS:
            self.status.config(text="DEAD", fg=RED)
        elif age > STALE_NS:
            self.status.config(text="STALE", fg=YELLOW)
        else:
            self.status.config(text="LIVE", fg=GREEN)

        state = "running" if not self.paused else "PAUSED"
        b = (frame.get("blast_left", 0.0), frame.get("blast_right", 0.0),
             frame.get("blast_top", 0.0), frame.get("blast_bottom", 0.0))
        blast = (f"   blast [{b[0]:.0f},{b[1]:.0f}]x[{b[3]:.0f},{b[2]:.0f}]"
                 if any(b) else "   blast ?")
        self.detail.config(
            text=f"{frame['fps']:5.1f} fps   age {age / 1e6:6.1f} ms   {state}"
        )
        flags = int(frame.get("flags", 0) or 0)
        stage = frame.get("stage_id", 0)
        stage_txt = f"stage {stage}" if flags & schema.HEADER_FLAG_STAGE_ID else "stage ?"
        ctrl_txt = "ctrl live" if flags & schema.HEADER_FLAG_CONTROL_LAYOUT else "ctrl ?"
        self.detail2.config(
            text=f"match {ms_name}  fight {'on' if frame['fight_on'] else 'off'}  "
                 f"fighters {frame['n_fighters']}  {stage_txt}  {ctrl_txt}  "
                 f"frame {frame.get('game_frame', 0)}{blast}"
        )

        rows = frame["rows"]
        for col in (0, 1):
            row = rows[col] if col < len(rows) else None
            if row is not None and row.get("raw_flags", 0) == 0:
                row = None                   # empty slot (player absent)
            self.update_column(col, row, age)

        if self.show_hex and rows:
            body = hex_bytes(rows[0]).split(" ")
            for i, hl in enumerate(self.hex_labels):
                chunk = " ".join(body[i * 4:(i + 1) * 4])
                hl.config(text=chunk)

    def update_column(self, col: int, row: dict | None, age: int):
        for i, (name, _dtype, _desc) in enumerate(schema.FIELDS):
            lab = self.cells[i][col]
            if row is None:
                lab.config(text="-", fg=DIM)
                continue
            if name == "pad":
                lab.config(text="-", fg=DIM)
                continue
            value = row.get(name)
            if value is None or not (row.get("raw_flags", 0) >> i) & 1:
                if name in DERIVED:
                    lab.config(text="n/a", fg=DIM)
                else:
                    lab.config(text="unverified", fg=RED)
                continue
            ok = PLAUSIBLE.get(name, lambda _v: True)(value)
            if not ok:
                colour = RED
            elif age > DEAD_NS:
                colour = RED
            elif age > STALE_NS:
                colour = YELLOW
            else:
                colour = GREEN
            
            # Special formatting for action_state: show name + number
            if name == "action_state" and value is not None:
                state_name = action_state_name(int(value))
                display = f"{value} ({state_name})"
            else:
                display = fmt_value(name, value)
            lab.config(text=display, fg=colour)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shm", default=schema.SHM_PATH)
    ap.add_argument("--no-topmost", action="store_true",
                    help="do not keep the window above the game")
    ap.add_argument("--geometry", default="+1500+40",
                    help="X11 geometry suffix, e.g. +1500+40")
    args = ap.parse_args()

    try:
        import tkinter as tk
    except ImportError:
        print("error: Tkinter is not available and neither PyQt6 nor PySide6 is installed",
              file=sys.stderr)
        return 2

    root = tk.Tk()
    root.geometry(f"980x1040{args.geometry}")
    App(root, args.shm, topmost=not args.no_topmost)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())