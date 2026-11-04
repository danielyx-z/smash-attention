"""External process-memory access for the recompiled SSBU.

Everything here is read-only and goes through /proc/<pid>/mem.  No ptrace
attach, no in-process code, no modification of the game.

Two address spaces matter:

  host module base   the ELF `ssbu_native` itself (ASLR'd).  Only needed if we
                     ever want host globals; the state we read is not there.

  guest image base   the recompiled Switch binary (`main.elf`) mapped as one
                     anonymous read-only region.  All game symbols are offsets
                     into this image, so guest_base + symbol_offset is the
                     address of a game global.

The guest base is discovered from /proc/<pid>/maps, never hardcoded.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    start: int
    end: int
    perms: str
    offset: int
    path: str

    @property
    def size(self) -> int:
        return self.end - self.start


def read_maps(pid: int) -> list[Region]:
    regions = []
    with open(f"/proc/{pid}/maps", "r") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            lo, hi = parts[0].split("-")
            regions.append(
                Region(
                    start=int(lo, 16),
                    end=int(hi, 16),
                    perms=parts[1],
                    offset=int(parts[2], 16),
                    path=parts[5] if len(parts) > 5 else "",
                )
            )
    return regions


def host_module_base(pid: int, needle: str = "ssbu_native") -> int | None:
    """First mapping of the host executable."""
    for r in read_maps(pid):
        if needle in r.path:
            return r.start
    return None


def guest_image_base(pid: int) -> int:
    """The recompiled guest image: the anonymous, read-only, offset-0 region.

    The recomp build maps main.elf as a single PROT_READ, MAP_ANONYMOUS region
    at offset 0 and keeps its data/bss in the write regions that follow it.
    We pick the largest such region and sanity-check it against the symbol
    table, so a layout change shows up as a clear error instead of wrong reads.
    """
    best = None
    for r in read_maps(pid):
        if r.path:            # file-backed: not the guest image
            continue
        if r.perms != "r--p" or r.offset != 0:
            continue
        if best is None or r.size > best.size:
            best = r
    if best is None:
        raise RuntimeError(
            f"pid {pid}: no anonymous r--p offset-0 region found; "
            "is this really the recompiled game?"
        )
    return best.start


class ProcessMemory:
    """Read-only view of another process's address space."""

    def __init__(self, pid: int):
        self.pid = pid
        self.fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        self._regions = None

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        return os.path.exists(f"/proc/{self.pid}/mem")

    # -- regions ----------------------------------------------------------
    @property
    def regions(self) -> list[Region]:
        if self._regions is None:
            self._regions = read_maps(self.pid)
        return self._regions

    def refresh_maps(self) -> None:
        self._regions = None

    def is_mapped(self, addr: int, size: int = 1) -> bool:
        for r in self.regions:
            if r.start <= addr and addr + size <= r.end:
                return True
        return False

    # -- raw --------------------------------------------------------------
    def read(self, addr: int, size: int) -> bytes | None:
        """Read `size` bytes, or None if the range is not readable.

        A pointer read from the guest can be garbage while the game tears a
        match down; such a value can be larger than C long, which makes
        os.pread raise OverflowError.  Treat that exactly like an unmapped
        page so a transition never kills the reader.
        """
        if addr <= 0 or size <= 0 or addr >= 1 << 63:
            return None
        try:
            return os.pread(self.fd, size, addr)
        except (OSError, OverflowError, ValueError):
            return None

    def read_into(self, addr: int, buf: memoryview | bytearray) -> bool:
        """Zero-copy read straight into a preallocated buffer (hot path)."""
        if addr <= 0 or addr >= 1 << 63:
            return False
        try:
            n = os.preadv(self.fd, [buf], addr)
            return n == len(buf)
        except (OSError, OverflowError, ValueError):
            return False

    # -- typed ------------------------------------------------------------
    def u8(self, addr: int) -> int | None:
        d = self.read(addr, 1)
        return d[0] if d else None

    def u32(self, addr: int) -> int | None:
        d = self.read(addr, 4)
        return struct.unpack("<I", d)[0] if d and len(d) == 4 else None

    def i32(self, addr: int) -> int | None:
        d = self.read(addr, 4)
        return struct.unpack("<i", d)[0] if d and len(d) == 4 else None

    def u64(self, addr: int) -> int | None:
        d = self.read(addr, 8)
        return struct.unpack("<Q", d)[0] if d and len(d) == 8 else None

    def f32(self, addr: int) -> float | None:
        d = self.read(addr, 4)
        return struct.unpack("<f", d)[0] if d and len(d) == 4 else None

    def f64(self, addr: int) -> float | None:
        d = self.read(addr, 8)
        return struct.unpack("<d", d)[0] if d and len(d) == 8 else None

    def peek(self, addr: int, size: int = 64) -> str:
        d = self.read(addr, size)
        if d is None:
            return "<unreadable>"
        return " ".join(f"{b:02x}" for b in d)


def find_game_pid(comm: str = "ssbu_native") -> int | None:
    """PID of the running game, by /proc/<pid>/comm."""
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/comm", "r") as fh:
                if fh.read().strip() == comm:
                    return int(entry)
        except OSError:
            continue
    return None


def ptrace_scope() -> int | None:
    try:
        with open("/proc/sys/kernel/yama/ptrace_scope", "r") as fh:
            return int(fh.read().strip())
    except OSError:
        return None


def check_access(pid: int) -> tuple[bool, str]:
    """Can we read this process's memory right now?  (open succeeds or not)"""
    try:
        fd = os.open(f"/proc/{pid}/mem", os.O_RDONLY)
        os.close(fd)
        return True, "ok"
    except PermissionError:
        scope = ptrace_scope()
        hint = (
            f"permission denied (ptrace_scope={scope}). Either start the game "
            "as a child of reader.py (--spawn), or run "
            "'sudo sysctl -w kernel.yama.ptrace_scope=0'."
        )
        return False, hint
    except FileNotFoundError:
        return False, f"pid {pid} does not exist"
