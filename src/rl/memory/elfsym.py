"""Minimal ELF64 AArch64 .dynsym reader.

The game's symbols live in the *guest* image (main.elf), not in the host
`ssbu_native` binary, so `nm ssbu_native` is useless for game state.  This
module reads the dynamic symbol table straight out of main.elf so that every
offset used by this toolset comes from the binary itself rather than being
typed in by hand.

No third-party dependencies: struct + mmap only.
"""

from __future__ import annotations

import struct

SHT_DYNSYM = 11
SHN_UNDEF = 0


class ElfSymbols:
    """name -> value (offset in the guest image, which is mapped at its base)."""

    def __init__(self, path: str):
        self.path = path
        self.symbols: dict[str, int] = {}
        self.sizes: dict[str, int] = {}
        self._load()

    def _load(self) -> None:
        with open(self.path, "rb") as fh:
            data = fh.read()
        if data[:4] != b"\x7fELF":
            raise ValueError(f"{self.path}: not an ELF file")
        if data[4] != 2:
            raise ValueError(f"{self.path}: not ELF64")
        (e_shoff,) = struct.unpack_from("<Q", data, 0x28)
        (e_shentsize,) = struct.unpack_from("<H", data, 0x3A)
        (e_shnum,) = struct.unpack_from("<H", data, 0x3C)

        sections = []
        for i in range(e_shnum):
            off = e_shoff + i * e_shentsize
            sh = struct.unpack_from("<IIQQQQIIQQ", data, off)
            sections.append(sh)  # name, type, flags, addr, offset, size, link, info, align, entsize

        for sh in sections:
            sh_type, sh_offset, sh_size, sh_link, sh_entsize = sh[1], sh[4], sh[5], sh[6], sh[9]
            if sh_type != SHT_DYNSYM or sh_entsize == 0:
                continue
            strtab = sections[sh_link]
            str_off, str_size = strtab[4], strtab[5]
            strings = data[str_off:str_off + str_size]
            for k in range(sh_size // sh_entsize):
                base = sh_offset + k * sh_entsize
                st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
                    "<IBBHQQ", data, base
                )
                if st_shndx == SHN_UNDEF or st_name == 0:
                    continue
                end = strings.find(b"\0", st_name)
                name = strings[st_name:end].decode("ascii", "replace")
                self.symbols[name] = st_value
                self.sizes[name] = st_size

    # -- lookup -----------------------------------------------------------
    def offset(self, name: str) -> int:
        if name not in self.symbols:
            raise KeyError(f"symbol not found in {self.path}: {name}")
        return self.symbols[name]

    def find(self, *needles: str, limit: int = 200) -> list[str]:
        """All symbol names containing every needle (case-insensitive)."""
        needles = tuple(n.lower() for n in needles)
        out = []
        for name in self.symbols:
            low = name.lower()
            if all(n in low for n in needles):
                out.append(name)
                if len(out) >= limit:
                    break
        return sorted(out)


if __name__ == "__main__":
    import sys

    syms = ElfSymbols(sys.argv[1])
    print(f"{len(syms.symbols)} defined symbols in {sys.argv[1]}")
    for name in sys.argv[2:]:
        try:
            print(f"  {name} = 0x{syms.offset(name):x}")
        except KeyError as exc:
            print(f"  {exc}")
