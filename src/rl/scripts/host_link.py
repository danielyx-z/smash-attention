#!/usr/bin/env python3
"""Hot-rebuild only the SSBU host binary (runtime/*.c changes).

The game itself lives in ``build-linux/work/ssbu_game_eh_o1.so`` (553 MB) and
is built by ``scripts/setup.py``.  Host sources (``runtime/debug_mode.c``,
``nnhid.c``, ``display.c``, ``gfx_options.c``, ...) are compiled into the much
smaller ``build-linux/work/ssbu_native`` executable.  Changing one of those
needs only:

1. copy ``runtime/<file>.c`` to ``build-linux/work/<file>.c`` (the ``stage``
   step of setup.py does this),
2. recompile the stale host objects,
3. link ``ssbu_native`` against the existing ``ssbu_game_eh_o1.so``.

It must **never** run the game-DLL link step: ``build.py link`` first calls
``build_game_dll()``, and if ``obj_eh_o1/`` has no usable objects that step
relinks the 553 MB library from just the data objects (3 MB), destroying it.
This tool monkeypatches that step out and verifies the library looks sane
before touching anything.

Usage (from ssbu-rl, or anywhere with the package installed)::

    .venv/bin/rl-host-link debug_mode.c
    .venv/bin/rl-host-link debug_mode.c nnhid.c
    .venv/bin/rl-host-link --stage-only debug_mode.c
    .venv/bin/rl-host-link --exe ssbu_native debug_mode.c

After it succeeds, restart the game (``src/rl/scripts/live.sh ...``).  The game
library is not touched; if it is missing or only a few MB, run the full
``scripts/setup.py --target linux --dump ...`` rebuild instead.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
import shutil
import subprocess
import sys
import time

MIN_SO_BYTES = 100 * 1024 * 1024
BUILD_LOCK = "/tmp/ssbu_build.lock"
BACKUP_DIR = "/tmp/opencode/host_backups"


def find_repo() -> str:
    env = os.environ.get("SSBU_RECOMP_DIR")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..",
                                        "ssbu-recomp"))


def full_build_running() -> str | None:
    """The command line of a running full build, if there is one."""
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "host_link" in line:
            continue
        if "scripts/setup.py" in line or "build_opt.py" in line:
            return line.strip()
    return None


def game_running() -> str | None:
    """The command line of a running game, if there is one."""
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True,
                             text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return None
    for line in out.splitlines():
        if "build-linux/work/ssbu_native" in line:
            return line.strip()
    return None


def acquire_lock(path: str, timeout: float = 120.0):
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                raise SystemExit(f"error: {path} is held by another build")
            time.sleep(0.2)


def stage_source(repo: str, work: str, src: str) -> str:
    """Copy runtime/<src> (or an explicit path) to build-linux/work/<rel>."""
    runtime = os.path.join(repo, "runtime")
    candidates = []
    if os.path.isabs(src):
        candidates.append(src)
    candidates += [
        os.path.join(runtime, src),
        os.path.join(work, src),
        os.path.abspath(src),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            break
    else:
        raise SystemExit(f"error: cannot find source {src!r} "
                         f"(looked in runtime/ and {work}/)")
    cand = os.path.abspath(cand)
    rel = os.path.basename(cand)
    if os.path.commonpath([cand, os.path.abspath(runtime)]) == \
            os.path.abspath(runtime):
        rel = os.path.relpath(cand, runtime)     # keep runtime/plat/... layout
    dst = os.path.join(work, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(cand, dst)
    os.utime(dst, None)                 # make the object stale
    return dst


def ensure_allow_shlib_undefined(work: str) -> None:
    """Make build.py's Linux host link tolerate a library's optional symbols.

    The full build's .so normally defines all its lifted symbols; the flag
    keeps a hot rebuild working after a partially relinked library as well.
    """
    path = os.path.join(work, "build.py")
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return
    if "--allow-shlib-undefined" in text:
        return
    old = ("r = subprocess.run([CLANG] + objs + game + ['-Wl,-rpath,$ORIGIN'] "
           "+ LIBS + ['-o', out], capture_output=True, text=True, "
           "errors='replace')")
    new = ("r = subprocess.run([CLANG] + objs + game + "
           "['-Wl,--allow-shlib-undefined', '-Wl,-rpath,$ORIGIN'] + LIBS + "
           "['-o', out], capture_output=True, text=True, errors='replace')")
    if old in text:
        open(path, "w", encoding="utf-8").write(text.replace(old, new))
        print("build.py: enabled --allow-shlib-undefined for the host link")


def load_build(work: str):
    """Import build-linux/work/build.py without running its __main__."""
    spec = importlib.util.spec_from_file_location(
        "ssbu_work_build", os.path.join(work, "build.py"))
    if spec is None or spec.loader is None:
        raise SystemExit("error: cannot load build-linux/work/build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sources", nargs="*", metavar="FILE.c",
                    help="runtime sources to stage, e.g. debug_mode.c; "
                         "no sources = rebuild whatever is stale")
    ap.add_argument("--stage-only", action="store_true",
                    help="only copy the sources into build-linux/work")
    ap.add_argument("--exe", default=os.environ.get("RECOMP_EXE", "ssbu_native"),
                    help="host executable name (default: ssbu_native)")
    ap.add_argument("--allow-small-so", action="store_true",
                    help="link even if ssbu_game_eh_o1.so is suspiciously small")
    ap.add_argument("--no-lock", action="store_true",
                    help="do not take /tmp/ssbu_build.lock")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="seconds to wait for the build lock")
    args = ap.parse_args(argv)

    repo = find_repo()
    work = os.path.join(repo, "build-linux", "work")
    so = os.path.join(work, "ssbu_game_eh_o1.so")
    exe = os.path.join(work, args.exe)
    if not os.path.isdir(work):
        print(f"error: {work} does not exist; build the game once first",
              file=sys.stderr)
        return 2
    if not os.path.exists(so):
        print(f"error: {so} is missing; run scripts/setup.py --target linux",
              file=sys.stderr)
        return 2
    so_size = os.path.getsize(so)
    if so_size < MIN_SO_BYTES and not args.allow_small_so:
        print(f"error: {so} is only {so_size / 1e6:.1f} MB (a healthy build is "
              f"~550 MB).  The game library is broken/incomplete; do a full "
              f"rebuild: deps/venv/bin/python scripts/setup.py --target linux "
              f"--dump <dump>", file=sys.stderr)
        return 2
    other = full_build_running()
    if other:
        print(f"error: a full build is running: {other}\n"
              f"       wait for it to finish before hot-rebuilding.",
              file=sys.stderr)
        return 3
    running = game_running()
    if running:
        print(f"error: the game is running: {running}\n"
              f"       stop it first (the old executable cannot be replaced "
              f"while it runs), then restart it after the link.",
              file=sys.stderr)
        return 3

    staged = []
    for src in args.sources:
        dst = stage_source(repo, work, src)
        staged.append(dst)
        print(f"staged {os.path.basename(dst)}")
    if args.stage_only:
        print("stage-only: nothing compiled")
        return 0

    if not args.no_lock:
        lock_fd = acquire_lock(BUILD_LOCK, args.timeout)
    else:
        lock_fd = None
    backup = None
    try:
        os.makedirs(BACKUP_DIR, exist_ok=True)
        if os.path.exists(exe):
            backup = os.path.join(
                BACKUP_DIR,
                f"{os.path.basename(exe)}.{time.strftime('%Y%m%d_%H%M%S')}")
            shutil.copy2(exe, backup)
            print(f"backup: {backup}")
        ensure_allow_shlib_undefined(work)

        os.environ["RECOMP_TARGET"] = "linux"
        os.environ.setdefault("RECOMP_CLANG", "/usr/bin/clang")
        os.environ["RECOMP_EXE"] = args.exe
        B = load_build(work)
        # NEVER let cmd_link() touch the game library.
        B.build_game_dll = lambda: 0
        t0 = time.time()
        rc = B.cmd_link()
        elapsed = time.time() - t0
        if rc != 0:
            print("host link failed", file=sys.stderr)
            if backup:
                shutil.copy2(backup, exe)
                print(f"restored the previous {os.path.basename(exe)}",
                      file=sys.stderr)
            return rc
        print(f"hot rebuild ok in {elapsed:.1f}s: "
              f"{os.path.basename(exe)} {os.path.getsize(exe) / 1e6:.1f} MB; "
              f"game library untouched ({so_size / 1e6:.0f} MB)")
        print("restart the game to use it (src/rl/scripts/live.sh ...)")
        return 0
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
