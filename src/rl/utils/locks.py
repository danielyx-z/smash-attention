"""Cross-process locks for the shared game/build/pad resources.

The repo has one game instance, one display, and one virtual pad.  Every live
test must hold the relevant lock so parallel agents cannot collide.
"""

from __future__ import annotations

import fcntl
import os
import time

GAME_LOCK = "/tmp/ssbu_game.lock"
BUILD_LOCK = "/tmp/ssbu_build.lock"
PAD_LOCK = "/tmp/ssbu_pad.lock"


class FileLock:
    """Advisory exclusive lock with a timeout, using flock(2)."""

    def __init__(self, path: str = GAME_LOCK, timeout: float = 900.0,
                 poll: float = 0.25):
        self.path = path
        self.timeout = timeout
        self.poll = poll
        self.fd: int | None = None
        self.owner = os.environ.get("SSBU_SESSION_OWNER", "?")

    def acquire(self) -> "FileLock":
        if self.path == GAME_LOCK and os.environ.get("SSBU_GAME_LOCK_HELD") == "1":
            self.fd = None          # the session manager already holds it
            return self
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o664)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                os.ftruncate(self.fd, 0)
                os.write(self.fd, f"{self.owner} {os.getpid()} {time.time():.0f}\n".encode())
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(self.fd)
                    self.fd = None
                    raise TimeoutError(f"lock {self.path} busy (owner {self.owner})")
                time.sleep(self.poll)

    def release(self) -> None:
        if self.fd is not None:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self) -> "FileLock":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def game_lock(timeout: float = 900.0) -> FileLock:
    return FileLock(GAME_LOCK, timeout)


def build_lock(timeout: float = 1800.0) -> FileLock:
    return FileLock(BUILD_LOCK, timeout)


def pad_lock(timeout: float = 120.0) -> FileLock:
    return FileLock(PAD_LOCK, timeout)
