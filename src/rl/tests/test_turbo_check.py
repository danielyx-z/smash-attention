#!/usr/bin/env python3
"""Headless tests for the turbo simulation-hash checker (no game)."""

from __future__ import annotations

import contextlib
import io
import json
import os
import struct
import tempfile
import threading
import time
import unittest

from rl.scripts import turbo_check as tc


def pack_block(*, frame: int = 0, state_hash: int = 0, flags: int = 0,
               seed: int = 1234, speed_milli: int = 1000,
               match_start: int = 0,
               version: int = 2, magic: bytes = tc.MAGIC) -> bytes:
    block = bytearray(tc.SIZE)
    block[:4] = magic
    struct.pack_into("<I", block, tc.VERSION_OFFSET, version)
    struct.pack_into("<I", block, tc.FLAGS_OFFSET, flags)
    struct.pack_into("<I", block, tc.GAME_FRAME_OFFSET, frame)
    struct.pack_into("<I", block, tc.STATE_HASH_OFFSET, state_hash)
    struct.pack_into("<I", block, tc.SEED_OFFSET, seed)
    struct.pack_into("<I", block, tc.SPEED_MILLI_OFFSET, speed_milli)
    struct.pack_into("<I", block, tc.MATCH_START_OFFSET, match_start)
    return bytes(block)


def write_block(path: str, **kwargs) -> None:
    data = pack_block(**kwargs)
    with open(path, "r+b" if os.path.exists(path) else "wb") as fh:
        fh.seek(0)
        fh.write(data)
        fh.flush()


VALID = tc.FLAG_FIGHT_ON | tc.FLAG_HASH_VALID


def write_run(path: str, frames, hashes, *, seed: int = 1234,
              speed_milli: int = 4000, fmt: str = tc.FORMAT) -> str:
    payload = {
        "format": fmt,
        "game_frame": list(frames),
        "state_hash": list(hashes),
        "seed": seed,
        "speed_milli": speed_milli,
        "source": "/dev/shm/ssbu_control.bin",
        "started_ns": 1,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


class CompareTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name

    def path(self, name: str) -> str:
        return os.path.join(self.tmp, name)

    def test_equal_runs_align_by_relative_frame(self):
        a = write_run(self.path("a.json"), [100, 101, 102, 103],
                      [10, 11, 12, 13], speed_milli=1000)
        b = write_run(self.path("b.json"), [900, 901, 902, 903],
                      [10, 11, 12, 13], speed_milli=8000)
        result = tc.compare_runs(tc.load_run(a), tc.load_run(b))
        self.assertTrue(result.passed)
        self.assertEqual(result.common, 4)
        self.assertEqual(result.missing, 0)
        self.assertIsNone(result.divergence)
        self.assertEqual(result.matched, 4)

    def test_phase_shift_is_tolerated_only_when_asked(self):
        a = write_run(self.path("a.json"), [100, 101, 102, 103],
                      [10, 11, 12, 13], speed_milli=1000)
        b = write_run(self.path("e.json"), [200, 201, 202, 203, 204],
                      [99, 10, 11, 12, 13], speed_milli=4000)
        strict = tc.compare_runs(tc.load_run(a), tc.load_run(b))
        self.assertFalse(strict.passed)
        loose = tc.compare_runs(tc.load_run(a), tc.load_run(b), max_shift=1)
        self.assertTrue(loose.passed)
        self.assertEqual(loose.shift, 1)
        self.assertTrue(any("phase-shifted" in w for w in loose.warnings))

    def test_single_differing_hash_fails_with_right_frame(self):
        a = write_run(self.path("a.json"), [100, 101, 102, 103],
                      [10, 11, 12, 13])
        b = write_run(self.path("b.json"), [200, 201, 202, 203],
                      [10, 99, 12, 13])
        result = tc.compare_runs(tc.load_run(a), tc.load_run(b))
        self.assertFalse(result.passed)
        self.assertEqual(result.common, 4)
        self.assertEqual(result.matched, 3)
        self.assertIsNotNone(result.divergence)
        self.assertEqual(result.divergence.rel_frame, 1)
        self.assertEqual(result.divergence.a_hash, 11)
        self.assertEqual(result.divergence.b_hash, 99)
        self.assertEqual(result.divergence.a_frame, 101)
        self.assertEqual(result.divergence.b_frame, 201)

        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = tc.main(["compare", a, b])
        self.assertEqual(rc, 1)
        self.assertIn("relative frame 1", out.getvalue())
        self.assertIn("FAIL", out.getvalue())

    def test_missing_frames_and_tolerance(self):
        a = write_run(self.path("a.json"), [0, 1, 2, 3], [10, 11, 12, 13])
        b = write_run(self.path("b.json"), [50, 51, 52], [10, 11, 12])
        strict = tc.compare_runs(tc.load_run(a), tc.load_run(b), max_missing=0)
        self.assertFalse(strict.passed)
        self.assertEqual((strict.only_a, strict.only_b), (1, 0))
        self.assertEqual(strict.missing, 1)
        self.assertIsNone(strict.divergence)

        loose = tc.compare_runs(tc.load_run(a), tc.load_run(b), max_missing=1)
        self.assertTrue(loose.passed)

        out, _ = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tc.main(["compare", a, b, "--max-missing", "2"])
        self.assertEqual(rc, 0)
        self.assertIn("PASS", out.getvalue())

    def test_missing_frames_on_both_sides(self):
        a = write_run(self.path("a.json"), [0, 2, 3], [10, 12, 13])
        b = write_run(self.path("b.json"), [0, 1, 3], [10, 11, 13])
        result = tc.compare_runs(tc.load_run(a), tc.load_run(b), max_missing=0)
        self.assertFalse(result.passed)
        self.assertEqual((result.only_a, result.only_b), (1, 1))
        self.assertEqual(result.missing, 2)
        self.assertEqual(result.common, 2)

    def test_duplicate_frames_deduped_first_wins(self):
        deduped = tc.run_from_payload({
            "format": tc.FORMAT,
            "game_frame": [5, 5, 6, 6, 7],
            "state_hash": [50, 999, 60, 888, 70],
            "seed": 1,
            "speed_milli": 1000,
        })
        self.assertEqual(deduped.frames, (5, 6, 7))
        self.assertEqual(deduped.hashes, (50, 60, 70))

        a = write_run(self.path("a.json"), [0, 0, 1, 1, 2],
                      [10, 999, 11, 888, 12])
        b = write_run(self.path("b.json"), [100, 101, 102], [10, 11, 12])
        result = tc.compare_runs(tc.load_run(a), tc.load_run(b))
        self.assertTrue(result.passed)
        self.assertEqual(result.common, 3)

    def test_json_report(self):
        a = write_run(self.path("a.json"), [10, 11], [1, 2], speed_milli=1000)
        b = write_run(self.path("b.json"), [20, 21], [1, 2], speed_milli=8000)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tc.main(["compare", a, b, "--json"])
        self.assertEqual(rc, 0)
        report = json.loads(out.getvalue())
        self.assertEqual(report["format"], tc.COMPARE_FORMAT)
        self.assertTrue(report["pass"])
        self.assertEqual(report["common"], 2)
        self.assertEqual(report["missing"], 0)
        self.assertIsNone(report["divergence"])
        self.assertEqual(report["b"]["speed_milli"], 8000)

    def test_match_start_aligns_absolute_frames(self):
        a = write_run(self.path("a.json"), [100, 101, 102], [10, 11, 12],
                      speed_milli=1000)
        b = write_run(self.path("b.json"), [500, 501, 502], [10, 11, 12],
                      speed_milli=4000)
        for path, start in ((a, 100), (b, 500)):
            payload = json.load(open(path, encoding="utf-8"))
            payload["match_start"] = start
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
        result = tc.compare_runs(tc.load_run(a), tc.load_run(b))
        self.assertTrue(result.passed)

    def test_warns_on_mismatched_seed_and_equal_speed(self):
        a = write_run(self.path("a.json"), [0, 1], [1, 2],
                      seed=1, speed_milli=1000)
        b = write_run(self.path("b.json"), [0, 1], [1, 2],
                      seed=2, speed_milli=1000)
        result = tc.compare_runs(tc.load_run(a), tc.load_run(b))
        self.assertTrue(result.passed)  # comparison itself still passes
        text = "\n".join(result.warnings)
        self.assertIn("seeds differ", text)
        self.assertIn("same speed", text)

    def test_invalid_json_or_format_is_bad_input(self):
        good = write_run(self.path("good.json"), [0, 1], [1, 2])
        bad_json = self.path("bad.json")
        with open(bad_json, "w", encoding="utf-8") as fh:
            fh.write("{not json")
        wrong_format = write_run(self.path("fmt.json"), [0, 1], [1, 2],
                                 fmt="something_else")
        mismatch = self.path("mismatch.json")
        with open(mismatch, "w", encoding="utf-8") as fh:
            json.dump({"format": tc.FORMAT, "game_frame": [0, 1],
                       "state_hash": [1], "seed": 1, "speed_milli": 1000}, fh)
        empty = self.path("empty.json")
        with open(empty, "w", encoding="utf-8") as fh:
            json.dump({"format": tc.FORMAT, "game_frame": [],
                       "state_hash": [], "seed": 1, "speed_milli": 1000}, fh)

        for bad in (bad_json, wrong_format, mismatch, empty, self.path("nope.json")):
            with self.subTest(path=bad):
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    rc = tc.main(["compare", bad, good])
                self.assertEqual(rc, 2, err.getvalue())

    def test_bad_max_missing_is_bad_input(self):
        a = write_run(self.path("a.json"), [0], [1])
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = tc.main(["compare", a, a, "--max-missing", "-1"])
        self.assertEqual(rc, 2)


class RecordTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = self._tmp.name
        self.block = os.path.join(self.tmp, "control.bin")
        self.out = os.path.join(self.tmp, "run.json")

    def test_parse_rejects_short_bad_magic_and_v1(self):
        with self.assertRaises(tc.TurboCheckError):
            tc.parse_sample(b"\x00" * (tc.SIZE - 1))
        with self.assertRaises(tc.TurboCheckError):
            tc.parse_sample(pack_block(magic=b"NOPE"))
        with self.assertRaises(tc.TurboCheckError):
            tc.parse_sample(pack_block(version=1))

    def test_parse_reads_contract_offsets(self):
        sample = tc.parse_sample(pack_block(frame=42, state_hash=0xDEADBEEF,
                                            flags=VALID, seed=777,
                                            speed_milli=8000))
        self.assertEqual(sample.game_frame, 42)
        self.assertEqual(sample.state_hash, 0xDEADBEEF)
        self.assertEqual(sample.seed, 777)
        self.assertEqual(sample.speed_milli, 8000)
        self.assertTrue(sample.fight_on)
        self.assertTrue(sample.hash_valid)

    def test_record_static_block(self):
        write_block(self.block, frame=321, state_hash=0xCAFEBABE,
                    flags=VALID, seed=99, speed_milli=8000)
        result = tc.record_run(self.out, frames=1, timeout=1.0,
                               path=self.block, poll=0.0005)
        self.assertEqual(result.count, 1)
        self.assertEqual(result.stop_reason, "frames")
        payload = json.loads(open(self.out, encoding="utf-8").read())
        self.assertEqual(payload["format"], tc.FORMAT)
        self.assertEqual(payload["game_frame"], [321])
        self.assertEqual(payload["state_hash"], [0xCAFEBABE])
        self.assertEqual(payload["seed"], 99)
        self.assertEqual(payload["speed_milli"], 8000)
        self.assertEqual(payload["source"], self.block)
        self.assertIsInstance(payload["started_ns"], int)

    def test_record_threaded_dedupes_by_frame(self):
        write_block(self.block, frame=100, state_hash=1000, flags=VALID,
                    seed=5, speed_milli=4000)

        def writer():
            for i in range(10):
                write_block(self.block, frame=100 + i, state_hash=1000 + i,
                            flags=VALID, seed=5, speed_milli=4000)
                time.sleep(0.002)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            result = tc.record_run(self.out, frames=5, timeout=2.0,
                                   path=self.block, poll=0.0005)
        finally:
            thread.join(timeout=2.0)
        self.assertEqual(result.count, 5)
        frames = result.payload["game_frame"]
        hashes = result.payload["state_hash"]
        self.assertEqual(len(set(frames)), 5, frames)
        self.assertEqual(frames, sorted(frames))
        self.assertEqual(hashes, [f + 900 for f in frames])

    def test_record_skips_hash_invalid_and_times_out(self):
        write_block(self.block, frame=1, state_hash=1,
                    flags=tc.FLAG_FIGHT_ON, seed=5)
        started = time.monotonic()
        with self.assertRaises(tc.TurboCheckError):
            tc.record_run(self.out, frames=10, timeout=0.05,
                          path=self.block, poll=0.001)
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(os.path.exists(self.out))

    def test_record_stops_when_fight_ends(self):
        write_block(self.block, frame=10, state_hash=1000, flags=VALID,
                    seed=5)

        def writer():
            time.sleep(0.10)
            write_block(self.block, frame=11, state_hash=1001, flags=0,
                        seed=5)
            time.sleep(0.30)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            started = time.monotonic()
            result = tc.record_run(self.out, frames=0, timeout=2.0,
                                   path=self.block, poll=0.0005)
            elapsed = time.monotonic() - started
        finally:
            thread.join(timeout=2.0)
        self.assertEqual(result.stop_reason, "fight_off")
        self.assertEqual(result.count, 1)
        self.assertEqual(result.payload["game_frame"], [10])
        self.assertLess(elapsed, 1.0)

    def test_record_no_frames_does_not_write_output(self):
        write_block(self.block, frame=1, state_hash=1, flags=tc.FLAG_FIGHT_ON)
        with self.assertRaises(tc.TurboCheckError):
            tc.record_run(self.out, frames=1, timeout=0.05,
                          path=self.block, poll=0.001)
        self.assertFalse(os.path.exists(self.out))

    def test_record_times_out_waiting_for_missing_block(self):
        with self.assertRaises(tc.TurboCheckError):
            tc.record_run(self.out, frames=1, timeout=0.05,
                          path=os.path.join(self.tmp, "missing.bin"),
                          poll=0.001)

    def test_record_honours_frame_cap(self):
        write_block(self.block, frame=7, state_hash=70, flags=VALID)
        result = tc.record_run(self.out, frames=1, timeout=1.0,
                               path=self.block, poll=0.0005)
        self.assertEqual(result.payload["game_frame"], [7])
        self.assertEqual(result.stop_reason, "frames")

    def test_record_skips_menu_frames(self):
        write_block(self.block, frame=1, state_hash=1,
                    flags=tc.FLAG_HASH_VALID, seed=0)

        def writer():
            time.sleep(0.02)
            write_block(self.block, frame=500, state_hash=500, flags=VALID,
                        seed=42, speed_milli=1000)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            result = tc.record_run(self.out, frames=1, timeout=1.0,
                                   path=self.block, poll=0.0005)
        finally:
            thread.join(timeout=1.0)
        self.assertEqual(result.payload["game_frame"], [500])
        self.assertEqual(result.payload["seed"], 42)

    def test_record_rejects_seed_zero_on_a_live_frame(self):
        write_block(self.block, frame=1, state_hash=1, flags=VALID, seed=0)
        with self.assertRaises(tc.TurboCheckError):
            tc.record_run(self.out, frames=1, timeout=0.2,
                          path=self.block, poll=0.0005)

    def test_record_accepts_seed_zero_when_allowed(self):
        write_block(self.block, frame=1, state_hash=1, flags=VALID, seed=0)
        result = tc.record_run(self.out, frames=1, timeout=0.2,
                               path=self.block, poll=0.0005,
                               allow_seed_zero=True)
        self.assertEqual(result.payload["seed"], 0)

    def test_record_rejects_a_stale_epoch(self):
        flags = VALID | ((1234 & tc.EPOCH_MASK) << tc.EPOCH_SHIFT)
        write_block(self.block, frame=1, state_hash=1, flags=flags, seed=7)
        with self.assertRaises(tc.TurboCheckError):
            tc.record_run(self.out, frames=1, timeout=0.2,
                          path=self.block, poll=0.0005,
                          expect_pid=5678)

    def test_record_accepts_the_matching_epoch(self):
        flags = VALID | ((5678 & tc.EPOCH_MASK) << tc.EPOCH_SHIFT)
        write_block(self.block, frame=1, state_hash=1, flags=flags, seed=7)
        result = tc.record_run(self.out, frames=1, timeout=0.2,
                               path=self.block, poll=0.0005,
                               expect_pid=5678)
        self.assertEqual(result.payload["epoch"], 5678)

    def test_record_reads_the_match_start(self):
        write_block(self.block, frame=500, state_hash=5, flags=VALID,
                    seed=7, match_start=100)
        result = tc.record_run(self.out, frames=1, timeout=0.2,
                               path=self.block, poll=0.0005, settle=10)
        self.assertEqual(result.payload["match_start"], 100)

    def test_record_checks_the_speed_echo(self):
        write_block(self.block, frame=1, state_hash=1, flags=VALID,
                    seed=7, speed_milli=1000)
        with self.assertRaises(tc.TurboCheckError):
            tc.record_run(self.out, frames=1, timeout=0.2,
                          path=self.block, poll=0.0005, expect_speed=8000)


class CliTests(unittest.TestCase):
    def test_self_test_passes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = tc.main(["--self-test"])
        self.assertEqual(rc, 0)
        self.assertIn("self-test OK", out.getvalue())

    def test_no_command_is_usage_error(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = tc.main([])
        self.assertEqual(rc, 2)
        self.assertIn("usage", err.getvalue())


if __name__ == "__main__":
    unittest.main()
