#!/usr/bin/env python3
"""Tests for rl.utils.match_config (no game, no menus)."""

from __future__ import annotations

import os
import tempfile
import unittest

from rl.utils.match_config import (MatchSpec, append_match, match_env,
                                   match_list_env, write_match_list)


class MatchSpecTests(unittest.TestCase):
    def test_defaults(self):
        spec = MatchSpec()
        self.assertEqual(spec.p1_type, "human")
        self.assertEqual(spec.p2_type, "cpu")
        self.assertIn("p1=human", spec.to_string())
        self.assertIn("p2=cpu", spec.to_string())
        self.assertIn("stock=3", spec.to_string())
        self.assertIn("items=off", spec.to_string())
        self.assertIn("time=5", spec.to_string())

    def test_characters_and_round_trip(self):
        spec = MatchSpec(p1="samus", p2="kirby", cpu=9, stage="end",
                         stock=3, items="off")
        text = spec.to_string()
        self.assertIn("p1=samus", text)
        self.assertIn("p2=kirby", text)
        self.assertEqual(MatchSpec.from_string(text), spec)

    def test_swapped_types(self):
        spec = MatchSpec(p1="mario", p2="donkey", p1_type="cpu",
                         p2_type="human")
        text = spec.to_string()
        self.assertIn("p1=cpu", text)
        self.assertIn("p2=human", text)
        parsed = MatchSpec.from_string(text)
        self.assertEqual(parsed.p1_type, "cpu")
        self.assertEqual(parsed.p2_type, "human")

    def test_time_can_be_omitted(self):
        self.assertNotIn("time=", MatchSpec(time=None).to_string())

    def test_optional_fighter_names(self):
        spec = MatchSpec(p1=None, p2=None)
        text = spec.to_string()
        self.assertNotIn("samus", text)
        self.assertEqual(MatchSpec.from_string(text).p1, None)

    def test_bad_values(self):
        with self.assertRaises(ValueError):
            MatchSpec(p1_type="player")
        with self.assertRaises(ValueError):
            MatchSpec(items="maybe")
        with self.assertRaises(ValueError):
            MatchSpec(cpu=0)
        with self.assertRaises(ValueError):
            MatchSpec(players=9)
        with self.assertRaises(ValueError):
            MatchSpec(stock=0)
        with self.assertRaises(ValueError):
            MatchSpec(form="hazards")
        with self.assertRaises(ValueError):
            MatchSpec(p1="samus,zero")

    def test_env_helpers(self):
        spec = MatchSpec(p1="samus")
        self.assertIn("p1=samus", match_env(spec)["SSBU_DEBUG_MATCH"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "matches.txt")
            env = match_list_env(path)
            self.assertEqual(env["SSBU_DEBUG_MATCH_LIST"], os.path.abspath(path))


class MatchListTests(unittest.TestCase):
    def test_write_and_append(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "matches.txt")
            written = write_match_list(path, [
                MatchSpec(p1="samus", p2="kirby"),
                MatchSpec(p1="mario", p2="donkey"),
            ])
            self.assertEqual(written, os.path.abspath(path))
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("p1=samus", lines[0])
            self.assertIn("p1=mario", lines[1])

            append_match(path, MatchSpec(p1="link", p2="ness"))
            with open(path, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
            self.assertEqual(len(lines), 3)
            self.assertIn("p1=link", lines[2])


class MatchConfigCliTests(unittest.TestCase):
    def test_cli_prints_env_string(self):
        from rl.utils import match_config
        from io import StringIO
        from contextlib import redirect_stdout
        out = StringIO()
        with redirect_stdout(out):
            rc = match_config.main(["--p1", "samus", "--p2", "kirby"])
        self.assertEqual(rc, 0)
        self.assertIn("p1=samus", out.getvalue())
        self.assertIn("p2=kirby", out.getvalue())

    def test_cli_parse(self):
        from rl.utils import match_config
        from io import StringIO
        from contextlib import redirect_stdout
        out = StringIO()
        with redirect_stdout(out):
            rc = match_config.main(
                ["--parse", "p1=samus,p1=human,p2=kirby,p2=cpu,cpu=9"])
        self.assertEqual(rc, 0)
        parsed = MatchSpec.from_string(out.getvalue().strip())
        self.assertEqual(parsed.p1, "samus")
        self.assertEqual(parsed.p2, "kirby")
        self.assertEqual(parsed.cpu, 9)


if __name__ == "__main__":
    unittest.main()
