#!/usr/bin/env python3
"""Headless tests for ``rl.scripts.live_checks`` (no game, no X11, no uinput).

The live check itself is intentionally not run here: these tests cover the
parsing helpers, the stock/item evaluation on synthetic frames, the focus
decision helper, the JSON shape and the result/exit-code model.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

from rl.extra import extra_state as ex
from rl.scripts import live_checks as lc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def snapshot(fight_on: int = 1, stocks=(3, 3), xs=(0.0, 0.0),
             seq: int = 1, ts_ns: int = 0, is_cpu=(0, 0),
             buttons=(0, 0), stick=(0.0, 0.0), raw_flags: int = 0x7FFFFFFF):
    """A minimal RawSnapshot-compatible object for sample_frame()."""
    fighters = [
        {
            "stock": int(stocks[slot]),
            "x": float(xs[slot]),
            "is_cpu": int(is_cpu[slot]),
            "buttons": int(buttons[slot]),
            "stick_x": float(stick[slot]),
        }
        for slot in (0, 1)
    ]
    return SimpleNamespace(
        fight_on=int(fight_on), n_fighters=2, fighters=fighters,
        seq=int(seq), ts_ns=int(ts_ns), raw_flags=int(raw_flags),
    )


def frame_from(snap) -> lc.FrameSample:
    return lc.sample_frame(snap)


def frames(*specs) -> list[lc.FrameSample]:
    """``frames((1, (3, 3)), (0, (2, 3)))`` -> FrameSample list."""
    out = []
    for index, (fight_on, stocks) in enumerate(specs):
        out.append(frame_from(snapshot(fight_on=fight_on, stocks=stocks,
                                       seq=index + 1, ts_ns=index *
                                       16_666_667)))
    return out


def find(checks, needle: str) -> lc.Check:
    for check in checks:
        if needle in check.name:
            return check
    raise AssertionError(
        f"no check containing {needle!r}; got {[c.name for c in checks]}")


# ---------------------------------------------------------------------------
# SSBU_DEBUG_MATCH parsing
# ---------------------------------------------------------------------------
class DebugMatchTests(unittest.TestCase):
    def test_parse_options(self):
        parsed = lc.parse_debug_match(
            "p1=mario, p2=donkey ,items=off,stock=3,cpu=0")
        self.assertEqual(parsed["p1"], "mario")
        self.assertEqual(parsed["p2"], "donkey")
        self.assertEqual(parsed["items"], "off")
        self.assertEqual(parsed["stock"], "3")
        self.assertEqual(parsed["cpu"], "0")

    def test_parse_empty_and_bare_tokens(self):
        self.assertEqual(lc.parse_debug_match(None), {})
        self.assertEqual(lc.parse_debug_match(""), {})
        self.assertEqual(lc.parse_debug_match(" , ,"), {})
        self.assertEqual(lc.parse_debug_match("cpu"), {"cpu": ""})
        self.assertEqual(lc.parse_debug_match("p1=a,p1=b"), {"p1": "b"})

    def test_expectations(self):
        exp = lc.debug_match_expectations("p1=mario,items=off,stock=3")
        self.assertEqual(exp.stock, 3)
        self.assertTrue(exp.items_off)
        self.assertEqual(exp.options["p1"], "mario")

    def test_items_on_and_missing(self):
        self.assertFalse(lc.debug_match_expectations("items=on").items_off)
        self.assertFalse(lc.debug_match_expectations("items=1").items_off)
        self.assertIsNone(lc.debug_match_expectations("p1=mario").items_off)
        self.assertIsNone(lc.debug_match_expectations("items=maybe").items_off)

    def test_bad_stock_is_ignored(self):
        self.assertIsNone(lc.debug_match_expectations("stock=x").stock)
        self.assertIsNone(lc.debug_match_expectations("stock=-2").stock)

    def test_resolve_expected_stock_precedence(self):
        self.assertEqual(lc.resolve_expected_stock(2, "stock=5"), 2)
        self.assertEqual(lc.resolve_expected_stock(None, "stock=5"), 5)
        self.assertEqual(lc.resolve_expected_stock(None, ""),
                         lc.DEFAULT_EXPECTED_STOCK)
        self.assertEqual(lc.resolve_expected_stock(None, None),
                         lc.DEFAULT_EXPECTED_STOCK)


# ---------------------------------------------------------------------------
# snapshot normalisation
# ---------------------------------------------------------------------------
class FrameSampleTests(unittest.TestCase):
    def test_sample_frame(self):
        sample = lc.sample_frame(snapshot(
            stocks=(3, 2), xs=(-20.0, 20.0), is_cpu=(0, 1),
            buttons=(0x1, 0), stick=(1.0, -1.0), seq=7, ts_ns=99))
        self.assertEqual(sample.seq, 7)
        self.assertEqual(sample.ts_ns, 99)
        self.assertTrue(sample.live)
        self.assertEqual(sample.stocks, (3, 2))
        self.assertEqual(sample.x, (-20.0, 20.0))
        self.assertEqual(sample.is_cpu, (0, 1))
        self.assertEqual(sample.buttons, (1, 0))
        self.assertEqual(sample.stick_x, (1.0, -1.0))

    def test_missing_fighter_defaults(self):
        snap = SimpleNamespace(fight_on=1, fighters=[
            {"stock": 3, "x": 1.0},
        ], seq=1, ts_ns=2)
        sample = lc.sample_frame(snap)
        self.assertEqual(sample.stocks, (3, 0))
        self.assertEqual(sample.x, (1.0, 0.0))
        self.assertEqual(sample.buttons, (0, 0))


# ---------------------------------------------------------------------------
# stock / match lifetime evaluation
# ---------------------------------------------------------------------------
class StockEvaluationTests(unittest.TestCase):
    def test_clean_run_without_end(self):
        checks, summary = lc.evaluate_stock_run(
            frames((1, (3, 3)), (1, (3, 3)), (1, (2, 3))), expected_stock=3)
        self.assertFalse(summary["ended"])
        self.assertEqual(summary["stock_drops"], [1, 0])
        self.assertTrue(find(checks, "slot 0 stock reads 3").ok)
        self.assertTrue(find(checks, "slot 1 stock reads 3").ok)
        self.assertTrue(find(checks, "match did not end").ok)
        self.assertFalse([c for c in checks if c.ok is False])

    def test_zero_stock_is_a_failure(self):
        checks, summary = lc.evaluate_stock_run(
            frames((1, (0, 0)), (1, (0, 0))), expected_stock=3)
        self.assertEqual(summary["stocks_initial"], [0, 0])
        self.assertFalse(find(checks, "slot 0 stock reads 3").ok)
        self.assertFalse(find(checks, "slot 1 stock reads 3").ok)

    def test_end_after_first_ko_fails(self):
        checks, summary = lc.evaluate_stock_run(
            frames((1, (3, 3)), (1, (2, 3)), (0, (2, 3))),
            expected_stock=3)
        self.assertTrue(summary["ended"])
        self.assertEqual(summary["end_stocks"], [2, 3])
        self.assertEqual(summary["ko_counts"], [1, 0])
        check = find(checks, "did not end at the first KO")
        self.assertFalse(check.ok)
        self.assertIn("1 KO", check.detail)

    def test_end_with_both_stocks_positive_fails(self):
        checks, _summary = lc.evaluate_stock_run(
            frames((1, (2, 2)), (0, (2, 2))), expected_stock=3)
        self.assertFalse(find(checks, "did not end at the first KO").ok)

    def test_zeroed_results_after_first_ko_still_fail(self):
        # The 0-stock rule can zero the results screen after one KO; that must
        # not make a 3-stock match look legitimate.
        checks, summary = lc.evaluate_stock_run(
            frames((1, (3, 3)), (1, (2, 3)), (0, (0, 0))),
            expected_stock=3)
        # end_stocks is the last live frame; a results-screen reset to 0 must
        # not erase the one-KO evidence.
        self.assertEqual(summary["end_stocks"], [2, 3])
        self.assertEqual(summary["ko_counts"], [1, 0])
        self.assertFalse(find(checks, "did not end at the first KO").ok)

    def test_one_stock_match_may_end_at_zero(self):
        checks, summary = lc.evaluate_stock_run(
            frames((1, (1, 1)), (1, (0, 1)), (0, (0, 1))),
            expected_stock=1)
        self.assertTrue(summary["ended"])
        self.assertTrue(find(checks, "match ended only after").ok)

    def test_three_stock_match_may_end_after_three_kos(self):
        checks, summary = lc.evaluate_stock_run(
            frames((1, (3, 3)), (1, (2, 3)), (1, (1, 3)),
                   (1, (0, 3)), (0, (0, 3))),
            expected_stock=3)
        self.assertTrue(summary["ended"])
        self.assertEqual(summary["ko_counts"], [3, 0])
        self.assertTrue(find(checks, "match ended only after").ok)

    def test_no_live_frames(self):
        checks, summary = lc.evaluate_stock_run(
            frames((0, (3, 3))), expected_stock=3)
        self.assertEqual(summary["live_frames"], 0)
        self.assertFalse(find(checks, "live match observed").ok)

    def test_unknown_expected_stock_still_requires_zero(self):
        checks, _summary = lc.evaluate_stock_run(
            frames((1, (2, 2)), (0, (2, 2))), expected_stock=-1)
        self.assertTrue(find(checks, "slot 0 stock reads > 0").ok)
        self.assertFalse(find(checks, "did not end at the first KO").ok)
        # A KO was not observed, but the info row still exists.
        self.assertIsNone(find(checks, "KO observed").ok)


# ---------------------------------------------------------------------------
# items / battle objects
# ---------------------------------------------------------------------------
class ItemEvaluationTests(unittest.TestCase):
    def _check(self, checks, needle):
        return find(checks, needle)

    def test_extra_unavailable_is_info_only(self):
        checks, summary = lc.evaluate_items([], [], extra_available=False,
                                            extra_error="FileNotFoundError")
        self.assertFalse(summary["extra_available"])
        self.assertTrue(all(c.ok is None for c in checks))
        self.assertIn("FileNotFoundError",
                      self._check(checks, "item_appear").detail)

    def test_clean_items_pass(self):
        checks, summary = lc.evaluate_items(
            [0, 0, 0], [[], []], extra_available=True, extra_samples=3)
        self.assertTrue(self._check(checks, "item_appear == 0").ok)
        self.assertTrue(self._check(checks, "no item battle objects").ok)
        self.assertEqual(summary["item_appear_values"], [0])

    def test_nonzero_item_appear_fails(self):
        checks, _summary = lc.evaluate_items(
            [0, 5], [], extra_available=True, extra_samples=2)
        self.assertFalse(self._check(checks, "item_appear == 0").ok)

    def test_missing_item_record_is_info(self):
        checks, _summary = lc.evaluate_items(
            [], [], extra_available=True, extra_samples=1)
        self.assertIsNone(self._check(checks, "item_appear").ok)

    def test_item_object_fails(self):
        obj = lc.BattleObject(pool=2, oid=0x30001, category=0)
        checks, summary = lc.evaluate_items(
            [0], [[obj]], extra_available=True, extra_samples=1)
        self.assertFalse(self._check(checks, "no item battle objects").ok)
        self.assertEqual(summary["item_objects"][0]["id"], 0x30001)

    def test_fighter_objects_are_filtered(self):
        pairs = {
            ("f.accessor", 1): 0x11110000,
            ("f.accessor", 2): 0x22220000,
            ("bom.obj.pool", 0): 0,
            ("bom.obj.id", 0): 0,
            ("bom.obj.accessor", 0): 0x11110000,
            ("bom.obj.pool", 1): 0,
            ("bom.obj.id", 1): 0x10000,
            ("bom.obj.accessor", 1): 0x22220000,
            ("bom.obj.pool", 2): 2,
            ("bom.obj.id", 2): 0x30001,
            ("bom.obj.accessor", 2): 0x33330000,
            ("bom.obj.category", 2): 0,
        }
        objects = lc.battle_object_candidates(pairs)
        self.assertEqual([o.oid for o in objects], [0x30001])
        self.assertEqual(objects[0].pool, 2)

    def test_fighter_ids_filtered_without_accessors(self):
        pairs = {
            ("bom.obj.pool", 0): 0,
            ("bom.obj.id", 0): 0,
            ("bom.obj.pool", 1): 0,
            ("bom.obj.id", 1): 0x10000,
            ("bom.obj.pool", 2): 0,
            ("bom.obj.id", 2): 0x40001,
        }
        objects = lc.battle_object_candidates(pairs)
        self.assertEqual([o.oid for o in objects], [0x40001])

    def test_extra_pairs_from_a_real_block(self):
        records = [
            ex.Rec("g.match.item_appear", ex.T_U32, 0, 0, True),
            ex.Rec("f.accessor", ex.T_I64, 0x11110000, 1, True),
            ex.Rec("bom.obj.pool", ex.T_U32, 2, 0, True),
            ex.Rec("bom.obj.id", ex.T_U32, 0x30001, 0, True),
        ]
        block = ex.unpack_block(ex.pack_block(records, 4, 5))
        pairs = lc.extra_pairs(block)
        self.assertEqual(lc.extra_get(pairs, "g.match.item_appear"), 0)
        self.assertEqual([o.oid for o in lc.battle_object_candidates(pairs)],
                         [0x30001])

    def test_read_extra_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            block, error = lc.read_extra(f"{tmp}/missing.bin")
        self.assertIsNone(block)
        self.assertIsNotNone(error)
        self.assertIn("FileNotFound", error)


# ---------------------------------------------------------------------------
# focus decision / environment
# ---------------------------------------------------------------------------
class FocusDecisionTests(unittest.TestCase):
    def test_no_game_window(self):
        decision = lc.decide_focus([], "0x1")
        self.assertEqual(decision.state, "no-window")
        self.assertEqual(decision.action, "skip")

    def test_game_focused(self):
        decision = lc.decide_focus(["0x42"], "0x42")
        self.assertEqual(decision.state, "game-focused")
        self.assertEqual(decision.action, "defocus")
        self.assertEqual(decision.game_window, "0x42")

    def test_other_window_focused(self):
        decision = lc.decide_focus(["0x42", "0x43"], "0x99")
        self.assertEqual(decision.state, "not-focused")
        self.assertEqual(decision.action, "proceed")

    def test_unknown_focus(self):
        decision = lc.decide_focus(["0x42"], None)
        self.assertEqual(decision.state, "unknown")
        self.assertEqual(decision.action, "skip")

    def test_no_xdotool(self):
        decision = lc.decide_focus(["0x42"], "0x99", xdotool_available=False)
        self.assertEqual(decision.action, "skip")
        self.assertIn("xdotool", decision.detail)

    def test_decision_is_json_safe(self):
        decision = lc.decide_focus(["0x42"], "0x99")
        data = decision.as_dict()
        self.assertEqual(set(data),
                         {"state", "action", "game_window",
                          "focused_window", "detail"})
        json.dumps(data)


class FocusEnvTests(unittest.TestCase):
    def test_env_truthy(self):
        for value in ("1", "true", "YES", "on", " On "):
            self.assertTrue(lc.env_truthy(value), value)
        for value in ("", "0", "false", "off", None):
            self.assertFalse(lc.env_truthy(value), value)

    def test_parse_environ(self):
        data = b"PATH=/bin\0SSBU_INPUT_NO_FOCUS=1\0EMPTY=\0bad\0"
        parsed = lc.parse_environ(data)
        self.assertEqual(parsed["PATH"], "/bin")
        self.assertEqual(parsed["SSBU_INPUT_NO_FOCUS"], "1")
        self.assertEqual(parsed["EMPTY"], "")
        self.assertNotIn("bad", parsed)

    def test_enabled_in_own_environment(self):
        enabled, why = lc.focus_env_enabled(
            {"SSBU_INPUT_NO_FOCUS": "1"}, pid=0)
        self.assertTrue(enabled)
        self.assertIn("this shell", why)

    def test_disabled_in_own_environment(self):
        enabled, why = lc.focus_env_enabled(
            {"SSBU_INPUT_NO_FOCUS": "0"}, pid=0)
        self.assertFalse(enabled)
        self.assertIn("not enabled", why)

    def test_missing_everywhere(self):
        enabled, why = lc.focus_env_enabled({}, pid=0)
        self.assertFalse(enabled)
        self.assertIn("not set", why)

    def test_enabled_in_game_environment(self):
        with mock.patch.object(lc, "game_env_value",
                               return_value="1"):
            enabled, why = lc.focus_env_enabled({}, pid=12345)
        self.assertTrue(enabled)
        self.assertIn("game pid 12345", why)

    def test_scripted_menus_skip_a_taps(self):
        self.assertFalse(lc.scripted_menus({}))
        self.assertTrue(lc.scripted_menus({"SSBU_SEQ": "@movie 30:Z"}))
        self.assertTrue(lc.scripted_menus({"SSBU_PLAY": "input_last.rec"}))
        self.assertFalse(lc.scripted_menus({"SSBU_SEQ": ""}))


# ---------------------------------------------------------------------------
# focus motion evaluation
# ---------------------------------------------------------------------------
class FocusMotionTests(unittest.TestCase):
    def test_moved_right_passes(self):
        samples = [
            frame_from(snapshot(stocks=(3, 3), xs=(x, 10.0), seq=i + 1,
                                ts_ns=i * 16_666_667, buttons=(1, 0),
                                stick=(1.0, 0.0)))
            for i, x in enumerate((0.0, 2.0, 5.0, 12.0))
        ]
        checks, summary = lc.evaluate_focus_motion(samples, slot=0)
        check = find(checks, "P1 moved right")
        self.assertTrue(check.ok)
        self.assertAlmostEqual(summary["dx"], 12.0)
        self.assertEqual(summary["max_buttons"], 1)
        self.assertAlmostEqual(summary["max_abs_stick_x"], 1.0)
        self.assertIsNone(find(checks, "expert buttons").ok)

    def test_no_movement_fails(self):
        samples = [
            frame_from(snapshot(xs=(x, 10.0), seq=i + 1))
            for i, x in enumerate((0.0, 1.0, 2.0))
        ]
        checks, summary = lc.evaluate_focus_motion(samples, slot=0)
        self.assertFalse(find(checks, "P1 moved right").ok)
        self.assertEqual(summary["dx"], 2.0)

    def test_no_live_frames(self):
        checks, _summary = lc.evaluate_focus_motion(
            [frame_from(snapshot(fight_on=0))], slot=0)
        self.assertFalse(find(checks, "P1 moved right").ok)

    def test_slot_one_is_supported(self):
        samples = [
            frame_from(snapshot(xs=(-10.0, 0.0), seq=1)),
            frame_from(snapshot(xs=(-10.0, 9.0), seq=2)),
        ]
        checks, summary = lc.evaluate_focus_motion(samples, slot=1)
        self.assertTrue(find(checks, "P1 moved right").ok)
        self.assertEqual(summary["slot"], 1)

    def test_no_received_input_is_reported(self):
        samples = [
            frame_from(snapshot(xs=(0.0, 0.0), seq=i + 1,
                                ts_ns=i * 16_666_667))
            for i in range(3)
        ]
        checks, summary = lc.evaluate_focus_motion(samples, slot=0)
        self.assertFalse(find(checks, "game received the pad input").ok)
        self.assertEqual(summary["input_frames"], 0)


class HoldCollectTests(unittest.TestCase):
    class Reader:
        def __init__(self, snaps):
            self.snaps = list(snaps)
            self.i = 0

        def read_latest(self, timeout=None):
            if self.i >= len(self.snaps):
                raise lc.SnapshotTimeout("done")
            snap = self.snaps[self.i]
            self.i += 1
            return snap

    class Pad:
        def __init__(self):
            self.actions = []

        def apply(self, action):
            self.actions.append(action)

    def test_hold_rides_out_a_start_stall(self):
        # The game stalls at match start (no input poll), then receives the
        # stick.  The hold must not stop inside the stall.
        snaps = [
            snapshot(xs=(0.0, 10.0), seq=i + 1,
                     ts_ns=i * 16_666_667, stick=(0.0, 0.0))
            for i in range(4)
        ] + [
            snapshot(xs=(float(i), 10.0), seq=4 + i + 1,
                     ts_ns=(4 + i) * 16_666_667, stick=(1.0, 0.0))
            for i in range(3)
        ]
        reader = self.Reader(snaps)
        pad = self.Pad()
        out = lc.hold_right_and_collect(reader, pad, hold=0.02, poll=0.0,
                                        grace=0.5)
        self.assertGreaterEqual(len(out), 5)
        self.assertTrue(any(f.stick_x[0] for f in out))
        self.assertEqual(pad.actions[0]["stick"], (1.0, 0.0))
        self.assertEqual(pad.actions[-1]["stick"], (0.0, 0.0))


# ---------------------------------------------------------------------------
# report / JSON / CLI
# ---------------------------------------------------------------------------
class ReportTests(unittest.TestCase):
    def test_exit_codes(self):
        report = lc.Report("rules")
        self.assertEqual(report.exit_code(), 0)
        report.add("info", None, "detail")
        self.assertEqual(report.exit_code(), 0)
        report.add("pass", True, "detail")
        self.assertEqual(report.exit_code(), 0)
        self.assertEqual(report.status, "pass")
        report.add("fail", False, "detail")
        self.assertEqual(report.exit_code(), 1)
        self.assertEqual(report.status, "fail")

    def test_skip_wins_over_failure(self):
        report = lc.Report("focus")
        report.add("fail", False, "detail")
        report.skip("no display")
        self.assertEqual(report.exit_code(), 2)
        self.assertEqual(report.status, "skip")
        self.assertIn("no display", report.skipped)

    def test_children_exit_codes(self):
        rules = lc.Report("rules")
        focus = lc.Report("focus")
        focus.skip("no SSBU_INPUT_NO_FOCUS")
        combined = lc.Report("all")
        combined.children = [rules, focus]
        self.assertEqual(combined.exit_code(), 2)
        focus2 = lc.Report("focus")
        rules.add("fail", False)
        focus2.add("fail", False)
        combined.children = [rules, focus2]
        self.assertEqual(combined.exit_code(), 1)

    def test_json_shape(self):
        report = lc.Report("rules")
        report.add("a check", True, "ok")
        report.add("an info", None, "note")
        report.summary["stocks"] = [3, 3]
        data = report.as_dict()
        self.assertEqual(data["tool"], "live_checks")
        self.assertEqual(data["mode"], "rules")
        self.assertEqual(data["status"], "pass")
        self.assertEqual(data["exit_code"], 0)
        self.assertEqual(set(data["checks"][0]),
                         {"name", "ok", "detail"})
        self.assertIsNone(data["checks"][1]["ok"])
        self.assertEqual(data["summary"]["stocks"], [3, 3])
        blob = json.dumps(data)
        self.assertEqual(json.loads(blob)["checks"][0]["name"], "a check")

    def test_all_json_has_reports(self):
        rules = lc.Report("rules")
        rules.add("a", True)
        focus = lc.Report("focus")
        focus.add("b", True)
        combined = lc.Report("all")
        combined.children = [rules, focus]
        data = combined.as_dict()
        self.assertEqual([r["mode"] for r in data["reports"]],
                         ["rules", "focus"])
        self.assertEqual(data["mode"], "all")

    def test_format_report(self):
        report = lc.Report("rules")
        report.add("stock check", True, "stock=3")
        report.add("item check", False, "item_appear=5")
        text = lc.format_report(report)
        self.assertIn("live_checks rules: FAIL", text)
        self.assertIn("[PASS] stock check", text)
        self.assertIn("[FAIL] item check", text)
        self.assertIn("1 pass, 1 fail, 0 info", text)

    def test_format_report_skip(self):
        report = lc.Report("focus")
        report.skip("SSBU_INPUT_NO_FOCUS is not enabled")
        text = lc.format_report(report)
        self.assertIn("focus: SKIP", text)
        self.assertIn("skipped:", text)


class CliTests(unittest.TestCase):
    def test_rules_defaults(self):
        args = lc.build_parser().parse_args(["rules"])
        self.assertEqual(args.mode, "rules")
        self.assertFalse(args.json)
        self.assertIsNone(args.expected_stock)
        self.assertEqual(args.timeout, lc.DEFAULT_TIMEOUT)
        self.assertEqual(args.path, lc.SHM_PATH)
        self.assertEqual(args.extra_path, lc.EXTRA_SHM_PATH)
        self.assertEqual(args.seconds, lc.DEFAULT_SECONDS)
        self.assertEqual(args.hold, lc.DEFAULT_HOLD)
        self.assertEqual(args.min_dx, lc.DEFAULT_MIN_DX)

    def test_focus_options(self):
        args = lc.build_parser().parse_args([
            "focus", "--json", "--expected-stock", "2", "--timeout", "5",
            "--path", "/tmp/state.bin", "--extra-path", "/tmp/extra.bin",
            "--seconds", "3", "--hold", "1", "--min-dx", "4",
            "--display", ":1",
        ])
        self.assertEqual(args.mode, "focus")
        self.assertTrue(args.json)
        self.assertEqual(args.expected_stock, 2)
        self.assertEqual(args.timeout, 5.0)
        self.assertEqual(args.path, "/tmp/state.bin")
        self.assertEqual(args.extra_path, "/tmp/extra.bin")
        self.assertEqual(args.seconds, 3.0)
        self.assertEqual(args.hold, 1.0)
        self.assertEqual(args.min_dx, 4.0)
        self.assertEqual(args.display, ":1")

    def test_all_mode_exists(self):
        args = lc.build_parser().parse_args(["all"])
        self.assertEqual(args.mode, "all")

    def test_options_may_precede_the_mode(self):
        args = lc.build_parser().parse_args([
            "--json", "--timeout", "3", "rules", "--expected-stock", "2"])
        self.assertEqual(args.mode, "rules")
        self.assertTrue(args.json)
        self.assertEqual(args.timeout, 3.0)
        self.assertEqual(args.expected_stock, 2)
        self.assertEqual(args.seconds, lc.DEFAULT_SECONDS)

    def test_mode_is_required(self):
        with self.assertRaises(SystemExit):
            lc.build_parser().parse_args([])

    def test_make_pad_passes_a_short_lock_timeout(self):
        class Pad:
            def __init__(self, lock_timeout=0.0):
                self.lock_timeout = lock_timeout

        class OldPad:
            def __init__(self):
                self.lock_timeout = None

        self.assertEqual(lc._make_pad(Pad, 4.0).lock_timeout, 4.0)
        self.assertIsNone(lc._make_pad(OldPad, 4.0).lock_timeout)

    def test_progress_goes_to_stderr_with_json(self):
        import contextlib
        import io

        args = lc.build_parser().parse_args(["rules", "--json"])
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(err):
            lc._progress(args, "hello")
        self.assertEqual(out.getvalue(), "")
        self.assertIn("hello", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
