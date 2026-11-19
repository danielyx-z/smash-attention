#!/usr/bin/env python3
"""Headless tests for the live stage view (no game, no window)."""

from __future__ import annotations

import contextlib
import io
import json
import math
import unittest

from rl.scripts import live_view as lv


class WorldMapTests(unittest.TestCase):
    def test_center_and_y_direction(self):
        m = lv.WorldMap(-240, 240, -140, 180, 800, 600, margin=20)
        cx, cy = m.px(0, 0)
        self.assertAlmostEqual(cx, 400.0, delta=1.0)
        self.assertAlmostEqual(cy, 300.0, delta=60.0)
        # world y up -> screen y down
        _, low = m.px(0, -100)
        _, high = m.px(0, 100)
        self.assertGreater(low, high)
        # equal world distances map to equal pixel distances
        a = m.px(0, 0)[0]
        b = m.px(100, 0)[0]
        c = m.px(200, 0)[0]
        self.assertAlmostEqual((b - a), (c - b), delta=1e-6)

    def test_aspect_is_preserved(self):
        m = lv.WorldMap(-100, 100, -50, 50, 1000, 300, margin=10)
        dx = m.px(1, 0)[0] - m.px(0, 0)[0]
        dy = m.px(0, 1)[1] - m.px(0, 0)[1]
        self.assertAlmostEqual(dx, -dy, delta=1e-6)

    def test_rect_orders_corners(self):
        m = lv.WorldMap(-100, 100, -100, 100, 400, 400)
        x0, y0, x1, y1 = m.rect(0, 0, 10, 20)
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)   # screen top is the head


class MiscTests(unittest.TestCase):
    def test_blast_fallback(self):
        self.assertEqual(lv.blast_box({"blast": (0, 0, 0, 0)}),
                         (-240.0, 240.0, 180.0, -140.0))
        self.assertEqual(lv.blast_box({"blast": (-240, 240, 180, -140)}),
                         (-240.0, 240.0, 180.0, -140.0))

    def test_state_badge(self):
        self.assertEqual(lv.state_badge({"action_state": 0, "on_ground": 1}), "WAIT")
        self.assertEqual(lv.state_badge({"action_state": 12, "on_ground": 0}),
                         "AIRBORNE")
        self.assertEqual(lv.state_badge({"shield": 1}), "SHIELD")
        self.assertEqual(lv.state_badge({"hitstun": 1}), "HITSTUN")
        self.assertEqual(lv.state_badge({"is_on_rebirth": 1}), "REBIRTH")

    def test_format_value(self):
        text, tag = lv.format_value("buttons", 0x8004)
        self.assertIn("0x8004", text)
        self.assertIn("X", text)
        text, tag = lv.format_value("raw_flags", 0x7FFFFFFF)
        self.assertEqual(text, "0x7fffffff")
        self.assertEqual(tag, "ok")
        text, tag = lv.format_value("action_state", 12)
        self.assertIn("12", text)
        self.assertIn("JUMP", text)
        text, tag = lv.format_value("facing", 500.0)
        self.assertEqual(tag, "bad")

    def test_damage_colour_moves_green_to_red(self):
        self.assertNotEqual(lv.damage_color(0), lv.damage_color(150))
        self.assertEqual(len(lv.damage_color(75)), 7)


class ExtraViewTests(unittest.TestCase):
    def _extras(self):
        return lv.ExtraView({
            ("f.accessor", 1): 0x11110000,
            ("f.accessor", 2): 0x22220000,
            ("bom.obj.pool", 0): 0,
            ("bom.obj.id", 0): 0,
            ("bom.obj.accessor", 0): 0x11110000,   # fighter 1: hide
            ("bom.obj.pos", 0): (10.0, 1.0, 0.0),
            ("bom.obj.pool", 1): 0,
            ("bom.obj.id", 1): 0x10000,
            ("bom.obj.accessor", 1): 0x22220000,   # fighter 2: hide
            ("bom.obj.pos", 1): (-10.0, 1.0, 0.0),
            ("bom.obj.pool", 2): 2,
            ("bom.obj.id", 2): 0x30001,            # projectile: keep
            ("bom.obj.accessor", 2): 0x33330000,
            ("bom.obj.pos", 2): (60.0, 25.0, 0.0),
            ("bom.obj.status_kind", 2): 3,
        })

    def test_filters_fighter_objects(self):
        objs = self._extras().objects()
        self.assertEqual(len(objs), 1)
        self.assertEqual(objs[0].oid, 0x30001)
        self.assertAlmostEqual(objs[0].x, 60.0)
        self.assertEqual(objs[0].label, "p2 30001 s3")

    def test_without_accessors_skips_pool0_fighters(self):
        e = lv.ExtraView({
            ("bom.obj.pool", 0): 0,
            ("bom.obj.id", 0): 0,
            ("bom.obj.pos", 0): (1.0, 2.0, 0.0),
            ("bom.obj.pool", 1): 3,
            ("bom.obj.id", 1): 0x55555,
            ("bom.obj.pos", 1): (5.0, 6.0, 0.0),
        })
        objs = e.objects()
        self.assertEqual([o.oid for o in objs], [0x55555])

    def test_ignores_non_finite_positions(self):
        e = lv.ExtraView({
            ("bom.obj.pool", 0): 1,
            ("bom.obj.id", 0): 9,
            ("bom.obj.pos", 0): (float("nan"), 0.0, 0.0),
        })
        self.assertEqual(e.objects(), [])


class MockSourceTests(unittest.TestCase):
    def test_frames_are_complete_and_animate(self):
        src = lv.MockSource(hz=60.0)
        a = src.read()
        for _ in range(30):
            b = src.read()
        self.assertEqual(len(a.fighters), 2)
        self.assertEqual(len(b.fighters), 2)
        self.assertGreaterEqual(len(b.objects), 1)
        self.assertNotEqual(a.fighters[0]["x"], b.fighters[0]["x"])
        for f in b.fighters:
            self.assertEqual(set(f), set(lv.FIELD_NAMES))
            for name in ("x", "y", "damage", "action_frame"):
                self.assertTrue(math.isfinite(float(f[name])), name)
        self.assertEqual(b.objects[0].key, (1, 0x30001))

    def test_to_json_is_serialisable(self):
        model = lv.MockSource().read()
        blob = json.dumps(model.to_json())
        back = json.loads(blob)
        self.assertEqual(len(back["fighters"]), 2)
        self.assertIn("blast", back)
        self.assertGreaterEqual(len(back["objects"]), 1)


class EventTrackerTests(unittest.TestCase):
    def test_state_and_fight_transitions(self):
        tr = lv.EventTracker(maxlen=3)
        m = lv.MockSource().read()
        tr.update(m)
        m2 = lv.MockSource().read()
        m2.header["fight_on"] = 0
        m2.fighters[0]["action_state"] = 54
        tr.update(m2)
        text = " ".join(tr.text())
        self.assertIn("fight_on 1 -> 0", text)
        self.assertIn("P1", text)
        self.assertLessEqual(len(tr.text()), 3)


class ShmSourceTests(unittest.TestCase):
    def test_reads_a_real_v7_block(self):
        from rl.env.gym_env import ShmFixture
        fix = ShmFixture(stage_id=3)
        fix.set_fighter(0, x=12.0, y=0.0, damage=11.0, kind=0,
                        on_ground=1, raw_flags=0x7FFFFFFF)
        fix.set_fighter(1, x=-12.0, y=0.0, damage=22.0, kind=1,
                        on_ground=1, raw_flags=0x7FFFFFFF)
        fix.publish()
        src = lv.ShmSource(fix.path, extra="off")
        try:
            model = src.read()
            self.assertIsNotNone(model)
            self.assertEqual(model.header["stage_id"], 3)
            self.assertEqual(len(model.fighters), 2)
            self.assertAlmostEqual(model.fighters[0]["x"], 12.0)
            self.assertAlmostEqual(model.fighters[1]["damage"], 22.0)
            self.assertEqual(model.objects, [])
            self.assertEqual(lv.state_badge(model.fighters[0]), "WAIT")
        finally:
            src.close()
            fix.close()


class AdjacentGeometryTest(unittest.TestCase):
    class Root:
        def winfo_screenwidth(self):
            return 1920

        def winfo_screenheight(self):
            return 1080

    def test_places_to_the_right_when_it_fits(self):
        from unittest import mock
        with mock.patch.object(lv, "_game_window_geometry",
                               return_value=(0, 0, 1280, 720)):
            w, h, x, y = lv.adjacent_geometry(self.Root(), 1440, 860)
        self.assertEqual((w, h, x, y), (632, 860, 1288, 0))

    def test_falls_back_below(self):
        from unittest import mock
        with mock.patch.object(lv, "_game_window_geometry",
                               return_value=(0, 0, 1800, 800)):
            w, h, x, y = lv.adjacent_geometry(self.Root(), 1440, 860)
        self.assertEqual((w, h, x, y), (1440, 272, 0, 808))

    def test_no_game_leaves_placement_alone(self):
        from unittest import mock
        with mock.patch.object(lv, "_game_window_geometry",
                               return_value=None):
            w, h, x, y = lv.adjacent_geometry(self.Root(), 1440, 860)
        self.assertEqual((w, h, x, y), (1440, 860, None, None))

    def test_fullscreen_stays_visible(self):
        from unittest import mock
        with mock.patch.object(lv, "_game_window_geometry",
                               return_value=(0, 0, 1920, 1080)):
            w, h, x, y = lv.adjacent_geometry(self.Root(), 1440, 860)
        self.assertEqual((w, h), (640, 860))
        self.assertEqual((x, y), (1280, 0))


class CliTests(unittest.TestCase):
    def test_mock_dump_once_json(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = lv.main(["--mock", "--dump-once", "--json"])
        self.assertEqual(rc, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(data["source"], "mock")
        self.assertEqual(len(data["fighters"]), 2)
        self.assertTrue(data["objects"])


if __name__ == "__main__":
    unittest.main()
