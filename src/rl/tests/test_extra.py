#!/usr/bin/env python3
"""Tests for tools/mem/extra_state.py (format + getter decoder + live probe).

The unit tests need no game and no lock.  The live probe only runs when a
match is actually live (fight_on == 1); otherwise it is skipped, so the test
file is safe to run at any time:

    python3 tools/mem/test_extra.py -v
    SSBU_EXTRA_LIVE=1 python3 tools/mem/test_extra.py -v   # require a match
"""

from __future__ import annotations

import math
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from rl.extra import extra_state as ex  # noqa: E402


class FormatTest(unittest.TestCase):
    def test_roundtrip_all_types(self):
        recs = [
            ex.Rec("g.test.i64", ex.T_I64, -1234567890123, 0, True),
            ex.Rec("g.test.f64", ex.T_F64, 3.25, 0, True),
            ex.Rec("g.test.u32", ex.T_U32, 0xDEADBEEF, 3, False),
            ex.Rec("g.test.bool", ex.T_BOOL, True, 1, True),
            ex.Rec("g.test.vec3", ex.T_VEC3, (1.0, -2.0, 0.5), 2, True),
            ex.Rec("g.test.vec4", ex.T_VEC4, (1.0, 2.0, 3.0, 4.0), 2, True),
            ex.Rec("g.test.bytes", ex.T_BYTES, b"\x00\x01\x02\xfe\xff", 0, True),
            ex.Rec("g.test.str", ex.T_STR, "hello", 0, False),
        ]
        blob = ex.pack_block(recs, seq=42, ts_ns=987654321, flags=3)
        self.assertEqual(len(blob), 32 + sum(56 + ((len(r.encoded()) + 7) // 8) * 8
                                             for r in recs))
        out = ex.unpack_block(blob)
        self.assertEqual(out["magic"], b"SSBX")
        self.assertEqual(out["version"], ex.VERSION)
        self.assertEqual(out["seq"], 42)
        self.assertEqual(out["ts_ns"], 987654321)
        self.assertEqual(out["count"], len(recs))
        self.assertEqual(out["flags"], 3)
        self.assertEqual(out["total_bytes"], len(blob))
        for a, b in zip(recs, out["records"]):
            self.assertEqual(a.name, b.name)
            self.assertEqual(a.type, b.type)
            self.assertEqual(a.index, b.index)
            self.assertEqual(a.verified, b.verified)
            if a.type in (ex.T_VEC3, ex.T_VEC4):
                for x, y in zip(a.value, b.value):
                    self.assertAlmostEqual(x, y, places=6)
            else:
                self.assertEqual(a.value, b.value)

    def test_names_are_truncated_to_fit(self):
        r = ex.Rec("x" * 100, ex.T_I64, 1)
        blob = ex.pack_block([r], 1, 1)
        out = ex.unpack_block(blob)
        self.assertEqual(len(out["records"][0].name), ex.NAME_MAX)

    def test_records_are_8_byte_aligned(self):
        for t, v in ((ex.T_I64, 1), (ex.T_F64, 1.5), (ex.T_BOOL, True),
                     (ex.T_U32, 7), (ex.T_VEC3, (1, 2, 3)),
                     (ex.T_VEC4, (1, 2, 3, 4)), (ex.T_BYTES, b"abc")):
            self.assertEqual(len(ex.Rec("n", t, v).packed()) % 8, 0)

    def test_bad_magic_and_version(self):
        blob = bytearray(ex.pack_block([ex.Rec("n", ex.T_I64, 1)], 1, 1))
        blob[0:4] = b"NOPE"
        with self.assertRaises(ValueError):
            ex.unpack_block(bytes(blob))
        blob = bytearray(ex.pack_block([ex.Rec("n", ex.T_I64, 1)], 1, 1))
        blob[4:8] = (99).to_bytes(4, "little")
        with self.assertRaises(ValueError):
            ex.unpack_block(bytes(blob))

    def test_truncated_block_rejected(self):
        blob = ex.pack_block([ex.Rec("n", ex.T_BYTES, b"abcdefgh")], 1, 1)
        with self.assertRaises(ValueError):
            ex.unpack_block(blob[:-8])

    def test_odd_even_seq_protocol_spec(self):
        # documented invariant: the body is only valid when seq is even
        blob = ex.pack_block([], 0, 0)
        self.assertEqual(ex.unpack_block(blob)["seq"], 0)
        blob = ex.pack_block([], 1, 0)
        self.assertEqual(ex.unpack_block(blob)["seq"], 1)

    def test_publisher_file(self):
        import tempfile
        path = tempfile.mktemp(prefix="ssbx_test_")
        try:
            pub = ex.Publisher(path)
            recs = [ex.Rec("g.x", ex.T_I64, 5, 0, True),
                    ex.Rec("f.pos", ex.T_VEC3, (1.0, 2.0, 3.0), 1, True)]
            pub.publish(recs, ex.FLAG_LIVE | ex.FLAG_DECODED)
            pub.publish(recs, 0)
            pub.close()
            self.assertEqual(os.path.getsize(path), ex.MAX_BLOCK)
            out = ex.read_block(path)
            self.assertEqual(out["seq"] % 2, 0, "published seq must be even")
            self.assertEqual(out["flags"], 0)
            self.assertEqual(out["count"], 2)
            self.assertLessEqual(out["total_bytes"], ex.MAX_BLOCK)
            self.assertEqual(out["records"][0].name, "g.x")
            self.assertAlmostEqual(out["records"][1].value[2], 3.0)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class DecoderTest(unittest.TestCase):
    RET = 0xD65F03C0

    @staticmethod
    def code(*words):
        import struct
        return b"".join(struct.pack("<I", w) for w in words) + b"\x00" * 8

    def test_direct_u8(self):
        # ldrb w0,[x0,#127] ; ret   (ShieldModule::is_shield's first step)
        dec = ex.decode_getter(self.code(0x3941FC00, self.RET))
        self.assertEqual(dec, ("direct", 0, 127, "u8"))

    def test_direct_u32(self):
        # ldr w0,[x0,#152] ; ret    (StatusModule::status_kind)
        dec = ex.decode_getter(self.code(0xB9409800, self.RET))
        self.assertEqual(dec, ("direct", 0, 152, "u32"))

    def test_direct_f32(self):
        # ldr s0,[x0,#80] ; ret     (GroundModule::offset_x)
        dec = ex.decode_getter(self.code(0xBD405000, self.RET))
        self.assertEqual(dec, ("direct", 0, 80, "f32"))

    def test_deref_i16(self):
        # ldr x8,[x0,#24] ; ldrsh w0,[x8,#6] ; ret  (ShieldModule::get_part_size)
        dec = ex.decode_getter(self.code(0xF9400C08, 0x79C00D00, self.RET))
        self.assertEqual(dec, ("deref", 24, 6, "i16"))

    def test_pointer_add(self):
        # add x0,x0,#0x30 ; ret     (PostureModule vtable[12] pos pointer)
        dec = ex.decode_getter(self.code(0x9100C000, self.RET))
        self.assertEqual(dec, ("addr", 0x30, 0, "addr"))

    def test_load_pointer(self):
        # ldr x0,[x0,#16] ; ret     (ShieldModule::get_group_num)
        dec = ex.decode_getter(self.code(0xF9400800, self.RET))
        self.assertEqual(dec, ("loadptr", 0, 16, "ptr"))

    def test_vector_load(self):
        # ldr q0,[x0,#0x30] ; ret
        dec = ex.decode_getter(self.code(0x3DC00C00, self.RET))
        self.assertEqual(dec, ("direct", 0, 0x30, "vec4"))

    def test_vec_add(self):
        # 0x91004000: add x0,x0,#0x10 ; ret   (kinetic velocity vector)
        dec = ex.decode_vec_getter(self.code(0x91004000, self.RET))
        self.assertEqual(dec, ("addr", 0x10))

    def test_vec_select(self):
        # live entry 1 getter: ldr w8,[x0,#0x70] ; add x9,x0,#0x60 ;
        # add x10,x0,#0x10 ; cmp w8,#0 ; csel x0,x9,x10,gt ; ret
        code = self.code(0xB9407008, 0x91018009, 0x9100400A, 0x7100011F,
                         0x9A8AC120, self.RET)
        dec = ex.decode_vec_getter(code)
        self.assertEqual(dec, ("select", 0x70, 0x60, 0x10))

    def test_vec_select_second_shape(self):
        # live entry 4 getter: ldr w8,[x0,#0xD0] ; add x9,x0,#0xC0 ;
        # add x10,x0,#0x10 ; cmp w8,#0 ; csel x0,x9,x10,gt
        code = self.code(0xB940D008, 0x91030009, 0x9100400A, 0x7100011F,
                         0x9A8AC120, self.RET)
        self.assertEqual(ex.decode_vec_getter(code),
                         ("select", 0xD0, 0xC0, 0x10))

    def test_complex_getter_rejected(self):
        # ldrb w8,[x0,#127] ; cbz w8,... (ShieldModule::is_shield): the first
        # load is not the return value, so the decoder must refuse.
        code = self.code(0x3941FC08, 0x34000068)
        self.assertIsNone(ex.decode_getter(code))

    def test_arithmetic_getter_rejected(self):
        # ldr x8,[x0,#24] ; mov w9,#52 ; smaddl ... ; ret
        code = self.code(0xF9400C08, 0x52800689, 0x9B292028)
        self.assertIsNone(ex.decode_getter(code))


class WideGetterTest(unittest.TestCase):
    def test_hash40_loadptr(self):
        import struct
        from types import SimpleNamespace

        class FakeMem:
            def __init__(self, value):
                self.value = value
            def read(self, addr, n):
                return struct.pack("<Q", self.value)[:n]
            def u64(self, addr):
                return 0

        obj = object.__new__(ex.ExtraState)
        obj.mem = FakeMem(0x47DEE83E5)          # hash40("wait")
        obj._module_getter = lambda module, slot: ("loadptr", 0, 104, "ptr")
        v, ok = obj._read_wide(0x1000, 39)
        self.assertEqual(v, 0x47DEE83E5)
        self.assertTrue(ok)

    def test_hash40_rejects_pointer(self):
        import struct

        class FakeMem:
            def read(self, addr, n):
                return struct.pack("<Q", 0x000077C788990000)[:n]
            def u64(self, addr):
                return 0

        obj = object.__new__(ex.ExtraState)
        obj.mem = FakeMem()
        obj._module_getter = lambda module, slot: ("loadptr", 0, 104, "ptr")
        v, ok = obj._read_wide(0x1000, 39)
        self.assertIsNone(v)
        self.assertFalse(ok)


class LiveProbe(unittest.TestCase):
    """Read a live game when a fight is running; otherwise skip."""

    def test_live_extras(self):
        from rl.memory.memapi import find_game_pid
        pid = find_game_pid()
        if not pid:
            self.skipTest("no ssbu_native process")
        try:
            gs = ex.ExtraState(pid)
        except Exception as exc:              # no permission / device gone
            self.skipTest(f"cannot attach: {exc}")
        try:
            if not gs.fight_on():
                if os.environ.get("SSBU_EXTRA_LIVE") == "1":
                    self.fail("SSBU_EXTRA_LIVE=1 but no fight is running")
                self.skipTest("no live match")
            import time
            recs = []
            deadline = time.time() + 5.0
            while time.time() < deadline:
                recs = gs.read()
                names = {r.name for r in recs}
                if any(n.startswith("f.info.") for n in names):
                    break
                time.sleep(0.2)
            names = {r.name for r in recs}
            keys = [(r.name, r.index) for r in recs]
            self.assertEqual(len(keys), len(set(keys)),
                             "duplicate (name,index) records")
            self.assertIn("g.base", names)
            self.assertIn("g.ready_go", names)
            self.assertIn("f.entry", names)
            bytes_recs = [r for r in recs if r.type == ex.T_BYTES]
            self.assertTrue(bytes_recs, "expected at least one raw blob")
            # every record must survive the wire format
            blob = ex.pack_block(recs, 2, 123)
            back = ex.unpack_block(blob)
            self.assertEqual(back["count"], len(recs))
            for r in back["records"]:
                self.assertLessEqual(len(r.name), ex.NAME_MAX)
            # at least the documented, disassembly-proven fields are present
            self.assertTrue(any(r.name.startswith("f.damage.") for r in recs))
            if not any(r.name.startswith("f.info.") for r in recs):
                if os.environ.get("SSBU_EXTRA_LIVE") == "1":
                    self.fail("no FighterInformation records in a live match")
                self.skipTest("fighter entries vanished during the read")
        finally:
            gs.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
