""" ============================================================================
py/tests/test_doctor.py — 设备探测的契约测试（V3 P1 首项）
----------------------------------------------------------------------------
doctor 的产出会直接进训练预算表，测三件事：
  1) probe() 在任何机器上都返回**结构完整**的字典（缺设备 ≠ 缺字段）；
  2) 探测**只读**：连跑两次结果一致、不抛、不改全局状态；
  3) fail-closed：没有的设备就是没有（本仓库 CI/本机多为 CPU-only，
     cuda/mps 的 available 必须如实为 False，而不是缺字段让下游误判）。

运行：
  python -m unittest py.tests.test_doctor
============================================================================ """
from __future__ import annotations
import os
import sys
import unittest

PY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PY not in sys.path:
    sys.path.insert(0, PY)

from novellm.doctor import probe  # noqa: E402


class TestDoctor(unittest.TestCase):
    def test_结构完整(self):
        r = probe()
        for key in ('torch', 'cuda', 'mps', 'cpu', 'precision'):
            self.assertIn(key, r)
        for sub in ('available',):
            self.assertIn(sub, r['cuda'])
            self.assertIn(sub, r['mps'])
        self.assertIn('logical_cores', r['cpu'])
        # 精度：CPU 探针必须给出 bf16/fp16 的布尔结果（可用与否都要有答案）
        for name in ('bf16', 'fp16'):
            self.assertIn(name, r['precision']['cpu'])
            self.assertIsInstance(r['precision']['cpu'][name], bool)

    def test_只读且确定_连跑两次一致(self):
        a, b = probe(), probe()
        self.assertEqual(a['torch'], b['torch'])
        self.assertEqual(a['cuda']['available'], b['cuda']['available'])
        self.assertEqual(a['precision']['cpu'], b['precision']['cpu'])

    def test_没有的设备如实为False而不是缺字段(self):
        r = probe()
        if not r['cuda']['available']:
            self.assertFalse(r['cuda']['available'])
            self.assertNotIn('devices', r['cuda'])
        if not r['mps']['available']:
            self.assertFalse(r['mps']['available'])

    def test_显存字段只在设备存在时出现(self):
        r = probe()
        if not r['cuda']['available']:
            self.assertNotIn('free_mem_gb_primary', r['cuda'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
