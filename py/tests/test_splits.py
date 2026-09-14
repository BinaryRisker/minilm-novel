""" ============================================================================
py/tests/test_splits.py — 切分身份校验的契约测试（V3 P0 出口条件）
----------------------------------------------------------------------------
V3 P0 要求「不合规数据和不匹配缓存非零退出」。本测试盯两件事：
  1) split 版本：manifest 变了但 data/split_report.json 没重建 → 必须报错
     （实测踩过：去重/抓取重跑后 manifest 是新的，data/splits/*.txt 是旧的，
      实验在旧切分上跑出不可复现的数字，却毫无提示）
  2) 家族/内容去重：manifest 里有逐字重复作品 → 必须报错
     （重复既虚增数据量，又可能把同一篇正文的两份切进不同 split 形成泄漏）

运行：
  python -m unittest py.tests.test_splits
============================================================================ """
from __future__ import annotations
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

from splits import (SplitIdentityError, duplicate_hashes, split_fingerprint,  # noqa
                    verify)


def work(sid, h, sp='train', bucket='b', title=None):
    return {'story_id': sid, 'hash': h, 'split': sp, 'bucket': bucket,
            'title': title or sid}


class TestFingerprint(unittest.TestCase):
    def test_稳定且与顺序无关(self):
        rows = [work('a_1', 'h1'), work('a_2', 'h2', 'test')]
        self.assertEqual(split_fingerprint(rows), split_fingerprint(list(reversed(rows))))

    def test_改一个split就变指纹(self):
        base = [work('a_1', 'h1'), work('a_2', 'h2')]
        moved = [work('a_1', 'h1'), work('a_2', 'h2', 'test')]
        self.assertNotEqual(split_fingerprint(base), split_fingerprint(moved))

    def test_长度与字符集(self):
        fp = split_fingerprint([work('a_1', 'h1')])
        self.assertEqual(len(fp), 16)
        self.assertTrue(all(c in '0123456789abcdef' for c in fp))


class TestDuplicateHashes(unittest.TestCase):
    def test_无重复返回空(self):
        self.assertEqual(duplicate_hashes([work('a_1', 'h1'), work('a_2', 'h2')]), [])

    def test_抓到重复并指明保留谁(self):
        dups = duplicate_hashes([work('a_1', 'h1'), work('a_2', 'hX'), work('a_9', 'h1')])
        self.assertEqual(dups, [('a_9', 'a_1')])   # story_id 最小的那部被保留

    def test_无hash不参与判重(self):
        self.assertEqual(duplicate_hashes([{'story_id': 'x', 'hash': None},
                                           {'story_id': 'y', 'hash': None}]), [])


class TestVerify(unittest.TestCase):
    """用临时目录造 manifest/report，不碰仓库真实数据。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='splits_test_')
        self.man = os.path.join(self.tmp, 'manifest.jsonl')
        self.rep = os.path.join(self.tmp, 'split_report.json')
        self.rows = [work('b_1', 'h1', 'train'), work('b_2', 'h2', 'train'),
                     work('b_3', 'h3', 'test')]
        self._write_manifest(self.rows)
        self._write_report(split_fingerprint(self.rows))

    def tearDown(self):
        for p in (self.man, self.rep):
            try:
                os.remove(p)
            except OSError:
                pass
        os.rmdir(self.tmp)

    def _write_manifest(self, rows):
        with open(self.man, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    def _write_report(self, fp):
        with open(self.rep, 'w', encoding='utf-8') as f:
            json.dump({'split_fingerprint': fp, 'total_works': 3}, f)

    def test_匹配时通过并回报指纹(self):
        got = verify(self.man, self.rep)
        self.assertEqual(got['fingerprint'], split_fingerprint(self.rows))
        self.assertEqual(got['works'], 3)

    def test_报告过期必须报错(self):
        # manifest 把 b_3 挪到 val，但 report 没重建 → 正是上轮踩到的失效模式
        self.rows[2]['split'] = 'val'
        self._write_manifest(self.rows)
        with self.assertRaises(SplitIdentityError) as ctx:
            verify(self.man, self.rep)
        self.assertIn('不是同一版', str(ctx.exception))

    def test_缺报告必须报错(self):
        os.remove(self.rep)
        with self.assertRaises(SplitIdentityError) as ctx:
            verify(self.man, self.rep)
        self.assertIn('build_splits', str(ctx.exception))

    def test_缺manifest必须报错(self):
        os.remove(self.man)
        with self.assertRaises(SplitIdentityError):
            verify(self.man, self.rep)

    def test_未去重必须报错(self):
        dup = list(self.rows) + [work('b_4', 'h1', 'test')]   # b_4 与 b_1 逐字相同
        self._write_manifest(dup)
        with self.assertRaises(SplitIdentityError) as ctx:
            verify(self.man, self.rep)
        msg = str(ctx.exception)
        self.assertIn('逐字重复', msg)
        self.assertIn('b_4 vs b_1', msg)


if __name__ == '__main__':
    unittest.main(verbosity=2)
