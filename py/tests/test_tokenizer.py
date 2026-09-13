"""py/tests/test_tokenizer.py — BPE tokenizer 基础断言"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from tokenizer import BPETokenizer  # noqa


class TestBPETokenizer(unittest.TestCase):
    def test_char_roundtrip(self):
        tok = BPETokenizer(num_merges=0)
        tok.fit('甲乙丙丁')
        ids = tok.encode('甲乙丙丁')
        self.assertEqual(tok.decode(ids), '甲乙丙丁')

    def test_chinese_common(self):
        text = '新长安的雨下了三天。城市上空的浮岛在雨幕里。'
        tok = BPETokenizer(num_merges=20)
        tok.fit(text)
        ids = tok.encode('新长安的雨')
        self.assertGreater(len(ids), 0)
        self.assertEqual(tok.decode(ids), '新长安的雨')

    def test_emoji_and_whitespace(self):
        text = '她笑了😊，然后  走开了。'
        tok = BPETokenizer(num_merges=0)
        tok.fit(text)
        ids = tok.encode(text)
        # emoji 应被拆成 surrogate，或作为独立 token；关键是不崩溃
        self.assertGreater(len(ids), 0)
        decoded = tok.decode(ids)
        # 由于 Python 字符串按 code point 处理，emoji 本身可以往返
        self.assertEqual(decoded, text)

    def test_oov_unknown(self):
        tok = BPETokenizer(num_merges=0)
        tok.fit('abc')
        # 用未见过的字符编码
        ids = tok.encode('xyz')
        self.assertEqual(len(ids), 3)
        self.assertEqual(tok.decode(ids), '<unk><unk><unk>')  # 三个 UNK


if __name__ == '__main__':
    unittest.main()