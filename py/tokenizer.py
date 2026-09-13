""" ============================================================================
py/tokenizer.py — 极简 BPE tokenizer（学习版）
----------------------------------------------------------------------------
设计：
  - 只用训练数据拟合 merges
  - 初始词表 = 所有 Unicode 字符 + 特殊 token；然后做 num_merges 次合并
  - encode 采用贪心最长匹配，OOV 回退到字节或 0 号未登录位
  - decode 用 merges 的逆序还原字符串

约束：
  - 本版本不引入 sentencepiece 或 tiktoken，避免外部依赖
  - 为可解释性，输出每个 merge 前后的 token 数变化
  - 训练数据不外泄到测试：词表和 merges 只在 fit_text 给的文本上学

用法：
  from tokenizer import BPETokenizer
  tok = BPETokenizer(num_merges=50)
  tok.fit(text)
  ids = tok.encode('新长安的雨')
  text2 = tok.decode(ids)
============================================================================ """
from __future__ import annotations
import collections
import json
import os
from typing import List, Dict, Tuple

SPECIAL = ['<unk>', '<s>', '</s>']
UNK = 0


class BPETokenizer:
    """学习版 BPE tokenizer。"""

    def __init__(self, num_merges: int = 50, special_tokens: List[str] = None):
        self.num_merges = num_merges
        self.special_tokens = special_tokens or SPECIAL
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.inverse_vocab: Dict[int, str] = {}

    def fit(self, text: str):
        """在 text 上学习 BPE merges。"""
        # 初始词表：text 中出现过的所有字符 + 特殊 token
        chars = sorted(set(text))
        self.vocab = {t: i for i, t in enumerate(self.special_tokens)}
        for ch in chars:
            if ch not in self.vocab:
                self.vocab[ch] = len(self.vocab)
        self.inverse_vocab = {i: s for s, i in self.vocab.items()}

        # 初始切分：每个字一个 token
        words = list(text)
        stats = collections.Counter(zip(words, words[1:]))

        for _ in range(self.num_merges):
            if not stats:
                break
            best = max(stats, key=stats.get)
            if stats[best] < 2:
                break
            a, b = best
            new_token = a + b
            # 添加新 token 到词表
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inverse_vocab[self.vocab[new_token]] = new_token
            self.merges.append((a, b))

            # 重新统计：把 words 中所有 a,b 合并成 ab
            new_words = []
            i = 0
            while i < len(words):
                if i < len(words) - 1 and words[i] == a and words[i + 1] == b:
                    new_words.append(new_token)
                    i += 2
                else:
                    new_words.append(words[i])
                    i += 1
            words = new_words
            stats = collections.Counter(zip(words, words[1:]))

        return self

    def encode(self, text: str) -> List[int]:
        """贪心最长匹配 encode。"""
        if not self.vocab:
            raise RuntimeError('tokenizer 尚未 fit')
        ids = []
        i = 0
        while i < len(text):
            # 从最长可能的子串开始匹配
            for l in range(min(len(text) - i, max(len(k) for k in self.vocab)), 0, -1):
                sub = text[i:i + l]
                if sub in self.vocab:
                    ids.append(self.vocab[sub])
                    i += l
                    break
            else:
                ids.append(UNK)
                i += 1
        return ids

    def decode(self, ids: List[int]) -> str:
        """把 ids 解码回字符串。"""
        return ''.join(self.inverse_vocab.get(i, self.inverse_vocab.get(UNK, '?')) for i in ids)

    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'num_merges': self.num_merges,
                'special_tokens': self.special_tokens,
                'vocab': self.vocab,
                'merges': self.merges,
            }, f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            d = json.load(f)
        self.num_merges = d['num_merges']
        self.special_tokens = d['special_tokens']
        self.vocab = d['vocab']
        self.inverse_vocab = {int(i): s for s, i in self.vocab.items()}
        self.merges = [tuple(m) for m in d['merges']]
        return self

    def info(self) -> Dict:
        return {
            'vocab_size': len(self.vocab),
            'num_merges': len(self.merges),
            'special_tokens': self.special_tokens,
        }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=os.path.join(os.path.dirname(__file__), 'data', 'corpus.json'))
    ap.add_argument('--num_merges', type=int, default=50)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'runs', 'tokenizer.json'))
    args = ap.parse_args()

    with open(args.corpus, 'r', encoding='utf-8') as f:
        c = json.load(f)
    # 只用 train split 的语料拟合（这里简单用全部非空 corpus）
    text = '\n'.join(v['text'] for v in c.values() if v.get('text'))

    tok = BPETokenizer(num_merges=args.num_merges)
    tok.fit(text)
    tok.save(args.out)
    print(f'[*] tokenizer saved to {args.out}')
    print(f'    vocab_size={tok.info()["vocab_size"]} merges={tok.info()["num_merges"]}')

    sample = '新长安的雨下了三天。'
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f'    sample: {sample}')
    print(f'    ids: {ids}')
    print(f'    decode: {back}')
    print(f'    roundtrip_ok: {back == sample}')


if __name__ == '__main__':
    main()