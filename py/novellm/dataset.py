""" ============================================================================
py/novellm/dataset.py — 预编码分片数据集（V3 P1；D1 全量训练的准入条件）
----------------------------------------------------------------------------
为什么必须有它（2026-09-16 实测）：
  用户批准 localugc 上云全量训练后，发现现行 `train.py` 会把 18.8 亿字一次性
  `encode()` 成 **Python list**——指针 14GB + int 对象 ≈49GB 内存，物理上不可行，
  且 Python 对象开销会让取 batch 也慢。正解就是预编码分片：

  1) **流式建词表**：只扫训练作品的字符频次（Counter 分块累计），词表只来自 train；
  2) **向量化编码**：utf-32 解出码点数组 → 查表（LUT）映射 token，GB/s 级，不跑 Python 逐字循环；
  3) **uint16 分片 + memmap**：词表 <65536 时每 token 仅 2 字节（24.5 亿字 ≈4.6GiB 落盘，
     随机取 batch 不占大内存）；文档边界单独记成 int64 偏移数组，不丢作品身份；
  4) **自带身份**：meta.json 记 manifest 指纹 / 词表指纹 / token 数，与本地/远端同一套契约。

编码语义与 `py/minilm_torch.py` 完全一致（unk→0、targets 右移一位、随机窗口起点
∈[0, len-T-1)），保证既有实验结论可迁移；novellm 的 BPE tokenizer 就绪后只替换编码函数，
本模块结构不变。

用法：
  python -X utf8 -m py.novellm.dataset build --split train
  python -X utf8 -m py.novellm.dataset info  --split train
  python -X utf8 -m py.novellm.dataset verify --split train   # 逐篇往返抽查
  from novellm.dataset import ShardedCorpus
============================================================================ """
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import sys
from typing import Dict, Iterator, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = os.path.dirname(HERE)
if PY not in sys.path:
    sys.path.insert(0, PY)

import splits  # noqa: E402

UNK = 0
SHARD_DIR = os.path.join(PY, 'runs', 'shards')      # 大文件：与其它数据产物一样不入库
DEFAULT_SHARD_TOKENS = 256 * 1024 * 1024            # 2.68 亿 token/片（uint16 ≈512MB/片）
READ_CHUNK = 4 * 1024 * 1024                        # 流式读文件的字符块


class DatasetError(RuntimeError):
    pass


# ---------------------------- 词表与流式读取 ----------------------------

def iter_split_texts(split: str, buckets: List[str] = None,
                     manifest_path: str = None,
                     processed_dir: str = None) -> Iterator[Tuple[str, str]]:
    """流式产出某 split 的作品正文（story_id, text）。按 story_id 稳定排序，可复现；
    一次只持有一篇正文，不把语料常驻内存。"""
    man, _rep = splits.resolve_current(manifest_path, None)
    rows = [r for r in splits.read_manifest(man)
            if r['split'] == split and (not buckets or r['bucket'] in buckets)]
    rows.sort(key=lambda r: r['story_id'])
    proc = processed_dir or splits.DEFAULT_PROCESSED
    for r in rows:
        p = os.path.join(proc, r['bucket'], splits.slug(r['title']) + '.txt')
        if not os.path.exists(p):
            raise FileNotFoundError(f'作品正文缺失：{p}（{r["story_id"]} {r["title"]}）')
        with open(p, 'r', encoding='utf-8') as f:
            yield r['story_id'], f.read()


def fit_char_vocab(split: str = 'train', min_freq: int = 3, buckets: List[str] = None,
                   manifest_path: str = None, processed_dir: str = None
                   ) -> Tuple[List[str], Dict[str, int]]:
    """流式统计训练集字符频次建词表。语义对齐 minilm_torch.build_vocab：
    频次 <min_freq → UNK=0；同频按字符稳定排序。词表**只来自 train**（评测协议 V1）。"""
    counter: collections.Counter = collections.Counter()
    for _sid, text in iter_split_texts(split, buckets, manifest_path, processed_dir):
        for i in range(0, len(text), READ_CHUNK):
            counter.update(text[i:i + READ_CHUNK])
    itos = ['', *(ch for ch, c in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
                  if c >= min_freq)]
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def vocab_fingerprint(itos: List[str]) -> str:
    return hashlib.sha256('\n'.join(itos).encode('utf-8')).hexdigest()[:16]


def _codepoints(text: str) -> np.ndarray:
    """utf-32 小端直接解出码点数组（无 Python 逐字循环）。"""
    return np.frombuffer(text.encode('utf-32-le'), dtype='<u4')


class CharEncoder:
    """码点 → token 查表（LUT 常驻 ≈2.2MB，与词表大小无关的常数开销）。"""

    def __init__(self, stoi: Dict[str, int]):
        lut = np.zeros(0x110000, dtype=np.uint16)
        for ch, i in stoi.items():
            if not ch:
                continue          # UNK 位是空串；LUT 已全零，未知码点天然映射到 0
            cp = ord(ch)
            if cp < 0x110000:
                lut[cp] = i
        self.lut = lut
        self.stoi = stoi

    def encode(self, text: str) -> np.ndarray:
        return self.lut[_codepoints(text)]


# ---------------------------- 分片构建 ----------------------------

def build_shards(split: str, itos: List[str], out_dir: str,
                 shard_tokens: int = DEFAULT_SHARD_TOKENS,
                 buckets: List[str] = None, manifest_path: str = None,
                 report_path: str = None, processed_dir: str = None) -> dict:
    """把某 split 的全部作品预编码成 uint16 分片 + 文档边界，落 out_dir/<split>_*。"""
    man, rep = splits.resolve_current(manifest_path, report_path)
    v = splits.verify(man, rep)                     # 数据身份不过关就地终止
    enc = CharEncoder({ch: i for i, ch in enumerate(itos)})
    os.makedirs(out_dir, exist_ok=True)

    shard_files: List[str] = []
    doc_offsets: List[int] = [0]                    # 只记**文档**边界；分片边界不是文档边界
    n_tokens = 0
    n_docs = 0
    buf = np.empty(shard_tokens, dtype=np.uint16)
    filled = 0
    idx = 0

    def flush():
        nonlocal idx, filled
        if filled == 0:
            return
        name = f'{split}_{idx:04d}.bin'
        with open(os.path.join(out_dir, name), 'wb') as f:
            f.write(buf[:filled].tobytes())
        shard_files.append(name)
        idx += 1
        filled = 0

    for _sid, text in iter_split_texts(split, buckets, man, processed_dir):
        toks = enc.encode(text)
        n = len(toks)
        pos = 0
        while pos < n:
            take = min(shard_tokens - filled, n - pos)
            buf[filled:filled + take] = toks[pos:pos + take]
            filled += take
            pos += take
            n_tokens += take
            if filled == shard_tokens:
                flush()
        doc_offsets.append(n_tokens)               # 文档边界（全局 token 偏移，单调递增）
        n_docs += 1
    flush()

    meta = {
        'generated_by': 'py/novellm/dataset.py build',
        'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'split': split,
        'buckets': sorted(buckets) if buckets else 'ALL',
        'source_manifest': os.path.basename(man),
        'source_split_fingerprint': v['fingerprint'],
        'vocab_fingerprint': vocab_fingerprint(itos),
        'vocab_size': len(itos),
        'tokens_dtype': 'uint16',
        'shard_tokens': shard_tokens,
        'shards': shard_files,
        'n_tokens': n_tokens,
        'n_docs': n_docs,
    }
    with open(os.path.join(out_dir, f'{split}_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, f'{split}_itos.json'), 'w', encoding='utf-8') as f:
        json.dump(itos, f, ensure_ascii=False)
    np.asarray(doc_offsets, dtype=np.int64).tofile(
        os.path.join(out_dir, f'{split}_doc_offsets.bin'))
    return meta


# ---------------------------- 读取侧 ----------------------------

class ShardedCorpus:
    """memmap 读取预编码分片。sample_batch 与 minilm_torch.make_batch 同语义
    （随机窗口起点 ∈[0, len-T-1)、targets 右移一位），但 O(1) 常驻内存——
    语料不进 Python list，随取随映。"""

    def __init__(self, shard_dir: str, split: str = 'train'):
        meta_path = os.path.join(shard_dir, f'{split}_meta.json')
        if not os.path.exists(meta_path):
            raise DatasetError(f'缺 {meta_path}（先跑 python -X utf8 -m py.novellm.dataset build）')
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.split = split
        self.shard_dir = shard_dir
        self.shard_tokens = self.meta['shard_tokens']
        self._files = [np.memmap(os.path.join(shard_dir, name), dtype='<u2', mode='r')
                       for name in self.meta['shards']]
        self.doc_offsets = np.fromfile(os.path.join(shard_dir, f'{split}_doc_offsets.bin'),
                                       dtype=np.int64)
        self.n_tokens = int(self.meta['n_tokens'])
        if sum(f.shape[0] for f in self._files) != self.n_tokens:
            raise DatasetError('分片长度与 meta 不符（文件缺失或被改动）')
        with open(os.path.join(shard_dir, f'{split}_itos.json'), 'r', encoding='utf-8') as f:
            self.itos: List[str] = json.load(f)

    def _locate(self, pos: int) -> Tuple[int, int]:
        acc = 0
        for fi, f in enumerate(self._files):
            if pos < acc + f.shape[0]:
                return fi, pos - acc
            acc += f.shape[0]
        raise IndexError(f'位置越界：{pos} ≥ {self.n_tokens}')

    def get_slice(self, start: int, length: int) -> np.ndarray:
        """取 [start, start+length) 的 token，跨分片透明拼接。"""
        if start < 0 or start + length > self.n_tokens:
            raise IndexError(f'越界：start={start} length={length} n={self.n_tokens}')
        out = np.empty(length, dtype=np.uint16)
        taken = 0
        while taken < length:
            fi, off = self._locate(start + taken)
            take = min(self._files[fi].shape[0] - off, length - taken)
            out[taken:taken + take] = self._files[fi][off:off + take]
            taken += take
        return out

    def get_window(self, start: int, ctx: int) -> np.ndarray:
        return self.get_slice(start, ctx + 1)

    def sample_batch(self, batch: int, ctx: int, rng: random.Random):
        """与 make_batch 同语义；语料太短同样报错。返回 torch long 的 (tokens, targets)。"""
        import torch
        max_start = self.n_tokens - ctx - 1
        if max_start <= 0:
            raise ValueError('语料太短，请增加文本长度或减小上下文窗口')
        x = np.empty((batch, ctx), dtype=np.int64)
        y = np.empty((batch, ctx), dtype=np.int64)
        for b in range(batch):
            w = self.get_window(rng.randrange(max_start), ctx).astype(np.int64)
            x[b], y[b] = w[:ctx], w[1:ctx + 1]
        return (torch.from_numpy(x), torch.from_numpy(y))

    def iter_windows(self, ctx: int, max_windows: int = None) -> Iterator[np.ndarray]:
        """顺序滑窗（验证用；与 holdout_eval 分窗方式一致：每 ctx 步一窗）。"""
        s = 0
        emitted = 0
        while s + ctx + 1 <= self.n_tokens:
            yield self.get_window(s, ctx)
            emitted += 1
            s += ctx
            if max_windows and emitted >= max_windows:
                break

    def decode(self, ids: np.ndarray) -> str:
        return ''.join(self.itos[int(i)] for i in ids)


# ---------------------------- CLI ----------------------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--split', default='train')
    b.add_argument('--buckets', default=None)
    b.add_argument('--shard-tokens', type=int, default=DEFAULT_SHARD_TOKENS)
    b.add_argument('--min-freq', type=int, default=3)
    b.add_argument('--out-dir', default=SHARD_DIR)
    i = sub.add_parser('info')
    i.add_argument('--split', default='train')
    i.add_argument('--dir', default=SHARD_DIR)
    v = sub.add_parser('verify')
    v.add_argument('--split', default='train')
    v.add_argument('--dir', default=SHARD_DIR)
    v.add_argument('--spot', type=int, default=3, help='往返抽查的文档数')
    args = ap.parse_args()

    if args.cmd == 'build':
        buckets = [x.strip() for x in args.buckets.split(',')] if args.buckets else None
        print(f'[*] 流式拟合词表（split={args.split}, min_freq={args.min_freq}）…')
        itos, _stoi = fit_char_vocab(args.split, args.min_freq, buckets)
        print(f'    V={len(itos)}（词表指纹 {vocab_fingerprint(itos)}）')
        meta = build_shards(args.split, itos, args.out_dir, args.shard_tokens, buckets)
        print(f'[*] 分片完成：{meta["n_tokens"]:,} tokens / {meta["n_docs"]} 部 → '
              f'{len(meta["shards"])} 片 @ {args.out_dir}')
        print(f'    磁盘占用 ≈ {meta["n_tokens"] * 2 / 2**30:.2f} GiB（uint16）')
        return 0

    if args.cmd == 'info':
        c = ShardedCorpus(args.dir, args.split)
        print(json.dumps({k: c.meta[k] for k in
                          ('split', 'n_tokens', 'n_docs', 'shard_tokens',
                           'vocab_size', 'source_split_fingerprint')},
                         ensure_ascii=False, indent=2))
        return 0

    if args.cmd == 'verify':
        c = ShardedCorpus(args.dir, args.split)
        offs = c.doc_offsets
        rng = random.Random(20260916)
        n_docs = c.meta['n_docs']
        picks = sorted(rng.sample(range(n_docs), min(args.spot, n_docs)))
        picked = set(picks)
        # 与 build 完全相同的桶过滤 + 排序 → 第 k 篇正文对应第 k 个文档边界。
        # （verify 的桶列表必须从 meta 读：build 时可能只选了部分桶，不过滤就会错位——实测踩过）
        mb = c.meta['buckets']
        build_buckets = None if mb == 'ALL' else mb
        texts = (t for i, (_sid, t) in
                 enumerate(iter_split_texts(args.split, build_buckets, None)) if i in picked)
        for k in picks:
            text = next(texts)
            start, end = int(offs[k]), int(offs[k + 1])
            stored = c.get_slice(start, end - start)
            # 正确的断言是 **token 级等值**：min_freq>1 时低频字按设计映射为 UNK(0)，
            # 字符串逐字相等反而不该成立（UNK 解码为空）。token 级相等 = 分片没丢/没错位。
            # 注意：成员判断的集合叫 picked、重编码结果叫 want_tokens——
            # 曾经同名（want），生成器迟绑定把第二个文档起的对账集合换成了 token 数组（实测踩过）。
            want_tokens = CharEncoder({ch: i for i, ch in enumerate(c.itos)}
                                      ).encode(text).astype(np.uint16)
            if not np.array_equal(stored, want_tokens):
                d = np.flatnonzero(stored != want_tokens)[:5]
                raise DatasetError(f'token 级往返不一致（文档 {k}，首差异位 {d.tolist()}）：'
                                   f'存储 {stored[d].tolist()} vs 重编码 {want_tokens[d].tolist()}')
            unk = int((stored == 0).sum())
            print(f'  ✓ 文档 {k} token 级一致（{len(text)} 字，UNK {unk}）')
        print(f'[*] 分片校验通过：{c.n_tokens:,} tokens / {c.meta["n_docs"]} 部 / '
              f'{len(c._files)} 片；抽查 {len(picks)} 篇 token 级一致'
              f'（UNK 为 min_freq 过滤所致，与既有静默丢弃口径一致）')
        return 0
    return 2


if __name__ == '__main__':
    sys.exit(main())
