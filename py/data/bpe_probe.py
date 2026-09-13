""" ============================================================================
py/data/bpe_probe.py — 字符级 vs BPE 子词的对照探针（PRD P2：tokenizer 实验）
----------------------------------------------------------------------------
回答三个问题：
  1) 压缩率：同样一段文本，BPE 的 token 数比"一字一 token"少多少
     （= 有效上下文延长多少倍）
  2) OOV 行为：留出段里训练时没见过的字，字符级的处理是**静默丢弃**
     （minilm.js encode→0、decode 跳过），BPE 也能把它们保留下来吗
  3) 往返完整性：decode(encode(x)) 是否还原原文

协议（防泄漏）：
  - merges **只用训练段**拟合（桶文本前 80%）
  - 评测在留出段（桶文本后 10%），与 py/tokenizer.py 的"训练数据不外泄"一致
  - 已知局限：桶文本是连续拼接，这里的 80/10 切分与 manifest 的按作品切分
    不是同一协议；结论只用于 tokenizer 对比，不用于模型评测

用法：
  python py/data/bpe_probe.py --bucket scifi --fit-chars 100000 --merges 300
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))

from tokenizer import BPETokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default=os.path.join(HERE, 'corpus.json'))
    ap.add_argument('--bucket', default='scifi')
    ap.add_argument('--fit-chars', type=int, default=100_000)
    ap.add_argument('--merges', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    with open(args.corpus, encoding='utf-8') as f:
        corpus = json.load(f)
    text = corpus[args.bucket]['text']
    n = len(text)
    n_fit = int(n * 0.8)
    n_val_end = int(n * 0.9)
    train_text = text[:n_fit]
    holdout = text[n_fit:n_val_end]
    print(f'桶 {args.bucket}：全文 {n:,} 字 | 训练段 {len(train_text):,} | '
          f'留出段 {len(holdout):,}')

    # ---- 字符级基线（对齐 minilm.js：词表 = 训练段字符 + \\x00，OOV 丢弃）----
    char_vocab = set(train_text)
    oov_chars = [ch for ch in set(holdout) if ch not in char_vocab]
    oov_positions = sum(1 for ch in holdout if ch not in char_vocab)
    print(f'\n[字符级] 训练段词表 {len(char_vocab)} 字')
    print(f'  留出段独立字符 {len(set(holdout))}，其中训练段未见 {len(oov_chars)} 字'
          f'：{"".join(sorted(oov_chars))[:40]!r}')
    print(f'  留出段中被静默丢弃的位置：{oov_positions:,} / {len(holdout):,} '
          f'= {100 * oov_positions / len(holdout):.2f}%')

    # ---- BPE：只在训练段拟合 ----
    print(f'\n[BPE] 在训练段前 {min(args.fit_chars, len(train_text)):,} 字上拟合 '
          f'{args.merges} 次合并…')
    fit_text = train_text[:args.fit_chars]
    t0 = time.time()
    tok = BPETokenizer(num_merges=args.merges).fit(fit_text)
    dt = time.time() - t0
    info = tok.info()
    print(f'  拟合耗时 {dt:.1f}s | 词表 {info["vocab_size"]} | 实际合并 {info["num_merges"]}')
    print('  前 12 个合并：', ' / '.join(a + b for a, b in tok.merges[:12]))

    # ---- 留出段上的 token 数与压缩率 ----
    t0 = time.time()
    ids = tok.encode(holdout)
    enc_dt = time.time() - t0
    ratio = len(ids) / len(holdout)
    print(f'\n[留出段] {len(holdout):,} 字')
    print(f'  字符级 token 数 = {len(holdout):,}（1.000 token/字）')
    print(f'  BPE token 数    = {len(ids):,}（{ratio:.3f} token/字，'
          f'压缩到 {100 * ratio:.1f}%，有效上下文 x{1 / ratio:.2f}）')
    unk = ids.count(0)
    print(f'  BPE UNK token：{unk}（{100 * unk / len(ids):.3f}%）')
    print(f'  encode 速度：{len(holdout) / max(enc_dt, 1e-9) / 1e3:.0f}K 字/s')

    # ---- 往返完整性 ----
    back = tok.decode(ids)
    exact = back == holdout
    print(f'\n[往返] BPE decode(encode(x)) == x ：{exact}')
    if not exact:
        for i, (a, b) in enumerate(zip(back, holdout)):
            if a != b:
                print(f'  首个差异 @{i}: {holdout[i:i + 20]!r} -> {back[i:i + 20]!r}')
                break
    js_back = ''.join(ch for ch in holdout if ch in char_vocab)
    print(f'[对照] 字符级（JS 行为：未见字丢弃）还原长度 {len(js_back):,} / '
          f'{len(holdout):,}（丢 {len(holdout) - len(js_back):,} 字）')

    # ---- 落盘 ----
    out_dir = os.path.join(os.path.dirname(HERE), 'runs')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'bpe_probe_{args.bucket}.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({
            'bucket': args.bucket, 'fit_chars': len(fit_text), 'merges': args.merges,
            'char_vocab': len(char_vocab), 'bpe_vocab': info['vocab_size'],
            'holdout_chars': len(holdout),
            'holdout_oov_types': len(oov_chars), 'holdout_oov_positions': oov_positions,
            'bpe_tokens': len(ids), 'ratio': ratio, 'unk': unk,
            'roundtrip_exact': exact, 'fit_seconds': round(dt, 1),
            'sample_merges': [a + b for a, b in tok.merges[:24]],
        }, f, ensure_ascii=False, indent=1)
    print(f'\n[*] 结果 -> {out}')


if __name__ == '__main__':
    main()
