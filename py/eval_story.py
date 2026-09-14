""" ============================================================================
py/eval_story.py — 场景写作评测（重写版：可复现 + 留出作品出题 + 指标不送分）
----------------------------------------------------------------------------
修复前的四个硬伤（本次自检发现，逐条对应下面的实现）：
  1. 默认提示词文件 eval/prompts.jsonl **根本不存在** → 脚本开箱即不能跑；
     而且提示词从哪来、有没有被训练过，报告里说不清（可能自问自答）。
     → 现在：提示词默认从**留出作品**（val/test split）自动生成，逐条记录
       story_id/title/字符偏移，写进输出，任何人可以回到原文核对。
  2. `torch.multinomial` 采样解码但**不设种子** → 同一 checkpoint 两次生成不同，
     评测结果无法复核。
     → 现在：`--seed` 统一驱动采样（py/repro.py），输出里记录 seed 与解码参数。
  3. 自动指标算在 `prompt + 生成` 上 → 「事实检查/事件完成度」被提示词本身送分
     （提示词里出现的关键词必然命中）；`generated_chars` 也把提示词算了进去。
     → 现在：指标只在**生成段**上算；同时把「旧口径（含提示词）」也算一遍，
       两个数字并排输出，让送分幅度可见（诚实展示旧口径虚高多少）。
  4. 没有任何基线对照 → 看不出「重复率高」到底是模型烂还是中文本来就这样。
     → 现在：`--baseline trigram` 用训练集的 trigram 采样续写，同一套解码参数、
       同一套指标、同一个种子，与模型并排。

用法：
  python py/eval_story.py --checkpoint py/runs/loop/story_s1234/checkpoint.pt \
      --bucket story --n_prompts 6 --seed 1234 --out py/runs/story_eval.json
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, decode, encode, build_vocab  # noqa
from repro import set_global_seed  # noqa
import splits as S  # noqa

UNK = 0


# ---------------------------- 出题（留出作品） ----------------------------

def make_prompts(bucket: str, n: int, prompt_chars: int, manifest: str = None,
                 report: str = None, split: str = 'test') -> List[Dict]:
    """从**留出作品**开头取提示词，并记录 provenance（story_id/标题/偏移）。

    刻意不用训练集作品：用训练过的文本出题等于让模型默写，重复率/新颖度全部失真。
    """
    kw = {}
    if manifest:
        kw['manifest_path'] = manifest
    if report:
        kw['report_path'] = report
    works = S.bucket_texts(bucket, **kw)[split]
    if not works:
        raise ValueError(f'桶 {bucket} 的 {split} split 没有作品（无法出题）')
    prompts = []
    for sid, title, txt in works[:n]:
        head = txt.strip()
        if len(head) < prompt_chars:
            continue
        prompts.append({'prompt_id': f'{sid}', 'corpus': bucket, 'scenario': 'heldout_prefix',
                        'story_id': sid, 'title': title, 'split': split, 'offset': 0,
                        'prompt': head[:prompt_chars], 'facts': {}, 'events': []})
    return prompts


def load_prompts_file(path: str) -> List[Dict]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------- 两种「笔」 ----------------------------

def sample_next(logits: torch.Tensor, ids: List[int], temp: float, topk: int,
                rep_pen: float) -> int:
    """统一的采样逻辑（模型与 trigram 基线共用，保证可比）。"""
    logit = logits.clone() / max(1e-6, temp)
    counts = Counter(ids)
    for tok, cnt in counts.items():
        if cnt > 0:
            logit[tok] -= math.log(rep_pen) * (cnt ** 0.5)
    vals, idxs = torch.topk(logit, min(topk, logit.size(0)))
    probs = torch.softmax(vals, dim=0)
    return int(idxs[torch.multinomial(probs, 1).item()].item())


def generate_model(model, prompt_ids: List[int], max_new: int, temp: float,
                   topk: int, rep_pen: float) -> List[int]:
    ids = list(prompt_ids)
    ctx = model.cfg['ctx']
    model.eval()
    with torch.no_grad():
        for _ in range(max_new):
            chunk = torch.tensor([ids[-ctx:]], dtype=torch.long)
            logits, _ = model(chunk)
            nxt = sample_next(logits[0, -1, :], ids, temp, topk, rep_pen)
            ids.append(nxt)
            if nxt == UNK:
                break
    # 只返回**新生成**的部分：修复前这里返回了含提示词的整段，调用方又拼了一次提示词
    return ids[len(prompt_ids):]


class TrigramGen:
    """训练集 trigram 采样基线（含 bigram/unigram 回退），与模型同一套解码参数。"""

    def __init__(self, train_ids: List[int], V: int):
        self.V = V
        self.tri = defaultdict(Counter)
        self.bi = defaultdict(Counter)
        self.uni = Counter()
        for a, b, c in zip(train_ids, train_ids[1:], train_ids[2:]):
            self.tri[(a, b)][c] += 1
            self.bi[b][c] += 1
        self.uni.update(train_ids)

    def logits(self, ids: List[int]) -> torch.Tensor:
        lg = torch.full((self.V,), -20.0)
        if len(ids) >= 2:
            c = self.tri.get((ids[-2], ids[-1]))
            if c:
                for t, n in c.items():
                    lg[t] = math.log(n)
                return lg
            c = self.bi.get(ids[-1])
            if c:
                for t, n in c.items():
                    lg[t] = math.log(n)
                return lg
        for t, n in self.uni.items():
            lg[t] = math.log(n)
        return lg

    def generate(self, prompt_ids: List[int], max_new: int, temp: float,
                 topk: int, rep_pen: float) -> List[int]:
        ids = list(prompt_ids)
        for _ in range(max_new):
            nxt = sample_next(self.logits(ids), ids, temp, topk, rep_pen)
            ids.append(nxt)
            if nxt == UNK:
                break
        return ids[len(prompt_ids):]


# ---------------------------- 指标（只算生成段） ----------------------------

def repetition_stats(text: str) -> Dict[str, float]:
    chars = list(text)
    n = len(chars)
    if n < 4:
        return {'dup4_ratio': 0.0, 'longest_repeat': 1 if n else 0}
    seen, dup = set(), 0
    for i in range(n - 3):
        g = ''.join(chars[i:i + 4])
        if g in seen:
            dup += 1
        seen.add(g)
    longest = cur = 1
    for i in range(1, n):
        if chars[i] == chars[i - 1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return {'dup4_ratio': dup / max(1, n - 3), 'longest_repeat': longest}


def novelty4(text: str, train_ngrams: set) -> float:
    """4-gram 新颖度：生成文本里有多少比例的 4-gram 在训练集里**没出现过**。
    这是「抄训练集」的直接度量（不是原创性判断，别过度解读）。"""
    chars = list(text)
    if len(chars) < 4:
        return 0.0
    grams = [''.join(chars[i:i + 4]) for i in range(len(chars) - 3)]
    if not grams:
        return 0.0
    return sum(1 for g in grams if g not in train_ngrams) / len(grams)


def train_ngrams(train_text: str) -> set:
    return {train_text[i:i + 4] for i in range(len(train_text) - 3)}


def keyword_checks(text: str, items: Dict | List) -> Dict:
    """提示词里给定的事实/事件关键词是否出现在 text 中（**在生成段上算**）。"""
    out = {}
    if isinstance(items, dict):
        for name in items:
            out[name] = name in text
    else:
        for ev in items:
            keys = re.findall(r'[\u4e00-\u9fff]{2,4}', ev)
            out[ev] = any(k in text for k in keys[:2])
    return out


def report_of(gen_ids: List[int], itos: List[str], train_ng: set) -> Dict:
    text = decode(gen_ids, itos)
    real = [i for i in gen_ids if i != UNK]
    return {
        'text': text,
        'generated_chars': len(text),
        'oov_unk_ratio': (len(gen_ids) - len(real)) / max(1, len(gen_ids)),
        **repetition_stats(text),
        'novelty4': novelty4(text, train_ng),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--bucket', default='story', help='出题的桶（提示词取自该桶留出作品）')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--prompts', default=None,
                    help='人工提示词 JSONL（可选）。给了就用它，逐行需含 prompt/prompt_id；'
                         '默认从留出作品自动出题')
    ap.add_argument('--n_prompts', type=int, default=6)
    ap.add_argument('--prompt_chars', type=int, default=40)
    ap.add_argument('--max_new', type=int, default=120)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--rep_pen', type=float, default=1.15)
    ap.add_argument('--seed', type=int, default=1234, help='采样种子（生成可复现的关键）')
    ap.add_argument('--baseline', default='trigram', choices=['trigram', 'none'])
    ap.add_argument('--out', default=os.path.join(HERE, 'runs', 'story_eval.json'))
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    itos, meta = ckpt['itos'], ckpt['meta']
    cfg = meta['cfg']
    stoi = {ch: i for i, ch in enumerate(itos)}
    model = MiniLM(len(itos), cfg)
    model.load_state_dict(ckpt['model'])

    # 训练语料：既用于 trigram 基线，也用于「新颖度」的 4-gram 参考集
    train_kw = {}
    if args.manifest:
        train_kw['manifest_path'] = args.manifest
    if args.report:
        train_kw['report_path'] = args.report
    bucket = meta.get('bucket') or args.bucket
    train_text = S.bucket_joined(bucket, **train_kw)['train']
    if meta.get('chars'):
        train_text = train_text[:int(meta['chars'])]
    train_ids = encode(train_text, stoi)
    train_ng = train_ngrams(train_text[:400000])

    prompts = (load_prompts_file(args.prompts) if args.prompts
               else make_prompts(bucket, args.n_prompts, args.prompt_chars,
                                 args.manifest, args.report))
    print(f'[*] checkpoint={os.path.basename(os.path.dirname(args.checkpoint))} '
          f'bucket={bucket} 提示词={len(prompts)} 条（来源：'
          f'{"人工文件" if args.prompts else "留出作品"}）'
          f' seed={args.seed} temp={args.temp} topk={args.topk} rep_pen={args.rep_pen}')
    print(f'[*] 训练语料 {len(train_text):,} 字，trigram 基线='
          f'{"开" if args.baseline == "trigram" else "关"}')

    tri = TrigramGen(train_ids, len(itos)) if args.baseline == 'trigram' else None

    results = []
    for p in prompts:
        set_global_seed(args.seed)                       # 每条前重置：同一 seed 可复现
        pids = encode(p['prompt'], stoi)
        gen_model = report_of(generate_model(model, pids, args.max_new, args.temp,
                                            args.topk, args.rep_pen), itos, train_ng)
        row = {'prompt_id': p['prompt_id'], 'corpus': p.get('corpus'), 'scenario': p.get('scenario'),
               'story_id': p.get('story_id'), 'title': p.get('title'),
               'split': p.get('split'), 'prompt': p['prompt'],
               'model': gen_model, 'human_score': {}}

        # 旧口径对照：把提示词拼进来再算一遍同样的指标（修复前就是这么算的）
        old_text = p['prompt'] + gen_model['text']
        row['model_old_protocol'] = {                          # 仅作对照，不要引用它做结论
            'note': '把提示词也算进去的旧口径，事实检查会被提示词送分',
            'generated_chars': len(old_text),
            **repetition_stats(old_text),
            'novelty4': novelty4(old_text, train_ng),
            'fact_check': keyword_checks(old_text, p.get('facts', {})),
            'fact_check_gen_only': keyword_checks(gen_model['text'], p.get('facts', {})),
            'event_completion': keyword_checks(old_text, p.get('events', [])),
            'event_completion_gen_only': keyword_checks(gen_model['text'], p.get('events', [])),
        }
        if tri is not None:
            set_global_seed(args.seed)                   # 同一 seed：两次采样起点一致
            row['baseline_trigram'] = report_of(
                tri.generate(pids, args.max_new, args.temp, args.topk, args.rep_pen),
                itos, train_ng)
        results.append(row)

    def agg(key, field):
        vals = [r[key][field] for r in results if key in r]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        'n_prompts': len(results),
        'model': {k: agg('model', k) for k in
                  ('generated_chars', 'dup4_ratio', 'longest_repeat', 'novelty4', 'oov_unk_ratio')},
    }
    if tri is not None:
        summary['baseline_trigram'] = {k: agg('baseline_trigram', k) for k in
                                       ('generated_chars', 'dup4_ratio', 'longest_repeat',
                                        'novelty4', 'oov_unk_ratio')}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'checkpoint': args.checkpoint,
                'bucket': bucket,
                'prompt_source': 'human_file' if args.prompts else f'heldout_{bucket}_test',
                'model_meta': {k: meta.get(k) for k in ('cfg', 'seed', 'bucket', 'chars', 'steps')},
                'generation': {'max_new': args.max_new, 'temp': args.temp, 'topk': args.topk,
                               'rep_pen': args.rep_pen, 'seed': args.seed},
                'metrics_scope': 'generated_span_only（旧口径对照另存 model_old_protocol）',
                'baseline': args.baseline,
            },
            'summary': summary,
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    print(f'\n== 汇总（{len(results)} 条，指标只在生成段上算）==')
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f'   {k}: ' + ' '.join(f'{kk}={vv}' for kk, vv in v.items()))
        else:
            print(f'   {k}: {v}')
    print(f'\n[*] 结果 -> {args.out}')


if __name__ == '__main__':
    main()
