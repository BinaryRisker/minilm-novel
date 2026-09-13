""" ============================================================================
py/eval_story.py — 场景写作评测
----------------------------------------------------------------------------
职责：
  - 读取 eval/prompts.jsonl 中的提示
  - 调用模型（或加载 checkpoint）为每个提示生成一段续写
  - 保存匿名样本、人物事实检查、事件完成度、重复与复制统计
  - 输出一份报告，自动指标和人工评分字段分列

用法：
  python py/eval_story.py --checkpoint py/runs/scifi_s1234_n400/checkpoint.pt \
                          --out py/runs/story_eval_scifi.json

注意：
  - 人工评分列初始为空，需要人类读者填写
  - 本脚本只做：生成 + 自动指标 + 保存便于盲评的格式
============================================================================ """
from __future__ import annotations
import argparse
import json
import math
import os
import re
import sys
from typing import Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, decode, encode, build_vocab  # noqa


def load_prompts(path: str) -> List[Dict]:
    prompts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    return prompts


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    itos = ckpt['itos']
    meta = ckpt['meta']
    cfg = meta['cfg']
    model = MiniLM(len(itos), cfg)
    model.load_state_dict(ckpt['model'])
    model.eval()
    stoi = {ch: i for i, ch in enumerate(itos)}
    return model, itos, stoi, meta


@torch.no_grad()
def generate(model, stoi, itos, prompt_text: str,
             max_new: int = 120, temp: float = 0.8, topk: int = 8,
             rep_pen: float = 1.15) -> str:
    """极简自回归生成（与 minilm.js generate 对齐核心逻辑）。"""
    ids = encode(prompt_text, stoi)
    if len(ids) > model.cfg['ctx']:
        ids = ids[-model.cfg['ctx']:]
    for _ in range(max_new):
        ctx_len = min(len(ids), model.cfg['ctx'])
        chunk = torch.tensor([ids[-ctx_len:]], dtype=torch.long)
        logits, _ = model(chunk)
        # 只取最后一个位置
        logit = logits[0, -1, :] / temp
        # 重复惩罚
        counts = {}
        for i in ids:
            counts[i] = counts.get(i, 0) + 1
        for tok, cnt in counts.items():
            if cnt > 0:
                logit[tok] -= math.log(rep_pen) * (cnt ** 0.5)
        # top-k
        vals, idxs = torch.topk(logit, min(topk, logit.size(0)))
        probs = torch.softmax(vals, dim=0)
        nxt = idxs[torch.multinomial(probs, 1).item()].item()
        ids.append(nxt)
        if nxt == 0:
            break
    return decode(ids, itos)


def fact_check(text: str, facts: Dict[str, str]) -> Dict[str, bool]:
    """极简单的事实检查：关键词是否在生成文本中出现。
    注意：这只是自动过滤，不能替代人工判断。"""
    checks = {}
    for name, desc in facts.items():
        checks[name] = name in text
    return checks


def event_completion(text: str, events: List[str]) -> Dict[str, bool]:
    """检查关键事件是否出现（关键词匹配）。"""
    out = {}
    for ev in events:
        # 取事件前 4 个汉字作为关键词
        keys = re.findall(r'[\u4e00-\u9fff]{2,4}', ev)
        out[ev] = any(k in text for k in keys[:2])
    return out


def repetition_stats(text: str) -> Dict[str, float]:
    """统计 4-gram 重复率和最长连续重复串。"""
    chars = list(text)
    n = len(chars)
    if n < 4:
        return {'dup4_ratio': 0.0, 'longest_repeat': 0}
    seen = set()
    dup = 0
    for i in range(n - 3):
        g = ''.join(chars[i:i + 4])
        if g in seen:
            dup += 1
        seen.add(g)
    # 最长连续重复：连续相同字
    longest = cur = 1
    for i in range(1, n):
        if chars[i] == chars[i - 1]:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 1
    return {
        'dup4_ratio': dup / max(1, n - 3),
        'longest_repeat': longest,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='checkpoint.pt 路径')
    ap.add_argument('--prompts', default=os.path.join(ROOT, 'eval', 'prompts.jsonl'))
    ap.add_argument('--out', default=os.path.join(HERE, 'runs', 'story_eval.json'))
    ap.add_argument('--max_new', type=int, default=120)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--rep_pen', type=float, default=1.15)
    args = ap.parse_args()

    model, itos, stoi, meta = load_model(args.checkpoint)
    prompts = load_prompts(args.prompts)

    results = []
    for p in prompts:
        text = generate(model, stoi, itos, p['prompt'],
                        max_new=args.max_new, temp=args.temp,
                        topk=args.topk, rep_pen=args.rep_pen)
        full = p['prompt'] + text
        auto = {
            'fact_check': fact_check(full, p.get('facts', {})),
            'event_completion': event_completion(full, p.get('events', [])),
            'repetition': repetition_stats(text),
            'generated_chars': len(text),
        }
        results.append({
            'prompt_id': p['prompt_id'],
            'corpus': p['corpus'],
            'scenario': p['scenario'],
            'prompt': p['prompt'],
            'generated': text,
            'auto': auto,
            'human_score': {},   # 人工填写
        })

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'checkpoint': args.checkpoint,
                'model_meta': meta,
                'generation': {
                    'max_new': args.max_new,
                    'temp': args.temp,
                    'topk': args.topk,
                    'rep_pen': args.rep_pen,
                },
            },
            'results': results,
        }, f, ensure_ascii=False, indent=2)
    print(f'[*] saved {args.out} ({len(results)} prompts)')


if __name__ == '__main__':
    main()