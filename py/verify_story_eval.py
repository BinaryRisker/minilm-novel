""" ============================================================================
py/verify_story_eval.py — 故事评测的三项可复核性验证（含旧口径送分实证）
----------------------------------------------------------------------------
验证三件事，全部用数字说话：

  T1 出题来源：提示词必须来自**留出作品**（val/test split），且逐条记录
     story_id/标题/偏移 —— 否则生成评测可能是在训练过的文本上默写。
  T2 生成可复现：同一 checkpoint + 同一 seed 两次生成**逐字相同**；
     换 seed 则不同（反证 T1 的「相同」不是因为采样退化成 argmax）。
  T3 旧口径送分：修复前把提示词拼进生成文本再算「事实检查/事件完成度」，
     提示词里的关键词必然命中 → 指标恒为满分。这里对同一批样本算两个口径，
     把送分的幅度摆出来。

用法：
  python py/verify_story_eval.py --checkpoint py/runs/loop/.../checkpoint.pt --bucket story
============================================================================ """
from __future__ import annotations
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from minilm_torch import MiniLM, encode  # noqa
from repro import set_global_seed  # noqa
import eval_story as ES  # noqa
import splits as S  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--bucket', default='story')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--n_prompts', type=int, default=3)
    ap.add_argument('--max_new', type=int, default=60)
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--out', default=os.path.join(ROOT, 'dev', 'verify', 'verify_story_eval.json'))
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    itos, meta = ck['itos'], ck['meta']
    stoi = {ch: i for i, ch in enumerate(itos)}
    model = MiniLM(len(itos), meta['cfg'])
    model.load_state_dict(ck['model'])

    kw = {}
    if args.manifest:
        kw['manifest_path'] = args.manifest
    if args.report:
        kw['report_path'] = args.report

    rep = {'checkpoint': args.checkpoint, 'bucket': args.bucket}

    # ---- T1 出题来源 ----
    prompts = ES.make_prompts(args.bucket, args.n_prompts, 40, args.manifest, args.report)
    splits = S.bucket_texts(args.bucket, **kw)
    train_ids_set = {sid for sid, _t, _x in splits['train']}
    test_ids_set = {sid for sid, _t, _x in splits['val'] + splits['test']}
    prov = [{'prompt_id': p['prompt_id'], 'title': p['title'], 'split': p['split'],
             'in_train': p['story_id'] in train_ids_set,
             'in_heldout': p['story_id'] in test_ids_set} for p in prompts]
    rep['T1_provenance'] = {'prompts': prov,
                            'all_from_heldout': all(x['in_heldout'] and not x['in_train'] for x in prov)}
    print('== T1 出题来源 ==')
    for x in prov:
        print(f'   {x["prompt_id"]}《{x["title"]}》 split={x["split"]} '
              f'在训练集里={x["in_train"]} 在留出集里={x["in_heldout"]}')
    print(f'   判定：{"全部来自留出作品 ✓" if rep["T1_provenance"]["all_from_heldout"] else "★有训练集作品★"}')

    # ---- T2 生成可复现 ----
    p0 = prompts[0]
    pids = encode(p0['prompt'], stoi)

    def gen(seed):
        set_global_seed(seed)
        return ES.generate_model(model, pids, args.max_new, 0.8, 8, 1.15)

    a1, a2, b = gen(args.seed), gen(args.seed), gen(args.seed + 1)
    txt = lambda ids: ES.decode(ids, itos)
    rep['T2_determinism'] = {
        'same_seed_identical': a1 == a2,
        'diff_seed_differs': a1 != b,
        'sample_same_seed': txt(a1)[:80],
        'sample_other_seed': txt(b)[:80],
    }
    print('\n== T2 生成可复现 ==')
    print(f'   同 seed 两次逐字相同：{a1 == a2}')
    print(f'   换 seed 后不同：{a1 != b}')
    print(f'   样本(seed={args.seed})：{txt(a1)[:60]}')
    print(f'   样本(seed={args.seed + 1})：{txt(b)[:60]}')

    # ---- T3 旧口径送分 ----
    # 构造「事实/事件关键词只出现在提示词里」的样本：模型生成段未必提到
    facts = {'莉莉': '主角名', '公园': '地点'}
    events = ['他们喜欢在公园骑自行车']
    rows = []
    for p in prompts:
        set_global_seed(args.seed)
        gids = ES.generate_model(model, encode(p['prompt'], stoi), args.max_new, 0.8, 8, 1.15)
        gen_text = txt(gids)
        old_text = p['prompt'] + gen_text
        rows.append({
            'prompt_id': p['prompt_id'], 'prompt': p['prompt'],
            'gen': gen_text,
            'old_fact_hit': sum(ES.keyword_checks(old_text, facts).values()),
            'new_fact_hit': sum(ES.keyword_checks(gen_text, facts).values()),
            'old_event_hit': sum(ES.keyword_checks(old_text, events).values()),
            'new_event_hit': sum(ES.keyword_checks(gen_text, events).values()),
        })
    n = len(rows)
    rep['T3_old_protocol'] = {
        'facts': list(facts), 'events': events, 'rows': rows,
        'old_fact_pass_rate': sum(r['old_fact_hit'] for r in rows) / (n * len(facts)),
        'new_fact_pass_rate': sum(r['new_fact_hit'] for r in rows) / (n * len(facts)),
        'old_event_pass_rate': sum(r['old_event_hit'] for r in rows) / (n * len(events)),
        'new_event_pass_rate': sum(r['new_event_hit'] for r in rows) / (n * len(events)),
    }
    t3 = rep['T3_old_protocol']
    print('\n== T3 旧口径（prompt+生成）vs 新口径（仅生成段）==')
    print(f'   事实关键词通过率：旧 {100 * t3["old_fact_pass_rate"]:.0f}%  '
          f'→ 新 {100 * t3["new_fact_pass_rate"]:.0f}%')
    print(f'   事件关键词通过率：旧 {100 * t3["old_event_pass_rate"]:.0f}%  '
          f'→ 新 {100 * t3["new_event_pass_rate"]:.0f}%')
    for r in rows[:3]:
        print(f'   {r["prompt_id"]}: 生成段「{r["gen"][:34]}…」')

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f'\n[*] 报告 -> {args.out}')
    ok = (rep['T1_provenance']['all_from_heldout'] and rep['T2_determinism']['same_seed_identical']
          and rep['T2_determinism']['diff_seed_differs'])
    print(f'[*] 判定：{"通过（出题来自留出集、生成可复现、旧口径送分已量化）" if ok else "★失败★"}')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
