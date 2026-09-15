"""Read-only audit plus disposable tiny training probes; no historical results overwritten."""
import collections
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'py'))
import torch
from data.build_splits import slug
# eval_story 已重写（V2 P0）：generate → generate_model（只返回**续写段**）；
# fact_check / event_completion → keyword_checks（在**生成段**上算关键词命中）。
from eval_story import generate_model, keyword_checks
from minilm_torch import MiniLM, build_vocab, decode, encode


def main():
    result = {'torch': torch.__version__, 'cuda': torch.cuda.is_available()}
    rows = [json.loads(s) for s in (ROOT / 'data/manifest.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
    groups = collections.defaultdict(list)
    missing, mismatches = [], []
    for r in rows:
        groups[r['hash']].append(r)
        f = ROOT / 'data/processed' / r['bucket'] / (slug(r['title']) + '.txt')
        if not f.exists():
            missing.append(r['story_id'])
            continue
        digest = hashlib.sha256(f.read_text(encoding='utf8').encode()).hexdigest()[:16]
        if digest != r['hash']:
            mismatches.append(r['story_id'])
    result['data'] = {'works': len(rows), 'chars': sum(r['chars'] for r in rows),
        'duplicate_hash_groups': sum(len(v) > 1 for v in groups.values()),
        'duplicate_extra_rows': sum(len(v)-1 for v in groups.values()),
        'cross_split_hash_groups': sum(len({r['split'] for r in v}) > 1 for v in groups.values()),
        'missing_files': missing, 'hash_mismatches': mismatches}
    with tempfile.TemporaryDirectory(prefix='minilm_review_') as td:
        paths = []
        for i in range(2):
            dest = pathlib.Path(td) / str(i)
            paths.append(dest)
            cmd = [sys.executable, '-X', 'utf8', 'py/train.py', '--corpus', 'scifi',
                '--chars', '256', '--steps', '1', '--batch', '1', '--seed', '1234',
                '--cfg', json.dumps({'ctx':8,'d':8,'layers':1,'heads':2}), '--out', str(dest)]
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf8')
            if p.returncode:
                raise RuntimeError(p.stderr)
        a, b = [torch.load(p / 'checkpoint.pt', map_location='cpu') for p in paths]
        result['same_seed_cli'] = {
            'equal': all(torch.equal(a['model'][k], b['model'][k]) for k in a['model']),
            'max_weight_diff': max((a['model'][k]-b['model'][k]).abs().max().item() for k in a['model'])}
        p = subprocess.run([sys.executable,'-X','utf8','py/train.py','--resume',
            str(paths[0]/'checkpoint.pt'),'--steps','2'], cwd=ROOT,
            capture_output=True,text=True,encoding='utf8')
        c = torch.load(paths[0]/'checkpoint.pt',map_location='cpu')
        result['resume'] = {'exit_code': p.returncode, 'step_before': a['meta']['step'],
            'step_after': c['meta']['step'], 'checkpoint_keys': list(c),
            'log': p.stdout.strip().splitlines()[-1]}
    prompt = '小明推开了门。'
    itos, stoi = build_vocab(prompt)
    model = MiniLM(len(itos), {'ctx':32,'d':8,'layers':1,'heads':2})
    zero_text = decode(generate_model(model, encode(prompt, stoi), max_new=0,
                                      temp=1.0, topk=1, rep_pen=1.0), itos)
    result['zero_generation'] = {'prompt': prompt, 'returned': zero_text,
        'empty': zero_text == ''}
    valid, invalid = [], []
    for n, s in enumerate((ROOT/'eval/prompts.jsonl').read_text(encoding='utf8').splitlines(),1):
        if not s.strip():
            continue
        try:
            valid.append(json.loads(s))
        except json.JSONDecodeError as e:
            invalid.append({'line':n,'error':str(e)})
    # 「空生成不获故事分」（P0 出口条件）：新协议只对**生成段**评分，
    # 空生成 → 事实/事件命中必须为 0；顺带量化旧协议把提示词拼进去能白拿多少分。
    empty_hits = facts_events = old_prompt_hits = 0
    for p in valid:
        facts, events = p.get('facts', {}), p.get('events', [])
        empty_hits += (sum(keyword_checks('', facts).values())
                       + sum(keyword_checks('', events).values()))
        old_prompt_hits += (sum(keyword_checks(p['prompt'], facts).values())
                            + sum(keyword_checks(p['prompt'], events).values()))
        facts_events += len(facts) + len(events)
    result['prompts'] = {'invalid': invalid, 'valid_count': len(valid),
        'empty_generation_hits': empty_hits, 'facts_and_events': facts_events,
        'empty_generation_scores_zero': empty_hits == 0,
        'old_protocol_prompt_only_hits': old_prompt_hits}
    result['contradiction_passes'] = keyword_checks('苏念穿着红色外套。', {'苏念': '穿蓝色外套'})
    out = ROOT/'runs/review_20260914'
    out.mkdir(exist_ok=True)
    (out/'probes.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
