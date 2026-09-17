"""Bounded review probes; synthetic fixtures only, no production data edits."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'py'))
import torch
import train
import splits
import run_identity
import sweep_worksplit
from novellm import doctor


def main():
    torch.set_num_threads(1)
    results = {}
    with tempfile.TemporaryDirectory(prefix='minilm-review-') as tmp:
        base = Path(tmp)
        rows = [dict(story_id='a', title='a', bucket='b', split='train',
                     hash=hashlib.sha256('原始正文'.encode()).hexdigest(), chars=4)]
        man, rep = base / 'manifest.jsonl', base / 'report.json'
        man.write_text(json.dumps(rows[0], ensure_ascii=False)+'\n', encoding='utf-8')
        rep.write_text(json.dumps({'split_fingerprint': splits.split_fingerprint(rows)}), encoding='utf-8')
        (base / 'b').mkdir()
        (base / 'b/a.txt').write_text('被改写的正文', encoding='utf-8')
        # Force the production verification branch, rather than fixture bypass.
        with patch.object(splits, 'DEFAULT_PROCESSED', str(base)):
            try:
                splits.bucket_texts('b', str(man), str(base), str(rep))
                results['body_hash_mismatch_rejected'] = False
            except splits.SplitIdentityError:
                results['body_hash_mismatch_rejected'] = True
        family_rows = rows + [dict(rows[0], story_id='b', hash='different', split='test')]
        for row in family_rows:
            row['parent_story_id'] = 'same-parent'
            row['dedup_cluster'] = 'same-cluster'
        rep.write_text(json.dumps({'split_fingerprint': splits.split_fingerprint(family_rows)}), encoding='utf-8')
        try:
            splits.verify(str(man), str(rep), family_rows)
            results['cross_split_family_rejected'] = False
        except splits.SplitIdentityError:
            results['cross_split_family_rejected'] = True

        did = dict(split_fingerprint='same', manifest='same', near_dup_merged=True)
        a = dict(cfg={'ctx':4}, seed=1, steps=10, batch=2, lr=.001, bucket='a', chars=100)
        b = dict(a, bucket='b', chars=200)
        with patch.object(run_identity, 'data_identity', return_value=did), patch.object(run_identity, 'distributable_share', return_value={}):
            run_identity.attach(a)
            run_identity.attach(b)
        results['different_bucket_and_chars_same_identity'] = a['run_identity'] == b['run_identity']

        cache = base / 'tri.json'
        cache.write_text('{"stale":true}', encoding='utf-8')
        try:
            got = sweep_worksplit.trigram_cached(str(cache), [1]*10, [2]*10, 3, 2, False, cache_key='new')
            results['stale_trigram_cache_recomputed'] = got.get('cache_key') == 'new'
        except Exception:
            results['stale_trigram_cache_recomputed'] = False

        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        doctor.probe()
        results['doctor_changes_torch_rng'] = not torch.equal(before, torch.get_rng_state())
        try:
            import psutil
            results['doctor_physical_cores'] = doctor.probe()['cpu']['physical_cores']
            results['psutil_physical_cores'] = psutil.cpu_count(logical=False)
        except ImportError:
            pass

        cfg = json.dumps(dict(ctx=4,d=8,layers=1,heads=2,ffMul=2))
        text = '甲乙丙丁戊己庚辛壬癸'*50
        def invoke(args):
            out = io.StringIO()
            with patch.object(sys, 'argv', ['train.py']+args), patch.object(train, 'load_corpus', return_value=text), contextlib.redirect_stdout(out):
                train.main()
            return out.getvalue()
        common = ['--steps','4','--cfg',cfg,'--chars','100','--allow_leaky']
        full, partial = base/'full', base/'partial'
        invoke(common+['--stop_at','3','--batch','2','--lr','.002','--out',str(full)])
        invoke(common+['--stop_at','1','--batch','2','--lr','.002','--out',str(partial)])
        ckpt = partial/'checkpoint.pt'
        resumed_log = invoke(['--resume',str(ckpt),'--steps','4','--stop_at','3'])
        full_ck = torch.load(full/'checkpoint.pt',weights_only=False)
        resumed = torch.load(ckpt,weights_only=False)
        results['resume_default_hparams'] = {
            'actual_log':next(x for x in resumed_log.splitlines() if x.startswith('seed=')),
            'stored_batch':resumed['meta']['batch'], 'stored_lr':resumed['meta']['lr'],
            'max_weight_delta':max((full_ck['model'][k]-resumed['model'][k]).abs().max().item() for k in full_ck['model'])}
        try:
            invoke(['--resume',str(ckpt),'--steps','5','--stop_at','4','--chars','200'])
            results['changed_data_resume_rejected'] = False
        except ValueError:
            results['changed_data_resume_rejected'] = True
    target = ROOT/'runs/review_20260916/probes.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
