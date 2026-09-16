""" ============================================================================
py/data/pack_remote.py — 把「允许出本地」的语料打包成远程训练数据包
----------------------------------------------------------------------------
给 AutoDL 集成（docs/AUTODL_INTEGRATION_V1.md §2/§6）用。两条硬规则：

1. **许可闸门（fail-closed）**：local_only / unknown 分类的桶**默认拒绝打包**——
   上云 = 数据落第三方磁盘，与「仅本地学习」的许可判断冲突。
   确要打包必须显式 `--allow_non_distributable`，且包内 manifest 会把这些桶标出来。
2. **包必须自带身份，且自洽**：包内 manifest 是全量清单的**子集**，其指纹必然 ≠ 全量指纹。
   因此包内 split_report.json 按**子集清单**重算指纹（远端 verify(包内manifest, 包内report)
   直接通过），全量清单的来源指纹另记在 PACK_INFO.json（`source_split_fingerprint`，谱系用）。
   两个指纹各有定义域，不要混用。

包内容（tar，成员镜像远端目录树，解到工作目录根即落位）：
  data/manifest.jsonl            过滤后的清单（仅所选桶，字段原样）
  data/split_report.json         与子集清单自洽的切分报告
  data/PACK_INFO.json            打包时间 / 来源与子集指纹 / 许可分级摘要
  data/processed/<bucket>/*.txt  作品正文

用法：
  python py/data/pack_remote.py --buckets gudian,wuxia,urban,xiandai,scifi,story
  python py/data/pack_remote.py --buckets story --out dist/remote/story.tar
============================================================================ """
from __future__ import annotations
import argparse
import datetime
import hashlib
import io
import json
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))   # py/data
PY = os.path.dirname(HERE)                          # py
ROOT = os.path.dirname(PY)                          # 仓库根（data/processed 在这里）
if PY not in sys.path:
    sys.path.insert(0, PY)

import splits  # noqa: E402
from splits import split_fingerprint  # noqa: E402
from license_policy import NOT_DISTRIBUTABLE, classify  # noqa: E402
from data.build_splits import slug  # noqa: E402


def build(buckets: list, out_path: str, manifest_path: str = None,
          report_path: str = None, allow_non_distributable: bool = False,
          processed_dir: str = None) -> dict:
    man, rep = splits.resolve_current(manifest_path, report_path)
    rows = splits.read_manifest(man)
    info_rows = [r for r in rows if r['bucket'] in buckets]

    missing = [b for b in buckets if not any(r['bucket'] == b for r in rows)]
    if missing:
        raise SystemExit(f'所选桶在清单中不存在：{missing}')

    # 许可闸门：fail-closed
    bad = {}
    for r in info_rows:
        cls = classify(r.get('license'))
        if cls in NOT_DISTRIBUTABLE:
            bad.setdefault(cls, set()).add(r['bucket'])
    if bad and not allow_non_distributable:
        detail = '; '.join(f'{c}: {sorted(bs)}' for c, bs in sorted(bad.items()))
        raise SystemExit(
            f'[!] 拒绝打包——所选桶含「不可分发」许可分类：{detail}\n'
            f'    「权利不清/仅本地学习」的数据不出本地（上云=数据落第三方磁盘）。\n'
            f'    确要打包请加 --allow_non_distributable，并在模型卡谱系里如实记录。')

    fp = splits.verify(man, rep)['fingerprint']
    # 包内清单是**全量清单的子集**，其指纹必然 ≠ 全量指纹。
    # 因此包内报告按子集重算指纹：远端 splits.verify(包内manifest, 包内report) 自洽通过；
    # 全量清单的来源指纹另记进 PACK_INFO（谱系用），两个指纹各有各的定义域。
    subset_fp = split_fingerprint(info_rows)
    by_key = {(r['bucket'], slug(r['title'])): r for r in info_rows}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    n_files, n_bytes = 0, 0
    mode = 'w:gz' if out_path.endswith(('.gz', '.tgz')) else 'w'
    with tarfile.open(out_path, mode) as tar:
        # 1) 过滤后的清单（字段原样，只裁行）
        man_bytes = ''.join(json.dumps(r, ensure_ascii=False) + '\n'
                            for r in info_rows).encode('utf-8')
        info = tarfile.TarInfo('data/manifest.jsonl')
        info.size = len(man_bytes)
        tar.addfile(info, io.BytesIO(man_bytes))
        # 2) 与子集清单**自洽**的切分报告（指纹 = 子集指纹，远端 verify 直接可用）
        rep_bytes = json.dumps({
            'generated_by': 'py/data/pack_remote.py (derived)',
            'note': '由全量切分报告派生，仅覆盖所选桶；split 归属原样保留。',
            'split_fingerprint': subset_fp,
            'total_works': len(info_rows),
            'source_split_fingerprint': fp,
        }, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('data/split_report.json')
        info.size = len(rep_bytes)
        tar.addfile(info, io.BytesIO(rep_bytes))
        # 3) 作品正文
        for (bucket, s), r in sorted(by_key.items()):
            src = os.path.join(processed_dir or os.path.join(ROOT, 'data', 'processed'),
                               bucket, s + '.txt')
            if not os.path.exists(src):
                raise FileNotFoundError(f'作品正文缺失：{src}（{r["story_id"]} {r["title"]}）')
            tar.add(src, arcname=f'data/processed/{bucket}/{s}.txt')
            n_files += 1
            n_bytes += os.path.getsize(src)
        # 4) 打包信息：两个指纹各有定义域——subset 指纹给远端 verify 用，source 指纹记谱系
        pack_info = {
            'generated_by': 'py/data/pack_remote.py',
            'created_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'buckets': sorted(buckets),
            'source_manifest': os.path.basename(man),
            'source_split_fingerprint': fp,
            'subset_split_fingerprint': subset_fp,
            'manifest_sha256_16': hashlib.sha256(man_bytes).hexdigest()[:16],
            'works': len(info_rows),
            'text_files': n_files,
            'text_bytes': n_bytes,
            'allow_non_distributable': allow_non_distributable,
            'license_summary': _license_summary(info_rows),
        }
        b = json.dumps(pack_info, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('data/PACK_INFO.json')
        info.size = len(b)
        tar.addfile(info, io.BytesIO(b))

    return {'out': out_path, 'pack_bytes': os.path.getsize(out_path),
            'works': len(info_rows), 'text_files': n_files,
            'text_bytes': n_bytes, 'subset_fingerprint': subset_fp,
            'source_fingerprint': fp,
            'allow_non_distributable': allow_non_distributable}


def _license_summary(rows) -> dict:
    out: dict = {}
    for r in rows:
        cls = classify(r.get('license'))
        out[cls] = out.get(cls, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--buckets', required=True, help='逗号分隔的桶列表')
    ap.add_argument('--out', default=None, help='输出 tar 路径（默认 dist/remote/data_<指纹前8>.tar）')
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--allow_non_distributable', action='store_true',
                    help='显式允许「权利不清/仅本地学习」桶出本地（会记录在包内 PACK_INFO.json）')
    ap.add_argument('--gzip', action='store_true',
                    help='tar.gz（中文文本压缩比 ~3-4×；全量 localugc 包建议开，解包命令不变）')
    args = ap.parse_args()

    buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]
    man, rep = splits.resolve_current(args.manifest, args.report)
    fp = splits.verify(man, rep)['fingerprint']
    suffix = '.tar.gz' if args.gzip else '.tar'
    out = args.out or os.path.join(ROOT, 'dist', 'remote', f'data_{fp[:8]}{suffix}')
    r = build(buckets, out, args.manifest, args.report, args.allow_non_distributable)
    print(f'[*] 数据包 -> {r["out"]}')
    print(f'    桶 {sorted(buckets)}：{r["works"]} 部 / 正文 {r["text_files"]} 文件 '
          f'{r["text_bytes"]:,} 字；包体积 {r["pack_bytes"]:,} B')
    print(f'    子集指纹（远端 verify 用）{r["subset_fingerprint"]} · '
          f'来源指纹（谱系用）{r["source_fingerprint"]}')
    if r['allow_non_distributable']:
        print('[!] 包内含「不可分发」许可数据（--allow_non_distributable）——'
              '模型卡谱系必须如实记录。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
