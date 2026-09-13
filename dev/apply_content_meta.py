""" ============================================================================
dev/apply_content_meta.py — 给 docs/content/文章*.md 统一加上「实验身份」行
与四行固定信息（复现 / 原始结果 / 本篇没有证明 / 下一步），满足
《文章质量审计与后续计划 V1》§三 的发布前规范。

幂等：已含「实验身份：」则跳过。
用法：python dev/apply_content_meta.py
============================================================================ """
import os
import re
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONTENT = os.path.join(ROOT, 'docs', 'content')

# 每篇：实验身份行 + 4 行 footer。字段用真实命令/路径/边界，不编造。
META = {
    '文章01': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜split=前90%/后10%｜params=55,400｜seed=编码示例｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md；交互见 index.html 的 tokMap 面板',
        notprove='只证明输入被替换，不证明子词分词一定会提高写作质量',
        next='字符级 vs 带字节回退子词分词的覆盖率与生成质量对照',
    ),
    '文章02': dict(
        id='model=MiniLM-char-v1｜corpus=三语料(记忆之海/剑雨楼/深夜面馆)｜split=前90%/后10%｜params=55,400｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md；py/runs/js_scifi_s42_n400.json（含 bigram/trigram 基线）',
        notprove='不证明 Transformer 无用，只证明该预算下赢不了 trigram',
        next='扩大数据量后重跑同一张对照表',
    ),
    '文章03': dict(
        id='model=MiniLM-char-v1｜corpus=三语料｜params=55,400｜seed=1234｜steps=400｜sizes=500/1000/2000/all｜protocol=统一协议(eval/protocol.js)',
        repro='python py/sweep_data_scale.py --sizes 500,1000,2000,all --steps 400',
        raw='py/runs/sweep_*.json；docs/img/sweep_data_scale.png；py/runs/sweep_report.md',
        notprove='未测万字以上，不能推广到「Transformer 打不过统计模型」',
        next='万字 / 十万字级数据量实验',
    ),
    '文章04': dict(
        id='model=MiniLM-char-v1(PyTorch 复刻)｜cfg=ctx=4 d=8 layers=1 heads=2 ffMul=2｜V=9｜precision=float64',
        repro='node dev/export_fixture.js && python py/verify_parity.py',
        raw='py/runs/parity_fixture.json；docs/experiments/2026-09-13-pytorch-parity.md',
        notprove='仅验证单步数学对齐，未验证长期训练趋势一致（趋势分布另见 align_report.md）',
        next='多种子训练趋势分布对比（JS vs PyTorch）',
    ),
    '文章05': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜seed=42(单次)｜steps=400（另含早期 700 步观察）｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md；docs/img/training.png',
        notprove='「0.8~2.5 甜蜜点」不是普适阈值，仅为本次实测观察区间',
        next='在公版大语料上重测 loss-重复率曲线',
    ),
    '文章06': dict(
        id='model=MiniLM-char-v1｜corpus=三语料｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md（失败样本与温度对照）',
        notprove='漂亮片段不能代表整体能力',
        next='建立 30 个场景的匿名 A/B 评测（eval/prompts.jsonl）',
    ),
    '文章07': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜seed=42(单次)｜steps=1/100/200/300/400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md（训练快照表，单次值）',
        notprove='不证明所有模型都先学标点——这是本预算下的观察',
        next='观察不同语料/规模下「标点期」出现的步数',
    ),
    '文章08': dict(
        id='model=MiniLM-char-v1｜corpus=三语料｜seeds=JS×3 / PyTorch×5｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus wuxia --seed 7 --steps 400',
        raw='py/align_report.md；LOG.md（L0 续二）',
        notprove='3~5 种子是最低探索起点，不是充分统计证据，不能单独支撑能力结论',
        next='关键结论补到 ≥5~10 种子并报置信区间',
    ),
    '文章09': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜seed=42｜steps=400｜协议=旧(js_seed_run 打补丁)→新(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --seed 42 --steps 400   # 对照 dev/js_runs_oldprotocol/',
        raw='dev/js_runs_oldprotocol/；py/runs/js_scifi_s42_n400.json',
        notprove='不证明哪个协议「更正确」，只证明不标注协议就无法比较',
        next='给评测协议加 version 字段',
    ),
    '文章10': dict(
        id='model=MiniLM-char-v1｜corpus=三语料｜params=55,400｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md；py/runs/*_s42_n400.json',
        notprove='归因「容量相对数据过大」来自密度账（≈29 参数/字），未做容量扫描',
        next='容量扫描（层数 / 宽度）',
    ),
    '文章11': dict(
        id='model=MiniLM-char-v1｜corpus=三语料｜temp=0.8｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md（质量指标表）；eval/metrics.js',
        notprove='重合率低只说明「没原样回放」，不代表「会写」',
        next='人工逐句批注缝合点 + 场景级评测（py/eval_story.py）',
    ),
    '文章12': dict(
        id='env=Intel N100 / Node 单线程｜bench=dev/bench.js 矩阵微基准｜对比=NumPy/OpenBLAS',
        repro='node dev/bench.js',
        raw='dev/evidence_new.md；README.md',
        notprove='124 倍是特定矩阵微基准，不能外推到所有算子或完整训练',
        next='用 benchmark.py 在 PyTorch 侧测同一形状',
    ),
    '文章13': dict(
        id='gradcheck cfg=ctx=8 d=8 layers=1 heads=2 ffMul=2 B=2 T=8 eps=1e-5｜float32 vs float64',
        repro='node test.js',
        raw='LOG.md；test.js(gradCheck)',
        notprove='688/800+ 分量数只属于该极小配置，换大模型会变',
        next='在更大配置上重复梯度检查并比较噪声占比',
    ),
    '文章14': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md；index.html（候选分布 / 模型结构面板）',
        notprove='不证明注意力「理解了」语义，只说明数值如何流动',
        next='逐层可视化一个 token 的流向',
    ),
    '文章15': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜seed=42(单次)｜steps=400｜UI=index.html 观察文本',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md；index.html（注意力热力图）；py/runs/*_s42_n400.json',
        notprove='权重≠重要性，热力图不是「思考过程」',
        next='屏蔽单头看输出变化（C 组对照）',
    ),
    '文章16': dict(
        id='model=MiniLM-char-v1｜cfg=ctx=32 d=40 layers=2 ffMul=4｜seeds=42,7,1234｜steps=800｜A=跨书本(新语料) B=同篇内(记忆之海)',
        repro='python py/ablation.py --corpus scifi --train_size 20000 --steps 800 --seeds 42,7,1234 --heads 1,2,4,8',
        raw='py/runs/ablation_scifi_t20000_s800.json；docs/experiments/2026-09-13-ablation-heads-pos.md',
        notprove='「头数差异不显著」不等于「head 没有分工」；未做单头屏蔽',
        next='C 组屏蔽单头 + ≥5 种子',
    ),
    '文章17': dict(
        id='model=MiniLM-char-v1（presets fast/std/deep/big）｜cost=按 GFLOPS 折算（估算）',
        repro='python py/benchmark.py --corpus scifi --steps 400',
        raw='py/runs/benchmark_report.md；docs/DATA_CANDIDATES.md',
        notprove='放大到 55M 参数的天数是按 6N FLOPs 折算的估算，未实测',
        next='实跑一段 55M 参数外推校准',
    ),
    '文章18': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海/深夜面馆｜temp=0.5/0.8｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus urban --steps 400',
        raw='dev/evidence_new.md（语料内 / 语料外生成原文）',
        notprove='「训练目标无规划」是归因假设，未做因果实验',
        next='人物卡 + 大纲 与 直接续写 的对照',
    ),
    '文章19': dict(
        id='model=MiniLM-char-v1｜corpus=多语料（各条目单独标注）｜seed=42/1234｜steps=400/800｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='各条目指向的实验记录（LOG.md / py/runs/ / docs/experiments/）',
        notprove='这是一份清单，本身不含新实验',
        next='把清单落成 docs/experiments/模板.md 的必填项校验',
    ),
    '文章20': dict(
        id='model=MiniLM-char-v1｜corpus=三语料｜seed=1234｜steps=400｜sizes=500/1000/2000/all｜protocol=统一协议(eval/protocol.js)',
        repro='python py/sweep_data_scale.py --sizes 500,1000,2000,all --steps 400',
        raw='py/runs/sweep_summary.json；py/runs/sweep_report.md；docs/img/sweep_data_scale.png',
        notprove='「十万字级仍不胜则转 SFT」是设定的停止判据，不是实测结论',
        next='跑十万字级，再决定是否转基座 + SFT',
    ),
    '文章21': dict(
        id='model=MiniLM-char-v1｜cfg=ctx=32 d=40 layers=2｜seeds=42,7,1234｜steps=800｜A=跨书本 B=同篇内；pos 置零并冻结',
        repro='python py/ablation.py --corpus scifi --train_size 20000 --steps 800 --seeds 42,7,1234 --heads 4',
        raw='py/runs/ablation_scifi_*.json；docs/experiments/2026-09-13-ablation-heads-pos.md',
        notprove='「位置编码没用」不成立，结论只到「收益不显著（≤种子波动）」',
        next='同作品内切分 + ≥5 种子 + 更大步数',
    ),
    '文章22': dict(
        id='model=MiniLM-char-v1｜corpus=记忆之海(2,668字)｜seed=42(单次)｜steps=400｜protocol=统一协议(eval/protocol.js)',
        repro='node test_novel.js --corpus scifi --steps 400',
        raw='dev/evidence_new.md（下一步候选分布 trace）；eval/metrics.js',
        notprove='loss 只评价概率，不评价故事质量',
        next='同一提示词展示 loss 与人类评分的分歧',
    ),
    '文章23': dict(
        id='无自研实验（论文白话解读）｜对照数据集=docs/DATA_CANDIDATES.md',
        repro='（论文引用，无运行命令）',
        raw='docs/DATA_CANDIDATES.md（候选数据集许可核对）',
        notprove='论文结论建立在英文受控数据上；中文 / 本项目规模未验证',
        next='受限词表重写对照 + TinyStoriesChinese 小模型对照',
    ),
}


def apply(fn, m):
    with open(fn, 'r', encoding='utf-8') as f:
        t = f.read()
    if '实验身份：' in t:
        return False
    key = os.path.basename(fn)[:4]   # '文章01' 等前 4 字
    meta = META.get(key)
    if not meta:
        print('  跳过（无元数据）', key)
        return False

    id_line = '实验身份：' + meta['id'] + '\n'
    footer = (
        '\n---\n\n'
        f'复现：`{meta["repro"]}`\n'
        f'原始结果：{meta["raw"]}\n'
        f'本篇没有证明：{meta["notprove"]}\n'
        f'下一步：{meta["next"]}\n'
    )

    # 1) 插入实验身份：紧跟「**系列：...**」行之后
    t = re.sub(r'(\*\*系列：.*\*\*\n)',
               lambda mo: mo.group(1) + '\n' + id_line, t, count=1)

    # 2) footer：若有「发布前补充」注记，插在它前面；否则追加
    if '发布前补充' in t:
        t = re.sub(r'(\n> 发布前补充)', '\n' + footer + r'\1', t, count=1)
    else:
        t = t.rstrip() + '\n' + footer

    with open(fn, 'w', encoding='utf-8') as f:
        f.write(t)
    return True


def main():
    n = 0
    for fn in sorted(glob.glob(os.path.join(CONTENT, '文章*.md'))):
        if apply(fn, None):
            n += 1
            print('✓', os.path.basename(fn))
    print(f'\n完成：{n} 篇已加实验身份块')


if __name__ == '__main__':
    main()
