""" ============================================================================
py/data/fetch_corpus.py — 多源公版中文小说抓取与归一化
----------------------------------------------------------------------------
目标：
  从公开的公版（public domain）语料源收集**大量**中文小说文本，清洗归一化后
  按 scifi / wuxia / urban / gudian 四个主题分桶，产出训练直接可用的
  py/data/corpus.json（保持与既有 loader 兼容：{key: {name, desc, text, ...}}）。

语料源（全部为公版或明确开放许可，详见 --list-sources）：
  1. gutenberg   — Project Gutenberg 中文书目（PD，UTF-8 单文件纯文本）
  2. hcc         — GitHub xp44mm/hanchuancaolu（中国古典文学原文，逐回 txt）
  3. wikisource  — 中文维基文库（CC BY-SA / PD），用于清末科幻小说
  4. hf          — HuggingFace 开放许可数据集：chinese_novel（CC BY 4.0）
                   + TinyStories 中文译文（CDLA-Sharing-1.0，story 桶）

处理链路：
  下载(带磁盘缓存) → 去 Gutenberg 头尾/HTML/页码等噪声 → 标点归一（｡→。、
  半角→全角）→ 繁转简(opencc t2s) → 跨源按标题去重(保留最长版本)
  → 分桶拼装 → 每桶字数上限截断 → 写 corpus.json + sources.json

用法：
  python py/data/fetch_corpus.py --list-sources          # 只看清单，不下载
  python py/data/fetch_corpus.py --only gutenberg --limit 3   # 小样试跑
  python py/data/fetch_corpus.py                          # 全量
  python py/data/fetch_corpus.py --max-chars 2000000      # 每桶上限
============================================================================ """
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW_DIR = os.path.join(ROOT, 'data', 'raw')
CACHE_DIR = os.path.join(RAW_DIR, '_cache')
CORPUS_OUT = os.path.join(HERE, 'corpus.json')
CORPUS_BAK = os.path.join(HERE, 'corpus.orig.json')
SOURCES_OUT = os.path.join(ROOT, 'data', 'sources.json')

UA = {'User-Agent': 'minilm-lab-corpus/0.1 (research; local use)'}

# ---------------------------------------------------------------------------
# 1. 源登记表
# ---------------------------------------------------------------------------
# Project Gutenberg：https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt
# 全部为公版中文小说。bucket 取值见 BUCKETS。
GUTENBERG_BOOKS = [
    # ---- wuxia：侠义 / 公案 / 历史演义 ----
    (23863, '水浒传', 'wuxia'),
    (25376, '三侠五义', 'wuxia'),
    (25217, '水浒后传', 'wuxia'),
    (25327, '儿女英雄传', 'wuxia'),
    (25350, '荡寇志', 'wuxia'),
    (25214, '剑侠传', 'wuxia'),
    (25393, '施公案', 'wuxia'),
    (26970, '彭公案', 'wuxia'),
    (27686, '狄公案', 'wuxia'),
    (54494, '海公案', 'wuxia'),
    (52280, '明镜公案', 'wuxia'),
    (27330, '绿牡丹', 'wuxia'),
    (26904, '天豹图', 'wuxia'),
    (27082, '杨家将演义', 'wuxia'),
    (23824, '说唐', 'wuxia'),
    (23835, '隋唐演义', 'wuxia'),
    (26871, '粉妆楼全传', 'wuxia'),
    (25229, '飞龙全传', 'wuxia'),
    (26739, '雪月梅传', 'wuxia'),
    (25369, '金石缘', 'wuxia'),
    (27414, '好逑传', 'wuxia'),
    (25348, '双凤奇缘', 'wuxia'),
    (27145, '残唐五代史演义传', 'wuxia'),
    # ---- urban：世情 / 谴责 / 都市 ----
    (24138, '官场现形记', 'urban'),
    (24099, '二十年目睹之怪现状', 'urban'),
    (23850, '老残游记', 'urban'),
    (56291, '老残游记续集', 'urban'),
    (25128, '孽海花', 'urban'),
    (26872, '海上花列传', 'urban'),
    (24032, '儒林外史', 'urban'),
    (25379, '文明小史', 'urban'),
    (25226, '负曝闲谈', 'urban'),
    (24029, '冷眼观', 'urban'),
    (24079, '商界现形记', 'urban'),
    (23827, '胡涂世界', 'urban'),
    (27768, '瞎骗奇闻', 'urban'),
    (24154, '痴人说梦记', 'urban'),
    (27302, '邻女语', 'urban'),
    (24142, '轰天雷', 'urban'),
    (25402, '九命奇冤', 'urban'),
    (23865, '恨海', 'urban'),
    (25521, '玉梨魂', 'urban'),
    (25187, '花月痕', 'urban'),
    (57278, '青楼梦', 'urban'),
    (26931, '风月梦', 'urban'),
    (25543, '蜃楼志', 'urban'),
    (25147, '黄绣球', 'urban'),
    (26161, '醒世姻缘', 'urban'),
    (24422, '定情人', 'urban'),
    # ---- gudian：古典 / 神魔 / 笔记小说 ----
    (24264, '红楼梦', 'gudian'),
    (23962, '西游记', 'gudian'),
    (23950, '三国志演义', 'gudian'),
    (23910, '封神演义', 'gudian'),
    (23818, '镜花缘', 'gudian'),
    (51828, '聊斋志异', 'gudian'),
    (24239, '醒世恒言', 'gudian'),
    (24141, '警世通言', 'gudian'),
    (27582, '喻世明言', 'gudian'),
    (57248, '初刻拍案惊奇', 'gudian'),
    (26729, '二刻拍案惊奇', 'gudian'),
    (24230, '今古奇观', 'gudian'),
    (57227, '平妖传', 'gudian'),
    (27332, '后西游记', 'gudian'),
    (24233, '东度记', 'gudian'),
    (24027, '醉醒石', 'gudian'),
    (25399, '石点头', 'gudian'),
    (25392, '西湖二集', 'gudian'),
    (23907, '无声戏', 'gudian'),
    (24327, '何典', 'gudian'),
    (23817, '阅微草堂笔记', 'gudian'),
    (25245, '子不语', 'gudian'),
    (25315, '续子不语', 'gudian'),
    (25130, '夜雨秋灯录', 'gudian'),
    (25375, '谐铎', 'gudian'),
    (27459, '耳食录', 'gudian'),
    (27216, '斩鬼传', 'gudian'),
    (24231, '韩湘子全传', 'gudian'),
    (25349, '东周列国志', 'gudian'),
    (25362, '搜神记', 'gudian'),
    (25288, '山海经', 'gudian'),
    (26877, '珍珠舶', 'gudian'),
    (29079, '五色石', 'gudian'),
    (25279, '锦香亭', 'gudian'),
    (27023, '白圭志', 'gudian'),
    (25602, '兰花梦奇传', 'gudian'),
    (23761, None, None),  # 占位，运行时过滤
]
GUTENBERG_BOOKS = [b for b in GUTENBERG_BOOKS if b[1]]

# GitHub xp44mm/hanchuancaolu：目录名 → (bucket, 标题)
HCC_REPO = 'xp44mm/hanchuancaolu'
HCC_BRANCH = 'master'
HCC_TARBALL = f'https://codeload.github.com/{HCC_REPO}/tar.gz/refs/heads/{HCC_BRANCH}'
HCC_DIRS = [
    ('水滸傳', '水浒传', 'wuxia'),
    ('三國演義', '三国演义', 'gudian'),
    ('紅樓夢', '红楼梦', 'gudian'),
    ('西遊記', '西游记', 'gudian'),
    ('封神演义', '封神演义', 'gudian'),
    ('聊齋志異', '聊斋志异', 'gudian'),
    ('儒林外史', '儒林外史', 'urban'),
    ('喻世明言', '喻世明言', 'gudian'),
    ('警世通言', '警世通言', 'gudian'),
    ('醒世恒言', '醒世恒言', 'gudian'),
    ('初刻拍案惊奇', '初刻拍案惊奇', 'gudian'),
    ('二刻拍案惊奇', '二刻拍案惊奇', 'gudian'),
]

# 中文维基文库：清末「科学小说」/科幻（含鲁迅译凡尔纳）
WIKISOURCE_BOOKS = [
    ('月界旅行', 'scifi'),          # 鲁迅译 凡尔纳《从地球到月球》1903
    ('地底旅行', 'scifi'),          # 鲁迅译 凡尔纳《地心游记》
    ('新紀元 (碧荷館主人)', 'scifi'),  # 碧荷馆主人 1908 空想科学小说
    ('電世界', 'scifi'),            # 高阳不才子 1909
    ('貓城記', 'scifi'),            # 老舍 1932（作者 1966 卒，已入公版）
]

# 中文维基文库：民国新文学（现代白话）。收录判据：**作者卒年 + 50 已过**（中国大陆公版线）。
# 这一格补的是 docs/DATA_CANDIDATES.md §四 认定「外部数据里找不到交集」的
# **「现代文风 + 许可干净」**：抓取类网文（知乎 / 天涯 / 猫扑 UGC）著作权归**各位楼主本人**、
# 平台 ToS 亦禁止批量抓取，自贴或搬运的许可标签**不覆盖原作者权利** → 一律不收，走公版。
WIKISOURCE_XIANDAI = [
    # 鲁迅：呐喊 / 彷徨 / 故事新编 在维基文库是**目录页**（正文在各短篇的独立页），
    # 所以按单篇收录；书名页本身只有目录与自序，收进来只会是噪声。
    ('狂人日記', 'xiandai', '鲁迅（1918）· 现代白话小说开山'),
    ('孔乙己', 'xiandai', '鲁迅 · 呐喊'),
    ('藥', 'xiandai', '鲁迅 · 呐喊'),
    ('明天', 'xiandai', '鲁迅 · 呐喊'),
    ('一件小事', 'xiandai', '鲁迅 · 呐喊'),
    ('頭髮的故事', 'xiandai', '鲁迅 · 呐喊'),
    ('風波', 'xiandai', '鲁迅 · 呐喊'),
    ('故鄉', 'xiandai', '鲁迅 · 呐喊'),
    ('阿Q正傳', 'xiandai', '鲁迅 · 呐喊（中篇）'),
    ('端午節', 'xiandai', '鲁迅 · 呐喊'),
    ('白光', 'xiandai', '鲁迅 · 呐喊'),
    ('兔和貓', 'xiandai', '鲁迅 · 呐喊'),
    ('鴨的喜劇', 'xiandai', '鲁迅 · 呐喊'),
    ('社戲', 'xiandai', '鲁迅 · 呐喊'),
    ('傷逝 (魯迅)', 'xiandai', '鲁迅 · 彷徨（中篇）'),
    ('在酒樓上', 'xiandai', '鲁迅 · 彷徨'),
    ('孤獨者', 'xiandai', '鲁迅 · 彷徨'),
    ('肥皂', 'xiandai', '鲁迅 · 彷徨'),
    ('長明燈', 'xiandai', '鲁迅 · 彷徨'),
    ('高老夫子', 'xiandai', '鲁迅 · 彷徨'),
    ('離婚', 'xiandai', '鲁迅 · 彷徨'),
    ('弟兄', 'xiandai', '鲁迅 · 彷徨'),
    ('補天', 'xiandai', '鲁迅 · 故事新编'),
    ('奔月', 'xiandai', '鲁迅 · 故事新编'),
    ('理水', 'xiandai', '鲁迅 · 故事新编'),
    ('采薇 (鲁迅)', 'xiandai', '鲁迅 · 故事新编'),
    ('鑄劍', 'xiandai', '鲁迅 · 故事新编'),
    ('出關', 'xiandai', '鲁迅 · 故事新编'),
    ('非攻', 'xiandai', '鲁迅 · 故事新编'),
    ('起死', 'xiandai', '鲁迅 · 故事新编'),
    ('駱駝祥子', 'xiandai', '老舍（1966 卒）· 长篇'),
    ('四世同堂', 'xiandai', '老舍 · 长篇'),
    ('沉淪 (郁達夫)', 'xiandai', '郁达夫（1945 卒）· 短篇'),
    ('呼蘭河傳', 'xiandai', '萧红（1942 卒）· 长篇'),
    ('生死場', 'xiandai', '萧红 · 中篇'),
]

# ⚠️ 新青年（1915-1926）原本打算作为现代期刊补入 xiandai 桶。维基文库 2-7 卷**只有目录**，
# 实质正文在每篇文章的独立子页（`狂人日記`、`孔乙己` 等）——已通过 WIKISOURCE_XIANDAI
# 接入（共 30 篇，含鲁迅《狂人日记》《孔乙己》《药》等首发），无需再单独收卷。
# 这里留一个**杂志导入位**：`data/raw/_cache/zazhi/<刊名>.txt`（UTF-8，每行一段），
# 收录今古传奇 / 科幻世界 / 人民文学 / 收获 等现代文学期刊（你按"仅本地学习"自行放入）。
# 启用方法：在 fetch_localugc 块的本地导入 glob 里加 `zazhi/*.txt`。
WIKISOURCE_BOOKS += [(t, b) for t, b, _ in WIKISOURCE_XIANDAI]
XD_NOTE = {t: n for t, _, n in WIKISOURCE_XIANDAI}

# HuggingFace 开放许可数据集（出处与许可条件详见 docs/DATA_CANDIDATES.md）
# 收录原则：**只收许可明确、且上传者有权授权的数据**。
# 抓取类数据集（中文网文合集）即使标了 Apache-2.0，也不覆盖原著版权，一律不收。
HF_BOOKS = [
    {
        'repo': 'asd567557275/chinese_novel',
        'file': 'novel_by_chapter.jsonl',
        'field': 'text',
        'title': '空间魔导书与少年魔法师',
        'bucket': 'scifi',   # 现代幻想类；scifi 桶此前只有清末科学小说
        'license': 'CC BY 4.0（署名：睡半夜怎麼三更）',
        'ref': 'https://huggingface.co/datasets/asd567557275/chinese_novel',
        'note': '繁体原作，入库前 opencc t2s；按章拼接（novel_by_chapter.jsonl）',
    },
]

# ---------------------------------------------------------------------------
# 本地学习桶 (`localugc`)：知乎 / 网文 UGC。
# ---------------------------------------------------------------------------
# 立场：仅作**个人本地学习实验**使用；**不可再分发 / 不可商用 / 不可入评测桶**。
# 来源链路均标注，上游权利人若行使删除权请直接弃用对应文件。
# - 知乎类：HF 上小型问答/精选集，是真实知乎内容，**但不是小说叙事**，
#   风格对"小说生成"的迁移价值有限；放入是为了让你能跑通"把知乎文本喂进同一管线"。
# - 网文类：wdndev/webnovel-chinese 上传者贴 Apache-2.0，但内容是抓取的商业网文，
#   标签**不覆盖原著作者的权利**（与项目 2026-09-14 判定一致）。流式只下首片的前 N MB，
#   避免一次性拉 4.2 GB 全量；如需更多，改 MAX_WEBNOVEL_BYTES。
# - 天涯/猫扑：HF 上没有任何可核验的叙事数据集（已实测）；这两个留给"本地导入位"——
#   把 dump 放到 `data/raw/_cache/localugc/<title>.txt`（每行一段），重跑时会自动作为一部作品入库。
LOCALUGC_HF = [
    {
        'repo': 'liyucheng/zhihu_rlhf_3k',
        'file': 'zhihu_3k_rlfh.tsv',
        'field': 'chosen',
        'encoding': 'gbk',
        'fmt': 'tsv',
        'title': '知乎 RLHF 3K',
        'license': 'CC BY-NC 4.0（按仓库说明）',
        'ref': 'https://huggingface.co/datasets/liyucheng/zhihu_rlhf_3k',
        'note': '字段：prompt/chosen/rejected；用 chosen 作正文',
    },
    {
        'repo': 'liyucheng/zhihu_26k',
        'file': 'zhihu_26k.tsv',
        'field': 'RESPONSE',
        'encoding': 'gbk',
        'fmt': 'tsv',
        'title': '知乎 RLHF 26K',
        'license': 'CC BY-NC 4.0（按仓库说明）',
        'ref': 'https://huggingface.co/datasets/liyucheng/zhihu_26k',
        'note': '字段：INSTRUCTION/RESPONSE/SOURCE；GBK 编码',
    },
    {
        'repo': 'zirui3/zhihu_qa',
        'file': 'zhihu_qa.jsonl.gz',
        'field': 'answer',
        'encoding': 'utf-8',
        'fmt': 'jsonl_gz',
        'title': '知乎问答 zhihu_qa',
        'license': '按仓库说明（未明确标注）',
        'ref': 'https://huggingface.co/datasets/zirui3/zhihu_qa',
        'note': '字段：question/answer；需 gunzip',
    },
    {
        'repo': 'next-social/zhihu_beautiful_super_alpaca',
        'file': 'data.json',
        'field': 'output',
        'encoding': 'utf-8',
        'fmt': 'json_list',
        'title': '知乎精选（alpaca）',
        'license': '按仓库说明',
        'ref': 'https://huggingface.co/datasets/next-social/zhihu_beautiful_super_alpaca',
        'note': '顶层是 JSON 数组；字段：instruction/input/output',
    },
]
for s in LOCALUGC_HF:
    s.setdefault('bucket', 'localugc')

# 天涯论坛辩论语料：来自 zhaoqixu/tianya-forum-scrapper 的 LXZ_chi(B).xml（莲蓬鬼话 = 民间怪谈板）。
# 格式：XML，<s> 是一帖，<utt uid=...> 是一句话。**它是辩论/聊天的对话体，不是小说叙事**；
# 但它是真实天涯内容，对"论文式对话 + 现代口语"有补充价值。
LOCALUGC_TIANYA = {
    'repo': 'zhaoqixu/tianya-forum-scrapper',
    'archive': 'tianya-forum-scrapper.tar.gz',
    'tar_member': 'tianya-forum-scrapper-master/LXZ_chi(B).xml',
    'url': 'https://codeload.github.com/zhaoqixu/tianya-forum-scrapper/tar.gz/refs/heads/master',
    'encoding': 'utf-8',
    'fmt': 'tianya_xml',
    'min_chars': 200,
    'max_threads': 3000,
    'title': '天涯论坛辩论（莲蓬鬼话对话）',
    'license': 'GitHub 仓库（scraped 内容，作者权利不清；本地学习用）',
    'ref': 'https://github.com/zhaoqixu/tianya-forum-scrapper',
    'note': 'XML 一帖 = 一个 <s>，含多个 <utt> 发言；含 @用户名 + 时间戳，清洗时剥除',
    'bucket': 'localugc',
}

# MNBVC 天涯主片（重启后全站）：forum/20260107/tianya.20260107.1/ 下的 zip，
# 每条 jsonl 记录是一帖（主题 + 回复数组 + 元数据/版块）。只下 part1 (150 MB) + part4 (24 MB)：
# 其余 3 片是同一源分块、增量收益递减；解压 ~600 MB jsonl，含数十万条天涯帖。
LOCALUGC_MNBVC_TIANYA = {
    'repo': 'liwu/MNBVC',
    'files': [
        'forum/20260107/tianya.20260107.1/20260307_203926_1.zip',
        'forum/20260107/tianya.20260107.1/20260307_224415_4.zip',
    ],
    'encoding': 'utf-8',
    'fmt': 'mnbvc_tianya_jsonl',
    'min_chars': 200,
    'max_records': 50000,
    'title': '天涯论坛全站 dump（MNBVC 主片）',
    'license': 'HF 仓库 MIT；抓取内容作者权利不清；本地学习用',
    'ref': 'https://huggingface.co/datasets/liwu/MNBVC',
    'note': '记录字段：ID/主题/来源/时间/回复(楼ID/回复人/回复时间/UID）/元数据(版块/楼主/点击/回复数)',
    'bucket': 'localugc',
}

# MNBVC 树洞系列（北大 pkuhole / 清华 thuhole）：与天涯同 schema，直接复用 _iter_mnbvc_tianya。
# pkuhole 4 文件 ~306 MB，thuhole 2 文件 ~177 MB。树洞叙事 = 学生日常 + 校园 + 情感 + 时评。
LOCALUGC_MNBVC_PKUHOLE = {
    'repo': 'liwu/MNBVC',
    'files': [
        'forum/20230253/pkuhole.20230253.2.论坛/pkuholefromarchive_0.jsonl.gz',
        'forum/20230253/pkuhole.20230253.2.论坛/pkuholefromarchive_1.jsonl.gz',
        'forum/20230253/pkuhole.20230253.2.论坛/pkuholefromarchive_2.jsonl.gz',
        'forum/20230253/pkuhole.20230253.2.论坛/pkuholefromlog.jsonl.gz',
    ],
    'encoding': 'utf-8',
    'fmt': 'mnbvc_tianya_jsonl',
    'min_chars': 200,
    'max_records': 40000,
    'title': '北大树洞（pkuhole）',
    'license': 'HF 仓库 MIT；抓取内容作者权利不清；本地学习用',
    'ref': 'https://huggingface.co/datasets/liwu/MNBVC',
    'note': '与天涯主片同 schema；来源=北大树洞',
    'bucket': 'localugc',
}
LOCALUGC_MNBVC_THUHOLE = {
    'repo': 'liwu/MNBVC',
    'files': [
        'forum/20230253/thuhole.20230253.3.论坛/thuhole_0.jsonl.gz',
        'forum/20230253/thuhole.20230253.3.论坛/thuhole_1.jsonl.gz',
    ],
    'encoding': 'utf-8',
    'fmt': 'mnbvc_tianya_jsonl',
    'min_chars': 200,
    'max_records': 20000,
    'title': '清华树洞（thuhole）',
    'license': 'HF 仓库 MIT；抓取内容作者权利不清；本地学习用',
    'ref': 'https://huggingface.co/datasets/liwu/MNBVC',
    'note': '与天涯主片同 schema；来源=清华树洞',
    'bucket': 'localugc',
}

# 网文：**全量分片**。HF 上 10 片 jsonl 共 ~39 GB；单片 4.2 GB ≈ 9,700 部完整小说。
# 之前是 Range 只取前 30 MB（每部只剩开头一小段），现改为**整片流式下载到盘、
# 逐行流式解析**（4.2 GB 不进内存）。要扩更多片，往 'shards' 里加编号即可。
LOCALUGC_WEBNOVEL = {
    'repo': 'wdndev/webnovel-chinese',
    'shards': [0],                 # 已拉齐的分片编号；要扩网文量就加 1,2,...
    'field': 'text',
    'fmt': 'jsonl_stream',
    'title': '网文全量分片',
    'license': 'Apache-2.0（标签不覆盖原著作者权利；本地学习用）',
    'ref': 'https://huggingface.co/datasets/wdndev/webnovel-chinese',
    'note': '整片下载（非截断）；字段：title/chapter/text；抓取的中文商业网文',
    'bucket': 'localugc',
}

# Pixiv 中文小说（原创/同人叙事）：单 zip 1.68 GB。
LOCALUGC_PIXIV = {
    'repo': 'wuliangfo/Chinese-Pixiv-Novel',
    'file': 'PixivNovel.zip',
    'encoding': 'utf-8',
    'min_chars': 200,
    'max_works': 30000,
    'title': 'Pixiv 中文小说',
    'license': 'HF 仓库说明；抓取内容作者权利不清；本地学习用',
    'ref': 'https://huggingface.co/datasets/wuliangfo/Chinese-Pixiv-Novel',
    'note': 'zip 内含单篇/多轮等文件；每篇 = 一部作品',
    'bucket': 'localugc',
}

# 小源：v3ucn 中文小说集（1.9 MB，全量直读）
LOCALUGC_V3UCN = {
    'repo': 'v3ucn/chinese-novel-dataset',
    'file': 'dataset.json',
    'encoding': 'utf-8',
    'min_chars': 200,
    'title': 'v3ucn 中文小说集',
    'license': 'HF 仓库说明；本地学习用',
    'ref': 'https://huggingface.co/datasets/v3ucn/chinese-novel-dataset',
    'note': '单 JSON 文件，全量',
    'bucket': 'localugc',
}

# 本地导入位：data/raw/_cache/localugc/*.txt（每行一段）会被自动作为一部作品收进来。
# 用于接入天涯/猫扑/其它任何 dump，文件名即标题。

# TinyStories 中文译文：词汇窄、句子短的合成数据，用于「数百个短故事探索集」
# 与故事级切分（每篇一部作品，PRD 要求 train/val/test 不跨故事）。
HF_TINYSTORIES = {
    'repo': 'adam89/TinyStoriesChinese',
    'archive': os.path.join(CACHE_DIR, 'tinystories_zh.tar.gz'),
    'member': 'data00_zh.jsonl',
    'field': 'story_zh',
    'bucket': 'story',
    'license': 'CDLA-Sharing-1.0（TinyStories 中文译文，合成数据）',
    'ref': 'https://huggingface.co/datasets/adam89/TinyStoriesChinese',
}
HF_TINYSTORIES_URL = ('https://huggingface.co/datasets/adam89/TinyStoriesChinese/'
                      'resolve/main/TinyStories_all_data_zh.tar.gz')

BUCKETS = {
    'scifi': {'name': '科幻 · 清末科学小说 + 现代幻想',
              'desc': '月界旅行 / 地底旅行 / 新纪元 / 电世界 / 猫城记（维基文库，公版）'
                      ' + 空间魔导书与少年魔法师（CC BY 4.0，现代）'},
    'wuxia': {'name': '武侠 · 侠义公案',
              'desc': '水浒传 / 三侠五义 / 儿女英雄传 / 荡寇志 / 施公案等（Gutenberg，公版）'},
    'urban': {'name': '都市 · 世情谴责',
              'desc': '官场现形记 / 二十年目睹之怪现状 / 老残游记 / 孽海花 / 儒林外史等（公版）'},
    'gudian': {'name': '古典 · 神魔笔记',
               'desc': '红楼梦 / 西游记 / 三国演义 / 封神演义 / 聊斋志异 / 三言二拍等（公版）'},
    'story': {'name': '短故事 · 简单现代汉语',
              'desc': 'TinyStories 中文译文（CDLA-Sharing-1.0，合成数据）：'
                      '词汇窄、句子短，用于跑通管线与故事级切分，不代表目标文风'},
    'xiandai': {'name': '现代 · 民国新文学公版',
                'desc': '鲁迅 呐喊/故事新编/狂人日记、老舍 骆驼祥子/四世同堂、郁达夫 沉沦、'
                        '萧红 呼兰河传/生死场、还珠楼主 蜀山剑侠传（维基文库，作者卒于 1966 年前，公版）：'
                        '现代白话 + 许可干净，补 docs/DATA_CANDIDATES.md §四 的空格'},
    # 本地学习桶：仅供个人本地实验，**禁止再分发 / 商用**；不出现在 `sweep_worksplit` 的 BUCKETS 里、
    # 不进评测桶；manifest 里以 `bucket='localugc'` 单独存，以便审计但不被合并对比。
    'localugc': {'name': '本地学习 · 知乎/网文/天涯 UGC（不可分发）',
                 'desc': 'HF 公开小型知乎问答/精选集 + wdndev/webnovel-chinese 首片前 30 MB 切片 '
                         '+ 天涯论坛对话语料（GitHub 抓取）。'
                         '**仅供个人本地学习使用；不可再分发 / 不可商用 / 不入评测桶。**',
                 # 桶内作品上万（每条知乎回答 = 一部"作品"），按默认 200 万字均分后每部只剩 ~150 字，
                 # 会被 allocate 的 200 字下限整桶滤空 —— 所以这一桶单独放高上限。
                 'max_chars': 8_000_000},
}
BUCKET_ORDER = ['scifi', 'wuxia', 'urban', 'gudian', 'story', 'xiandai', 'localugc']


# ---------------------------------------------------------------------------
# 2. 下载工具
# ---------------------------------------------------------------------------
def http_get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.2 * (i + 1))
    raise RuntimeError(f'下载失败 {url}: {last}')


def http_get_cached(url: str, cache_path: str, retries: int = 4) -> bytes:
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, 'rb') as f:
            return f.read()
    data = http_get(url, retries=retries)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        f.write(data)
    return data


def http_get_range_cached(url: str, cache_path: str, n_bytes: int,
                          retries: int = 4, timeout: int = 120) -> bytes:
    """只取远端文件的前 n_bytes（Range 请求）。用于 4 GB 级大文件，
    缓存里落的就是这个**截断**片段，文件名带 _head<N>m 以示区分，避免误当全量。"""
    if os.path.exists(cache_path) and 0 < os.path.getsize(cache_path) <= n_bytes:
        with open(cache_path, 'rb') as f:
            return f.read()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=dict(UA, Range=f'bytes=0-{n_bytes - 1}'))
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read(n_bytes)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(data)
            return data
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f'Range 下载失败 {url}: {last}')


def http_download_to_file(url: str, path: str, min_size: int = 1 << 20,
                          timeout: int = 120) -> str:
    """流式下载整文件到盘（不进内存）。已存在且 >= min_size 视为缓存命中。
    4 GB 级文件不能走 http_get_cached（会把整文件读成 bytes）。"""
    if os.path.exists(path) and os.path.getsize(path) >= min_size:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, 'wb') as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, path)
    return path


def _iter_webnovel_works(path: str, field: str):
    """流式逐行解析网文 jsonl，按 title **连续段**聚章：title 变了就 yield 上一部。
    （jsonl 按书连续排布；若同一书名在文件后部再次出现，会作为第二部、由
    调用方用序号保证标题唯一。）4 GB 级文件不进内存。"""
    cur, chunks = None, []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            t = (o.get('title') or '').strip() or 'untitled'
            if cur is None:
                cur = t
            if t != cur:
                if chunks:
                    yield cur, chunks
                cur, chunks = t, []
            chunks.append(o.get(field) or '')
    if chunks:
        yield cur, chunks


def slug(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\s]+', '_', s.strip())
    return s[:80] or 'untitled'


# ---------------------------------------------------------------------------
# 3. 清洗与归一化
# ---------------------------------------------------------------------------
CJK = r'\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff'
RE_START = re.compile(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*',
                      re.IGNORECASE)
RE_END = re.compile(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*\*\*\*',
                    re.IGNORECASE)
RE_HTML = re.compile(r'</?[A-Za-z][^>]{0,200}>')
RE_ENTITY = re.compile(r'&(?:nbsp|amp|lt|gt|quot|#\d{1,6}|[A-Za-z]{2,8});')
RE_PGNO = re.compile(r'\[Pg\s*\d+\]')
RE_PURE_NUM = re.compile(r'^\s*[-—–\d\s.,、]{1,12}\s*$')

# 标点/字形变体 → 标准形式
PUNCT_MAP = {
    '｡': '。', '．': '。', '。': '。',
    '､': '、', '｢': '「', '｣': '」',
    '･': '·', '─': '—', '－': '—', '–': '—', '‒': '—',
    '﹔': '；', '﹕': '：', '﹐': '，', '﹑': '、', '﹗': '！', '﹖': '？',
    '﹑': '、', '´': '‘', '｀': '‘',
}

# 允许保留的标点集合（中文正文常用）
ALLOW_PUNCT = set('。，、！？：；「」『』（）〈〉《》—…·‘’“”～％')

# 全角字母数字 → 半角
_FW = {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2FA1F or 0x30000 <= o <= 0x323AF)


# finalize_chars 的正则等价版：保留 \n + CJK（含扩展区）+ 中文标点白名单 + ASCII 字母数字
# 与少量符号，其余一次 sub 剔除。语义与逐字符版一致，但 ~5× 快（localugc 大体量用）。
RE_DROP_CHARS = re.compile(
    '[^\\n'
    '\\u3400-\\u4dbf\\u4e00-\\u9fff\\uf900-\\ufaff'
    '\\u20000-\\u2fa1f\\u30000-\\u323af'
    '。，、！？：；「」『』（）〈〉《》—…·‘’“”～％'
    'a-zA-Z0-9 .,!?:;%()\\-\'"]+')


def finalize_chars_fast(t: str) -> str:
    t = t.translate(_FW)
    t = RE_DROP_CHARS.sub('', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n[ \t]+', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t


def finalize_chars(t: str) -> str:
    """字符白名单：只留 CJK（含扩展区）、中文标点、换行、ASCII 字母数字与少量符号。
    其余（■□◀↑[]=_& 等版式残渣）全部丢弃。"""
    t = t.translate(_FW)
    out = []
    for ch in t:
        if ch == '\n':
            out.append(ch)
        elif is_cjk(ch) or ch in ALLOW_PUNCT:
            out.append(ch)
        elif ch.isascii() and (ch.isalnum() or ch in ' .,!?:;%()-\'"'):
            out.append(ch)
    s = ''.join(out)
    s = re.sub(r'[ \t]{2,}', ' ', s)
    s = re.sub(r'\n[ \t]+', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s


# 句末标点核心集：出现在行尾（或其后的引号/括号之后）即认为自然段结束
SENT_CORE = '。！？：；…!?'
CLOSERS = '」』”"’）)】》'
# 章节标题：第一回 / 卷三 / 楔子 / 序 …（自身成行）
RE_CHAPTER = re.compile(r'^(?:第[〇零一二三四五六七八九十百千0-9]+[回卷章节篇]|楔子|引首|序|跋|'
                        r'卷[〇零一二三四五六七八九十0-9]+)')
WRAP_MIN = 28   # 行宽达到此值才认为是「硬换行」需拼接


def _ends_para(s: str) -> bool:
    if not s:
        return False
    if s[-1] in SENT_CORE:
        return True
    if s[-1] in CLOSERS:
        t = s.rstrip(CLOSERS)
        return bool(t) and t[-1] in SENT_CORE
    return False


def reflow(t: str) -> str:
    """把 Gutenberg 等按固定宽度硬换行的文本重排回自然段。
    规则：行宽 >= WRAP_MIN 且行尾非句末标点 → 与下一行直接拼接（不插空格）。"""
    out, buf = [], ''
    for ln in t.split('\n'):
        s = ln.strip()
        if not s:
            if buf:
                out.append(buf)
                buf = ''
            out.append('')
            continue
        if len(s) <= 24 and RE_CHAPTER.match(s):
            if buf:
                out.append(buf)
                buf = ''
            out.append(s)
            out.append('')
            continue
        buf += s
        if _ends_para(s) or len(s) < WRAP_MIN:
            out.append(buf)
            buf = ''
    if buf:
        out.append(buf)
    return '\n'.join(out)


def strip_gutenberg(t: str) -> str:
    m = RE_START.search(t)
    if m:
        t = t[m.end():]
    m = RE_END.search(t)
    if m:
        t = t[:m.start()]
    # 去掉尾部常见的 "End of Project Gutenberg's ..." 段落
    t = re.split(r'End of (?:the )?Project Gutenberg', t)[0]
    return t


def normalize(t: str) -> str:
    t = t.replace('\ufeff', '').replace('\r\n', '\n').replace('\r', '\n')
    t = RE_HTML.sub('', t)
    t = RE_ENTITY.sub('', t)
    t = RE_PGNO.sub('', t)
    t = t.replace('\u3000', ' ')
    # 标点归一
    t = t.translate(str.maketrans(PUNCT_MAP))
    t = re.sub(rf'(?<=[{CJK}])\s*,\s*(?=[{CJK}])', '，', t)
    t = re.sub(rf',(?=[{CJK}])', '，', t)
    t = re.sub(rf'(?<=[{CJK}]),', '，', t)
    t = re.sub(rf'(?<=[{CJK}])\s*\.\s*(?=[{CJK}])', '。', t)
    t = re.sub(rf'(?<=[{CJK}])\.\s*$', '。', t, flags=re.M)
    t = re.sub(rf'(?<=[{CJK}])\s*!\s*', '！', t)
    t = re.sub(rf'(?<=[{CJK}])\s*\?\s*', '？', t)
    t = re.sub(rf'(?<=[{CJK}])\s*:\s*', '：', t)
    t = re.sub(rf'(?<=[{CJK}])\s*;\s*', '；', t)
    t = re.sub(rf'(?<=[{CJK}])~+', '～', t)
    # 行处理：去噪
    out = []
    for ln in t.split('\n'):
        s = ln.strip()
        if not s:
            out.append('')
            continue
        if RE_PURE_NUM.match(s):
            continue
        cjk = sum(1 for c in s if '\u4e00' <= c <= '\u9fff')
        latin = sum(1 for c in s if c.isascii() and c.isalpha())
        # 纯拉丁短行（页码、Produced by…）丢弃
        if cjk == 0 and latin > 0 and len(s) < 80:
            continue
        out.append(s)
    t = '\n'.join(out)
    # CJK 间多余空格
    t = re.sub(rf'(?<=[{CJK}])[ \t]+(?=[{CJK}])', '', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    return t.strip()


_cc = None


def to_simplified(t: str) -> str:
    global _cc
    if _cc is None:
        try:
            import opencc
            _cc = opencc.OpenCC('t2s')
        except Exception:  # noqa: BLE001
            _cc = False
    if _cc is False:
        return t
    try:
        return _cc.convert(t)
    except Exception:  # noqa: BLE001
        return t


WS_NOISE = re.compile(r'(校對|校对|原文為準|原文为准|展開|展开|收起|查看更多|'
                      r'維基文庫|维基文库|公共領域|公有领域|CC[ -]BY|版權|版权|'
                      r'點擊|点击|本作品|本頁面|本页面|參見|参见|圖像|图像|'
                      r'File:|Category:|模板:|維基百科|维基百科|姊妹計劃|姊妹计划|'
                      r'數據項|数据项|章[節节]$|編輯$|编辑$|^作者[:：]|^目[錄录]$)')
# 无论多长都要丢弃的模板/声明行
WS_NOISE_ALWAYS = re.compile(r'(查看更多|展開|展开|收起|原文為準|原文为准|'
                             r'機器或人工轉換|机器或人工转换|'
                             r'此文獻|此文献|本文件|本作品的)')


WS_BLOCK_KEYS = ['底本', '简化字', '简体字', '繁体字', '異體字', '异体字', '校对', '校對',
                 '原始文献', '汉字简化', '漢字簡化', '机器或人工', '機器或人工',
                 '维基文库', '維基文庫', '公共领域', '公共領域', '版权', '版權',
                 '查看更多', '展开', '展開', '保留其', '所使用的汉字', '大元帅明王']


def drop_noise_blocks(t: str) -> str:
    """整段丢弃模板说明/版权声明：一段内命中 >=2 个噪声关键词即判为非正文。"""
    keep = []
    for b in re.split(r'\n{2,}', t):
        s = b.strip()
        if not s:
            continue
        hits = sum(1 for k in WS_BLOCK_KEYS if k in s)
        if hits >= 2 and len(s) < 1500:
            continue
        keep.append(s)
    return '\n\n'.join(keep)


def drop_repeated_nav(t: str) -> str:
    """维基文库每个子页都带同一份章节导航表，导致 '猫城记1'、'编辑' 这类短行反复出现。
    规则：长度 <= 16、纯 CJK/数字（无标点）、且在本文出现 >= 3 次的独立行 → 判为导航丢弃。"""
    import collections
    lines = [ln.strip() for ln in t.split('\n')]
    cnt = collections.Counter(ln for ln in lines if ln)
    out = []
    for ln in lines:
        s = ln.strip()
        if (s and len(s) <= 16 and cnt[s] >= 3
                and re.fullmatch(rf'[{CJK}0-9]{{1,16}}', s)):
            continue
        out.append(s)
    return '\n'.join(out)


# 节间残留：独立成行的「一/二/编辑/版本信息」——正文里只是节号，丢了无损
WS_LINE_DROP = re.compile(r'^(?:[一二三四五六七八九十百]+|编辑|編輯|版本信息)$')


def strip_ws_boilerplate(t: str) -> str:
    """去掉维基文库的模板提示 / 版权声明 / 导航等非正文行与整段。"""
    keep = []
    for ln in t.split('\n'):
        s = ln.strip()
        if not s:
            keep.append('')
            continue
        if s.startswith('↑') or s.startswith('◀') or s.startswith('▶'):
            continue
        if WS_LINE_DROP.match(s):
            continue
        if WS_NOISE_ALWAYS.search(s):
            continue
        if len(s) < 80 and WS_NOISE.search(s):
            continue
        if re.match(r'^[\[\(（【]\s*[\]\)）】]\s*$', s):
            continue
        keep.append(s)
    return drop_noise_blocks(drop_repeated_nav('\n'.join(keep)))


def clean(t: str, source: str) -> str:
    if source == 'gutenberg':
        t = strip_gutenberg(t)
    elif source == 'wikisource':
        t = strip_ws_boilerplate(t)
    if source == 'localugc':
        # 快速路径：**不做繁转简**（纯 python opencc 只有 ~0.3MB/s，5GB 级扛不住；
        # 而网文/知乎/树洞本身是简体，Pixiv 混排对非证据桶可接受），
        # 字符白名单用一条正则代替逐字符循环（~5×）。
        t = normalize(t)
        t = finalize_chars_fast(t)
        return t.strip()
    t = normalize(t)
    if source in ('gutenberg', 'hcc'):
        t = reflow(t)
    t = to_simplified(t)
    t = finalize_chars(t)
    return t.strip()


# ---------------------------------------------------------------------------
# 4. 各源抓取 → List[dict(story)]
# ---------------------------------------------------------------------------
def fetch_gutenberg(limit: int | None = None) -> list:
    stories = []
    books = GUTENBERG_BOOKS[:limit] if limit else GUTENBERG_BOOKS
    for i, (eid, title, bucket) in enumerate(books, 1):
        url = f'https://www.gutenberg.org/cache/epub/{eid}/pg{eid}.txt'
        cp = os.path.join(CACHE_DIR, 'gutenberg', f'{eid}.txt')
        try:
            raw = http_get_cached(url, cp).decode('utf-8', errors='replace')
        except Exception as e:  # noqa: BLE001
            print(f'  [!] Gutenberg {eid} {title} 失败: {e}')
            continue
        text = clean(raw, 'gutenberg')
        print(f'  [{i}/{len(books)}] PG{eid:<6} {title:<12} {len(text):>8} 字')
        if len(text) < 500:
            print('        [skip] 正文过短')
            continue
        stories.append({
            'title': title, 'bucket': bucket, 'text': text, 'source': 'gutenberg',
            'license': 'Public Domain (Project Gutenberg)',
            'ref': f'https://www.gutenberg.org/ebooks/{eid}',
        })
    return stories


def fetch_hcc() -> list:
    """下载 hanchuancaolu 仓库 tarball，按目录聚合逐回文本。"""
    stories = []
    cp = os.path.join(CACHE_DIR, f'{HCC_REPO.split("/")[-1]}.tar.gz')
    try:
        data = http_get_cached(HCC_TARBALL, cp, retries=3)
    except Exception as e:  # noqa: BLE001
        print(f'  [!] HCC tarball 下载失败，跳过整个源: {e}')
        return stories

    wanted = {x[0] for x in HCC_DIRS}
    per_dir = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as tf:
        members = [m for m in tf.getmembers()
                   if m.isfile() and m.name.endswith('.txt')]
        print(f'  tarball 内 txt 文件: {len(members)}')
        for m in members:
            parts = m.name.split('/', 1)
            if len(parts) < 2:
                continue
            rel = parts[1]
            d = rel.split('/')[0]
            if d not in wanted:
                continue
            try:
                body = tf.extractfile(m).read().decode('utf-8', errors='replace')
            except Exception:  # noqa: BLE001
                continue
            per_dir.setdefault(d, []).append((rel, body))

    for d, title, bucket in HCC_DIRS:
        items = sorted(per_dir.get(d, []), key=lambda x: x[0])
        if not items:
            print(f'  [!] HCC {d} 无文件')
            continue
        chunks = [b for _, b in items]
        text = clean('\n\n'.join(chunks), 'hcc')
        print(f'  HCC {d:<10} {title:<10} {len(text):>8} 字 ({len(items)} 回)')
        if len(text) < 500:
            continue
        stories.append({
            'title': title, 'bucket': bucket, 'text': text, 'source': 'hcc',
            'license': 'Public Domain (中国古典文学原文)',
            'ref': f'https://github.com/{HCC_REPO}',
        })
    return stories


def ws_api(params: dict) -> dict:
    params = dict(params, format='json')
    url = 'https://zh.wikisource.org/w/api.php?' + urllib.parse.urlencode(params)
    return json.loads(http_get(url, timeout=45).decode('utf-8'))


RE_SCRIPT = re.compile(r'<(script|style)[^>]*>.*?</\1>', re.S | re.I)
RE_BLOCK = re.compile(r'<(?:br|/p|/div|/h[1-6]|/li|/tr|/td)[^>]*>', re.I)
RE_TAG = re.compile(r'<[^>]+>')


def html_to_text(h: str) -> str:
    """MediaWiki 渲染 HTML → 纯文本。
    不用子树级跳过（MediaWiki 存在不闭合标签，会让深度计数失步），
    改为：正则剥标签 + 行级/段级噪声过滤。"""
    import html as _h
    h = RE_SCRIPT.sub('', h)
    h = RE_BLOCK.sub('\n', h)
    h = RE_TAG.sub('', h)
    h = _h.unescape(h)
    keep = []
    for ln in h.split('\n'):
        s = ln.strip()
        if not s:
            keep.append('')
            continue
        if '←' in s or '→' in s:          # 上一回/下一回 导航
            continue
        if s.startswith('↑') or s.startswith('◀') or s.startswith('▶'):
            continue
        if re.match(r'^(上一[回卷章节篇]|下一[回卷章节篇]|目[录錄])', s):
            continue
        keep.append(s)
    return '\n'.join(keep)


def _is_body_line(s: str) -> bool:
    """正文行判据：含句读或足够长。导航行（篇名/日期/节号）既短又无句读。"""
    return '。' in s or '，' in s or len(s) >= 20


def _drop_leading_nav(t: str) -> str:
    """丢掉页首导航块：第一个「像正文的行」之前的内容。
    维基文库单篇页顶部是「《呐喊》自序 / 狂人日记 / 1918年4月 / 孔乙己」这种
    **相邻篇名 + 日期**，混进正文会让模型学到错误的篇名邻接。"""
    lines = t.split('\n')
    at = next((i for i, s in enumerate(lines) if _is_body_line(s.strip())), 0)
    return '\n'.join(lines[at:])


def ws_page_text(title: str) -> str:
    d = ws_api({'action': 'parse', 'page': title, 'prop': 'text',
                'formatversion': 2, 'redirects': 1})
    return _drop_leading_nav(html_to_text(d.get('parse', {}).get('text', '')))


_WS_NUM = re.compile(r'/第(\d+)回')
_WS_CN = re.compile(r'/第([一二三四五六七八九十百]+)回')
_CN_DIGITS = {c: i for i, c in enumerate('零一二三四五六七八九', 0)}


def _ws_key(t: str):
    m = _WS_NUM.search(t)
    if m:
        return (0, int(m.group(1)))
    m = _WS_CN.search(t)
    if m:
        s = m.group(1)
        n = 0
        for ch in s:
            n = n * 10 + _CN_DIGITS.get(ch, 0)
        return (0, n)
    return (1, t)


def fetch_wikisource(limit: int | None = None) -> list:
    stories = []
    books = WIKISOURCE_BOOKS[:limit] if limit else WIKISOURCE_BOOKS
    for title, bucket in books:
        cp = os.path.join(CACHE_DIR, 'wikisource', f'{slug(title)}.json')
        if os.path.exists(cp) and os.path.getsize(cp) > 0:
            with open(cp, 'r', encoding='utf-8') as f:
                pages = json.load(f)
        else:
            try:
                d = ws_api({'action': 'query', 'list': 'prefixsearch',
                            'pssearch': title, 'pslimit': 500})
                subs = [p['title'] for p in d.get('query', {}).get('prefixsearch', [])
                        if p['title'] == title or p['title'].startswith(title + '/')]
                if not subs:
                    subs = [title]
                subs = sorted(set(subs), key=_ws_key)
                # 有子页时丢弃索引页本身（其上只有书名/作者/目录等导航内容）
                real = [x for x in subs if x.startswith(title + '/')]
                if real:
                    subs = real
                pages = []
                for t in subs:
                    try:
                        pages.append({'title': t, 'text': ws_page_text(t)})
                    except Exception:  # noqa: BLE001
                        pages.append({'title': t, 'text': ''})
                    time.sleep(0.3)  # 维基文库有速率限制（实测偶发 429），礼貌退避
            except Exception as e:  # noqa: BLE001
                print(f'  [!] Wikisource {title} 失败: {e}')
                continue
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            with open(cp, 'w', encoding='utf-8') as f:
                json.dump(pages, f, ensure_ascii=False)

        text = clean('\n\n'.join(p['text'] for p in pages if p.get('text')), 'wikisource')
        print(f'  WS {title:<18} {len(text):>8} 字 ({len(pages)} 页)')
        if len(text) < 500:
            print('        [skip] 正文过短（维基文库尚未录入完）')
            continue
        stories.append({
            'title': title, 'bucket': bucket, 'text': text, 'source': 'wikisource',
            'license': 'CC BY-SA 4.0 / Public Domain (中文维基文库)',
            'ref': 'https://zh.wikisource.org/wiki/' + urllib.parse.quote(title),
            'note': XD_NOTE.get(title, ''),
        })
    return stories


# ---------------------------------------------------------------------------
# 4c. NiuTrans/Classical-Modern（327 本公版古籍原文 / 612 MB）
# ---------------------------------------------------------------------------
# 许可：仓库标注（见 README）数据均来自互联网并标注出处；引用者按"使用即接受许可"。
# 已测：327 部古文原文覆盖 211 部（含章/卷/节），结构为 "古文原文/<部>/<章节>/text.txt"。
# 与 gutenberg/hcc/wikisource/xiandai 会有大量重叠（红楼梦/三国/聊斋 等），交由 dedupe 按
# 「保留最长版本」处理；新增的笔记小说/传奇短篇/三刻拍案惊奇 自动入桶。
NUTRANS_URL = ('https://codeload.github.com/NiuTrans/Classical-Modern/'
               'zip/refs/heads/main')
NUTRANS_ZIP = os.path.join(CACHE_DIR, 'nutrans-classical-modern.zip')

# 强制归到 gudian 桶的「笔记小说 / 传奇短篇」清单：
# - 避免长篇白话小说（红楼梦/三国 等）参与 → 保留现有更长版本
# - 增量为 NiuTrans 独有、现有桶没有的短篇与笔记类
NUTRANS_FORCE = {
    # 笔记小说（公版，南北朝至清）
    '世说新语', '容斋随笔', '酉阳杂俎', '新齐谐', '池北偶谈',
    '幽明录', '西京杂记', '搜神后记', '神仙传', '华阳国志',
    '武林旧事', '蛮书', '唐才子传', '鹤林玉露', '桯史',
    '万历野获编', '五灯会元', '涌幢小品', '坚瓠集', '茶余客话',
    '梦溪笔谈', '北梦琐言', '鹤山笔录', '野客丛书', '麈史',
    # 传奇短篇
    '剪灯新话', '剪灯余话', '剪灯别话',
    # 三刻
    '三刻拍案惊奇',
    # 地理/风物志（含叙事成分）
    '大唐西域记',
}


def _nutrans_zip() -> 'zipfile.ZipFile':
    """缓存式下载/解压 NiuTrans/Classical-Modern 主分支 zip，返回打开的 ZipFile。"""
    import zipfile as _zf
    if not os.path.exists(NUTRANS_ZIP) or os.path.getsize(NUTRANS_ZIP) < 1024:
        data = http_get(NUTRANS_URL, retries=2, timeout=180)
        os.makedirs(os.path.dirname(NUTRANS_ZIP), exist_ok=True)
        with open(NUTRANS_ZIP, 'wb') as f:
            f.write(data)
    return _zf.ZipFile(NUTRANS_ZIP, 'r')


def fetch_nutrans() -> list:
    """按 zip 内 "古文原文/<部>/.../text.txt" 路径枚举。每部拼所有章节为一条。

    路径深度不一：短篇（西游记回）3 段（'卷一'）；长卷（聊斋章/篇）4-5 段；
    卷中卷（剪灯新话 26 篇）5 段；少数 6 段。**统一按"古文原文/<作品>/..."** 切，
    作品名是 parts[2]，所有同名 `text.txt` 都聚到该作品。"""
    import collections as _c
    try:
        z = _nutrans_zip()
    except Exception as e:  # noqa: BLE001
        print(f'  [!] NiuTrans zip 下载/打开失败: {e}')
        return []
    per_work: dict = _c.OrderedDict()
    for name in z.namelist():
        parts = name.split('/')
        if '古文原文' not in parts or parts[-1] != 'text.txt':
            continue
        try:
            idx = parts.index('古文原文')
        except ValueError:
            continue
        work = parts[idx + 1]
        if not work:
            continue
        try:
            text = z.read(name).decode('utf-8', errors='replace')
        except Exception:  # noqa: BLE001
            continue
        per_work.setdefault(work, []).append(text)
    stories = []
    for work, chunks in per_work.items():
        if not chunks:
            continue
        body = clean('\n\n'.join(chunks), 'gutenberg')
        if len(body) < 500:
            continue
        # 白名单才入桶（避免 dedupe 算冗余 hash）；未在白名单的部会被记
        # 在 sources.json 的 works 列表里、被 dedupe 视为另一版本
        if work not in NUTRANS_FORCE:
            continue
        stories.append({
            'title': work, 'bucket': 'gudian', 'text': body,
            'source': 'github:NiuTrans/Classical-Modern',
            'license': '仓库注明来源；引用者按许可（公版古籍，CC BY-NC-SA4.0 仓库）',
            'ref': f'https://github.com/NiuTrans/Classical-Modern（古文原文/{work}/）',
            'note': f'{len(chunks)} 章节，由 zip 拼接',
        })
    print(f'  [NiuTrans 古文  ] {len(stories)} 部 / '
          f'{sum(len(s["text"]) for s in stories)} 字')
    return stories


def fetch_huggingface(limit: int | None = None) -> list:
    """HuggingFace 开放许可数据集：下载 jsonl → 拼接 → 清洗（含繁转简）。"""
    stories = []
    books = HF_BOOKS[:limit] if limit else HF_BOOKS
    for i, spec in enumerate(books, 1):
        url = (f"https://huggingface.co/datasets/{spec['repo']}"
               f"/resolve/main/{spec['file']}")
        cp = os.path.join(CACHE_DIR, 'hf',
                          spec['repo'].replace('/', '__') + '__' + spec['file'])
        try:
            raw = http_get_cached(url, cp).decode('utf-8', errors='replace')
        except Exception as e:  # noqa: BLE001
            print(f'  [!] HF {spec["repo"]} 失败: {e}')
            continue
        parts = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            t = o.get(spec['field']) or ''
            if t:
                parts.append(t)
        text = clean('\n\n'.join(parts), 'hf')
        print(f'  [{i}/{len(books)}] {spec["title"]:<24} {len(parts):>5} 段 '
              f'{len(text):>9} 字')
        if len(text) < 500:
            print('        [skip] 正文过短')
            continue
        stories.append({
            'title': spec['title'], 'bucket': spec['bucket'], 'text': text,
            'source': 'huggingface', 'license': spec['license'],
            'ref': spec['ref'], 'note': spec.get('note', ''),
        })
    return stories


def fetch_tinystories(n: int) -> list:
    """从已下载的 TinyStories 中文包取前 n 篇，**每篇作为一部独立作品**：
    这样 PRD 要求的「故事家族不跨集合」在切分时天然成立。
    只读 tar 内第一个分片，不整包解压（整包 2.6 GB / 48 分片）。"""
    spec = HF_TINYSTORIES
    if not os.path.exists(spec['archive']):
        print(f'  [!] 未找到 {spec["archive"]}，跳过 TinyStories')
        print(f'      下载：curl -L -C - -o {spec["archive"]} {HF_TINYSTORIES_URL}')
        return []
    stories = []
    with tarfile.open(spec['archive'], 'r:gz') as tf:
        fh = tf.extractfile(spec['member'])
        for ln in io.TextIOWrapper(fh, encoding='utf-8', errors='replace'):
            if len(stories) >= n:
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            t = (o.get(spec['field']) or '').strip()
            if len(t) < 200:   # 与 allocate 的 200 字下限对齐，避免进 sources 却进不了桶
                continue
            stories.append({
                'title': f'ts_zh_{len(stories) + 1:04d}', 'bucket': 'story',
                'text': clean(t, 'hf'), 'source': 'huggingface',
                'license': spec['license'], 'ref': spec['ref'],
                'note': 'TinyStories 中文译文（合成）',
            })
    total = sum(len(s['text']) for s in stories)
    print(f'  [tiny] {len(stories)} 篇短故事，合计 {total} 字'
          f'（均长 {total // max(1, len(stories))} 字）')
    return stories


# ---------------------------------------------------------------------------
# 4b. 本地学习桶：知乎（HF）+ 网文切片（流式）+ 本地导入位（天涯/猫扑等）
# ---------------------------------------------------------------------------
def _iter_tianya_threads(xml_text: str, min_chars: int, max_threads: int):
    """天涯 XML → 一帖一条。每条把各 <utt> 发言按行拼起来。
    发言前缀形如 `@whc20109898：2017-09-1713:10:31`（**时间戳常不带空格**，实测如此），
    必须剥掉：否则模型会把「用户名 + 时间戳」当作正文的高频片段学走。"""
    head = re.compile(r'^@[^\s：:]{1,32}[：:]\s*\d{4}-\d{2}-\d{2}\s*\d{0,2}:?\d{0,2}:?\d{0,2}\s*')
    tail = re.compile(r'\d{4}-\d{2}-\d{2}\s*\d{0,2}:?\d{0,2}:?\d{0,2}\s*')
    n = 0
    for block in re.findall(r'<s>(.*?)</s>', xml_text, re.S):
        if n >= max_threads:
            return
        turns = []
        for utt in re.findall(r'<utt[^>]*>(.*?)</utt>', block, re.S):
            u = head.sub('', utt.strip())
            u = tail.sub('', u).strip()
            if u:
                turns.append(u)
        text = '\n'.join(turns)
        if len(text) < min_chars:
            continue
        n += 1
        yield {'text': text}


def _iter_mnbvc_tianya(raw: str, min_chars: int, max_records: int):
    """MNBVC 天涯重启主片 → 一帖一条。每条含主题+所有楼回复。
    记录里的"扩展字段"是真 JSON-as-string（双层 json.loads），要：
      1) 抽出版块、回复人、回复时间等元数据；
      2) 在文本里**剥掉**整段 JSON，避免模型学到乱码字符。
    """
    n = 0
    for ln in raw.split('\n'):
        if n >= max_records:
            return
        ln = ln.strip()
        if not ln:
            continue
        try:
            o = json.loads(ln)
        except Exception:  # noqa: BLE001
            continue
        title = (o.get('主题') or '').strip()
        replies = o.get('回复') or []
        if not isinstance(replies, list):
            continue
        parts = [title] if title else []
        for r in replies:
            if not isinstance(r, dict):
                continue
            txt = (r.get('回复') or '').strip()
            if not txt:
                continue
            parts.append(txt)
        text = '\n'.join(parts)
        # 抽取并删掉"扩展字段"JSON-as-string（防止模型学到 `{\"回复人\":...}`）
        # 注：MNBVC 的"扩展字段"在楼回复里是 dict-as-string；
        # "版块"在元数据.扩展字段里（同样是 JSON-as-string）。
        bk = ''
        meta = o.get('元数据') or {}
        if isinstance(meta, dict):
            meta_ext = meta.get('扩展字段', '')
            if isinstance(meta_ext, str) and meta_ext.startswith('{'):
                try:
                    md = json.loads(meta_ext)
                    bk = md.get('版块', '') if isinstance(md, dict) else ''
                except Exception:
                    bk = ''
        ext_match = re.search(r'"扩展字段"\s*:\s*"(\{.*?\})"\s*[,}]', text)
        if ext_match:
            text = text.replace(ext_match.group(0), '')
        if len(text) < min_chars:
            continue
        n += 1
        yield {
            'text': text,
            '板块': bk,
            '回复数': o.get('回复'),
        }


def _iter_records(raw: bytes, spec: dict):
    """按 spec['fmt'] 把下载字节变成一条条 dict。"""
    fmt = spec.get('fmt', 'jsonl')
    enc = spec.get('encoding', 'utf-8')
    if fmt == 'jsonl_gz':
        import gzip
        with gzip.open(io.BytesIO(raw), 'rt', encoding=enc, errors='replace') as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        yield json.loads(ln)
                    except Exception:  # noqa: BLE001
                        continue
    elif fmt == 'json_list':
        try:
            for o in json.loads(raw.decode(enc, errors='replace')):
                if isinstance(o, dict):
                    yield o
        except Exception:  # noqa: BLE001
            return
    elif fmt == 'tsv':
        import csv
        txt = raw.decode(enc, errors='replace')
        for row in csv.DictReader(io.StringIO(txt), delimiter='\t'):
            yield row
    else:  # jsonl（含 Range 截断的尾行不完整 → json.loads 失败即丢）
        txt = raw.decode(enc, errors='replace')
        for ln in txt.split('\n'):
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:  # noqa: BLE001
                continue


def fetch_localugc(per_source_cap: int = 4000, min_chars: int = 100) -> list:
    """知乎（HF）+ 网文**全量分片** + Pixiv/v3ucn + MNBVC + 本地导入位。

    产出 `bucket='localugc'` 的作品：知乎/树洞/天涯类**每条记录一篇**（故事级切分
    天然成立），网文/Pixiv 类**每部小说一篇**；本地导入位每个文件一篇。"""
    stories: list = []

    for spec in LOCALUGC_HF:
        url = f"https://huggingface.co/datasets/{spec['repo']}/resolve/main/{spec['file']}"
        cp = os.path.join(CACHE_DIR, 'localugc',
                          slug(spec['repo']) + '__' + slug(spec['file']))
        try:
            raw = http_get_cached(url, cp)
        except Exception as e:  # noqa: BLE001
            print(f'  [!] localugc {spec["repo"]} 失败: {e}')
            continue
        n, chars = 0, 0
        for o in _iter_records(raw, spec):
            if n >= per_source_cap:
                break
            t = (o.get(spec['field']) or '').strip()
            if len(t) < min_chars:
                continue
            t = clean(t, 'localugc')
            if len(t) < min_chars:
                continue
            n += 1
            chars += len(t)
            stories.append({
                'title': f'{slug(spec["title"])}_{n:04d}', 'bucket': 'localugc',
                'text': t, 'source': f'huggingface:{spec["repo"]}',
                'license': spec['license'], 'ref': spec['ref'],
                'note': spec.get('note', ''),
            })
        print(f'  [{spec["title"]:<18}] {n:>5} 条 {chars:>9} 字'
              f'（{spec["file"]}，{spec.get("encoding", "utf-8")}）')

    # 网文：**整片全量**（流式下载到盘、逐行流式解析、按 title 连续段聚章）。
    # 之前是 Range 取前 30 MB（每部只剩开头）；现为完整分片。
    wn = LOCALUGC_WEBNOVEL
    wn_works = wn_chars = 0
    try:
        for sh in wn['shards']:
            remote = f'data/webnovel_{sh}.jsonl'
            url = urllib.parse.quote(
                f"https://huggingface.co/datasets/{wn['repo']}/resolve/main/{remote}",
                safe=':/?&=')
            cp = os.path.join(CACHE_DIR, 'localugc', f'webnovel_{sh}.jsonl')
            http_download_to_file(url, cp, min_size=1 << 30)
            for title, chunks in _iter_webnovel_works(cp, wn['field']):
                text = clean('\n\n'.join(chunks), 'localugc')
                if len(text) < 500:
                    continue
                wn_works += 1
                wn_chars += len(text)
                # 序号入题避免同名书 slug 相互顶替（跨几千部同名很常见）
                stories.append({
                    'title': f'wn{sh}_{wn_works:05d}_{slug(title)[:40]}',
                    'bucket': 'localugc', 'text': text,
                    'source': f'huggingface:{wn["repo"]}', 'license': wn['license'],
                    'ref': wn['ref'], 'note': f'{wn["note"]}·{len(chunks)} 章',
                })
            print(f'  [网文分片{sh} 全量   ] 累计 {wn_works:>5} 部 {wn_chars:>9} 字')
    except Exception as e:  # noqa: BLE001
        print(f'  [!] localugc 网文全量失败: {e}')

    # v3ucn 小源（续写对：input 前文 + output 续文 → 拼为一篇）
    v3 = LOCALUGC_V3UCN
    try:
        url = urllib.parse.quote(
            f"https://huggingface.co/datasets/{v3['repo']}/resolve/main/{v3['file']}",
            safe=':/?&=')
        cp = os.path.join(CACHE_DIR, 'localugc', 'v3ucn_dataset.json')
        raw = http_get_cached(url, cp).decode(v3['encoding'], errors='replace')
        v3_n = v3_chars = 0
        for i, o in enumerate(json.loads(raw), 1):
            t = ((o.get('input') or '').strip() + '\n' + (o.get('output') or '').strip()).strip()
            if len(t) < v3['min_chars']:
                continue
            t = clean(t, 'localugc')
            if len(t) < v3['min_chars']:
                continue
            v3_n += 1
            v3_chars += len(t)
            stories.append({
                'title': f'v3ucn_{i:04d}', 'bucket': 'localugc', 'text': t,
                'source': f'huggingface:{v3["repo"]}', 'license': v3['license'],
                'ref': v3['ref'], 'note': v3['note'],
            })
        print(f'  [v3ucn 小说集     ] {v3_n:>5} 条 {v3_chars:>9} 字')
    except Exception as e:  # noqa: BLE001
        print(f'  [!] localugc v3ucn 失败: {e}')

    # Pixiv 中文小说（zip 1.76 GB）：结构为 `PixivNovel/<用户ID>/<作品ID>.txt`（+ 同名 -meta.txt）。
    # 每个非 meta 的 .txt = 一部作品。实测 307,188 个成员。
    px = LOCALUGC_PIXIV
    try:
        url = urllib.parse.quote(
            f"https://huggingface.co/datasets/{px['repo']}/resolve/main/{px['file']}",
            safe=':/?&=')
        cp = os.path.join(CACHE_DIR, 'localugc', 'PixivNovel.zip')
        http_download_to_file(url, cp, min_size=1 << 30)
        import zipfile as _zf2
        px_n = px_chars = 0
        with _zf2.ZipFile(cp, 'r') as zf:
            for m in zf.infolist():
                if px_n >= px['max_works'] or m.is_dir():
                    continue
                name = m.filename
                if not name.lower().endswith('.txt') or name.endswith('-meta.txt'):
                    continue
                body = zf.read(m).decode(px['encoding'], errors='replace')
                if len(body) < px['min_chars']:
                    continue
                body = clean(body, 'localugc')
                if len(body) < px['min_chars']:
                    continue
                px_n += 1
                px_chars += len(body)
                stories.append({
                    'title': f'pixiv_{px_n:05d}', 'bucket': 'localugc', 'text': body,
                    'source': f'huggingface:{px["repo"]}', 'license': px['license'],
                    'ref': px['ref'], 'note': f'{px["note"]}·{name}',
                })
        print(f'  [Pixiv 小说       ] {px_n:>5} 条 {px_chars:>9} 字')
    except Exception as e:  # noqa: BLE001
        print(f'  [!] localugc Pixiv 失败: {e}')

    # 天涯论坛（GitHub 抓取语料，XML；辩论/对话体）
    ty = LOCALUGC_TIANYA
    cp = os.path.join(CACHE_DIR, 'localugc', ty['archive'])
    try:
        blob = http_get_cached(ty['url'], cp)
        with tarfile.open(fileobj=io.BytesIO(blob), mode='r:gz') as tf:
            xml_text = tf.extractfile(ty['tar_member']).read().decode(
                ty['encoding'], errors='replace')
        ty_n = ty_chars = 0
        for i, o in enumerate(_iter_tianya_threads(xml_text, ty['min_chars'],
                                                  ty['max_threads']), 1):
            t = clean(o['text'], 'localugc')
            if len(t) < ty['min_chars']:
                continue
            ty_n += 1
            ty_chars += len(t)
            stories.append({
                'title': f'tianya_{i:05d}', 'bucket': 'localugc', 'text': t,
                'source': f"github:{ty['repo']}", 'license': ty['license'],
                'ref': ty['ref'], 'note': ty['note'],
            })
        print(f'  [天涯论坛        ] {ty_n:>5} 条 {ty_chars:>9} 字'
              f'（{ty["tar_member"]}）')
    except Exception as e:  # noqa: BLE001
        print(f'  [!] localugc 天涯语料失败: {e}')

    # MNBVC 一组源：天涯主片 / 北大树洞 / 清华树洞（同 schema，直接复用 _iter_mnbvc_tianya）
    import zipfile as _zipfile
    import gzip as _gzip
    mnb_groups = [
        (LOCALUGC_MNBVC_TIANYA, '天涯主片', 'mnbvc_ty'),
        (LOCALUGC_MNBVC_PKUHOLE, '北大树洞', 'mnbvc_pk'),
        (LOCALUGC_MNBVC_THUHOLE, '清华树洞', 'mnbvc_th'),
    ]
    for mty, tag, prefix in mnb_groups:
        mnb_n = mnb_chars = 0
        try:
            for sub_path in mty['files']:
                fn = os.path.basename(sub_path)
                cp = os.path.join(CACHE_DIR, 'localugc', 'mnbvc_' + fn)
                # 路径含中文（如 pkuhole.20230253.2.论坛）必须逐段 quote
                url = f"https://huggingface.co/datasets/{mty['repo']}/resolve/main/{sub_path}"
                url = urllib.parse.quote(url, safe=':/?&=')
                blob = http_get_cached(url, cp, retries=3)
                # MNBVC 多数文件是 .jsonl.gz 直接 gzip；天涯主片是 zip 包一个 jsonl。
                # 先按 .gz 解，不行再按 zip 解。
                raw = ''
                try:
                    raw = _gzip.decompress(blob).decode(mty['encoding'], errors='replace')
                except Exception:
                    try:
                        with _zipfile.ZipFile(io.BytesIO(blob), 'r') as zf:
                            zname = fn.replace('.zip', '.jsonl')
                            with zf.open(zname) as fp:
                                raw = fp.read().decode(mty['encoding'], errors='replace')
                    except Exception as ee:
                        raise RuntimeError(f'既非 gzip 也非 zip: {ee}')
                for i, o in enumerate(_iter_mnbvc_tianya(raw, mty['min_chars'],
                                                         mty['max_records']), 1):
                    t = clean(o['text'], 'localugc')
                    if len(t) < mty['min_chars']:
                        continue
                    mnb_n += 1
                    mnb_chars += len(t)
                    stories.append({
                        'title': f'{prefix}_{mnb_n:05d}', 'bucket': 'localugc', 'text': t,
                        'source': f"huggingface:{mty['repo']}",
                        'license': mty['license'],
                        'ref': mty['ref'] + f'/{sub_path}',
                        'note': f"{mty['note']}·{o.get('板块','?')}·{o.get('回复数','?')} 回",
                    })
            print(f'  [MNBVC {tag:<8}] {mnb_n:>5} 条 {mnb_chars:>9} 字'
                  f'（{len(mty["files"])} 文件）')
        except Exception as e:  # noqa: BLE001
            print(f'  [!] localugc MNBVC {tag} 失败: {e}')

    # 本地导入位：data/raw/_cache/localugc/*.txt —— 天涯/猫扑/知乎/网文 dump 的接入口。
    import glob as _glob
    local = sorted(_glob.glob(os.path.join(CACHE_DIR, 'localugc', '*.txt')))
    for p in local:
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                raw_txt = f.read()
        except Exception as e:  # noqa: BLE001
            print(f'  [!] 本地导入失败 {p}: {e}')
            continue
        title = os.path.splitext(os.path.basename(p))[0]
        text = clean(raw_txt, 'localugc')
        print(f'  [本地导入 {title:<12}] {len(text):>9} 字')
        if len(text) < 500:
            continue
        stories.append({
            'title': title, 'bucket': 'localugc', 'text': text,
            'source': 'local-import', 'license': '本地文件（权利状态自辨，不可分发）',
            'ref': p, 'note': '本地导入位：天涯/猫扑等 dump 放这里',
        })
    if not local:
        print('  [本地导入位       ] 空 —— 把天涯/猫扑等 dump 存成 '
              f'{os.path.join(CACHE_DIR, "localugc")}/<标题>.txt（UTF-8）即自动入库')

    # 杂志导入位：data/raw/_cache/zazhi/*.txt —— 现代文学期刊
    # （今古传奇 / 科幻世界 / 人民文学 / 收获 / 十月 / 萌芽 等）。
    # **仅个人本地学习用**：这些期刊均不公版（创刊时间晚于公版线），平台 ToS 也禁止批量抓取，
    # 按本仓库项目立场 `docs/DATA_CANDIDATES.md` §二属同一风险等级；接入仅在你已获得合法 dump
    # （自存、机构数据库离线、OCR 纸本等）的前提下启用。
    zazhi = sorted(_glob.glob(os.path.join(CACHE_DIR, 'zazhi', '*.txt')))
    for p in zazhi:
        try:
            with open(p, 'r', encoding='utf-8', errors='replace') as f:
                raw_txt = f.read()
        except Exception as e:  # noqa: BLE001
            print(f'  [!] 杂志导入失败 {p}: {e}')
            continue
        title = os.path.splitext(os.path.basename(p))[0]
        text = clean(raw_txt, 'localugc')
        print(f'  [杂志导入 {title:<14}] {len(text):>9} 字')
        if len(text) < 500:
            continue
        stories.append({
            'title': title, 'bucket': 'localugc', 'text': text,
            'source': 'local-import:magazine', 'license': '本地文件（期刊非公版；按你已获授权的方式使用）',
            'ref': p, 'note': '杂志导入位：今古传奇/科幻世界/人民文学/收获等 dump 放这里',
        })
    if not zazhi:
        print('  [杂志导入位       ] 空 —— 把今古传奇/科幻世界/人民文学/收获等 dump '
              f'存成 {os.path.join(CACHE_DIR, "zazhi")}/<刊名>.txt（UTF-8）即自动入库')
    return stories


# ---------------------------------------------------------------------------
# 5. 去重 + 分桶 + 落盘
# ---------------------------------------------------------------------------
def dedupe(stories: list) -> list:
    """同一标题多源时保留最长版本，并记录全部来源。"""
    best = {}
    for s in stories:
        k = s['title']
        prev = best.get(k)
        if prev is None:
            s = dict(s, all_sources=[s['source']])
            best[k] = s
        else:
            if len(s['text']) > len(prev['text']):
                s = dict(s, all_sources=prev['all_sources'] + [s['source']])
                best[k] = s
                print(f'       [dedupe] {k}: 采用 {s["source"]} '
                      f'({len(s["text"])}字) 覆盖 {prev["source"]} ({len(prev["text"])}字)')
            else:
                prev['all_sources'].append(s['source'])
                print(f'       [dedupe] {k}: 保留 {prev["source"]} '
                      f'({len(prev["text"])}字)，丢弃 {s["source"]} ({len(s["text"])}字)')
    return list(best.values())


def allocate(items: list, max_chars: int) -> list:
    """把每桶的字数额度分给各作品。两种模式：

    - **均分**（作品数少，默认）：每部配额 = 上限/部数，短作品用不完的额度轮转补给
      被截断的长作品，保证覆盖面。
    - **贪心**（作品数极多，均分后每部 <200 字会被下限整桶滤空——`localugc` 踩过：
      20 万部均分 800 万上限，每部 40 字 → 实际入库 0 字）：改按字数从大到小
      依次整部装入，装满上限为止。保进桶的都是完整作品。"""
    if not items:
        return []
    quota = max(0, max_chars // len(items))
    if quota < 200:
        taken_total = 0
        out = []
        for s in sorted(items, key=lambda x: -len(x['text'])):
            if taken_total >= max_chars:
                break
            take = min(len(s['text']), max_chars - taken_total)
            if take < 200:
                continue
            taken_total += take
            out.append((s, s['text'][:take]))
        return out
    taken = {s['title']: min(len(s['text']), quota) for s in items}
    used = sum(taken.values())
    guard = 0
    while max_chars - used > 512 and guard < 60:
        guard += 1
        room = [s for s in items if taken[s['title']] < len(s['text'])]
        if not room:
            break
        share = max(1, (max_chars - used) // len(room))
        progressed = False
        for s in room:
            add = min(share, len(s['text']) - taken[s['title']], max_chars - used)
            if add <= 0:
                continue
            taken[s['title']] += add
            used += add
            progressed = True
        if not progressed:
            break
    return [(s, s['text'][:taken[s['title']]])
            for s in items if taken[s['title']] >= 200]


def assemble(stories: list, max_chars: int) -> dict:
    """分桶拼装。`max_chars` 是默认桶上限；BUCKETS 元里写了 `max_chars` 的桶单独覆盖。"""
    corpus = {}
    for b in BUCKET_ORDER:
        items = [s for s in stories if s['bucket'] == b]
        items.sort(key=lambda s: -len(s['text']))
        bucket_cap = BUCKETS.get(b, {}).get('max_chars', max_chars)
        parts, used, chars = [], [], 0
        for s, take in allocate(items, bucket_cap):
            parts.append(take)
            used.append({'title': s['title'], 'source': s['source'],
                         'chars': len(take), 'ref': s['ref']})
            chars += len(take)
        text = '\n\n'.join(parts)
        meta = BUCKETS[b]
        corpus[b] = {
            'name': meta['name'],
            'desc': meta['desc'],
            'text': text,
            'source': '; '.join(sorted({u['source'] for u in used})),
            'license': 'Public Domain / CC BY-SA（详见 data/sources.json）',
            'note': f'{len(used)} 部作品，{chars} 字（均分上限 {max_chars}）',
            'stories': used,
        }
    return corpus


def dump_works(stories: list, out_root: str) -> int:
    """把每部作品清洗后的正文单独落盘到 data/processed/<bucket>/<title>.txt，
    便于人工抽查与后续按作品切分（build_splits/sweep 都按这个逐作品文件取正文）。
    注：localugc 桶十几万条小文件、全量写一次要十几分钟——不是卡死，等它写完。"""
    n = 0
    for st in stories:
        d = os.path.join(out_root, st['bucket'])
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, slug(st['title']) + '.txt'), 'w',
                  encoding='utf-8') as f:
            f.write(st['text'])
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='gutenberg,hcc,wikisource,hf,nutrans,localugc',
                    help='逗号分隔：gutenberg,hcc,wikisource,hf,nutrans,localugc')
    ap.add_argument('--limit', type=int, default=None, help='每个源最多抓 N 本（试跑用）')
    ap.add_argument('--tiny-n', type=int, default=1000,
                    help='TinyStories 中文短故事取前 N 篇（story 桶）')
    ap.add_argument('--ugc-cap', type=int, default=2500,
                    help='每个知乎源最多收 N 条（localugc 桶；网文按部收）')
    ap.add_argument('--max-chars', type=int, default=2_000_000,
                    help='每个主题桶保留的最大字数')
    ap.add_argument('--list-sources', action='store_true', help='只打印源清单')
    ap.add_argument('--out', default=CORPUS_OUT)
    ap.add_argument('--sources-out', default=SOURCES_OUT,
                    help='源审计落盘路径。**试跑必须改这个**：它默认写规范文件，'
                         '而该文件不随 --out 走（--only 某源时会把其它源的记录覆盖掉）')
    ap.add_argument('--processed-dir', default=os.path.join(ROOT, 'data', 'processed'),
                    help='每部作品清洗后正文的落盘目录（可人工抽查）')
    ap.add_argument('--no-dump', action='store_true', help='不落盘逐作品文本')
    args = ap.parse_args()

    if args.list_sources:
        print(f'Gutenberg（{len(GUTENBERG_BOOKS)} 本）:')
        for eid, t, b in GUTENBERG_BOOKS:
            print(f'  {eid:>6} [{b:<6}] {t}')
        print(f'\nGitHub {HCC_REPO}（{len(HCC_DIRS)} 部）:')
        for d, t, b in HCC_DIRS:
            print(f'  [{b:<6}] {d} -> {t}')
        print(f'\n维基文库（{len(WIKISOURCE_BOOKS)} 部）:')
        for t, b in WIKISOURCE_BOOKS:
            print(f'  [{b:<6}] {t}')
        print(f'\nHuggingFace（{len(HF_BOOKS)} 个数据集 + TinyStories 短故事）:')
        for s in HF_BOOKS:
            print(f"  [{s['bucket']:<6}] {s['repo']} -> {s['title']}（{s['license']}）")
        print(f"  [story ] {HF_TINYSTORIES['repo']} -> 前 N 篇短故事"
              f"（{HF_TINYSTORIES['license']}）")
        print(f'\n本地学习桶 localugc（⚠️ 仅个人本地学习；不可再分发 / 商用 / 入评测桶）:')
        for s in LOCALUGC_HF:
            print(f"  [{s['bucket']:<8}] {s['repo']}/{s['file']} -> {s['title']}"
                  f"（{s['license']}）")
        wn = LOCALUGC_WEBNOVEL
        print(f"  [{wn['bucket']:<8}] {wn['repo']} -> 整片全量分片 "
              f"{wn['shards']}（每片 4.2 GB，{wn['license']}）")
        for extra in (LOCALUGC_PIXIV, LOCALUGC_V3UCN):
            print(f"  [{extra['bucket']:<8}] {extra['repo']} -> {extra['title']}"
                  f"（{extra['license']}）")
        ty = LOCALUGC_TIANYA
        print(f"  [{ty['bucket']:<8}] {ty['repo']} -> {ty['title']}"
              f"（{ty['license']}）")
        print(f"  [{wn['bucket']:<8}] 本地导入位：{os.path.join(CACHE_DIR, 'localugc')}"
              f"/*.txt（天涯/猫扑等 dump 放这里即自动入库）")
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    only = {x.strip() for x in args.only.split(',') if x.strip()}
    stories = []
    if 'gutenberg' in only:
        print('== 1/6 Project Gutenberg ==')
        stories += fetch_gutenberg(args.limit)
    if 'hcc' in only:
        print('== 2/6 GitHub hanchuancaolu ==')
        stories += fetch_hcc()
    if 'wikisource' in only:
        print('== 3/6 中文维基文库 ==')
        stories += fetch_wikisource(args.limit)
    if 'hf' in only:
        print('== 4/6 HuggingFace 开放许可数据集 ==')
        stories += fetch_huggingface(args.limit)
        stories += fetch_tinystories(args.tiny_n)
    if 'nutrans' in only:
        print('== 5/6 NiuTrans 古文原文（笔记/传奇/三刻，公版）==')
        stories += fetch_nutrans()
    if 'localugc' in only:
        print('== 6/6 本地学习桶（知乎 / 网文切片 / 本地导入位）==')
        print('    注意：仅限个人本地学习；不可再分发 / 商用 / 入评测桶')
        stories += fetch_localugc(per_source_cap=args.ugc_cap)

    print('\n== 跨源去重 ==')
    stories = dedupe(stories)

    corpus = assemble(stories, args.max_chars)

    if not args.no_dump:
        # 树洞类（localugc + mnbvc_/天涯）量大时写 16+ 万个小文件极慢；
        # dump_works 改成"每桶一个 _combined.jsonl" + 小作品逐个 .txt：
        # - 树洞 16 万条 → 1 个 jsonl（快）
        # - 其他桶逐个 .txt（保留人工抽查粒度）
        n = dump_works(stories, args.processed_dir)
        print(f'[*] 逐作品正文 -> {args.processed_dir} ({n} 个 txt)')

    # 备份原始 corpus.json（只备份一次，且始终备份真实的语料文件），保留 custom 桶
    if os.path.exists(CORPUS_OUT) and not os.path.exists(CORPUS_BAK):
        with open(CORPUS_OUT, 'r', encoding='utf-8') as f:
            old = json.load(f)
        with open(CORPUS_BAK, 'w', encoding='utf-8') as f:
            json.dump(old, f, ensure_ascii=False, indent=1)
        print(f'[*] 原始 corpus.json 已备份 -> {CORPUS_BAK}')
    if os.path.exists(args.out):
        with open(args.out, 'r', encoding='utf-8') as f:
            try:
                old = json.load(f)
            except Exception:  # noqa: BLE001
                old = {}
        if 'custom' in old:
            corpus['custom'] = old['custom']
            print('[*] 保留 custom 桶')
    if 'custom' not in corpus:
        corpus['custom'] = {'name': '自定义语料', 'desc': '', 'text': '',
                            'source': None, 'license': None}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, ensure_ascii=False)
    print(f'\n[*] corpus -> {args.out} '
          f'({os.path.getsize(args.out) / 1e6:.1f} MB)')

    # 源审计
    srcs = []
    for s in stories:
        srcs.append({
            'title': s['title'], 'bucket': s['bucket'], 'source': s['source'],
            'all_sources': s.get('all_sources', [s['source']]),
            'chars': len(s['text']), 'sha256': hashlib.sha256(
                s['text'].encode('utf-8')).hexdigest()[:16],
            'license': s['license'], 'ref': s['ref'],
        })
    srcs.sort(key=lambda x: (x['bucket'], -x['chars']))
    audit = {
        'generated_by': 'py/data/fetch_corpus.py',
        'note': '全部为公版或开放许可中文文本；繁转简(opencc t2s)；跨源按标题去重保留最长版本',
        'total_works': len(srcs),
        'total_chars': sum(x['chars'] for x in srcs),
        'by_bucket': {b: {
            'works': sum(1 for x in srcs if x['bucket'] == b),
            'chars': sum(x['chars'] for x in srcs if x['bucket'] == b),
        } for b in BUCKET_ORDER},
        'works': srcs,
    }
    os.makedirs(os.path.dirname(args.sources_out), exist_ok=True)
    with open(args.sources_out, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)

    print('\n== 分桶统计 ==')
    for b in BUCKET_ORDER:
        st = audit['by_bucket'][b]
        print(f'  {b:<8} {st["works"]:>3} 部  {st["chars"]:>9} 字  '
              f'(实际入库 {len(corpus[b]["text"])} 字, 独立字符 '
              f'{len(set(corpus[b]["text"]))})')
    print(f'  custom   保留（UI 上传用）')
    print(f'\n[*] sources -> {args.sources_out}')


if __name__ == '__main__':
    main()
