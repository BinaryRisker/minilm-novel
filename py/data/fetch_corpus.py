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

BUCKETS = {
    'scifi': {'name': '科幻 · 清末科学小说',
              'desc': '月界旅行 / 地底旅行 / 新纪元 / 电世界 / 猫城记（维基文库，公版）'},
    'wuxia': {'name': '武侠 · 侠义公案',
              'desc': '水浒传 / 三侠五义 / 儿女英雄传 / 荡寇志 / 施公案等（Gutenberg，公版）'},
    'urban': {'name': '都市 · 世情谴责',
              'desc': '官场现形记 / 二十年目睹之怪现状 / 老残游记 / 孽海花 / 儒林外史等（公版）'},
    'gudian': {'name': '古典 · 神魔笔记',
               'desc': '红楼梦 / 西游记 / 三国演义 / 封神演义 / 聊斋志异 / 三言二拍等（公版）'},
}
BUCKET_ORDER = ['scifi', 'wuxia', 'urban', 'gudian']


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


def ws_page_text(title: str) -> str:
    d = ws_api({'action': 'parse', 'page': title, 'prop': 'text',
                'formatversion': 2, 'redirects': 1})
    return html_to_text(d.get('parse', {}).get('text', ''))


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
        })
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
    """把每桶的字数额度**均分**给各作品，保证覆盖面；短作品用不完的额度再轮转
    补给被截断的长作品。返回 [(story, text)]，只保留 >=200 字的片段。"""
    if not items:
        return []
    quota = max(0, max_chars // len(items))
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
    corpus = {}
    for b in BUCKET_ORDER:
        items = [s for s in stories if s['bucket'] == b]
        items.sort(key=lambda s: -len(s['text']))
        parts, used, chars = [], [], 0
        for s, take in allocate(items, max_chars):
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
    便于人工抽查与后续按作品切分。"""
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
    ap.add_argument('--only', default='gutenberg,hcc,wikisource',
                    help='逗号分隔：gutenberg,hcc,wikisource')
    ap.add_argument('--limit', type=int, default=None, help='每个源最多抓 N 本（试跑用）')
    ap.add_argument('--max-chars', type=int, default=2_000_000,
                    help='每个主题桶保留的最大字数')
    ap.add_argument('--list-sources', action='store_true', help='只打印源清单')
    ap.add_argument('--out', default=CORPUS_OUT)
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
        return

    os.makedirs(CACHE_DIR, exist_ok=True)
    only = {x.strip() for x in args.only.split(',') if x.strip()}
    stories = []
    if 'gutenberg' in only:
        print('== 1/3 Project Gutenberg ==')
        stories += fetch_gutenberg(args.limit)
    if 'hcc' in only:
        print('== 2/3 GitHub hanchuancaolu ==')
        stories += fetch_hcc()
    if 'wikisource' in only:
        print('== 3/3 中文维基文库 ==')
        stories += fetch_wikisource(args.limit)

    print('\n== 跨源去重 ==')
    stories = dedupe(stories)

    corpus = assemble(stories, args.max_chars)

    if not args.no_dump:
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
    os.makedirs(os.path.dirname(SOURCES_OUT), exist_ok=True)
    with open(SOURCES_OUT, 'w', encoding='utf-8') as f:
        json.dump(audit, f, ensure_ascii=False, indent=1)

    print('\n== 分桶统计 ==')
    for b in BUCKET_ORDER:
        st = audit['by_bucket'][b]
        print(f'  {b:<8} {st["works"]:>3} 部  {st["chars"]:>9} 字  '
              f'(实际入库 {len(corpus[b]["text"])} 字, 独立字符 '
              f'{len(set(corpus[b]["text"]))})')
    print(f'  custom   保留（UI 上传用）')
    print(f'\n[*] sources -> {SOURCES_OUT}')


if __name__ == '__main__':
    main()
