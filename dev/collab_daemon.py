#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zcode 侧后台协作 daemon。

设计原则：
  1. 只读 .collab/inbox/zcode/，只写 .collab/outbox/zcode/（zcode 协议边界）
  3. 不写 .collab/inbox/zcode/（那是 root 写的位，写入会破坏对方'未读'语义）
  4. 不抢同事工作线（文件在 dev/，不动 py/、data/、docs/ 等）
  5. 不强求常驻：one-shot + 可选 watch 由用户启动
  6. 跨平台：polling（无 fsnotify 依赖）；文件锁用 helper 同款 msvcrt/fcntl

对偶关系：
  - dev/collab_monitor.py（root 侧）监听 outbox/zcode/ → 写 inbox/zcode/
  - 本脚本（zcode 侧）监听 inbox/zcode/ → 写 outbox/zcode/

子命令：
  once              单次扫描 inbox，stdout 输出建议（不动 status/log）
  peek              纯摘要（不改任何文件）
  watch --interval 5  后台轮询，新消息追加到 .collab/log/zcode_watch.log

退出码：
  0 = 正常
  1 = 参数错误
  2 = inbox 有 task/blocker/decision 等需 zcode 关注（one-shot 模式
      主动返回 2 方便上层 shell 触发自动执行）
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
COLLAB_DIR = ROOT / ".collab"
INBOX = COLLAB_DIR / "inbox" / "zcode"
OUTBOX = COLLAB_DIR / "outbox" / "zcode"
LOG_FILE = COLLAB_DIR / "log" / "zcode_watch.log"


# ---------- 工具 ----------

def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(p: Path) -> Dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def list_inbox() -> List[Path]:
    if not INBOX.exists():
        return []
    return sorted([p for p in INBOX.iterdir() if p.suffix == ".json"])


def summarize(msg: Dict[str, Any]) -> Tuple[str, str, str]:
    """把一条 inbox 消息浓缩成 (kind, action, summary) 三元组。

    kind  ∈ {task, decision, blocker, question, ack, other}
    action ∈ {run, wait, decide, answer, ignore}

    action 是"raw 建议"——上层（shell / zcode 会话）需要交叉看 inbox
    内多条消息的语义（如 task 与 decision 的 status 冲突时取 decision）。
    daemon 不做最终判定；做的是"把每条消息浓缩好，让上层一眼看清"。
    """
    t = msg.get("type", "other")
    if t == "task":
        return (
            "task",
            "run",  # raw 建议是 run；上层如果看到同 task 的 blocked decision 则降为 wait
            f"派工 task_id={msg.get('task_id') or msg.get('id')} "
            f"status={msg.get('status', 'queued')} "
            f"objective={(msg.get('objective') or msg.get('instruction', ''))[:80]}",
        )
    if t == "decision":
        st = msg.get("status", "n/a")
        # decision 本身就是权威状态——它会覆盖同名 task 的 raw run 建议
        return (
            "decision",
            "wait",
            f"decision={msg.get('decision')} reply_to={msg.get('reply_to')} "
            f"status={st} (覆盖 task, 改用本条)",
        )
    if t == "blocker":
        return ("blocker", "decide", f"blocker id={msg.get('id')} 等 zcode 决策回复")
    if t == "question":
        return ("question", "answer", f"question id={msg.get('id')}")
    if t in ("ack",):
        return ("ack", "ignore", "root 端 ack；无需 zcode 主动响应")
    return ("other", "ignore", f"type={t} id={msg.get('id')}")


def cross_check(msgs: List[Dict[str, Any]]) -> Dict[str, str]:
    """跨消息一致性检查：task 被 blocked decision 覆盖后，effective action=wait。

    返回 {task_id: effective_action}
    """
    eff: Dict[str, str] = {}
    # 先记所有 task 的 raw 建议
    for m in msgs:
        if m.get("type") == "task":
            tid = m.get("task_id") or m.get("id")
            eff[tid] = "run"
    # 用 decision 覆盖
    for m in msgs:
        if m.get("type") == "decision":
            st = (m.get("status") or "").lower()
            rt = m.get("reply_to") or ""
            # decision 可能引用 blocker 而非 task；映射是 blocker_id 关联到 task_id 需要 schema 升级
            # 当前简化：任何 status 以 blocked 开头的 decision 都全局降级为 wait
            if st.startswith("blocked"):
                for tid in list(eff.keys()):
                    eff[tid] = "wait"
    return eff


# ---------- 子命令 ----------

def cmd_once(args: argparse.Namespace) -> int:
    msgs_paths = list_inbox()
    if not msgs_paths:
        print(f"[{_now_iso()}] inbox empty")
        return 0
    msgs: List[Dict[str, Any]] = []
    for p in msgs_paths:
        try:
            msgs.append(_read_json(p))
        except Exception as e:
            print(f"[{_now_iso()}] [parse-error] {p.name}: {e}")
    eff = cross_check(msgs)
    has_actionable = False
    for p, msg in zip(msgs_paths, msgs):
        kind, _action, summary = summarize(msg)
        # 决定 effective action
        if kind == "task":
            tid = msg.get("task_id") or msg.get("id")
            eff_action = eff.get(tid, "run")
        else:
            eff_action = _action
        prefix = {
            ("task", "run"):      "[NEED-RUN] ",
            ("task", "wait"):     "[WAIT-TASK] ",
            ("decision", "wait"): "[ROOT-DECISION] ",
            ("blocker", "decide"): "[ROOT-BLOCKER] ",
            ("question", "answer"): "[NEED-ANS] ",
            ("ack", "ignore"):    "[ACK]      ",
            ("other", "ignore"):  "[UNKNOWN]  ",
        }.get((kind, eff_action), "[UNKNOWN]  ")
        print(f"{prefix}{p.name}  {summary}")
        if eff_action in ("run", "answer", "decide"):
            has_actionable = True
    return 2 if has_actionable else 0


def cmd_peek(args: argparse.Namespace) -> int:
    msgs_paths = list_inbox()
    msgs: List[Dict[str, Any]] = []
    for p in msgs_paths:
        try:
            msgs.append(_read_json(p))
        except Exception as e:
            print(f"  {p.name}: PARSE-ERROR {e}")
            continue
    eff = cross_check(msgs)
    print(f"inbox count = {len(msgs)}")
    for p, msg in zip(msgs_paths, msgs):
        kind, action, summary = summarize(msg)
        if kind == "task":
            tid = msg.get("task_id") or msg.get("id")
            eff_action = eff.get(tid, action)
        else:
            eff_action = action
        print(f"  [{kind:<8}/{eff_action:<7}] {p.name}  {summary}")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """后台轮询：新消息追加到 log；已处理消息归档到 .processed/。"""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    seen: Dict[str, int] = {}
    interval = float(args.interval)
    print(f"[{_now_iso()}] watching {INBOX} (interval={interval}s)")
    while True:
        msgs = list_inbox()
        for p in msgs:
            mt = p.stat().st_mtime_ns
            if seen.get(p.name) == mt:
                continue
            try:
                msg = _read_json(p)
            except Exception as e:
                # 原子写完成前可能短暂看到空/半成品；不标记为已处理，
                # 下一轮用同一文件的新内容重试。
                _log(f"[parse-error] {p.name}: {e}")
                continue
            seen[p.name] = mt
            msgs = []
            for other in list_inbox():
                try:
                    msgs.append(_read_json(other))
                except Exception:
                    pass
            eff = cross_check(msgs)
            kind, action, summary = summarize(msg)
            if kind == "task":
                tid = msg.get("task_id") or msg.get("id")
                action = eff.get(tid, action)
            _log(f"[{kind}/{action}] {p.name}  {summary}")
        time.sleep(interval)


def _log(line: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{_now_iso()}] {line}\n")


# ---------- main ----------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("once", help="单次扫描 inbox 并输出建议（返回 2 表示有待处理）")
    sub.add_parser("peek", help="纯摘要，不动任何文件")

    w = sub.add_parser("watch", help="后台轮询，新消息追加到 .collab/log/zcode_watch.log")
    w.add_argument("--interval", default=5.0, type=float, help="轮询间隔秒")

    args = ap.parse_args()
    if args.cmd == "once":
        return cmd_once(args)
    if args.cmd == "peek":
        return cmd_peek(args)
    if args.cmd == "watch":
        return cmd_watch(args)
    ap.error(f"unknown cmd: {args.cmd}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
