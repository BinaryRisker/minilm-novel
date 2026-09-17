#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".collab/ 协议读写工具（zcode 侧）。

为什么向 root 写"结果"而不是直接写 inbox：
- inbox/<role>/ 是发给对应 agent 的入站消息；root 自己的入站
  应该由 root 工具/进程读取并处理，zcode 写入会让 inbox 失去
  "未读"语义。
- 所以 zcode 写结果 → outbox/zcode/（协议明确）；root 轮询
  outbox/zcode/ 把已读结果移走或归档。

本 helper 提供：
  1. read_protocol()            — 读 .collab/protocol.json
  2. read_task(task_id)         — 读 tasks/<id>.json
  3. validate_result(...)       — 校验 result 字段是否符合 schema
  4. write_result(...)           — 写 outbox/zcode/result-<task_id>.json
  5. update_my_status(...)       — 更新 status/zcode.json 的 state / current_task
  6. acquire_lock / release_lock — 短文件锁，避免 zcode/root 同时改同文件

设计原则：
- 不读 inbox（那是 root 的事）
- 不写 inbox（zcode 只写 outbox）
- 不主动删文件；只 move/rename 由 root 或 orchestrator 决定
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 文件锁：Windows 用 msvcrt，POSIX 用 fcntl
if os.name == "nt":
    import msvcrt  # type: ignore

    def _lock_ex(fd: int) -> None:
        # 上锁：先尝试锁 1 字节；会抛 OSError if locked
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _lock_un(fd: int) -> None:
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
else:
    import fcntl  # type: ignore

    def _lock_ex(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _lock_un(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


ROOT = Path(__file__).resolve().parent.parent
COLLAB_DIR = ROOT / ".collab"


# ---------- 协议字段（与 protocol.json 保持一致；改 schema 时两边一起改）----------

REQUIRED_TASK_FIELDS = [
    "id", "from", "to", "type", "created_at",
    "objective", "allowed_paths", "forbidden_paths", "acceptance",
]

REQUIRED_RESULT_FIELDS = [
    "id", "from", "to", "type", "reply_to",
    "status", "commit", "changed_files", "tests",
]

VALID_RESULT_STATUS = {"completed", "partial", "failed"}
VALID_TASK_TYPES_FOR_RESULT = {"task"}


# ---------- 工具函数 ----------

def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_protocol() -> Dict[str, Any]:
    return _read_json(COLLAB_DIR / "protocol.json")


def read_task(task_id: str) -> Dict[str, Any]:
    p = COLLAB_DIR / "tasks" / f"{task_id}.json"
    if not p.exists():
        # 兼容：也可能写在 current.json
        cur = COLLAB_DIR / "tasks" / "current.json"
        if cur.exists() and _read_json(cur).get("id") == task_id:
            return _read_json(cur)
        raise FileNotFoundError(f"task not found: {task_id}")
    return _read_json(p)


def validate_result(task: Dict[str, Any], result: Dict[str, Any]) -> List[str]:
    """返回错误列表；空表示通过。"""
    errs: List[str] = []
    for k in REQUIRED_RESULT_FIELDS:
        if k not in result:
            errs.append(f"missing required field: {k}")
    if result.get("type") != "result":
        errs.append(f"type must be 'result', got {result.get('type')!r}")
    if result.get("status") not in VALID_RESULT_STATUS:
        errs.append(f"status must be one of {VALID_RESULT_STATUS}, got {result.get('status')!r}")
    if result.get("reply_to") != task.get("id"):
        errs.append(f"reply_to must equal task.id={task.get('id')!r}, got {result.get('reply_to')!r}")
    if result.get("from") != task.get("to"):
        errs.append(f"from must equal task.to={task.get('to')!r}（执行方视角）, got {result.get('from')!r}")
    if result.get("to") != task.get("from"):
        errs.append(f"to must equal task.from={task.get('from')!r}（发任务方视角）, got {result.get('to')!r}")
    if not isinstance(result.get("commit"), str):
        errs.append("commit must be string（'none' 表示未提交）")
    if not isinstance(result.get("changed_files"), list):
        errs.append("changed_files must be list")
    if not isinstance(result.get("tests"), list):
        errs.append("tests must be list")
    # forbidden_paths 校验
    forbidden = set(task.get("forbidden_paths") or [])
    bad = [f for f in (result.get("changed_files") or []) if f in forbidden]
    if bad:
        errs.append(f"changed_files 命中 forbidden_paths: {bad}")
    # max_runtime_sec 提醒（不阻断）
    return errs


def write_result(task_id: str, result_body: Dict[str, Any]) -> Path:
    """把 result 写到 outbox/zcode/。会自动补 created_at。"""
    if "created_at" not in result_body:
        result_body["created_at"] = _now_iso()
    out_dir = COLLAB_DIR / "outbox" / "zcode"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"result-{task_id}.json"
    # 原子写：写临时文件再 rename
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(result_body, f, ensure_ascii=False, indent=2, sort_keys=False)
        f.write("\n")
    os.replace(tmp, out_path)
    return out_path


def update_my_status(state: str, current_task: Optional[str] = None,
                     extras: Optional[Dict[str, Any]] = None) -> Path:
    """更新 status/zcode.json 的 state / current_task / last_committed。"""
    p = COLLAB_DIR / "status" / "zcode.json"
    s = _read_json(p) if p.exists() else {"role": "zcode"}
    s["state"] = state
    s["updated_at"] = _now_iso()
    if current_task is not None:
        s["current_task"] = current_task
    if extras:
        s.update(extras)
    tmp = p.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, p)
    return p


# ---------- 文件锁（短临界区，避免与 root 同时改同一文件）----------

class FileLock:
    def __init__(self, path: Path, timeout_sec: float = 30.0, poll: float = 0.2):
        self.path = path
        self.timeout = timeout_sec
        self.poll = poll
        self.fd: Optional[int] = None

    def __enter__(self):
        deadline = time.time() + self.timeout
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(str(self.path), os.O_CREAT | os.O_RDWR, 0o644)
        while True:
            try:
                _lock_ex(self.fd)
                return self
            except OSError:
                if time.time() >= deadline:
                    os.close(self.fd)
                    self.fd = None
                    raise TimeoutError(f"acquire lock timeout: {self.path}")
                time.sleep(self.poll)

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                _lock_un(self.fd)
            finally:
                os.close(self.fd)
                self.fd = None


# ---------- CLI ----------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_protocol = sub.add_parser("read-protocol", help="打印 protocol.json")
    p_read_task = sub.add_parser("read-task", help="读 tasks/<id>.json（或 current.json）")
    p_read_task.add_argument("task_id")

    p_validate = sub.add_parser("validate-result", help="校验 result JSON 是否符合 schema")
    p_validate.add_argument("task_id")
    p_validate.add_argument("result_json_path")

    sub.add_parser("print-status", help="打印 status/zcode.json")

    args = ap.parse_args()

    if args.cmd == "read-protocol":
        print(json.dumps(read_protocol(), ensure_ascii=False, indent=2))
    elif args.cmd == "read-task":
        print(json.dumps(read_task(args.task_id), ensure_ascii=False, indent=2))
    elif args.cmd == "validate-result":
        task = read_task(args.task_id)
        result = _read_json(Path(args.result_json_path))
        errs = validate_result(task, result)
        if errs:
            print("INVALID:")
            for e in errs:
                print(f"  - {e}")
            return 1
        print("VALID")
    elif args.cmd == "print-status":
        p = COLLAB_DIR / "status" / "zcode.json"
        if not p.exists():
            print("(no status yet)")
        else:
            print(p.read_text(encoding="utf-8"))
    else:
        ap.error(f"unknown cmd: {args.cmd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())