#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地双 agent 协作监控器。

只读监控 .collab/outbox/zcode、.collab/status/zcode.json 和当前任务，
不修改代码、数据或消息文件。一次模式适合 CI/人工检查，--watch 模式适合
由本地终端或进程持续轮询。消息仍由 root/zcode 各自按协议写入。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
COLLAB = ROOT / ".collab"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def current_task() -> dict[str, Any] | None:
    path = COLLAB / "tasks" / "current.json"
    return read_json(path) if path.exists() else None


def inspect_once(seen: set[str]) -> int:
    """打印当前状态；返回 0 正常、2 blocker、3 无效回执。"""
    task = current_task()
    status_path = COLLAB / "status" / "zcode.json"
    status = read_json(status_path) if status_path.exists() else {}
    outbox = COLLAB / "outbox" / "zcode"
    files = sorted(outbox.glob("*.json")) if outbox.exists() else []

    print(f"[collab] task={task.get('id') if task else '-'} "
          f"zcode_state={status.get('state', 'unknown')} "
          f"updated={status.get('updated_at', '-')}")
    if not files:
        print("[collab] no zcode messages")
        return 0

    code = 0
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            msg = read_json(path)
        except Exception as exc:  # noqa: BLE001
            print(f"[collab] INVALID {path.name}: {type(exc).__name__}: {exc}")
            code = max(code, 3)
            continue
        kind = msg.get("type")
        if kind == "blocker":
            print(f"[collab] BLOCKER {path.name}: {msg.get('blocker', '-')}")
            for option in msg.get("proposed_options", []):
                print(f"  option {option.get('key', '?')}: {option.get('title', '-')}")
            code = max(code, 2)
        elif kind == "result":
            print(f"[collab] RESULT {path.name}: status={msg.get('status')} "
                  f"commit={msg.get('commit')} files={len(msg.get('changed_files', []))}")
            if task:
                required = ("id", "from", "to", "type", "reply_to", "status",
                            "commit", "changed_files", "tests")
                missing = [k for k in required if k not in msg]
                if missing or msg.get("reply_to") != task.get("id"):
                    print(f"  INVALID result schema: missing={missing} "
                          f"reply_to={msg.get('reply_to')}")
                    code = max(code, 3)
        else:
            print(f"[collab] MESSAGE {path.name}: type={kind or 'unknown'}")
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true", help="持续轮询")
    ap.add_argument("--interval", type=float, default=5.0, help="轮询秒数，默认 5")
    args = ap.parse_args()
    seen: set[str] = set()
    if not args.watch:
        return inspect_once(seen)
    try:
        while True:
            inspect_once(seen)
            time.sleep(max(0.2, args.interval))
    except KeyboardInterrupt:
        print("[collab] stopped")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
