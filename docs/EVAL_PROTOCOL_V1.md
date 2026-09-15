# 评测协议 V1：val 选型 / 锁定测试族一次性使用 / 历史 test 数字标 exploratory

日期：2026-09-15。配套：[`docs/TODO_NOVEL_MODEL_LEARNING_V3.md`](TODO_NOVEL_MODEL_LEARNING_V3.md) P0 第 4 条、
`py/locked_test.py`、`py/splits.py`。

## 三条规则

1. **调参 / 选模型 / 定超参 / 定停止点：只准看 val。**
   `py/train.py` 已按此实现：训练过程中只有 val 指标，test 仅在结束时报一次。
2. **锁定测试族只在最终验收时评一次。**
   冻结：`python py/locked_test.py --freeze --per-bucket 20` → `data/locked_test.json`
   （每桶按**内容哈希**取前 N 部 test 作品；story_id 会随去重/重跑重新编号，不作为身份）。
   校验：`python py/locked_test.py --verify` —— 清单换代只作注记；锁定作品消失 / split 归属变化 /
   文件被手改 → **非零退出**。锁定失效后旧锁定族上的数字全部作废，须重新冻结并声明。
3. **反复读过 test 的历史数字 = exploratory（探索性），不得作为验收或选型依据。**
   包括：`wsplit_*` / `datascale` / `budget` / `ctx` 各轮 sweep 的 Top-1/NLL 收敛曲线与
   「预算阈值」「收敛步数」类结论 —— 它们是看着 test 曲线定下来的。
   这些数字**保留、不删**（`LOG.md` 与 `docs/experiments/` 里的原始记录是回溯证据），
   但引用时必须带 exploratory 标注；要变成结论，需在 val 上重新选型、锁定族上一次性复测。

## 为什么现在才立这条

V2 审计与 2026-09-15 自检确认：`py/sweep_worksplit.py` 在**每个 eval 步号**读 test 画收敛曲线，
「Top-1 缺口在 N 步转正」这类结论隐含了对 test 的多轮读取；同时清单多次换代
（32,036 → 192,823 → 225,756 部），不同批次数字的口径并不相同。
在 test 被反复读过的前提下，「模型超过 trigram」既可能是真泛化，也可能是选择效应——无法区分。

## 给未来的操作顺序

1. 训练 / 调参：只看 val（`train.py --bucket X` 的日志里 val 指标）。
2. 选型完成、准备验收：`python py/locked_test.py --verify` 通过后，在锁定族上**评一次**，记录数字与
   `lock_fingerprint`。
3. 数据管线变更（去重 / 换代）：重跑 `--verify`；漂移即重新冻结，并声明旧验收数字作废。
4. 发布 / 导出（P4）：另过 `python py/license_policy.py` 的可分发检查。
