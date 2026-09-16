# dev/autodl — AutoDL 远程训练薄封装

方案与安全阀全文：[`docs/AUTODL_INTEGRATION_V1.md`](../../docs/AUTODL_INTEGRATION_V1.md)。
这里只是六只薄脚本。**所有付费动作都由人手动触发，脚本只负责把这一步做对。**

## 准备（一次性，本地）

```bash
cp dev/autodl/instance.example.sh dev/autodl/instance.local.sh
# 编辑 instance.local.sh：填控制台「复制SSH登录指令」的参数与工作目录
```

`instance.local.sh` 已被 .gitignore——主机/端口永远不进仓库。

## 流程（对应方案 §7 的 B→E 阶段）

| 步骤 | 命令 | 跑在哪 | 花钱吗 |
|---|---|---|---|
| 打包数据 | `python py/data/pack_remote.py --buckets gudian,wuxia,urban,xiandai,scifi,story` | 本地 | 否（许可闸门在此：local_only/unknown 直接拒绝） |
| 环境准备 | `bash dev/autodl/10_setup_env.sh` | →实例 | 无卡模式即可 |
| 上传 | `bash dev/autodl/30_upload.sh` | 本地→实例 | 否 |
| **探测** | `bash dev/autodl/40_benchmark.sh` | →实例(GPU) | 是（约 1 GPU 时） |
| **训练** | `bash dev/autodl/50_train.sh --bucket story --steps 20000 --out py/runs/…` | →实例(GPU) | 是（`AUTODL_MAX_HOURS` 硬上限，结束自动关机） |
| 回传 | `bash dev/autodl/60_fetch_results.sh` | 本地←实例 | 否（关机前做，或无卡模式开机后做） |

## 三条不可破的规矩（脚本已内置，不要绕）

1. **数据包必须出自 `pack_remote.py`**：许可闸门 + 自带身份（子集指纹给远端 verify，
   来源指纹记谱系）。手搓 scp 整个 data/ 目录 = 绕过许可红线，禁止。
2. **远端结果必须带身份回传**：`60_fetch_results.sh` 校验 meta 里的
   `data_identity.split_fingerprint` 与数据包一致，不一致拒收。
3. **锁定测试族（`data/locked_test.json`）不上云**：test 只在本地一次性验收（评测协议 V1）。

## 当前状态

- 脚本与打包器已就绪（62 项测试含 4 项打包契约测试）；
- **尚未在真实 AutoDL 实例上跑过**——首次跑按 40_benchmark → 50_train 的顺序，
  先小后大；预算表出来后回填 `docs/AUTODL_INTEGRATION_V1.md` §7 的成本列。
