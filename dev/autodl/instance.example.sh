# 复制为 instance.local.sh 并填写。该文件已被 .gitignore，**绝不入库**。
# 主机/端口/密钥这类信息永远只留在本地。

# AutoDL 控制台「复制SSH登录指令」的完整参数（不含 ssh 命令本身）
export AUTODL_SSH="root@connect.region.autodl.com -p 12345"

# 实例上的工作目录（放数据盘 autodl-tmp，实例关机不丢；释放实例才丢）
export AUTODL_DIR="/root/autodl-tmp/minilm"

# 成本安全阀（docs/AUTODL_INTEGRATION_V1.md §9-②，默认值可改）
export AUTODL_MAX_HOURS="${AUTODL_MAX_HOURS:-10}"

# 时价（元/小时），只用于 benchmark 的费用估算打印
export AUTODL_PRICE_PER_HOUR="${AUTODL_PRICE_PER_HOUR:-2.0}"
