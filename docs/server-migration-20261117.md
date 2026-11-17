# P40 服务器迁移（2026-11-17）

新连接为 `ssh -p 47541 root@cn-north-b.ssh.damodel.com`。
旧服务器的 Autellix 目录约 133 GB，整体迁入新实例；已完成的实验结果保留原始硬件
记录，新采集和后续配对在新实例上继续。

| 项目 | 新实例 |
| --- | --- |
| GPU | 2×Tesla P40，各 23040 MiB，合计 45 GiB 可见显存 |
| CPU 配额 / 内存 | 12 核 / 48 GiB |
| 系统 / 驱动 | Ubuntu 22.04.5 / 550.120 |
| 统一项目入口 | `/root/workspace/autellix/` |
| 代码、数据、历史日志和实验归档 | 20 GB 持久工作区内的对应子目录 |
| Python 环境 | `autellix/env/` → `/root/damodel-tmp/autellix/env/` |
| Llama-3.1-8B 权重 | `autellix/models/Meta-Llama-3.1-8B-Instruct/` → 50 GB 数据盘 |
| Qwen3-14B/32B、SmolLM2 权重 | `autellix/models/` → `/root/autellix-system/models/` |
| SQLite、IPC 与运行 trace | `autellix/tmp/` → `/tmp/autellix-runtime/` |
| 历史绝对路径入口 | `/mnt/masfiles/autellix` → `/root/workspace/autellix` |

目录链接保留统一入口，历史计划中的模型、数据、环境和状态路径继续可用。
完整迁移包含四个模型、两套 Python 环境、数据与上游固定版本、全部结果和归档。
迁移记录、文件内容 SHA256 和校验结果存放在 `autellix/logs/`。

采集器的 `--resume` 按完整程序检查点接续 BFCL/LATS；已保存程序及实际输入、
生成长度、工具结果继续保留。迁移时尚未完成的程序从该程序开头重新采集。
`run_paper_followups.py --gpu` 标识原实验流程，`--device` 指定新实例的物理卡：

```bash
cd /root/workspace/autellix/code
python_bin=/root/workspace/autellix/env/sglang/bin/python
"$python_bin" -u scripts/run_paper_followups.py --gpu 2 --device 0
"$python_bin" -u scripts/run_paper_followups.py --gpu 1 --device 1
```

两条命令分别安排 BFCL 与 LATS/Mixed，可并行执行。两种调度策略在同一物理卡上
使用相同输入 trace 与资源配置进行配对。此前三张卡上的 ShareGPT 和 PP=3
实验已完成，结果见[论文工作负载报告](paper-workload-experiment.md)。
