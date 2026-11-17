# 服务器目录

2026-11-17 的新实例使用 `/root/workspace/autellix/` 为统一入口，连接端口为 47541。
完整迁移与存储映射见[新 P40 服务器迁移](server-migration-20261117.md)。下文记录
原三卡服务器的组织方式；新实例保留 `/mnt/masfiles/autellix` 链接以解析历史计划。

Autellix 的持久化根目录为 `/mnt/masfiles/autellix/`。

```text
autellix/
├── code/                 项目源码、启动和实验脚本
├── env/sglang/           SGLang / Torch / Python 环境
├── env/vllm/             vLLM 兼容性探测环境
├── models/               Llama-3.1-8B、Qwen3-14B/32B、SmolLM2 权重
├── datasets/             原始数据、固定版本上游代码、采集轨迹和数据清单
├── plans/                每组实验的模型、GPU、到达率和调度参数
├── runs/                 原始结果、运行日志和历史实验
├── logs/                 数据准备、目录迁移和服务日志
├── archives/             完整结果归档、旧安装缓存和历史临时目录
├── cache/                后续模型库与编译缓存
├── tmp/                  本机运行状态入口，指向 /tmp/autellix-runtime
└── layout-migration.json 原路径到新路径的映射
```

SQLite WAL、IPC 和实时调度 trace 使用本机文件系统。持久盘的类型为 `yrfs`；
`tmp/` 用符号链接提供统一入口，完成的调度 trace 随实验归档保存到 `runs/`。
模型权重、源码、Python 环境、数据和正式结果均在持久盘根目录下。

Qwen3-14B 和 Qwen3-32B 的实体目录已迁入 `autellix/models/`。其他项目也引用这两份
权重，因此原 `/mnt/masfiles/models/` 中保留对应符号链接。Autellix 专属的旧环境、
代码、数据和结果目录已迁走；81 个历史 `/tmp/autellix*` 项目文件/目录已收入
`archives/legacy-tmp/`。旧结果 JSON 和日志保留采集时的原路径，迁移映射用于定位这些文件。

## 启动服务

```bash
cd /mnt/masfiles/autellix/code
bash scripts/serve_p40_three_gpu.sh
```

`scripts/server_env.sh` 统一配置路径、缓存和临时文件位置。可通过 `AUTELLIX_HOME`、
`AUTELLIX_PYTHON`、`AUTELLIX_MODEL` 覆盖默认设置。原三卡服务使用 Qwen3-32B FP16、PP=3。迁移后实例为两张 P40，
双卡启动入口为 `bash scripts/serve_p40_two_gpu.sh`，使用已有 Qwen3-14B FP16、
PP=2、20/20 层、B=8、KV=65536 和 512-token MLP 分块。论文实验期间两张卡
分别运行配对回放；双卡服务在实验结束后验收。

## 准备论文工作负载

```bash
cd /mnt/masfiles/autellix/code
source scripts/server_env.sh
python_bin="$AUTELLIX_HOME/env/sglang/bin/python"
"$python_bin" scripts/fetch_paper_sources.py --output "$AUTELLIX_HOME/datasets"
"$python_bin" scripts/prepare_sharegpt_replay.py \
  --source "$AUTELLIX_HOME/datasets/paper/sharegpt.json" \
  --model "$AUTELLIX_HOME/models/Meta-Llama-3.1-8B-Instruct" \
  --programs 128 --seed 42 --output "$AUTELLIX_HOME/datasets/sharegpt-whole128.json"
bash scripts/run_paper_suite.sh
"$python_bin" scripts/evaluate_bfcl_capture.py \
  --source "$AUTELLIX_HOME/datasets/bfcl-v3-source" \
  --capture "$AUTELLIX_HOME/datasets/bfcl-captured64.json" \
  --output "$AUTELLIX_HOME/runs/bfcl-capture-quality.json"
```

入口脚本为每轮创建独立的 `runs/` 和 `plans/` 子目录。三个 GPU 分别运行一个
ShareGPT 配对点，随后继续第二随机种子、BFCL 采集/回放、LATS 采集/回放及 Mixed。
这些任务完成后，三张卡共同运行 Llama-3.1-8B PP=3、大 KV 预算的配对实验。
每组完成后立即保存 JSON 和退出码；完整命令、种子和输入 trace 哈希写入结果。
单个矩阵中断后，可用原计划和输出文件重新执行 `run_sglang_matrix.py --resume`，
复用已经完整结束的策略项，再执行余下项。

`scripts/organize_server.py` 保存此次迁移逻辑；迁移内容限定为项目专属目录和上述
四个模型。重复执行时跳过已迁移项，共用 Qwen 模型的旧入口保持可用。
