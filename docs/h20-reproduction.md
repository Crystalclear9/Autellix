# Autellix 论文复现尝试 · H20

`h20-env-zy-fjr` 用于代码和数据同步，`h20-gpu-zy-fjr` 用于计算。代码位于
`/home/nice/qiuzeyu/stu/fjr/code/autellix`，数据、计划、结果保存在
`/home/nice/qiuzeyu/stu/fjr/data/autellix`，环境为 `fjr` 的 `autellix-sglang`。
SQLite 和进程间通信使用本机 `/tmp/fjr-autellix-*`，运行记录和调度轨迹在结束后保存到共享目录。

实验直接读取共享模型。8B 回放使用 `Meta-Llama-3.1-8B`，大模型配对使用
`Meta-Llama-3.1-70B`，接口验证使用 `Llama-3.2-1B-Instruct`。四个工作负载使用从
P40 迁移的完整 ShareGPT、BFCL、LATS 及其混合集合。Mixed 默认抽取 12 个程序，
每个来源 4 个，混排后按泊松过程到达。每个配对共用程序、到达时间、
GPU、模型、KV 容量和批次上限；候选组使用 PLAS 或 ATLAS。正式配对在 BF16 下使用
SGLang 的 Triton attention，保留 radix 缓存。两组关闭 overlap、CUDA Graph 和
chunked prefill。到达率按 H20 的执行速度设置，结果包含实际积压和排空时间。
汇总同时记录 SGLang 报告的缓存命中 token 比例，便于对照前缀复用和调度开销。
`--native-default` 增加一组采用 SGLang 上游默认优化的对照，使用相同模型权重、批次和
KV 上限。`--device '2;3' --routing-ablation` 在两个单卡副本上，固定调度策略对比
程序亲和性、最少调用数和轮询。多卡单副本使用 `--device '2,3'`。上述卡号为命令示例。

下面的命令在 H20 上执行，`gpu` 使用本次分配且空闲的物理卡号。

```bash
source /home/nice/qiuzeyu/stu/fjr/tools/activate.sh
conda activate autellix-sglang
cd /home/nice/qiuzeyu/stu/fjr/code/autellix
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="$(mktemp -d /tmp/fjr-autellix-XXXXXX)"
data=/home/nice/qiuzeyu/stu/fjr/data/autellix
model=/data-hdd/opt/models/llama3.1/Meta-Llama-3.1-8B
python scripts/run_h20_paper.py sharegpt --model "$model" --datasets "$data/datasets" \
  --run-dir "$data/runs/h20-sharegpt" --device "$gpu"
```

长任务用 `fjr-autellix-*` 命名的 tmux 会话运行。接口回归启动的 HTTP 服务会随测试退出，
每次配对结束后关闭引擎。结果保存后，通过 `nvidia-smi` 核对任务进程是否已释放 GPU。

论文第 6 节按程序级秒/token 对照可承载的程序到达率，另报 P95 和 P99。结果汇总记录
这三项，并记录实际程序完成率、输出 token/s、到达阶段积压和排空时间。8B 使用单张
H20，70B 在本实验使用流水线并行，论文使用 4 张 A100。论文的主要对照包括 vLLM
0.6.1、启用分块 prefill 和前缀缓存的 vLLM-opt，以及 MLFQ。
[Autellix 论文，第 6 节](https://arxiv.org/html/2502.13965v1#S6)
