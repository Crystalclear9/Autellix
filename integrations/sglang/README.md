# SGLang 后端

本项目在论文 vLLM 方案之外增加的独立适配，固定使用 **SGLang 0.4.9.post6**。安装、服务、客户端和验证入口见 [安装与首次使用](../../docs/getting-started.md)。

## 支持范围

- 单 GPU 副本或 PP 多 GPU 副本，文本生成，每次调用一个输出序列。
- MHA KV token pool、radix 前缀缓存、page size 1。
- 非重叠执行、不分块 prefill；允许多个独立副本由真实协调器管理。
- Autellix 的模型并行使用 PP；TP、DP attention、推测解码、LoRA 和 HiCache 尚未接入。

`backend.py` 启动真实 Engine，`scheduler.py` 在生成进程内安装钩子，不修改已安装的 SGLang 文件。程序和调用信息随请求传递，用户无需提供执行时间估计或 DAG。

## 调度

默认 `paper` 模式在窗口边界按 PLAS/ATLAS 等策略选择请求集合，混合执行已接纳的 prefill 和 decode。纯解码窗口内跳过原生 prefill 接纳；显存压力下按窗口中的优先级选择抢占对象，不采用原生按长度排序的回退规则。

N=8、K=1 是项目默认参数。预备请求的已计算前缀通过 radix 锁保留，活动请求结束后可补位。`--implementation compat` 保留旧的原生调度钩子路径。

### 三卡流水线

`--device-groups '0,1,2'` 与 `pp_size=3` 将一个模型按层拆到三张 GPU。
`--devices 0,1,2` 则启动三个完整模型副本，用于程序路由实验。

`pipeline.py` 在原生 PP 微批次循环中传递首阶段的程序表读取结果与策略时钟。
取消会覆盖全部在途微批次，包括尚未合并到 running batch 的 prefill。
首阶段维护全局程序历史，其余阶段复用同一策略输入，各自管理本阶段的 KV。
每个微批次有独立的 N 步窗口和 K 个预备请求，B 在微批次间均分。
模型执行时间由三个阶段的 CUDA 计时累加，随采样结果返回首阶段并统一记账。
跨阶段的 `batch`、`execute`、`swap_out`、`swap_in` 与完成事件可逐条核对。

P40 路径同时修复了固定版本 Llama 模型在 PP 下的 tied embedding 加载：
首阶段加载输入 embedding，末阶段从同一权重加载输出 head。

运行 `bash scripts/serve_p40_three_gpu.sh` 启动 Qwen3-32B FP16、PP=3。
服务的 64 层按 `20,23,21` 分配，同时考虑 GPU 0 少 1.5 GiB 和末阶段输出 head 的占用。
服务使用 KV=8192、prefill 上限 4096 和 `autellix_attention_score_elements=16777216`，
控制接近上下文上限的输入所需的 attention 临时空间。
完整配置与对照结果见[三卡实验](../../docs/three-gpu-experiment.md)。

## KV 所有权与恢复

抢占保留输出 token，将已计算 KV 打包到锁页 CPU 内存后释放 GPU 槽位。`--engine-args` 中的 `autellix_swap_space` 控制主机 KV 预算，单位 GiB，默认 1。预算不足会明确报错。

换入、换出按层和 token 分块，每块最多 16 MiB；每块传输完成即释放临时 GPU
张量。长对话的整段 KV 保存在锁页 CPU 内存，GPU 无需额外容纳所有层的副本。
`scripts/check_host_swap_gpu.py` 验证大于 2 GiB 的 KV 往返及临时显存峰值。

恢复时锁住已有共享前缀，仅分配和传输缺失后缀。恢复成功的前缀会保持锁定到原生接纳取得自己的锁，避免后续恢复逐出前面的请求。接纳失败时释放临时锁，并保留 CPU 副本供重试。

新调用的已有缓存也在恢复低优先级请求之前锁定。若低优先级恢复项挤占了高优先级
新 prefill 的空间，按队列后缀释放临时锁并重试接纳；已晋升的低优先级预备项同样
可以换出。`scripts/check_prefill_priority_gpu.py` 在实际 GPU 上验证此路径与生成一致性。

预备请求离开窗口或因显存压力被释放时，先保存 CPU 副本再解锁。备份失败保留原有 GPU 锁；替换主机备份失败也保留旧副本。成功接纳或取消后回收主机副本。

显存不足时先释放预备请求，再按冻结的队列顺序换出低优先级后缀；若单个请求仍无法容纳则报错。实现继续依赖原生 token pool 和 radix cache 的所有权规则。

## 验证

### P40 实验性兼容模式

`--engine-args` 中设置 `"autellix_legacy_gpu": true`，可显式启用 Pascal
兼容路径：默认 FP32（可指定 `dtype=float16`）、`torch_native` attention、PyTorch sampling，关闭 CUDA
Graph、自定义 all-reduce 和 Dynamo 编译。调度子进程内将标准 RMSNorm、激活和
旋转位置编码切换到原生 PyTorch 实现，并仅移除固定版本模型加载器中的 sm75
限制及其强制 FP16 逻辑。补丁在调度进程启动时安装。

环境固定为包含 Pascal 内核的 Torch 2.7.1/CUDA 12.6。支持标准稠密文本模型。
缓存前缀 attention 只计算新增 Query，原生 SGLang 对照和 Autellix 共用这条路径。
长 prefill 按 Query 分块，每块使用绝对前缀位置构造因果遮罩，控制临时 attention
矩阵大小。FP16/FP32 CUDA 数值回归覆盖此路径。
`autellix_prefix_attention=false` 可复测原来的完整 Query 填充实现。
`autellix_attention_score_elements` 调整前缀 attention 的分块预算，默认 `67108864`；
原生基线和 Autellix 使用同一实现及参数传递路径。

持久盘部署示例：`bash scripts/serve_p40.sh`。通过 `AUTELLIX_PYTHON`、
`AUTELLIX_MODEL`、`AUTELLIX_DEVICES`、`AUTELLIX_PORT`、`AUTELLIX_ENGINE_ARGS` 覆盖配置；SQLite
状态默认保存在统一入口 `/mnt/masfiles/autellix/tmp/` 下的新建目录，可通过
`AUTELLIX_STATE_DIR` 显式指定；每个服务使用独立的本地状态目录。
GPU 回归使用 `AUTELLIX_TEST_LEGACY_GPU=1`；仍需设置后端和模型环境变量。

模型、精度、上下文与 GPU 数值测试的最新结果见
[三卡论文复现实验报告](../../docs/three-gpu-experiment.md)。

### 原生调度对照

`autellix_native_baseline=true` 仅用于对照：跳过 Autellix scheduler 钩子、
程序服务历史调度和主机 KV 换出实现，沿用 SGLang 的调度与 KV 管理。
要求 `PolicyConfig.overprovision=0`；返回 metrics 中的 `native_sglang=true`
用于标记这条路径，不包含 Autellix 的执行/继承统计。外层 IPC 与会话封装仍在。
若同时启用 `autellix_legacy_gpu`，只安装相同的硬件兼容算子，不安装调度钩子。
双方使用相同的 overlap/chunked prefill 设置。`policy=fcfs` 是 Autellix 内部策略；
原生调度对照使用 `autellix_native_baseline=true`。

配对实验、输入构造限制和结论见 [论文对照审计](../../docs/autellix-paper-audit.md)。

[开发与验证](../../docs/development.md)中的 GPU 测试覆盖真实生成、抢占恢复、程序服务继承和预备请求补位。`AUTELLIX_TEST_REPLICAS=1` 额外测试两个独立副本的路由、取消及清理；设置 `AUTELLIX_TEST_DEVICES=0,1` 可覆盖两张 GPU。HTTP 验证包含客户端流式输出和会话释放。
