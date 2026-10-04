# vLLM 后端

本项目独立实现的 Autellix 适配，固定使用 **vLLM 0.6.1**。安装、服务、客户端和验证入口见 [安装与首次使用](../../docs/getting-started.md)。

## 支持范围

- Decoder-only 文本生成，每次调用仅支持一个输出序列（`n=1`、`best_of=1`）。
- 前缀缓存、GPU/CPU KV 交换、批量 KV 传输和独立引擎副本。
- 可选张量并行：每个 rank 安装传输钩子，尚未完成多 GPU 硬件验证。
- 不支持流水线并行；推测解码等扩展不属于已验证的运行配置。

后端检查版本，并在自己的进程内安装钩子，不修改已安装的 vLLM 文件。

## 调度与显存

默认 `paper` 模式由 `paper.py` 按全局优先级接纳请求，不使用原生 `_schedule()` 决定批次。窗口开始时预留后续 KV 槽位；窗口内更新 token 和 block 元数据，预备请求可在活动请求结束后补位。

启用混合 prefill/decode 执行，避免原生阶段偏好覆盖请求优先级。接纳按整段 prefill 计算；输入超过 `max_num_batched_tokens` 会在接纳前报错，需增大该参数。预备请求占用 KV 显存，在补位前不消耗 decode token 配额。

抢占沿用原生 block 所有权转换。换出和换入分开处理，保留生成序列；实际批量传输由 `autellix/runtime/swap.py` 完成。预留窗口所需显存不足时会明确报错，需调整模型、显存预算、批次或窗口长度。

## 配置

[安装文档](../../docs/getting-started.md)的启动命令使用默认 `paper`、N=8、K=1。`--implementation compat` 选择旧包装实现：单 GPU 且 K=0 时使用原生缓存多步执行，其余情况在筛选后的请求集合中调用原生调度。

`--devices 0,1` 启动两个独立副本。张量并行使用 `--device-groups '0,1;2,3'`，并在 `--engine-args` 中设置 `"tensor_parallel_size":2`；每组设备数必须匹配并行度。

## 原生基准对照

真实工作负载命令可选择以下原生模式，保留原生调度和 KV 传输，仅观测执行时间：

| 模式 | 配置 |
| --- | --- |
| `vllm` | 基础原生配置 |
| `vllm-opt` | 前缀缓存和分块 prefill |
| `vllm-opt-multistep` | 前缀缓存和原生多步执行 |

vLLM 0.6.1 不允许同时开启分块 prefill 与原生多步执行，因此后两者分开提供，不声称等于论文中的组合优化基准。`--native-opt-steps` 配置原生多步长度。

`autellix_adapter.py` 保留旧元数据接口，其 `create_backend()` 可构造真实后端。回归测试包括抢占前后贪心结果一致、程序历史继承和无效输入隔离；双副本及张量并行测试需显式启用，见[开发与验证](../../docs/development.md)。
