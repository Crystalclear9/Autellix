# KV 传输微基准

测量本项目独立实现的 GPU/CPU KV 批量传输，不代表作者原始 CUDA kernel 或论文性能复现。安装与运行环境见 [安装与首次使用](../docs/getting-started.md)。

## 运行

在有 CUDA 的 vLLM 环境中执行：

```bash
python cuda/batched_swap_benchmark.py --blocks 128 --layers 4 --iterations 10
```

默认对照为 vLLM 编译后的 `swap_blocks`。`--baseline python` 改为逐 block 的 Python 复制，包含 Python 调用开销，不能当成同一个对照。

## 测量内容

- 执行真实 GPU/CPU 传输，使用非连续映射验证 KV 往返结果完全一致。
- 预热后同步 CUDA 计时，报告中位数；不预设批量传输在所有负载下都更快。
- 批量路径使用 `autellix/runtime/swap.py`：聚合各层 K/V 为连续载荷，经锁页内存一次传输后写入目标 block。
- 支持连续的 `[2, blocks, ...]` KV 布局；不兼容的形状、越界映射和重复目标会被拒绝。

此路径使用 PyTorch CUDA 操作，不需单独编译自定义 kernel。没有 CUDA 时直接报错，不回退到 CPU 模拟。后端生成正确性仍需运行[开发与验证](../docs/development.md)中的真实推理测试，微基准不能替代这些测试。
