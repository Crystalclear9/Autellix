# KV 传输微基准

测量本项目使用 PyTorch 实现的 GPU/CPU KV 批量传输。安装与运行环境见 [安装与首次使用](../docs/getting-started.md)。

## 运行

在有 CUDA 的 vLLM 环境中执行：

```bash
python cuda/batched_swap_benchmark.py --blocks 128 --layers 4 --iterations 10
```

默认对照为 vLLM 编译后的 `swap_blocks`。`--baseline python` 改为逐 block 的 Python 复制，其测量包含 Python 调用开销；结果记录所选基线。

## 测量内容

- 执行真实 GPU/CPU 传输，使用非连续映射验证 KV 往返结果完全一致。
- 预热后同步 CUDA 计时，报告中位数和相对于原生搬运的加速比。
- 批量路径使用 `autellix/runtime/swap.py`：直接将各层 K/V 打包进连续缓冲区，经锁页内存传输后写入目标块。单次载荷超过 1 GiB 时按块分批。
- 使用连续的 `[2, blocks, ...]` KV 布局。多个映射指向同一目标块时，按原生复制顺序保留最后一次写入。

此路径使用 PyTorch CUDA 操作，无需单独编译自定义 kernel。传输性能由本微基准测量，生成与抢占恢复正确性由[开发与验证](../docs/development.md)中的真实推理测试检查。
