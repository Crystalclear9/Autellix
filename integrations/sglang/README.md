# SGLang 后端

本项目在论文 vLLM 方案之外增加的独立适配，固定使用 **SGLang 0.4.9.post6**。安装、服务、客户端和验证入口见 [安装与首次使用](../../docs/getting-started.md)。

## 支持范围

- 每个副本一张 GPU，文本生成，每次调用一个输出序列。
- MHA KV token pool、radix 前缀缓存、page size 1。
- 非重叠执行、不分块 prefill；允许多个独立副本由真实协调器管理。
- 不支持 TP/PP/DP、推测解码、LoRA、HiCache 或关闭 radix 缓存。

`backend.py` 启动真实 Engine，`scheduler.py` 在生成进程内安装钩子，不修改已安装的 SGLang 文件。程序和调用信息随请求传递，用户无需提供执行时间估计或 DAG。

## 调度

默认 `paper` 模式在窗口边界按 PLAS/ATLAS 等策略选择请求集合，混合执行已接纳的 prefill 和 decode。纯解码窗口内跳过原生 prefill 接纳；显存压力下按窗口中的优先级选择抢占对象，不采用原生按长度排序的回退规则。

N=8、K=1 是项目默认参数。预备请求的已计算前缀通过 radix 锁保留，活动请求结束后可补位。`--implementation compat` 保留旧的原生调度钩子路径。

## KV 所有权与恢复

抢占保留输出 token，将已计算 KV 打包到锁页 CPU 内存后释放 GPU 槽位。`--engine-args` 中的 `autellix_swap_space` 控制主机 KV 预算，单位 GiB，默认 1。预算不足会明确报错。

恢复时锁住已有共享前缀，仅分配和传输缺失后缀。恢复成功的前缀会保持锁定到原生接纳取得自己的锁，避免后续恢复逐出前面的请求。接纳失败时释放临时锁，并保留 CPU 副本供重试。

预备请求离开窗口或因显存压力被释放时，先保存 CPU 副本再解锁。备份失败保留原有 GPU 锁；替换主机备份失败也保留旧副本。成功接纳或取消后回收主机副本。

显存不足时先释放预备请求，再按冻结的队列顺序换出低优先级后缀；若单个请求仍无法容纳则报错。实现继续依赖原生 token pool 和 radix cache 的所有权规则。

## 验证

### P40 实验性兼容模式

`--engine-args` 中设置 `"autellix_legacy_gpu": true`，可显式启用 Pascal
兼容路径：强制 FP32、`torch_native` attention、PyTorch sampling，关闭 CUDA
Graph、自定义 all-reduce 和 Dynamo 编译。调度子进程内将标准 RMSNorm、激活和
旋转位置编码切换到原生 PyTorch 实现，并仅移除固定版本模型加载器中的 sm75
限制及其强制 FP16 逻辑。不修改已安装的 SGLang 文件，也不伪造 GPU 算力。

这要求 PyTorch 本身包含 Pascal 内核（本项目固定的 Torch 2.7.1/CUDA 12.6），
不能复用不含 sm61 支持的新版 CUDA wheel。目标是标准稠密文本模型的小规模
功能验证，不承诺其他模型、量化、MoE 或性能。默认优化路径不受影响。

持久盘部署示例：`bash scripts/serve_p40.sh`。通过 `AUTELLIX_PYTHON`、
`AUTELLIX_MODEL`、`AUTELLIX_DEVICES`、`AUTELLIX_PORT` 覆盖配置；SQLite
状态默认保存在新建的本地 `/tmp/autellix-p40-XXXXXX` 目录，可通过
`AUTELLIX_STATE_DIR` 显式指定；不要放到网络持久盘或与活动服务共用。
GPU 回归使用 `AUTELLIX_TEST_LEGACY_GPU=1`；仍需设置后端和模型环境变量。

当前仅有 SmolLM2-135M 的 P40 实测证据。其他模型、量化、低精度、长上下文、
高并发及长时间运行均未完成验证。原生算子与独立 Transformers 基线的数值比较
尚未完成；已通过的是同一适配引擎的抢占/不中断贪心输出一致性。
运行时补丁依赖固定版本的内部源码结构，不能直接用于其他 SGLang 版本。
完整证据与未修复项见 [P40 验收记录](../../docs/p40-validation-20261115.md)。

[开发与验证](../../docs/development.md)中的 GPU 测试覆盖真实生成、抢占恢复、程序服务继承和预备请求补位。`AUTELLIX_TEST_REPLICAS=1` 额外测试同一张 GPU 上两个独立模型进程的路由、取消及清理，不等同于多 GPU 验证。HTTP 验证包含客户端流式输出和会话释放。
