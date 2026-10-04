# 开发与验证

[文档导航](README.md) · [项目首页](../README.md)

本文档中的命令均从仓库根目录执行。

## 验证

不加载模型的测试：

```bash
python -m unittest discover -s tests -v
```

真实 GPU 测试，在对应后端环境执行：

```bash
AUTELLIX_GPU_BACKEND=vllm AUTELLIX_TEST_MODEL=/absolute/path/to/model \
  AUTELLIX_TEST_STEPS=8 AUTELLIX_TEST_RESERVE=1 \
  python -m unittest discover -s tests -p test_gpu_runtime.py -v
```

SGLang 将后端变量改为 `sglang`。可选项：`AUTELLIX_TEST_REPLICAS=1` 验证双进程路由和取消；`AUTELLIX_TEST_CUDA_SWAP=1` 验证实际 KV 传输；`AUTELLIX_TEST_TP=1` 启用需要两张 GPU 的 vLLM 张量并行测试；`AUTELLIX_TEST_IMPLEMENTATION=compat` 验证旧模式。

完整的小模型验证入口会依次运行测试、真实 HTTP 流程和四策略 DAG 工作负载：

```bash
bash scripts/validate_backend.sh vllm /absolute/path/to/model
```

SGLang 同样替换后端名称。可用 `AUTELLIX_PYTHON` 指定 Python 路径，`AUTELLIX_ENGINE_ARGS` 调整 HTTP 和 DAG 验证参数。生成结果写入 `outputs/validation/`，不纳入版本控制。KV 传输单独测量见 [cuda/README.md](../cuda/README.md)。

## 开发约束与修改流程

### 开发时必须保持的约束

1. **服务统计与优先级窗口分开**：防饥饿可以重置局部等待/运行窗口，不能清除调用整个生命周期的实际执行统计。
2. **按当前队列顺序移动请求**：不能用最初到达顺序替代降级、提升后的 FIFO 顺序。
3. **窗口内不随意重新选批次**：新调用等待边界，提前完成由窗口已有预备请求补位；容量不足按有定义的顺序释放资源。
4. **先保全 KV，再释放所有权**：CPU 备份失败不能丢失原有 GPU KV；恢复数据在正式接纳前不能被其他恢复操作逐出。
5. **传输屏障保持顺序**：不要让新换入覆盖尚未完成换出的 GPU block，也不要重复重放原生多步的换入。
6. **取消以确认完成为准**：Future 被取消不代表 GPU 或数据库已回收，错误和取消路径都必须完成负载及会话清理。
7. **路由不能使用未知输出长度**：仅按实际输入 token 数判断 2048 阈值。
8. **保留接口分层**：模拟器数据、估算耗时和 fake worker 不能替代真实后端验证。

### 修改与升级流程

先阅读受影响模块及对应测试，在现有目录内修改；通常不需要增加新的说明文件。修改策略应同时检查真实 runtime 与模拟器，若两者有刻意差异，在架构文档中解释。

升级 vLLM/SGLang 时，在独立环境验证原生 scheduler、block/token pool、radix lock、输入输出结构、worker 启动方式和 CUDA 计时接口。更新版本检查、锁文件及适配代码后，再执行下列验证，不能仅放宽版本检查。

| 修改范围 | 至少需要的验证 |
| --- | --- |
| 文档、示例命令 | 路径/链接检查、对应 `--help`、Python 示例语法检查 |
| PLAS/ATLAS、窗口、路由 | `test_runtime.py`、`test_runtime_regressions.py`、`test_paper_plan.py`、`test_simulator_paper.py` |
| 客户端/HTTP | `test_client_lifecycle.py`、`test_client_stream.py`、`test_http.py`、`scripts/http_smoke.py` |
| KV 或 SGLang 锁/抢占 | host swap、reserve、decode pressure、SGLang scheduler 测试，加真实 GPU 恢复一致性测试 |
| IPC、取消、副本协调 | `test_transport.py`，加 `AUTELLIX_TEST_REPLICAS=1` 真实双进程测试 |
| TP 适配 | 两张及以上 GPU 的 `AUTELLIX_TEST_TP=1`，没有设备时明确保留未验证状态 |

定向运行示例：

```bash
python -B -m unittest discover -s tests -p test_runtime.py -v
python -B -m unittest discover -s tests -p test_http.py -v
python -B scripts/http_smoke.py --help
```

增加新策略需更新 `PolicyConfig` 的合法值、继承和完成统计、队列行为、服务/benchmark CLI 的策略选项及测试；若需要模拟支持，再修改 `autellix/core/schedulers.py`。新增后端还需实现工作进程使用的初始化、分词、接纳、step、cancel、has_work 和 close 接口，并接入副本配置及 CLI 校验。当前没有自动注册后端的插件接口。

## 后续开发边界

现有测试和文档提供继续开发的起点，不能保证未来依赖、任意模型和硬件升级后自动兼容。继续维护时，应优先保留固定环境中的可运行基线，再逐项扩展支持范围。

当前需要额外开发或验证的方向包括：多 GPU TP 的硬件覆盖、更广模型/KV 布局支持、后端版本迁移，以及鉴权、限流、失败副本重启等服务能力。论文大规模性能实验是独立验证目标，不是现有功能测试已经完成的结论。对外说明应持续区分“已经实现”“已经验证”和“尚未支持”。

## 一次开发变更的完成标准

1. 明确触发条件和预期行为，用最小输入描述问题；读取当前实现，不凭后端新版本的接口印象修改固定版本钩子。
2. 为有行为变化的边界补回归测试，尤其是取消/失败后的资源回收；不只检查返回文本是否非空。
3. 先跑受影响的 CPU 测试，再按上表选择真实后端测试。修改 GPU KV 所有权不能仅凭 mock 测试交付。
4. 输出一致性使用同一个加载引擎、同一提示和贪心参数比较中断/不中断执行；不要把跨环境文本差异直接归因于调度。
5. 检查 `git diff --check` 和变更文件。模型、PDF、虚拟环境、缓存、`outputs/` 不入库。
6. 提交时说明改了什么、验证了哪些条件、哪些条件仍未验证；必要时同步接口、配置和后端文档。

文档改动通常只需检查链接、示例语法和对应 CLI，不需要重跑耗时的 GPU 测试。本地临时输出按需保留，但不要将其充当论文规模性能证明。

## 后端迁移清单

迁移至少涉及以下调用契约，不能只让 import 成功：

- worker 的启动方式、CUDA 可见设备和模型初始化顺序；子进程能否正确安装调度钩子。
- 请求进入、完成、取消和失败时的回调；每条路径是否正确更新程序统计。
- vLLM block 分配/追加/换入换出方法及 SchedulerOutputs；混合阶段的元数据顺序。
- SGLang token pool 与 radix cache 的锁引用、释放规则、重排后的 tensor 与 request 对齐。
- KV tensor 形状、dtype、stride 和索引映射；原生复制与批量传输是否保留相同内容。
- 批次执行计时和事件同步；是不是仍只把执行时间计入参与调用。
- 流式输出结构、结束原因和 token 数；客户端不能吞掉错误或把中断流当成功。

验证后再更新锁文件和版本校验。现有适配依赖内部 API，不保证可直接适用于较新后端。SGLang 的 TP 支持、其他 KV 布局和多模态等都需要独立设计与测试，而非添加一个 CLI 参数即可完成。

## 示例命令验证

下列命令只查看入口参数，不启动模型：

```bash
python -B -m autellix.runtime.server --help
python -B -m autellix.runtime.benchmark --help
python -B examples/real_inference.py --help
python -B scripts/http_smoke.py --help
```

执行示例脚本前，应按安装文档在当前环境完成 `pip install -e .` 或后端安装脚本。未安装项目时，直接执行子目录中的脚本可能报 `No module named autellix`；不要因为根目录下 `python -m autellix...` 能运行就认定已经安装。完整真实验证仍需激活对应的固定版本环境。CPU 环境中模块 import 成功不代表后端依赖已经可用。
