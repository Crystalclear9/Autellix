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

SGLang 将后端变量改为 `sglang`。`AUTELLIX_TEST_REPLICAS=1` 验证双进程路由和取消。`AUTELLIX_TEST_CUDA_SWAP=1` 验证实际 KV 传输。`AUTELLIX_TEST_IMPLEMENTATION=compat` 验证旧模式。

vLLM 的分块 prefill 和抢占恢复回归可通过 `AUTELLIX_TEST_VLLM_MEMORY=.86`、
`AUTELLIX_TEST_VLLM_DTYPE=bfloat16` 使用单卡 Qwen3-32B。用 `AUTELLIX_TEST_DEVICE`
指定物理卡，`AUTELLIX_TEST_FUSED_SWAP=1` 走批量 CUDA KV 搬运。

`AUTELLIX_TEST_TP=1` 验证 vLLM 张量并行中的抢占和 KV 恢复。用 `AUTELLIX_TEST_TP_DEVICES=0,2` 指定两张物理卡。`AUTELLIX_TEST_TP_NATIVE_SWAP=1` 使用原生 KV 搬运执行同一回归。

三张 P40 的流水线回归：

```bash
AUTELLIX_TEST_PP_MODEL=/mnt/masfiles/autellix/models/SmolLM2-135M-Instruct \
  python -m unittest discover -s tests -p test_pipeline_gpu.py -v
```

该测试默认运行真实 PP=3；设置 `AUTELLIX_TEST_PP_DEVICES=0,1` 使用 PP=2。
测试强制 KV 换出和恢复，比较中断与连续生成，检查服务历史继承、取消清理以及
所有流水线阶段逐条一致的调度记录。

完整的小模型验证入口会依次运行测试、真实 HTTP 流程和四策略 DAG 工作负载：

```bash
bash scripts/validate_backend.sh vllm /absolute/path/to/model
```

SGLang 同样替换后端名称。可用 `AUTELLIX_PYTHON` 指定 Python 路径，`AUTELLIX_ENGINE_ARGS` 调整 HTTP 和 DAG 验证参数。生成结果写入 `outputs/validation/`，不纳入版本控制。KV 传输单独测量见 [cuda/README.md](../cuda/README.md)。

P40 需为 GPU unittest 另外设置 `AUTELLIX_TEST_LEGACY_GPU=1`，并在 HTTP/DAG
的 `AUTELLIX_ENGINE_ARGS` 中设置 `autellix_legacy_gpu=true`；两处分别控制对应入口。
当前三卡配置、检查结果和待办见[三卡大模型实验](three-gpu-experiment.md)，
早期单卡记录见 [P40 验收记录](p40-validation-20261115.md)。
专项运行会复用测试文件；汇总时按独立用例计数。

## 开发约束与修改流程

### 开发时必须保持的约束

1. **服务统计与优先级窗口分开**：防饥饿重置局部等待/运行窗口，保留调用整个生命周期的实际执行统计。
2. **按当前队列顺序移动请求**：降级和提升后使用更新后的 FIFO 顺序。
3. **窗口内不随意重新选批次**：新调用等待边界，提前完成由窗口已有预备请求补位；容量不足按有定义的顺序释放资源。
4. **先保全 KV，再释放所有权**：CPU 备份完成后释放原有 GPU KV；恢复数据保持前缀锁直到正式接纳。
5. **传输屏障保持顺序**：换出完成后复用 GPU block，原生多步的换入只执行一次。
6. **取消以确认完成为准**：等待 worker 确认，再检查 GPU、负载和会话清理。
7. **路由使用可观察输入**：按实际输入 token 数判断 2048 阈值。
8. **保留接口分层**：模拟器、fake worker 与真实后端分别测试和记录结果。

### 修改与升级流程

先阅读受影响模块及对应测试，在现有目录内修改；通常不需要增加新的说明文件。修改策略应同时检查真实 runtime 与模拟器，若两者有刻意差异，在架构文档中解释。

升级 vLLM/SGLang 时，在独立环境验证原生 scheduler、block/token pool、radix lock、输入输出结构、worker 启动方式和 CUDA 计时接口。同步更新版本检查、锁文件及适配代码，然后执行下列验证。

| 修改范围 | 至少需要的验证 |
| --- | --- |
| 文档、示例命令 | 路径/链接检查、对应 `--help`、Python 示例语法检查 |
| PLAS/ATLAS、窗口、路由 | `test_runtime.py`、`test_runtime_regressions.py`、`test_paper_plan.py`、`test_simulator_paper.py` |
| 客户端/HTTP | `test_client_lifecycle.py`、`test_client_stream.py`、`test_http.py`、`scripts/http_smoke.py` |
| KV 或 SGLang 锁/抢占 | host swap、reserve、decode pressure、SGLang scheduler 测试，加真实 GPU 恢复一致性测试 |
| SGLang 分块续跑和恢复批次 | 检查当前窗口的接纳顺序、KV 所有权、分块余量与生成一致性，配对校验逐调用 token 数 |
| SGLang 解码步数 | `test_sglang_scheduler.py` 检查混合批次计数和降级，真实 GPU 核对首次降级前的解码步数与生成文本 |
| SGLang 部分解码保留 | `test_sglang_admission.py` 检查按优先级截断和 radix 锁配对，定向 GPU 验证抢占数量和生成文本 |
| IPC、取消、副本协调 | `test_transport.py`，加 `AUTELLIX_TEST_REPLICAS=1` 真实双进程测试 |
| TP 适配 | 两张及以上 GPU 的 `AUTELLIX_TEST_TP=1`，没有设备时明确保留未验证状态 |

定向运行示例：

```bash
python -B -m unittest discover -s tests -p test_runtime.py -v
python -B -m unittest discover -s tests -p test_http.py -v
python -B scripts/http_smoke.py --help
```

增加新策略需更新 `PolicyConfig` 的合法值、继承和完成统计、队列行为、服务/benchmark CLI 的策略选项及测试；若需要模拟支持，再修改 `autellix/core/schedulers.py`。新增后端还需实现工作进程使用的初始化、分词、接纳、step、cancel、has_work 和 close 接口，并接入副本配置及 CLI 校验。当前没有自动注册后端的插件接口。

## 后续开发方向

在固定环境上维护回归基线，逐项扩展模型、设备和后端版本。

当前需要额外开发或验证的方向包括：多 GPU TP 的硬件覆盖、更广模型/KV 布局支持、后端版本迁移，以及鉴权、限流、失败副本重启等服务能力。实验报告记录各项实现、测试结果和待办。

## 一次开发变更的完成标准

1. 明确触发条件和预期行为，用最小输入描述问题；读取当前实现，不凭后端新版本的接口印象修改固定版本钩子。
2. 为有行为变化的边界补回归测试，尤其是取消/失败后的资源回收；不只检查返回文本是否非空。
3. 先跑受影响的 CPU 测试，再按上表选择真实后端测试；GPU KV 所有权修改包含实际换出/恢复验证。
4. 输出一致性使用同一个加载引擎、同一提示和贪心参数比较中断/不中断执行，并记录模型精度和批次配置。
5. 检查 `git diff --check` 和变更文件。模型、PDF、虚拟环境、缓存、`outputs/` 不入库。
6. 提交时说明改了什么、验证了哪些条件、哪些条件仍未验证；必要时同步接口、配置和后端文档。

文档改动通常只需检查链接、示例语法和对应 CLI，不需要重跑耗时的 GPU 测试。实验输出保存在 `outputs/`。

## 后端迁移清单

迁移涉及以下调用契约：

- worker 的启动方式、CUDA 可见设备和模型初始化顺序；子进程能否正确安装调度钩子。
- 请求进入、完成、取消和失败时的回调；每条路径是否正确更新程序统计。
- vLLM block 分配/追加/换入换出方法及 SchedulerOutputs；混合阶段的元数据顺序。
- SGLang token pool 与 radix cache 的锁引用、释放规则、重排后的 tensor 与 request 对齐。
- KV tensor 形状、dtype、stride 和索引映射；原生复制与批量传输是否保留相同内容。
- 批次执行计时和事件同步；是不是仍只把执行时间计入参与调用。
- 流式输出结构、结束原因和 token 数，以及客户端的完成与错误事件。

验证后更新锁文件和版本校验。现有适配依赖固定版本的内部 API；升级时按上述契约迁移。SGLang TP、其他 KV 布局和多模态分别扩展对应模型执行与缓存路径，并补充硬件测试。

## 示例命令验证

下列命令只查看入口参数，不启动模型：

```bash
python -B -m autellix.runtime.server --help
python -B -m autellix.runtime.benchmark --help
python -B examples/real_inference.py --help
python -B scripts/http_smoke.py --help
```

执行子目录中的示例脚本前，先在对应后端环境完成 `pip install -e .` 或运行后端安装脚本，再执行真实 GPU 检查。
