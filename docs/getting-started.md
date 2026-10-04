# 安装与首次使用

[文档导航](README.md) · [项目首页](../README.md)

本文档中的命令均从仓库根目录执行。

## 安装

真实后端需要 Linux 或 Ubuntu WSL、可用的 NVIDIA GPU，以及 Python 3.10。两个后端使用不同依赖环境，不要安装到同一个环境。

| 后端 | 固定版本 | 说明 |
| --- | --- | --- |
| vLLM | 0.6.1 | [支持范围与实现](../integrations/vllm/README.md) |
| SGLang | 0.4.9.post6 | [支持范围与实现](../integrations/sglang/README.md) |

在仓库根目录执行，需预先安装 `uv`：

```bash
sudo apt-get update && sudo apt-get install -y build-essential python3-dev
bash scripts/setup_backend.sh vllm
source ~/autellix-envs/vllm/bin/activate
```

安装 SGLang 时将 `vllm` 换成 `sglang`。脚本使用 `requirements/` 中的完整依赖锁文件，并以可编辑方式安装本项目。`AUTELLIX_ENV_DIR` 可指定环境路径；环境内的安装元数据应保留。

## 环境检查与开发安装

下面的命令在 Ubuntu/WSL 中执行。服务端需要 GPU，CPU 模拟器和大部分逻辑测试不需要 GPU。

```bash
nvidia-smi
python --version
command -v uv
```

后端安装完成并激活环境后，确认当前解释器确实来自该环境：

```bash
which python
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -c 'from importlib.metadata import version; print(version("vllm"))'
```

SGLang 环境将最后一行的 `vllm` 改为 `sglang`。不要仅根据终端提示符判断环境。锁文件是固定版本依赖清单，不包含模型权重、操作系统、驱动、下载源或包文件哈希，因此不能单独保证所有机器上完全一致的安装结果。

只开发模拟器、策略或 HTTP 协议时，可以创建独立 CPU 环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[server]' httpx matplotlib
python -B -m unittest discover -s tests -v
```

Windows PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`。没有 Torch 时，依赖 CPU tensor 的测试会跳过；没有显式 GPU 环境变量时，真实推理测试会跳过。检查输出中的跳过原因，不能把 `OK (skipped=...)` 当成所有路径均已验证。

可编辑安装使源码修改立即生效，但运行中的服务和工作进程仍需重启。修改 `pyproject.toml`、入口点或依赖后需重新安装；升级后端版本不能只修改锁文件中的版本号。

## 启动真实推理服务

以下配置用于小模型功能验证。`--model` 可以是模型名称或本地模型目录，显存和上下文参数需按模型调整。聊天请求要求模型 tokenizer 提供 chat template。

### vLLM

```bash
autellix-serve --backend vllm --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 --state-dir /tmp/autellix-server \
  --engine-args '{"max_model_len":512,"max_num_seqs":4,"gpu_memory_utilization":0.35,"swap_space":0.25}'
```

### SGLang

在 SGLang 环境执行：

```bash
autellix-serve --backend sglang --model HuggingFaceTB/SmolLM2-135M-Instruct \
  --policy atlas --devices 0 \
  --engine-args '{"context_length":512,"max_running_requests":4,"mem_fraction_static":0.35}'
```

`--devices 0,1` 表示两个独立单 GPU 副本。vLLM 张量并行的配置和限制见其后端说明。共享 SQLite 状态应放在 Linux 本地文件系统，不要放在网络盘或 WSL 的 Windows 挂载路径。

## 客户端

真实推理使用 `autellix.runtime`。同一客户端的自动会话可供多个线程共享；上下文退出时结束会话，已接纳的调用完成或取消后释放程序状态。

```python
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    answer = client.chat(
        [{"role": "user", "content": "What is the capital of France?"}],
        temperature=0, max_tokens=32,
    )
    print(answer["choices"][0]["message"]["content"])
```

流式响应返回迭代器。提前停止时应关闭迭代器，让服务端能够取消未完成生成：

```python
from contextlib import closing
from autellix.runtime import InferenceClient

with InferenceClient("http://127.0.0.1:8000") as client:
    with closing(client.chat(
        [{"role": "user", "content": "Explain program-aware scheduling."}],
        stream=True, max_tokens=64,
    )) as chunks:
        for chunk in chunks:
            print(chunk["choices"][0]["delta"].get("content", ""), end="", flush=True)
```

`client.session()` 可显式创建独立会话作用域；自定义程序 ID 支持中文和 `/`。fork 后需创建新客户端，已关闭的客户端不能继续发请求。流式错误或缺少 `[DONE]` 会抛出异常；清理失败不会掩盖已有的应用异常。解释器退出清理是尽力执行，建议使用上下文管理器。

绕过 HTTP 的真实 Python 示例见 [examples/real_inference.py](../examples/real_inference.py)。`InferenceEngine.submit()` 返回由工作进程结果完成的 Future；创建引擎的代码需放在 `if __name__ == "__main__":` 中。

| HTTP 接口 | 用途 |
| --- | --- |
| `POST /v1/chat/completions` | 普通或 SSE 流式聊天 |
| `GET /v1/models` | 已加载模型 |
| `POST /sessions`、`GET /sessions` | 创建会话、查看程序状态 |
| `DELETE /sessions/{id}` | 结束会话 |
| `DELETE /requests/{id}` | 取消活动请求 |
| `GET /health` | 副本健康状态和负载 |

原始 HTTP 请求若不携带 `session_id`，每个请求会成为独立程序，无法继承此前调用的服务历史。
