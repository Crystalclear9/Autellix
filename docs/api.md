# Python 与 HTTP 接口

[文档导航](README.md) · [项目首页](../README.md)

本文档中的命令均从仓库根目录执行。

## 直接 Python 接入

下面是一个完整的真实推理脚本骨架。保存为 Python 脚本运行，不要在交互式解释器中直接创建多进程引擎。

```python
from autellix.runtime import InferenceEngine, PolicyConfig, ReplicaConfig


def main():
    replica = ReplicaConfig(
        backend="vllm",
        model="HuggingFaceTB/SmolLM2-135M-Instruct",
        device="0",
        engine_args={
            "max_model_len": 512,
            "max_num_seqs": 2,
            "gpu_memory_utilization": 0.35,
            "swap_space": 0.25,
        },
    )
    policy = PolicyConfig(policy="atlas", schedule_interval=8, overprovision=1)
    with InferenceEngine([replica], policy=policy) as engine:
        pid = engine.start_session()
        future = None
        try:
            future = engine.submit(
                pid,
                prompt="The capital of France is",
                call_id="first",
                thread_id="main",
                sampling={"temperature": 0, "max_tokens": 16},
            )
            result = future.result(timeout=180)
            print(result["text"])
            print(result["engine_id"], result["metrics"])
        finally:
            if future is not None and not future.done():
                future.cancel()
            engine.end_session(pid)
            engine.wait_idle(timeout=30)


if __name__ == "__main__":
    main()
```

### 对象与生命周期

| 接口 | 行为与约束 |
| --- | --- |
| `ReplicaConfig(backend, model, device, engine_args)` | 描述一个副本；同一协调器内各副本须使用同一模型标识 |
| `InferenceEngine(replicas, policy=..., state_dir=..., locality_threshold=2048)` | 启动独立工作进程、IPC 队列和结果监听线程 |
| `wait_ready(timeout=600)` | 等待所有副本完成启动；引擎上下文进入时自动调用 |
| `start_session(program_id=None)` | 创建程序，未指定 ID 时自动生成；未结束的 ID 不能重复创建 |
| `submit(pid, ...)` | `prompt`、`messages`、`input_ids` 三选一；返回 Future |
| `tokenize(prompt=... / messages=...)` | 使用真实后端 tokenizer；两种输入只能选一种 |
| `end_session(pid)` | 禁止继续向该程序提交；不会主动取消已接纳调用 |
| `wait_idle(timeout=600)` | 等待整个引擎的活动调用及取消确认完成，不只等待某个程序 |
| `shutdown()` | 停止工作进程、结束会话并关闭 IPC 和数据库；上下文退出时自动调用 |

`submit()` 对模型执行是异步的，但字符串/消息输入需要先完成分词，提交本身不保证立即返回。`input_ids` 可省去这次分词，调用者须确保 token 与所加载模型匹配。

`call_id` 在一个程序会话内必须唯一，包括已完成的调用；省略时自动生成。`thread_id` 省略时由当前线程标识生成。`metadata` 用于随请求记录可 JSON 序列化的信息，不参与优先级公式。

一个会话代表一个逻辑程序，不是全体用户的公共桶。不同程序应使用不同会话，同一程序的连续和并行调用应复用会话。服务历史只在调用完成后更新；并行提交的调用可能继承相同历史，这是在线算法的行为。

### 返回值、超时和取消

真实结果主要包括 `text`、`token_ids`、`prompt_tokens`、`finish_reason`、`metrics`、`program_id`、`call_id` 和 `engine_id`。SGLang 还可能提供 `completion_tokens`，其 `token_ids` 可能为空；直接接口中的 `finish_reason` 也可能是后端结构，不能假定所有后端返回同一种字符串。HTTP 层将结束原因统一为 `stop` 或 `length`。

`future.result(timeout=...)` 超时仅停止调用者等待，不会自动取消 GPU 请求。需要主动调用 `future.cancel()`。取消请求发出后，必须等待工作进程确认才能认定负载和程序状态已回收；可用 `wait_idle()` 验证。

进程失败会使对应未完成调用以异常结束。协调器不会自动重启失败副本，也不会把失败的在途调用重新提交到其他引擎。是否重试、怎样处理业务重复执行，需要由上层应用明确决定。

## HTTP 请求约定

这是项目实现的聊天接口子集，并非完整 OpenAI API。未实现的工具调用、结构化输出、多模态及其他扩展字段，不应作为可用功能依赖。

| 请求字段 | 用途 |
| --- | --- |
| `messages` | 必需，非空消息列表；格式还必须适合模型 chat template |
| `model` | 可省略；提供时须与服务配置的模型标识一致 |
| `session_id` | 已创建的程序会话 ID；省略则自动创建一次性会话 |
| `request_id` | 可选的 HTTP 请求标识；活动请求间不能重复，可用于取消 |
| `call_id`、`thread_id` | 程序内调用标识、线程标识，非空字符串 |
| `metadata` | 可选 JSON 对象 |
| `temperature`、`top_p`、`top_k`、`max_tokens`、`stop`、`n` | 当前转发给后端的采样字段；支持范围由后端约束，`n` 只能为 1 |
| `stream` | true 时返回 SSE；Python 客户端返回可迭代的数据块 |

HTTP `request_id` 与协调器内部请求 ID 是两个标识，日志关联时还应保留 `call_id`。HTTP 取消入口使用 HTTP 请求 ID，不使用内部 ID。未列出的采样字段不会自动转发，不能假定请求中的字段都已生效。

完整的最小 HTTP 会话流程：

```bash
curl -sS http://127.0.0.1:8000/sessions \
  -H 'Content-Type: application/json' -d '{"program_id":"demo"}'

curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","request_id":"demo-request-1","call_id":"first","messages":[{"role":"user","content":"Hello"}],"max_tokens":16}'

curl -sS -X DELETE http://127.0.0.1:8000/sessions/demo
```

会话 ID 在 URL 路径中必须正确编码。流式请求可给 curl 加 `-N` 并在 JSON 中设置 `"stream":true`；流以 `data: [DONE]` 结束，中途错误可能作为 `error` 事件返回。

常见状态码：400 表示输入或会话参数问题，404 表示模型不匹配或取消目标不存在，409 表示活动请求 ID 冲突，499 表示请求取消，500 表示执行失败；没有健康副本时健康接口返回 503。部分后端参数错误发生在工作进程内，会表现为执行失败，不能仅凭状态码判断能否重试。

## 多线程与会话作用域

自动会话由同一个客户端实例共享，适合一个逻辑程序的并行调用。显式 `client.session()` 使用上下文变量；新建线程不会自动继承当前线程的显式作用域。跨线程传递此类会话时，应将作用域返回的 ID 显式传给 `chat(session_id=...)`，并在所有调用结束后退出作用域。

客户端 `timeout` 默认 600 秒，作为 HTTP 连接的超时参数使用，不是整个程序或整个 SSE 流的总执行期限。程序总超时、重试和重试后的业务去重由调用方管理。客户端尚无自动重试、异步 API 或完整 SDK 对象体系，返回值是普通字典/迭代器。

普通响应包含 `choices`、`usage`、`session_id` 和 `autellix` 调度信息；流式块主要是 `choices[0].delta` 与最终 `finish_reason`，不承诺包含普通响应里的所有统计字段。需要完整逐调用指标时使用直接 Python 结果或服务端状态/trace。
