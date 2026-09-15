from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
import queue
from contextlib import asynccontextmanager

from .engine import InferenceEngine, ReplicaConfig
from .policy import PolicyConfig


def create_app(engine):
    from fastapi import FastAPI, HTTPException, Request
    # Request must be available when FastAPI resolves postponed annotations.
    globals()["Request"] = Request

    @asynccontextmanager
    async def lifespan(app):
        try:
            await asyncio.to_thread(engine.wait_ready)
            yield
        finally:
            await asyncio.to_thread(engine.shutdown)

    app = FastAPI(title="Autellix", lifespan=lifespan)
    requests = {}

    @app.get("/health")
    async def health():
        if engine.closed or len(engine.failed) == len(engine.replicas):
            raise HTTPException(503, "no healthy inference replicas")
        return {"status": "ok", "backend": [r.backend for r in engine.replicas],
                "policy": engine.policy.policy, "loads": list(engine.loads)}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": [{"id": engine.model, "object": "model", "owned_by": "local"}]}

    @app.post("/sessions")
    async def start(body: dict):
        try:
            return {"session_id": engine.start_session(body.get("program_id"))}
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.delete("/sessions/{pid}")
    async def end(pid: str):
        engine.end_session(pid)
        return {"session_id": pid, "status": "closing"}

    @app.delete("/requests/{rid}")
    async def cancel(rid: str):
        future = requests.get(rid)
        if future is None:
            raise HTTPException(404, "request not active")
        future.cancel()
        return {"request_id": rid, "status": "cancel_requested"}

    @app.post("/v1/chat/completions")
    async def chat(body: dict, request: Request):
        if body.get("model", engine.model) != engine.model:
            raise HTTPException(404, "model is not loaded")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a nonempty list")
        automatic = not body.get("session_id")
        pid = engine.start_session() if automatic else body["session_id"]
        rid = body.get("request_id") or uuid.uuid4().hex
        if rid in requests:
            if automatic:
                engine.end_session(pid)
            raise HTTPException(409, "request ID already active")
        sampling = {k: body[k] for k in ("temperature", "top_p", "top_k", "max_tokens", "stop", "n") if k in body}
        future = None
        streaming = False
        try:
            future = await asyncio.to_thread(engine.submit, pid, messages=messages,
                                             sampling=sampling, call_id=body.get("call_id"),
                                             thread_id=body.get("thread_id"), metadata=body.get("metadata"))
            requests[rid] = future
            if body.get("stream"):
                from fastapi.responses import StreamingResponse
                async def stream():
                    previous = ""
                    try:
                        while True:
                            if await request.is_disconnected():
                                future.cancel()
                                break
                            try:
                                part = future.progress.get_nowait()
                            except queue.Empty:
                                part = None
                            if part is not None:
                                text = part.get("text", "")
                                if not text.startswith(previous):
                                    raise RuntimeError("backend revised previously streamed text")
                                delta = text[len(previous):]
                                previous = text
                                chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                                         "model": engine.model, "choices": [{"index": 0, "delta": {"content": delta},
                                                                              "finish_reason": None}]}
                                yield "data: " + json.dumps(chunk) + "\n\n"
                            if future.done() and future.progress.empty():
                                result = future.result()
                                reason = result.get("finish_reason", "stop")
                                if isinstance(reason, dict):
                                    reason = reason.get("type", "stop")
                                chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                                         "model": engine.model, "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "length" if reason == "length" else "stop"}]}
                                yield "data: " + json.dumps(chunk) + "\n\n"
                                yield "data: [DONE]\n\n"
                                break
                            await asyncio.sleep(.01)
                    except asyncio.CancelledError:
                        future.cancel()
                        raise
                    except Exception as exc:
                        yield "data: " + json.dumps({"error": {"message": str(exc)}}) + "\n\n"
                    finally:
                        if not future.done():
                            future.cancel()
                        requests.pop(rid, None)
                        if automatic:
                            engine.end_session(pid)
                streaming = True
                return StreamingResponse(stream(), media_type="text/event-stream")
            wrapped = asyncio.wrap_future(future)
            while not wrapped.done():
                if await request.is_disconnected():
                    future.cancel()
                    raise HTTPException(499, "client disconnected")
                await asyncio.wait({wrapped}, timeout=.1)
            result = await wrapped
            completion_tokens = result.get("completion_tokens", len(result["token_ids"]))
            finish_reason = result["finish_reason"]
            if isinstance(finish_reason, dict):
                finish_reason = finish_reason.get("type", "stop")
            if finish_reason not in {"length", "stop"}:
                finish_reason = "stop"
            return {"id": rid, "object": "chat.completion", "created": int(time.time()),
                    "model": engine.model, "session_id": pid,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": result["text"]},
                                 "finish_reason": finish_reason}],
                    "usage": {"prompt_tokens": result["prompt_tokens"], "completion_tokens": completion_tokens,
                              "total_tokens": result["prompt_tokens"] + completion_tokens},
                    "autellix": {"engine_id": result["engine_id"], "metrics": result["metrics"]}}
        except HTTPException:
            raise
        except asyncio.CancelledError:
            if future:
                future.cancel()
            raise
        except (ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(500, str(exc)) from exc
        finally:
            if not streaming:
                requests.pop(rid, None)
                if automatic:
                    engine.end_session(pid)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run real Autellix inference replicas")
    parser.add_argument("--backend", choices=("vllm", "sglang"), default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--devices", default="0", help="comma separated GPU IDs, one replica per entry")
    parser.add_argument("--policy", choices=("fcfs", "mlfq", "plas", "atlas"), default="atlas")
    parser.add_argument("--engine-args", default="{}", help="JSON arguments for the pinned backend")
    parser.add_argument("--state-dir")
    parser.add_argument("--schedule-interval", type=int, default=1)
    parser.add_argument("--overprovision", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    replicas = [ReplicaConfig(args.backend, args.model, d.strip(), json.loads(args.engine_args))
                for d in args.devices.split(",")]
    engine = InferenceEngine(replicas, state_dir=args.state_dir,
                             policy=PolicyConfig(policy=args.policy, schedule_interval=args.schedule_interval,
                                                 overprovision=args.overprovision))
    import uvicorn
    try:
        uvicorn.run(create_app(engine), host=args.host, port=args.port)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
