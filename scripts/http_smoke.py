"""Real service lifecycle + streaming test using an actual local TCP server."""
import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="vllm")
    parser.add_argument("--model", required=True)
    parser.add_argument("--engine-args", default="{}")
    parser.add_argument("--output", default="outputs/validation/http.json")
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="autellix-http-") as directory:
        command = [sys.executable, "-m", "autellix.runtime.server", "--backend", args.backend,
                   "--model", args.model, "--engine-args", args.engine_args, "--state-dir", directory,
                   "--port", str(port)]
        log = open(Path(directory, "server.log"), "w+")
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        def request(path, body=None, method=None):
            data = json.dumps(body).encode() if body is not None else None
            return urlopen(Request(f"http://127.0.0.1:{port}" + path, data=data,
                                  headers={"Content-Type": "application/json"}, method=method), timeout=120)
        try:
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("server exited during startup")
                try:
                    with request("/health") as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(.5)
            else:
                raise TimeoutError("server startup timed out")
            with request("/sessions", {}) as response:
                sid = json.load(response)["session_id"]
            body = dict(model=args.model, session_id=sid, temperature=0, max_tokens=20,
                        messages=[{"role": "user", "content": "What is the capital of France?"}])
            with request("/v1/chat/completions", body) as response:
                first = json.load(response)
            assert first["usage"]["completion_tokens"] > 0
            assert first["choices"][0]["message"]["content"]
            with request("/v1/chat/completions", {**body, "stream": True}) as response:
                stream = response.read().decode()
            assert "data: [DONE]" in stream
            with request("/sessions/" + sid, method="DELETE") as response:
                json.load(response)
            from autellix.runtime.client import InferenceClient
            with InferenceClient(f"http://127.0.0.1:{port}") as client:
                automatic_first = client.chat(body["messages"], temperature=0, max_tokens=4)
                automatic_next = client.chat(body["messages"], temperature=0, max_tokens=4)
                assert automatic_first["session_id"] == automatic_next["session_id"]
                assert automatic_next["autellix"]["metrics"]["inherited"] > 0
                assert automatic_next["autellix"]["metrics"]["metadata"]["thread_id"]
                with request("/sessions") as response:
                    active = json.load(response)[automatic_first["session_id"]]
                assert active["last_arrival"] is not None and active["last_completion"] is not None
            with request("/sessions") as response:
                assert json.load(response) == {}
            import sqlite3
            with sqlite3.connect(str(Path(directory, "programs.sqlite"))) as db:
                assert db.execute("SELECT count(*) FROM programs").fetchone()[0] == 0
            result = dict(backend=args.backend, health=health, completion=first,
                          streaming_completed=True, session_table_empty=True,
                          automatic_session_inherits_service=True, shared_timestamps_visible=True)
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(json.dumps(result, indent=2))
        except BaseException:
            log.flush()
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()


if __name__ == "__main__":
    main()
