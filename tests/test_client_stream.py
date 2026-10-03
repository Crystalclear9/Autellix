"""Real client SSE parsing and response lifetime without a GPU model."""
import io
import json
import unittest
from unittest.mock import patch

from autellix.runtime.client import InferenceClient


class ClientStreamTests(unittest.TestCase):
    def test_stream_parsing_metadata_and_completion_close(self):
        response = io.BytesIO((
            ': heartbeat\r\n\r\n'
            'data: {"choices":\r\n'
            'data: [{"delta": {"content": "你好"}}]}\r\n\r\n'
            'data: [DONE]\r\n\r\n').encode())
        with InferenceClientForTest() as client:
            with patch("autellix.runtime.client.urlopen", return_value=response) as open_response:
                chunks = list(client.chat([], session_id="p", stream=True))
                body = json.loads(open_response.call_args.args[0].data)
                self.assertEqual(body["session_id"], "p")
                self.assertTrue(body["call_id"])
                self.assertTrue(body["thread_id"])
                self.assertTrue(body["stream"])
            self.assertEqual(chunks[0]["choices"][0]["delta"]["content"], "你好")
            self.assertTrue(response.closed)

    def test_early_close_releases_http_response(self):
        response = io.BytesIO(b'data: {"choices": []}\n\ndata: [DONE]\n\n')
        with InferenceClientForTest() as client:
            with patch("autellix.runtime.client.urlopen", return_value=response):
                stream = client.chat([], stream=True)
                self.assertEqual(next(stream), {"choices": []})
                stream.close()
                self.assertTrue(response.closed)

    def test_server_error_and_truncation_are_not_success(self):
        for payload, message in ((b'data: {"error": {"message": "worker failed"}}\n\n', "worker failed"),
                                 (b'data: {"choices": []}\n\n', "before \\[DONE\\]")):
            with self.subTest(payload=payload), InferenceClientForTest() as client:
                response = io.BytesIO(payload)
                with patch("autellix.runtime.client.urlopen", return_value=response):
                    with self.assertRaisesRegex(RuntimeError, message):
                        list(client.chat([], stream=True))
                self.assertTrue(response.closed)

    def test_deferred_stream_cannot_start_after_client_closes(self):
        with InferenceClientForTest() as client:
            stream = client.chat([], stream=True)
        with patch("autellix.runtime.client.urlopen") as request:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                next(stream)
            request.assert_not_called()


class InferenceClientForTest(InferenceClient):
    def _request(self, method, path, body=None):
        return {"session_id": "p"}
