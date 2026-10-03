"""Session ownership and error cleanup without a running HTTP service."""
import unittest
from unittest.mock import patch

from autellix.runtime.client import InferenceClient


class ClientLifecycleTests(unittest.TestCase):
    def test_closed_client_rejects_explicit_and_context_sessions(self):
        client = InferenceClient()
        client.close()
        with patch.object(client, "_request") as request:
            with self.assertRaisesRegex(RuntimeError, "closed"):
                client.chat([], session_id="explicit")
            token = client._session.set("scoped")
            try:
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    client.chat([])
            finally:
                client._session.reset(token)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                with client.session():
                    self.fail("closed client opened a session")
            request.assert_not_called()

    def test_forked_client_never_touches_inherited_lock_or_network(self):
        client = InferenceClient()
        try:
            with patch("autellix.runtime.client.os.getpid", return_value=client._owner + 1), \
                    patch.object(client, "_request") as request, \
                    patch.object(client, "_lock") as lock:
                for action in (lambda: client.chat([], session_id="explicit"),
                               lambda: client.chat([]),
                               lambda: client.session().__enter__()):
                    with self.assertRaisesRegex(RuntimeError, "forking"):
                        action()
                client.close()
                request.assert_not_called()
                lock.__enter__.assert_not_called()
        finally:
            client.close()

    def test_cleanup_failure_preserves_application_error_and_cause(self):
        for scoped in (False, True):
            with self.subTest(scoped=scoped):
                client = InferenceClient()
                application = ValueError("application")
                cleanup = OSError("server unavailable")
                def request(method, path, body=None):
                    if method == "DELETE":
                        raise cleanup
                    return {"session_id": "program"}
                try:
                    with patch.object(client, "_request", side_effect=request):
                        with self.assertRaises(ValueError) as caught:
                            with (client.session() if scoped else client):
                                raise application
                    self.assertIs(caught.exception, application)
                    self.assertIs(caught.exception.__cause__, cleanup)
                    self.assertIsNone(client._session.get())
                finally:
                    with patch.object(client, "_request"):
                        client.close()

    def test_cleanup_failure_without_application_error_is_visible(self):
        client = InferenceClient()
        try:
            with patch.object(client, "_request", side_effect=[
                    {"session_id": "program"}, OSError("cleanup")]):
                with self.assertRaisesRegex(OSError, "cleanup"):
                    with client.session():
                        pass
            self.assertIsNone(client._session.get())
        finally:
            client.close()

    def test_scope_exit_in_child_does_not_end_parent_session(self):
        client = InferenceClient()
        try:
            with patch.object(client, "_request", return_value={"session_id": "parent"}) as request:
                scope = client.session()
                self.assertEqual(scope.__enter__(), "parent")
                with patch("autellix.runtime.client.os.getpid", return_value=client._owner + 1):
                    scope.__exit__(None, None, None)
                self.assertEqual(request.call_count, 1)
                self.assertIsNone(client._session.get())
        finally:
            client.close()
