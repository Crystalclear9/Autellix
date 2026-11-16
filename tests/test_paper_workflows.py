import concurrent.futures
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


bfcl, lats = script("prepare_bfcl_replay"), script("collect_lats")
collect_bfcl = script("collect_bfcl")


class PaperWorkflows(unittest.TestCase):
    def test_bfcl_function_decoding_accepts_only_literal_tool_calls(self):
        parse = lambda text: collect_bfcl.parse_action(text, {"mkdir", "cd"})
        self.assertEqual(parse("```python\nmkdir(dir_name='reports')\n```"), "mkdir(dir_name='reports')")
        self.assertEqual(parse('{"name":"cd","parameters":{"folder":"reports"}}'), "cd(folder='reports')")
        self.assertIsNone(parse("Task completed."))
        self.assertIsNone(parse("mkdir(__import__('os').getcwd())"))
        self.assertIsNone(parse("__import__('os').getcwd()"))
        actions = collect_bfcl.parse_actions('<|python_tag|>[{"name":"cd","parameters":{"folder":"reports"}},'
                                            '{"name":"mkdir","parameters":{"dir_name":"new"}}]', {"cd", "mkdir"})
        self.assertEqual(actions, ["cd(folder='reports')", "mkdir(dir_name='new')"])
        self.assertEqual(collect_bfcl.parse_actions("Done.", {"cd"}), [])
        self.assertEqual(collect_bfcl.parse_actions(
            '{"name":"cd","parameters":{"folder":"a;b"}}; {"name":"cd","parameters":{"folder":"c"}}', {"cd"}),
            ["cd(folder='a;b')", "cd(folder='c')"])

    def test_bfcl_reference_tool_result_feeds_next_call(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(range(len(text.split()))),
            apply_chat_template=lambda messages, **kwargs: "\n".join(m["content"] for m in messages))
        executions = []
        def execute(actions, *args, **kwargs):
            executions.extend(actions)
            return ["created directory"], {}
        task = dict(id="example", initial_config={}, involved_classes=["Files"],
                    question=[[dict(role="user", content="Create and inspect a directory")]])
        answer = dict(ground_truth=[["mkdir('reports')", "ls()"]])
        row = bfcl.convert(task, answer, {"Files": []}, execute, tokenizer, 1024)
        self.assertEqual(executions, ["mkdir('reports')", "ls()"])
        self.assertEqual(row["calls"][1]["parents"], ["0"])
        self.assertIn("created directory", row["calls"][1]["prompt"])
        self.assertNotIn("created directory", row["calls"][0]["prompt"])
        self.assertIsNone(bfcl.convert(task, answer, {"Files": []}, execute, tokenizer, 1))

    def test_lats_parallel_group_has_join_dependencies(self):
        submitted = []
        def submit(*args, **kwargs):
            submitted.append(kwargs)
            future = concurrent.futures.Future()
            future.set_result(dict(text="Action: Search[Moon]", completion_tokens=7))
            return future
        engine = SimpleNamespace(start_session=lambda: "session", submit=submit)
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])
        recorder = lats.Recorder(engine, tokenizer, max_calls=3)
        recorder.begin()
        recorder.gpt("expand", n=2)
        recorder.gpt("evaluate", n=1)
        self.assertEqual([c["parents"] for c in recorder.calls], [[], [], ["0", "1"]])
        self.assertEqual([c["max_tokens"] for c in recorder.calls], [7, 7, 7])
        self.assertFalse(any(c["append_parent_outputs"] for c in recorder.calls))
        with self.assertRaises(lats.SearchBudget):
            recorder.gpt("over budget")
        self.assertEqual(len(submitted), 3)

    def test_hotpot_document_lookup_and_answer_normalization(self):
        env = lats.DocumentEnv([dict(question="Which satellite?", answer="The Moon",
                               context=dict(title=["Moon"], sentences=[["The Moon orbits Earth.", "Moon facts."]]))])
        self.assertEqual(env.reset(0), "Question: Which satellite?")
        self.assertIn("orbits Earth", env.step("search[Moon]")[0])
        self.assertIn("Result 1/2", env.step("lookup[Moon]")[0])
        self.assertIn("Result 2/2", env.step("lookup[Moon]")[0])
        self.assertEqual(env.step("finish[moon]")[1:3], (1., True))


if __name__ == "__main__":
    unittest.main()
