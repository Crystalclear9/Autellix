import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location("sharegpt_replay",Path(__file__).parents[1]/"scripts/prepare_sharegpt_replay.py")
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class Tokenizer:
    def encode(self,text,**kwargs): return text.split()
    def apply_chat_template(self,messages,**kwargs):
        return " ".join(m["content"] for m in messages)+" assistant:"


class ShareGPTReplayTests(unittest.TestCase):
    def test_original_conversation_restored_by_message_offsets(self):
        messages = self.row()["conversations"]
        fragments = [dict(id="p_2", conversations=messages[2:]),
                     dict(id="p_0", conversations=messages[:3])]
        self.assertEqual(module.reconstruct_conversations(fragments), [self.row()])
        broken = [dict(id="p_0", conversations=messages[:1]), dict(id="p_2", conversations=messages[2:])]
        with self.assertRaisesRegex(ValueError, "missing"):
            module.reconstruct_conversations(broken)
        fragments[0]["conversations"] = [dict(from_="changed")]
        with self.assertRaisesRegex(ValueError, "conflicting"):
            module.reconstruct_conversations(fragments)

    def row(self):
        return {"id":"p","conversations":[{"from":role,"value":text} for role,text in
                [("human","first question"),("gpt","reference answer one"),
                 ("human","second question"),("gpt","reference answer two")]]}

    def test_preserves_turn_dependencies_and_reference_history(self):
        result=module.convert(self.row(),Tokenizer(),100,2)
        self.assertEqual(len(result["calls"]),2)
        a,b=result["calls"]
        self.assertEqual(a["parents"],[])
        self.assertEqual(b["parents"],["0"])
        self.assertIn("reference answer one",b["prompt"])
        self.assertFalse(b["append_parent_outputs"])
        self.assertEqual(b["reference_output_tokens"],3)
        self.assertEqual(b["max_tokens"],2)
        self.assertTrue(b["sampling"]["ignore_eos"])

    def test_oversized_later_turn_drops_whole_record(self):
        self.assertIsNone(module.convert(self.row(),Tokenizer(),6,2))

    def test_uncapped_reference_lengths(self):
        calls=module.convert(self.row(),Tokenizer(),100,0)["calls"]
        self.assertEqual([c["max_tokens"] for c in calls],[3,3])

    def test_template_is_encoded_without_second_special_token_pass(self):
        class RecordingTokenizer(Tokenizer):
            def encode(self, text, **kwargs):
                self_test.assertFalse(kwargs["add_special_tokens"])
                return super().encode(text, **kwargs)
        self_test = self
        calls = module.convert(self.row(), RecordingTokenizer(), 100, 0)["calls"]
        self.assertEqual(calls[0]["input_ids"], calls[0]["prompt"].split())
        self.assertEqual(calls[0]["input_tokens"], len(calls[0]["input_ids"]))

    def test_incomplete_or_unknown_roles_are_not_silently_repaired(self):
        row=self.row(); row["conversations"].pop()
        self.assertIsNone(module.convert(row,Tokenizer(),100,2))
        row=self.row(); row["conversations"][0]["from"]="tool"
        self.assertIsNone(module.convert(row,Tokenizer(),100,2))
