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

    def test_incomplete_or_unknown_roles_are_not_silently_repaired(self):
        row=self.row(); row["conversations"].pop()
        self.assertIsNone(module.convert(row,Tokenizer(),100,2))
        row=self.row(); row["conversations"][0]["from"]="tool"
        self.assertIsNone(module.convert(row,Tokenizer(),100,2))
