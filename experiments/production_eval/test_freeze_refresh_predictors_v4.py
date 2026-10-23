"""Construction-label scope and representation provenance for frozen proxies."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.production_eval.freeze_refresh_predictors_v4 import scoped_answers, grade_construction, load_features
from experiments.production_eval.synthetic_tasks import digest
from experiments.production_eval.refresh_run_contract_v4 import sha


class FreezePredictorTests(unittest.TestCase):
    def setUp(self):
        self.expected = {key: {"example_id": key, "family_id": "family_"+key, "input_hash": digest(key)} for key in ("a", "b")}
        self.answers = {"a": '{"n":1}', "b": '{"n":2}'}

    def gold(self):
        return [{"example_id": key, "family_id": self.expected[key]["family_id"], "answer": value, "answer_hash": digest(value)} for key, value in self.answers.items()]

    def responses(self):
        return [{**row, "config_id": "model", "partition": "construction", "status": "success", "run_manifest_sha256": "frozen",
                 "response": self.answers[key], "response_sha256": digest(self.answers[key])} for key, row in self.expected.items()]

    def test_future_gold_fields_are_never_consulted(self):
        # The future sentinel has no answer/hash/family fields: access would fail.
        self.assertEqual(scoped_answers(self.gold()+[{"example_id": "future"}], self.expected), self.answers)
        for rows in (self.gold()[:1], self.gold()+self.gold()[:1]):
            with self.assertRaises(ValueError): scoped_answers(rows, self.expected)
        changed=self.gold(); changed[0]["answer"]='{"n":99}'
        with self.assertRaises(ValueError): scoped_answers(changed, self.expected)

    def test_invalid_completed_format_is_zero_and_unknown_outcome_fails(self):
        rows=self.responses();rows[0].update(response="invalid",response_sha256=digest("invalid"))
        self.assertEqual(grade_construction(rows,self.expected,self.answers,"model","frozen"),{"family_a":0.,"family_b":1.})
        for field,value in (("status","api_error"),("partition","future"),("config_id","other"),("run_manifest_sha256","other"),("input_hash","other"),("response_sha256","other")):
            changed=copy.deepcopy(rows);changed[0][field]=value
            with self.assertRaises(ValueError): grade_construction(changed,self.expected,self.answers,"model","frozen")

    def test_missing_duplicate_and_wrong_family_construction_rows_fail(self):
        rows=self.responses()
        for changed in (rows[:1],[rows[0],rows[0]],[{**rows[0],"family_id":"family_b"},rows[1]]):
            with self.assertRaises(ValueError): grade_construction(changed,self.expected,self.answers,"model","frozen")

    def test_features_must_match_inputs_and_cannot_be_gold_augmented(self):
        rows=list(self.expected.values())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"features.npz"
            def write(ids=("a","b"), values=((1.,0.),(0.,1.))):
                np.savez(path,example_ids=np.array(ids),input_hashes=np.array([r["input_hash"] for r in rows]),embeddings=np.array(values))
            write()
            entry={"uses_gold":False,"model_outcomes_used":False,"embedding_file":str(path),"embedding_sha256":sha(path),"family_ids":[r["family_id"] for r in rows]}
            np.testing.assert_array_equal(load_features(entry,rows),np.eye(2))
            for field in ("uses_gold","model_outcomes_used"):
                with self.assertRaises(ValueError): load_features({**entry,field:True},rows)
            write(ids=("b","a"));entry["embedding_sha256"]=sha(path)
            with self.assertRaises(ValueError): load_features(entry,rows)
            write(values=((float("nan"),0.),(0.,1.)));entry["embedding_sha256"]=sha(path)
            with self.assertRaises(ValueError): load_features(entry,rows)


if __name__ == "__main__":
    unittest.main()
