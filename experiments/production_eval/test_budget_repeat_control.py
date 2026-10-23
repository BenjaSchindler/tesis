import copy
import unittest
from unittest.mock import patch

from experiments.production_eval import experiment_inference as inference
from experiments.production_eval.budget_repeat_control import repeat_config, verify_protocol
from experiments.production_eval.test_budget_sensitivity import baseline_config, record


class RepeatControlTest(unittest.TestCase):
    def test_only_bookkeeping_changes_no_gold_leak(self):
        base = baseline_config(); original = copy.deepcopy(base)
        for repeat in (1, 2):
            config = repeat_config(base, repeat)
            a, b = verify_protocol(record(), base, config)
            self.assertNotEqual(a["cache_key"], b["cache_key"])
            self.assertEqual(config["max_output_tokens"], 1024)
            secret = {**record(), "answer": "SECRET", "metadata": {"gold": "SECRET"}}
            self.assertEqual(b, inference.prepare(secret, config))
        self.assertEqual(original, base)

    def test_profile_seed_ceiling_and_prompt_changes_rejected(self):
        base = baseline_config()
        for key, value in (("seed", 1), ("max_output_tokens", 2048), ("profile", "direct")):
            config = repeat_config(base, 1); config[key] = value
            with self.assertRaises(ValueError):
                verify_protocol(record(), base, config)
        for repeat in (0, 3):
            with self.assertRaises(ValueError):
                repeat_config(base, repeat)

    def test_model_endpoint_body_identical_across_baseline_and_repeats(self):
        bodies = []
        class Response:
            status_code = 200
            def __init__(self, value): self.value = value
            def raise_for_status(self): pass
            def json(self): return self.value
        class Session:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def mount(self, *args): pass
            def post(self, url, json, timeout):
                if url.endswith("/apply-template"): return Response({"prompt": "same prompt"})
                if url.endswith("/tokenize"): return Response({"tokens": [1, 2]})
                bodies.append(json)
                return Response({"choices": [{"message": {"content": '{"work":"a","answer":"True"}'}, "finish_reason": "stop"}], "usage": {"completion_tokens": 7}})
        base = baseline_config()
        with patch("requests.Session", Session):
            for config in (base, repeat_config(base, 1), repeat_config(base, 2)):
                inference.local_call(inference.prepare(record(), config), config)
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(bodies[0], bodies[2])
        self.assertEqual(bodies[0]["max_tokens"], 1024)
        self.assertNotIn("config_id", bodies[0])
        self.assertNotIn("repeat_control_revision", bodies[0])


if __name__ == "__main__":
    unittest.main()
