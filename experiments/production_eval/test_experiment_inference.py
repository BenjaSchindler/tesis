import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from experiments.production_eval.experiment_inference import (
    DEFAULTS, VERSION, ContextLimitError, RunJournal, digest, gemini_call, infer_one, local_call,
    main, parse_output, prepare, read_jsonl, recover_local_output_parse_error, schema_for,
)


def config(provider="local", profile="direct"):
    return {**DEFAULTS, "config_id": provider + "_" + profile, "provider": provider,
            "profile": profile, "model": "fixture-local" if provider == "local" else "gemini-3.1-flash-lite",
            "model_revision": "fixed-fixture-revision", "runner_version": VERSION}


def row():
    return {"example_id": "id-1", "input_text": "How many is two plus three?",
            "input_hash": digest("How many is two plus three?"), "answer": "SECRET-GOLD-ANSWER",
            "answer_type": "decimal", "task_family": "arithmetic_word_problem",
            "metadata": {"answer": "SECRET-METADATA", "source": "SECRET-SOURCE"}}


def received(text='{"answer":"5"}'):
    return {"raw_response": {"choices": [{"message": {"content": text}}]},
            "raw_output_text": text, "usage": {"prompt_token_count": 10,
                "candidates_token_count": 5, "thoughts_token_count": 2, "total_token_count": 17},
            "finish_reason": "stop", "actual_model_version": "fixture"}


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / "out.jsonl"

    def journal(self, cfg=None, budget=5_000_000_000):
        journal = RunJournal(self.output, cfg or config(), budget)
        self.addCleanup(journal.close)
        return journal

    def test_gold_metadata_do_not_affect_messages_schema_or_hash(self):
        first, second = row(), row()
        second["answer"], second["metadata"] = "DIFFERENT", {"giant_gold": [42] * 20}
        a, b = prepare(first, config()), prepare(second, config())
        self.assertEqual(a["request"], b["request"])
        self.assertEqual(a["cache_key"], b["cache_key"])
        self.assertNotIn("SECRET", json.dumps(a["request"]))
        self.assertNotIn("id-1", json.dumps(a["request"]))

    def test_json_schema_fixed_independent_of_gold_keys_lengths_values(self):
        a, b = row(), row()
        a.update(answer_type="json", task_family="inventory_tracking", answer='{"wrong":123}')
        b.update(answer_type="json", task_family="inventory_tracking", answer='{"stock":[],"total_units":0}')
        self.assertEqual(prepare(a, config())["request"]["schema"], prepare(b, config())["request"]["schema"])
        schema_text = json.dumps(prepare(a, config())["request"]["schema"])
        for forbidden in ("enum", "minItems", "maxItems", "const", "wrong", "123"):
            self.assertNotIn(forbidden, schema_text)

    def test_full_hash_changes_for_profile_seed_model_revision(self):
        baseline = prepare(row(), config())["cache_key"]
        for key, value in (("profile", "reason_then_answer"), ("seed", 9), ("model_revision", "changed")):
            cfg = config(); cfg[key] = value
            self.assertNotEqual(baseline, prepare(row(), cfg)["cache_key"])

    def test_invalid_format_is_one_completed_outcome_no_retry_and_cached(self):
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        transport = MagicMock(return_value=received("not JSON"))
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "success")
        self.assertEqual(transport.call_count, 1)
        output = read_jsonl(self.output)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0]["format_status"], "invalid_json")
        self.assertIsNone(output[0]["answer"])
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "cached")
        self.assertEqual(transport.call_count, 1)
        journal.close()
        resumed = self.journal(cfg)
        self.assertIn(item["cache_key"], resumed.completed)

    def test_transport_retry_retains_attempts_but_one_terminal_output(self):
        cfg = config("gemini"); item = prepare(row(), cfg); journal = self.journal(cfg)
        transport = MagicMock(side_effect=[TimeoutError("SECRET CREDENTIAL"), received()])
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "success")
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(len(read_jsonl(self.output)), 1)
        attempts = read_jsonl(journal.attempt_file)
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]["terminal"])
        self.assertEqual(attempts[0]["charged_nanodollars"], item["reservation_nanodollars"])
        self.assertEqual(attempts[1]["charged_nanodollars"], 10 * 250 + 7 * 1500)
        self.assertNotIn("SECRET", journal.attempt_file.read_text())

    def test_uncertain_reservation_survives_resume_and_enforces_budget(self):
        cfg = config("gemini"); item = prepare(row(), cfg)
        journal = self.journal(cfg, item["reservation_nanodollars"])
        self.assertIsNotNone(journal.reserve(item))
        journal.close()
        resumed = self.journal(cfg, item["reservation_nanodollars"])
        self.assertIsNone(resumed.reserve(item))
        self.assertEqual(resumed.summary()["unsettled_attempts"], 1)
        self.assertEqual(resumed.summary()["ledger_accounted_usd"], item["reservation_nanodollars"] / 1e9)

    def test_two_transport_failures_are_one_unknown_terminal_outcome(self):
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        transport = MagicMock(side_effect=TimeoutError("private transport detail"))
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "api_error")
        self.assertEqual(transport.call_count, 2)
        self.assertEqual(len(read_jsonl(self.output)), 1)
        self.assertEqual(read_jsonl(self.output)[0]["format_status"], "not_received")
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "cached")
        self.assertEqual(transport.call_count, 2)

    def test_configuration_change_rejected_on_reuse(self):
        cfg = config(); journal = self.journal(cfg); journal.close()
        changed = copy.deepcopy(cfg); changed["model_revision"] = "another-revision"
        with self.assertRaises(ValueError):
            RunJournal(self.output, changed, readonly=True)

    def test_local_compute_has_no_fake_api_price(self):
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        infer_one(item, cfg, journal, transport=lambda *_: received())
        events = read_jsonl(journal.ledger)
        self.assertEqual(events[0]["compute_request_reservation"], 1)
        self.assertNotIn("pricing", events[0])
        self.assertEqual(journal.summary()["ledger_accounted_usd"], 0)
        self.assertEqual(read_jsonl(self.output)[0]["accounting_basis"], "local_compute_no_api_charge")

    def test_local_mock_disables_native_thinking_and_sends_fixed_schema(self):
        session = MagicMock()
        session.__enter__.return_value = session
        responses = []
        for value in ({"prompt": "rendered input"}, {"tokens": [1, 2, 3]},
                      {"model": "fixture", "usage": {"prompt_tokens": 3},
                       "choices": [{"message": {"content": '{"answer":"5"}'}, "finish_reason": "stop"}]}):
            response = MagicMock(); response.json.return_value = value; responses.append(response)
        session.post.side_effect = responses
        with patch("requests.Session", return_value=session):
            output = local_call(prepare(row(), config()), config())
        self.assertEqual(output["raw_output_text"], '{"answer":"5"}')
        self.assertEqual(session.post.call_count, 3)
        payload = session.post.call_args_list[-1].kwargs["json"]
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["json_schema"], prepare(row(), config())["request"]["schema"])
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["max_tokens"], 1024)
        self.assertFalse(session.trust_env)
        self.assertNotIn("SECRET", json.dumps(payload))

    def test_gemini_token_count_uses_contents_without_unsupported_system_config(self):
        client = MagicMock()
        client.__enter__.return_value = client
        client.models.count_tokens.return_value.total_tokens = 25
        response = MagicMock()
        response.model_dump.return_value = {"candidates": [], "model_version": "fixture"}
        response.candidates = []
        response.usage_metadata = None
        response.model_version = "fixture"
        client.models.generate_content.return_value = response
        cfg = config("gemini"); item = prepare(row(), cfg)
        with patch("google.genai.Client", return_value=client):
            output = gemini_call(item, cfg, "test-only-key")
        count_call = client.models.count_tokens.call_args.kwargs
        self.assertEqual(set(count_call), {"model", "contents"})
        self.assertIn(item["request"]["system_prompt"], count_call["contents"])
        self.assertIn(item["request"]["input_text"], count_call["contents"])
        self.assertNotIn("SECRET", json.dumps(count_call))
        generation = client.models.generate_content.call_args.kwargs
        self.assertEqual(generation["config"].system_instruction, item["request"]["system_prompt"])
        self.assertEqual(generation["contents"], item["request"]["input_text"])
        self.assertEqual(output["context_preflight_prompt_tokens"], 25)

    def test_context_limit_is_terminal_before_generation(self):
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        transport = MagicMock(side_effect=ContextLimitError())
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "context_limit")
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(read_jsonl(self.output)[0]["format_status"], "not_received")

    def test_local_truncated_parser_output_is_completed_without_retry(self):
        raw = {"error": {"code": 500, "type": "server_error",
                         "message": 'Failed to parse input at pos 0: {"answer":{"stock":[{"item":"x"'}}
        recovered = recover_local_output_parse_error(500, raw)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["raw_output_text"], '{"answer":{"stock":[{"item":"x"')
        self.assertIsNone(recovered["finish_reason"])
        self.assertIsNone(recovered["usage"])
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        transport = MagicMock(return_value=recovered)
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "success")
        self.assertEqual(transport.call_count, 1)
        result = read_jsonl(self.output)[0]
        self.assertEqual(result["format_status"], "server_parser_invalid")
        self.assertEqual(result["raw_response"], raw)
        self.assertIsNone(result["answer"])
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "cached")
        self.assertEqual(transport.call_count, 1)

    def test_generic_http500_is_not_recovered_as_model_output(self):
        import requests
        for status, raw in (
            (500, {"error": {"type": "server_error", "message": "CUDA allocation failed"}}),
            (500, {"error": {"type": "invalid_request", "message": "Failed to parse input at pos 0: x"}}),
            (400, {"error": {"type": "server_error", "message": "Failed to parse input at pos 0: x"}}),
            (500, {"error": {"type": "server_error", "message": "Failed to parse messages: x"}}),
        ):
            self.assertIsNone(recover_local_output_parse_error(status, raw))
        response = requests.Response(); response.status_code = 500
        error = requests.HTTPError(response=response)
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        transport = MagicMock(side_effect=error)
        self.assertEqual(infer_one(item, cfg, journal, transport=transport), "api_error")
        self.assertEqual(transport.call_count, 2)
        result = read_jsonl(self.output)[0]
        self.assertEqual(result["format_status"], "not_received")
        self.assertIsNone(result["answer"])

    def test_local_http500_generation_payload_recovered_before_raise(self):
        session = MagicMock(); session.__enter__.return_value = session
        responses = []
        raw = {"error": {"code": 500, "type": "server_error",
                         "message": 'Failed to parse input at pos 0: {"answer":'}}
        for status, value in ((200, {"prompt": "rendered"}), (200, {"tokens": [1, 2]}), (500, raw)):
            response = MagicMock(); response.status_code = status; response.json.return_value = value
            responses.append(response)
        responses[-1].raise_for_status.side_effect = AssertionError("Generated outcome incorrectly raised")
        session.post.side_effect = responses
        with patch("requests.Session", return_value=session):
            result = local_call(prepare(row(), config()), config())
        self.assertTrue(result["provider_output_parse_error"])
        self.assertEqual(result["raw_response"], raw)
        responses[-1].raise_for_status.assert_not_called()

    def test_changed_request_same_id_cannot_append_a_second_outcome(self):
        cfg = config(); journal = self.journal(cfg); item = prepare(row(), cfg)
        infer_one(item, cfg, journal, transport=lambda *_: received())
        changed = row(); changed["input_text"] = "different"; changed["input_hash"] = digest("different")
        with self.assertRaises(ValueError):
            journal.reserve(prepare(changed, cfg))

    def test_crash_after_attempt_before_output_recovers_without_requery(self):
        cfg = config(); item = prepare(row(), cfg); journal = self.journal(cfg)
        infer_one(item, cfg, journal, transport=lambda *_: received())
        self.output.unlink()
        journal.close()
        resumed = self.journal(cfg)
        self.assertEqual(len(read_jsonl(self.output)), 1)
        self.assertIn(item["cache_key"], resumed.completed)

    def test_strict_output_schema_types_and_duplicate_keys(self):
        schema = schema_for("json", "structured_records", "direct")
        text = '{"answer":{"selected_ids":[],"total_units":0,"total_value":0}}'
        self.assertEqual(parse_output(text, schema)[2], "valid")
        self.assertEqual(parse_output(text.replace('"total_units":0', '"total_units":false'), schema)[2], "schema_invalid")
        self.assertEqual(parse_output('{"answer":"a","answer":"b"}', schema)[2], "invalid_json")

    def test_dry_run_no_network_credentials_or_output_files(self):
        directory = Path(self.directory.name)
        cfg = config(); cfg.pop("runner_version")
        (directory / "config.json").write_text(json.dumps(cfg))
        (directory / "input.jsonl").write_text(json.dumps(row()) + "\n")
        args = ["runner", "--config", str(directory / "config.json"), "--input", str(directory / "input.jsonl"),
                "--output", str(self.output), "--dry-run"]
        with patch("sys.argv", args), patch("requests.Session", side_effect=AssertionError("network used")):
            self.assertEqual(main(), 0)
        self.assertFalse(self.output.exists())
        self.assertFalse(Path(str(self.output) + ".ledger.jsonl").exists())
        self.assertFalse(Path(str(self.output) + ".run.json").exists())


if __name__ == "__main__":
    unittest.main()
