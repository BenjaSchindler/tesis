import json
from pathlib import Path
import tempfile
import unittest
from experiments.production_eval.resume_sft_v3 import read_prefix, validate_steps
from experiments.production_eval.synthetic_sft_v3 import PARAMETERS


class RecoverySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'responses.jsonl'
        self.rows = [dict(example_id=str(i), family_id='family' + str(i), input_hash='input' + str(i),
                          domain='structured_records', role='reference', regime='stationary', variant='base',
                          answer='{"x":1}') for i in range(3)]

    def response(self, index):
        return {**{k: v for k, v in self.rows[index].items() if k != 'answer'},
                'response': '{"x":1}', 'exact_match': True}

    def write(self, rows):
        self.path.write_text(''.join(json.dumps(row) + '\n' for row in rows))

    def test_empty_or_valid_prefix_is_preserved(self):
        self.assertEqual(read_prefix(self.path, self.rows), [])
        self.write([self.response(0), self.response(1)])
        before = self.path.read_bytes()
        self.assertEqual(len(read_prefix(self.path, self.rows)), 2)
        self.assertEqual(self.path.read_bytes(), before)

    def test_reordered_or_duplicated_prefix_rejected(self):
        for indices in ([1, 0], [0, 0]):
            self.write([self.response(i) for i in indices])
            with self.assertRaises(ValueError):
                read_prefix(self.path, self.rows)

    def test_input_or_grade_changed_rejected(self):
        for key, value in [('input_hash', 'wrong'), ('exact_match', False), ('exact_match', 1)]:
            row = self.response(0)
            row[key] = value
            self.write([row])
            with self.assertRaises(ValueError):
                read_prefix(self.path, self.rows)

    def test_truncated_final_jsonl_line_rejected_without_mutation(self):
        self.path.write_text(json.dumps(self.response(0)))
        before = self.path.read_bytes()
        with self.assertRaises(ValueError):
            read_prefix(self.path, self.rows)
        self.assertEqual(self.path.read_bytes(), before)

    def test_longer_than_evaluation_rejected(self):
        self.write([self.response(0)] * 4)
        with self.assertRaises(ValueError):
            read_prefix(self.path, self.rows)

    def steps(self):
        return [dict(step=i, cumulative_example_presentations=i * 8,
                     learning_rate=PARAMETERS['learning_rate'] * min(1., i / 2),
                     cumulative_input_tokens=768, cumulative_target_tokens=512) for i in range(1, 33)]

    def test_complete_training_requires_matching_token_totals_and_schedule(self):
        examples = [dict(prompt_ids=[1, 2, 3], answer_ids=[4, 5])] * 64
        self.assertEqual(validate_steps(self.steps(), examples)['step'], 32)
        for mutation in ('partial', 'lr', 'tokens', 'presentations'):
            rows = self.steps()
            if mutation == 'partial': rows.pop()
            if mutation == 'lr': rows[0]['learning_rate'] *= 2
            if mutation == 'tokens': rows[-1]['cumulative_target_tokens'] += 1
            if mutation == 'presentations': rows[4]['cumulative_example_presentations'] += 1
            with self.assertRaises(ValueError):
                validate_steps(rows, examples)


if __name__ == '__main__':
    unittest.main()
