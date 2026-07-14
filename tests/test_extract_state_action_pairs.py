import tempfile
import unittest
from pathlib import Path

from scripts.extract_state_action_pairs import (
    SourceRecord,
    build_error_for_record,
    module_path,
    pairs_from_trace,
)


class PairExtractionTests(unittest.TestCase):
    def test_keeps_outer_tactic_for_duplicate_nested_transition(self) -> None:
        source = 'rw [h] at hypothesis\n'
        state_before = 'h : x = y\nhypothesis : P x\n⊢ P y'
        state_after = 'h : x = y\nhypothesis : P y\n⊢ P y'
        trace = {
            'tactics': [
                {
                    'pos': {'byteIdx': 0},
                    'endPos': {'byteIdx': 20},
                    'stateBefore': state_before,
                    'stateAfter': state_after,
                },
                {
                    'pos': {'byteIdx': 4},
                    'endPos': {'byteIdx': 5},
                    'stateBefore': state_before,
                    'stateAfter': state_after,
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = module_path(workspace, 1)
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding='utf-8')

            results = pairs_from_trace(
                {1: trace},
                [SourceRecord(1, source)],
                workspace,
            )

        self.assertEqual(
            results[1].pairs,
            ({'state': state_before, 'action': 'rw [h] at hypothesis'},),
        )

    def test_keeps_genuine_nested_proof_tactic(self) -> None:
        source = 'have h : P := by simp\n'
        trace = {
            'tactics': [
                {
                    'pos': {'byteIdx': 0},
                    'endPos': {'byteIdx': 21},
                    'stateBefore': '⊢ Q',
                    'stateAfter': 'h : P\n⊢ Q',
                },
                {
                    'pos': {'byteIdx': 17},
                    'endPos': {'byteIdx': 21},
                    'stateBefore': '⊢ P',
                    'stateAfter': 'no goals',
                },
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            path = module_path(workspace, 1)
            path.parent.mkdir(parents=True)
            path.write_text(source, encoding='utf-8')

            results = pairs_from_trace(
                {1: trace},
                [SourceRecord(1, source)],
                workspace,
            )

        self.assertEqual(
            results[1].pairs,
            ({'state': '⊢ P', 'action': 'simp'},),
        )


class BuildErrorTests(unittest.TestCase):
    def test_returns_only_diagnostics_for_requested_record(self) -> None:
        output = '\n'.join([
            'error: Proof_000000001.lean:1:0: first error',
            'error: Proof_000000002.lean:2:0: second error',
            'error: Proof_000000002.lean:3:0: another second error',
        ])

        self.assertEqual(
            build_error_for_record(output, 2),
            '\n'.join([
                'error: Proof_000000002.lean:2:0: second error',
                'error: Proof_000000002.lean:3:0: another second error',
            ]),
        )


if __name__ == '__main__':
    unittest.main()
