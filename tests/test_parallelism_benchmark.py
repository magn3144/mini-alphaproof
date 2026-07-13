import argparse
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.parallelism_benchmark import load_cohort, run_benchmark


class BenchmarkTests(unittest.TestCase):
    def write_source(self, path: Path, count: int = 20) -> None:
        with path.open('w', encoding='utf-8') as source_file:
            for index in range(count):
                source_file.write(
                        json.dumps(
                                {
                                        'id': f'source-{index}',
                                        'problem': f'problem {index}',
                                        'question_type': 'proof',
                                }
                        )
                        + '\n'
                )

    def test_cohort_rows_are_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / 'source.jsonl'
            self.write_source(source_path)

            cohort = load_cohort(source_path, 'proof', 16, seed=3)

        original_ids = [row['benchmark']['original_id'] for row in cohort]
        self.assertEqual(len(original_ids), len(set(original_ids)))
        self.assertEqual(len({row['id'] for row in cohort}), 16)

    def test_sweep_stops_only_the_mode_that_ooms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / 'source.jsonl'
            output_dir = root / 'benchmark'
            self.write_source(source_path)
            args = argparse.Namespace(
                    input_path=source_path,
                    output_dir=output_dir,
                    modes=['balanced', 'data'],
                    batch_sizes=[8, 16],
                    cohort_size=16,
                    question_type='proof',
                    seed=0,
                    max_model_batch_size=None,
                    wandb=False,
                    wandb_project='test',
                    group='test',
            )
            calls = []

            def run(command, **_kwargs):
                mode = command[command.index('--parallelism') + 1]
                batch_size = int(command[command.index('--batch-size') + 1])
                metrics_path = Path(command[command.index('--metrics-path') + 1])
                calls.append((mode, batch_size))
                metrics_path.write_text(
                        json.dumps(
                                {
                                        'out_of_memory': mode == 'balanced',
                                        'model': {},
                                        'peak_memory': {},
                                }
                        ),
                        encoding='utf-8',
                )
                return SimpleNamespace(returncode=0)

            with patch('scripts.parallelism_benchmark.subprocess.run', side_effect=run):
                run_benchmark(args)

        self.assertEqual(calls, [('balanced', 8), ('data', 8), ('data', 16)])


if __name__ == '__main__':
    unittest.main()
