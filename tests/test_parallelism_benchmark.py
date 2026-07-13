import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.parallelism_benchmark import (
        BATCH_SIZES,
        MODES,
        PROMPT,
        TARGET_TEXT,
        run_sweep,
        worker_command,
)


class ParallelismBenchmarkTests(unittest.TestCase):
    def test_prompt_has_fixed_character_length(self) -> None:
        self.assertEqual(len(PROMPT), 2000)
        self.assertEqual(len(TARGET_TEXT), 1000)
        self.assertIn(TARGET_TEXT, PROMPT)
        self.assertIn('Output exactly the text', PROMPT)
        self.assertNotIn('token', PROMPT.lower())

    def test_worker_commands_use_module_execution(self) -> None:
        result_path = Path('/tmp/result.json')

        single_process = worker_command('none', 8, result_path)
        distributed = worker_command('tensor', 8, result_path)

        self.assertIn('-m', single_process)
        self.assertIn('scripts.parallelism_benchmark', single_process)
        self.assertIn('--module', distributed)
        self.assertIn('scripts.parallelism_benchmark', distributed)

    def test_sweep_stops_one_mode_after_oom(self) -> None:
        calls = []

        def run_configuration(mode, batch_size, _output_dir):
            calls.append((mode, batch_size))
            status = 'oom' if mode == 'none' and batch_size == 16 else 'success'
            return {
                    'parallelism': mode,
                    'global_batch_size': batch_size,
                    'generation_seconds': 1.0,
                    'generation_characters_per_second': 1.0,
                    'generation_tokens_per_second': 1.0,
                    'end_to_end_seconds': 1.0,
                    'end_to_end_characters_per_second': 1.0,
                    'end_to_end_tokens_per_second': 1.0,
                    'status': status,
                    'return_code': 0,
                    'log_path': 'log',
            }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                    'scripts.parallelism_benchmark.run_configuration',
                    side_effect=run_configuration,
            ):
                with patch('builtins.print'):
                    run_sweep(Path(directory))

        expected = [('none', 8), ('none', 16)]
        expected.extend(
                (mode, batch_size)
                for mode in MODES[1:]
                for batch_size in BATCH_SIZES
        )
        self.assertEqual(calls, expected)


if __name__ == '__main__':
    unittest.main()
