import unittest
from unittest.mock import patch

from alphaproof.formalize.data_cleaning.metrics import aggregate_model_metrics
from alphaproof.formalize.data_cleaning.model_calls import parse_each
from alphaproof.formalize.data_cleaning.parallel import ParallelContext
from alphaproof.formalize.data_cleaning.pipeline import CleanResult, Timers
from alphaproof.formalize.qwen3 import ModelCall, Qwen3, TensorParallelError


class QwenTests(unittest.TestCase):
    def test_model_loading_arguments_are_mutually_exclusive(self) -> None:
        balanced = Qwen3(model_dir='/tmp/model', parallelism='balanced')
        tensor = Qwen3(model_dir='/tmp/model', parallelism='tensor')

        self.assertEqual(balanced._model_load_kwargs()['device_map'], 'balanced')
        self.assertNotIn('tp_plan', balanced._model_load_kwargs())
        self.assertEqual(tensor._model_load_kwargs()['tp_plan'], 'auto')
        self.assertNotIn('device_map', tensor._model_load_kwargs())

    def test_max_batch_size_chunks_and_preserves_order(self) -> None:
        model = Qwen3(model_dir='/tmp/model', max_batch_size=2)
        model.tokenizer = object()
        model.model = object()
        prompts = ['a', 'b', 'c', 'd', 'e']

        with patch.object(
                model,
                '_sample_batch',
                side_effect=lambda batch, *args: [f'out-{prompt}' for prompt in batch],
        ) as sample_batch:
            outputs = model.sample(prompts)

        self.assertEqual(outputs, [f'out-{prompt}' for prompt in prompts])
        self.assertEqual(
                [call.args[0] for call in sample_batch.call_args_list],
                [['a', 'b'], ['c', 'd'], ['e']],
        )

    def test_tensor_completion_disagreement_is_fatal(self) -> None:
        model = Qwen3(model_dir='/tmp/model', parallelism='tensor')

        def gather(gathered, _local):
            gathered[:] = [['same'], ['different']]

        with patch('alphaproof.formalize.qwen3.dist.is_initialized', return_value=True):
            with patch('alphaproof.formalize.qwen3.dist.get_world_size', return_value=2):
                with patch(
                        'alphaproof.formalize.qwen3.dist.all_gather_object',
                        side_effect=gather,
                ):
                    with self.assertRaises(TensorParallelError):
                        model._verify_tensor_completions(['same'])

    def test_model_metrics_include_actual_chunk_sizes(self) -> None:
        model = Qwen3(model_dir='/tmp/model')
        model.model_calls = [
                ModelCall('one', 4, 10, 5, 2.0),
                ModelCall('one', 2, 4, 3, 1.0),
        ]

        metrics = model.metrics()

        self.assertEqual(metrics['calls'], 2)
        self.assertEqual(metrics['max_model_call_batch_size'], 4)
        self.assertEqual(metrics['generated_tokens_per_second'], 8 / 3)


class ParsingAndMetricsTests(unittest.TestCase):
    def test_outputs_are_parsed_independently(self) -> None:
        outputs = parse_each(['1', 'bad', '3'], int)

        self.assertEqual(outputs[0], 1)
        self.assertIsInstance(outputs[1], ValueError)
        self.assertEqual(outputs[2], 3)

    def test_data_parallel_tokens_use_slower_rank_time(self) -> None:
        rank_template = {
                'calls': 1,
                'max_model_call_batch_size': 4,
                'stages': {},
                'model_calls': [],
        }
        rank_metrics = [
                {
                        **rank_template,
                        'prompt_tokens': 10,
                        'generated_tokens': 20,
                        'generation_seconds': 2.0,
                },
                {
                        **rank_template,
                        'prompt_tokens': 15,
                        'generated_tokens': 30,
                        'generation_seconds': 3.0,
                },
        ]

        metrics = aggregate_model_metrics('data', rank_metrics)

        self.assertEqual(metrics['generated_tokens_per_second'], 50 / 3)
        self.assertEqual(metrics['total_tokens_per_second'], 75 / 3)


class DataParallelMergeTests(unittest.TestCase):
    def test_strided_results_are_merged_in_original_order(self) -> None:
        records = [{'id': str(index)} for index in range(4)]
        context = ParallelContext(
                mode='data',
                rank=0,
                world_size=2,
                initialized=True,
        )

        def clean(local_records, _model, _timers):
            return [CleanResult(output_rows=[record]) for record in local_records]

        def gather(local_results, gathered, dst):
            self.assertEqual(dst, 0)
            gathered[0] = local_results
            gathered[1] = [
                    (1, CleanResult(output_rows=[records[1]])),
                    (3, CleanResult(output_rows=[records[3]])),
            ]

        with patch(
                'alphaproof.formalize.data_cleaning.parallel.clean_records',
                side_effect=clean,
        ):
            with patch(
                    'alphaproof.formalize.data_cleaning.parallel.dist.gather_object',
                    side_effect=gather,
            ):
                results = context.clean_batch(records, object(), Timers())

        self.assertIsNotNone(results)
        self.assertEqual(
                [result.output_rows[0]['id'] for result in results],
                ['0', '1', '2', '3'],
        )


if __name__ == '__main__':
    unittest.main()
