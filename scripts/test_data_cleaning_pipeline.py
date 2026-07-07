import argparse
import io
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from alphaproof.formalize import data_cleaning as dc
from alphaproof.formalize.filter_problems import (
        NUMINA_MATH_1_5_PATH,
        filtered_record,
        keep_record,
)
from alphaproof.formalize.qwen3 import Qwen3_8B


TEST_RECORDS = {
        'numina_math_1_5_0261840': 'missing-answer word problem',
        'numina_math_1_5_0187755': 'LaTeX JSON escaping',
        'numina_math_1_5_0426280': 'MCQ option removal',
}


def load_records(record_ids: set[str]) -> dict[str, dict]:
    """Load selected filtered records from the raw NuminaMath JSONL."""
    records = {}
    with NUMINA_MATH_1_5_PATH.open(encoding='utf-8') as input_file:
        for row_ix, line in enumerate(input_file):
            record_id = f'numina_math_1_5_{row_ix:07d}'
            if record_id not in record_ids:
                continue

            record = json.loads(line)
            if not keep_record(record):
                raise ValueError(f'{record_id} does not pass keep_record().')

            records[record_id] = filtered_record(record, record_id)
            if len(records) == len(record_ids):
                return records

    missing_ids = sorted(record_ids - records.keys())
    raise ValueError(f'Could not find records: {missing_ids}')


def print_record_header(record_id: str, record: dict) -> None:
    """Print compact identifying information for a test record."""
    print()
    print(f'=== {record_id}: {TEST_RECORDS[record_id]} ===')
    print(
            f'{record["question_type"]} | '
            f'{record["problem_type"]} | '
            f'{record["source"]} | '
            f'answer={record.get("answer")!r}'
    )


def write_proof_rows(record: dict, problem: str, answer: object, model: Qwen3_8B) -> list[dict]:
    """Run add_proof_problem_rows and return the emitted JSONL rows."""
    output = io.StringIO()
    rows_written = dc.add_proof_problem_rows(
            record,
            problem,
            answer,
            model,
            output,
            'test',
    )
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    if rows_written != len(rows):
        raise AssertionError(f'Expected {rows_written} rows, got {len(rows)}.')
    return rows


def test_missing_answer_record(record: dict, model: Qwen3_8B) -> None:
    """Check that missing-answer problems still produce answer-free proof rows."""
    rows = write_proof_rows(record, record['problem'], record.get('answer'), model)
    if len(rows) != 1:
        raise AssertionError(f'Expected one answer-free row, got {len(rows)}.')
    if rows[0]['answer'] is not None:
        raise AssertionError(f'Expected answer=None, got {rows[0]["answer"]!r}.')
    if not rows[0]['id'].endswith('_without_answer'):
        raise AssertionError(f'Expected without-answer row id, got {rows[0]["id"]}.')

    print('PASS missing answer: wrote one answer-free proof row')
    print(f'  {rows[0]["problem"][:240]}')


def test_latex_json_record(record: dict, model: Qwen3_8B) -> None:
    """Check that model JSON with LaTeX backslashes can be parsed."""
    split_problems = dc.split_into_several_problems(record, model)
    if not split_problems:
        raise AssertionError('Expected at least one split problem.')

    rows = write_proof_rows(
            record,
            split_problems[0]['problem'],
            split_problems[0].get('answer'),
            model,
    )
    if not rows:
        raise AssertionError('Expected at least one proof row.')

    print('PASS LaTeX JSON: split/proof JSON parsed successfully')
    print(f'  split problem: {split_problems[0]["problem"][:240]}')
    print(f'  first proof row: {rows[0]["problem"][:240]}')


def test_remove_options_record(record: dict, model: Qwen3_8B) -> None:
    """Check that MCQ rewriting hides the answer from the model prompt."""
    cleaned = dc.remove_options(record, model)
    expected_answer = dc.selected_option_answer(record)
    if cleaned['answer'] != expected_answer:
        raise AssertionError(
                f'Expected answer {expected_answer!r}, got {cleaned["answer"]!r}.'
        )
    if cleaned['answer'] == record.get('answer'):
        raise AssertionError('Expected answer label to be replaced by option value.')

    print('PASS remove options: rewrote problem and recovered answer locally')
    print(f'  problem: {cleaned["problem"][:240]}')
    print(f'  answer: {cleaned["answer"]!r}')


def run_tests(model: Qwen3_8B) -> None:
    """Run the small real-model data-cleaning smoke test."""
    records = load_records(set(TEST_RECORDS))

    missing_answer_id = 'numina_math_1_5_0261840'
    print_record_header(missing_answer_id, records[missing_answer_id])
    test_missing_answer_record(records[missing_answer_id], model)

    latex_json_id = 'numina_math_1_5_0187755'
    print_record_header(latex_json_id, records[latex_json_id])
    test_latex_json_record(records[latex_json_id], model)

    remove_options_id = 'numina_math_1_5_0426280'
    print_record_header(remove_options_id, records[remove_options_id])
    test_remove_options_record(records[remove_options_id], model)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
            description='Run a small real-model test of the data-cleaning pipeline.',
    )
    parser.add_argument(
            '--quantization',
            default='8bit',
            choices=('8bit', '4bit', 'none'),
            help='Model quantization to use. Default: 8bit.',
    )
    return parser.parse_args()


def check_cuda_device() -> None:
    """Fail early when the active GPU cannot run this Torch/bnb build."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Run this script inside a GPU shell.')

    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 5):
        raise RuntimeError(
                f'Active GPU is {device_name} with compute capability '
                f"{capability[0]}.{capability[1]}. This environment's "
                'Torch/bitsandbytes build does not support V100/CC 7.0. '
                'Run this test on an A100 node, for example with a100sh.'
        )


def main() -> None:
    """Load Qwen3-8B and run the data-cleaning smoke tests."""
    args = parse_args()
    check_cuda_device()
    quantization = None if args.quantization == 'none' else args.quantization
    model = Qwen3_8B(quantization=quantization)
    model.load()
    print(f'Loaded {model.model_name} from {model.model_dir} on {model.device}.')

    run_tests(model)
    print()
    print('All data-cleaning smoke tests passed.')


if __name__ == '__main__':
    main()
