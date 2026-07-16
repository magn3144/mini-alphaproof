import json
import random
from pathlib import Path

from alphaproof.core.paths import DATASET_DIR


NUMINA_MATH_1_5_PATH = DATASET_DIR / 'numina_math_1_5.jsonl'
FILTERED_NUMINA_MATH_1_5_PATH = DATASET_DIR / 'numina_math_1_5_filtered.jsonl'
DATASET_ID_PREFIX = 'numina_math_1_5'

GOOD_PROBLEM_TYPES = {
    'Algebra',
    'Calculus',
    'Combinatorics',
    'Inequalities',
    'Logic and Puzzles',
    'Number Theory',
}

GOOD_QUESTION_TYPES = {
    'proof',
    'math-word-problem',
    'MCQ',
}

GOOD_SOURCES = {
    'amc_aime',
    'olympiads',
    'olympiads_ref',
    'aops_forum',
    'inequalities',
    'number_theory',
    'cn_k12',
    'cn_contest',
    'orca_math',
    'synthetic_math',
}
MAX_PROBLEM_LENGTH = 1200


def keep_record(record: dict) -> bool:
    """Return whether a NuminaMath record should be kept for formalization."""
    problem = record.get('problem', '').strip()

    if not problem:
        return False

    if record.get('problem_is_valid') != 'Yes':
        return False

    if record.get('question_type') not in GOOD_QUESTION_TYPES:
        return False

    if record.get('problem_type') not in GOOD_PROBLEM_TYPES:
        return False

    if record.get('source') not in GOOD_SOURCES:
        return False

    if len(problem) > MAX_PROBLEM_LENGTH:
        return False

    return True


def filtered_record(record: dict, problem_id: str) -> dict:
    """Build the filtered dataset record used as autoformalization input."""
    return {
            'id': problem_id,
            'problem': record['problem'].strip(),
            'source': record['source'],
            'problem_type': record['problem_type'],
            'question_type': record['question_type'],
            'answer': record.get('answer'),
            'synthetic': record['synthetic'],
    }


def filter_numina_math_1_5(
        input_path: Path = NUMINA_MATH_1_5_PATH,
        output_path: Path = FILTERED_NUMINA_MATH_1_5_PATH,
) -> int:
    """Filter NuminaMath 1.5 and write selected problems to a new JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    filtered_rows = []
    with input_path.open(encoding='utf-8') as input_file:
        for row_ix, line in enumerate(input_file):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            if keep_record(record):
                problem_id = f'{DATASET_ID_PREFIX}_{row_ix:07d}'
                filtered_rows.append(filtered_record(record, problem_id))

    random.shuffle(filtered_rows)

    with output_path.open('w', encoding='utf-8') as output_file:
        for row in filtered_rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + '\n')

    return len(filtered_rows)


if __name__ == '__main__':
    rows = filter_numina_math_1_5()
    print(f'Wrote {rows} filtered problems to {FILTERED_NUMINA_MATH_1_5_PATH}')
