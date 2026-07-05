import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'data'
DATASET_DIR = DATA_DIR / 'dataset'
NUMINA_MATH_1_5_PATH = DATASET_DIR / 'numina_math_1_5.jsonl'
FILTERED_NUMINA_MATH_1_5_PATH = DATASET_DIR / 'numina_math_1_5_filtered.jsonl'

GOOD_PROBLEM_TYPES = {
    'Algebra',
    'Number Theory',
    'Inequalities',
}

GOOD_QUESTION_TYPES = {
    'proof',
}

GOOD_SOURCES = {
    'olympiads',
    'olympiads_ref',
    'aops_forum',
    'inequalities',
    'number_theory',
}
MAX_PROBLEM_LENGTH = 1200


def looks_like_multi_part_problem(problem: str) -> bool:
    """Return whether a problem appears to contain multiple subproblems."""
    lowered = problem.lower()
    numbered_markers = [
            '(1)',
            '(2)',
            '1.',
            '2.',
            '(a)',
            '(b)',
            'a)',
            'b)',
            'part a',
            'part b',
    ]
    markers_found = sum(marker in lowered for marker in numbered_markers)
    return markers_found >= 2


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

    if looks_like_multi_part_problem(problem):
        return False

    if len(problem) > MAX_PROBLEM_LENGTH:
        return False

    return True


def filtered_record(record: dict) -> dict:
    """Build the filtered dataset record used as autoformalization input."""
    return {
            'problem': record['problem'].strip(),
            'source': record['source'],
            'problem_type': record['problem_type'],
            'question_type': record['question_type'],
            'synthetic': record['synthetic'],
    }


def filter_numina_math_1_5(
        input_path: Path = NUMINA_MATH_1_5_PATH,
        output_path: Path = FILTERED_NUMINA_MATH_1_5_PATH,
) -> int:
    """Filter NuminaMath 1.5 and write selected problems to a new JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with input_path.open(encoding='utf-8') as input_file:
        with output_path.open('w', encoding='utf-8') as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue

                record = json.loads(line)
                if keep_record(record):
                    output_file.write(
                            json.dumps(filtered_record(record), ensure_ascii=False)
                            + '\n'
                    )
                    rows_written += 1

    return rows_written


if __name__ == '__main__':
    rows = filter_numina_math_1_5()
    print(f'Wrote {rows} filtered problems to {FILTERED_NUMINA_MATH_1_5_PATH}')
