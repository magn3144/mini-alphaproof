import argparse
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from alphaproof.core.paths import (
    CLEANED_NUMINA_MATH_LEAN_PATH,
    NUMINA_MATH_LEAN_PATH,
)


DATASET_URL = (
    'https://huggingface.co/datasets/AI-MO/NuminaMath-LEAN/'
    'resolve/main/data/train-00000-of-00001.parquet'
)
STRIP_PREFIXES = ('import ', 'set_option ', '#check ')


def download_dataset(output_path: Path = NUMINA_MATH_LEAN_PATH) -> None:
    """Download the NuminaMath-LEAN parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    urllib.request.urlretrieve(DATASET_URL, temporary_path)
    temporary_path.replace(output_path)


def process_statement(statement: str) -> str | None:
    """Convert one NuminaMath-LEAN formal statement into a proof problem."""
    lines = [
        line
        for line in statement.strip().splitlines()
        if not line.lstrip().startswith(STRIP_PREFIXES)
    ]
    statement = '\n'.join(lines).strip()

    if statement.endswith(':= by'):
        statement += ' sorry'
    elif statement.endswith(':='):
        statement += ' by sorry'
    else:
        return None

    if statement.count('sorry') != 1:
        return None
    return statement


def clean_dataset(
    input_path: Path = NUMINA_MATH_LEAN_PATH,
    output_path: Path = CLEANED_NUMINA_MATH_LEAN_PATH,
) -> dict[str, int]:
    """Write cleaned NuminaMath-LEAN theorem problems as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    counts = {'input': 0, 'duplicates': 0, 'rejected': 0, 'output': 0}

    parquet = pq.ParquetFile(input_path)
    with output_path.open('w', encoding='utf-8') as output_file:
        for batch in parquet.iter_batches(columns=['uuid', 'formal_statement']):
            columns = batch.to_pydict()
            for problem_id, statement in zip(
                columns['uuid'], columns['formal_statement'], strict=True
            ):
                counts['input'] += 1
                if problem_id in seen_ids:
                    counts['duplicates'] += 1
                    continue
                seen_ids.add(problem_id)

                theorem = process_statement(statement)
                if theorem is None:
                    counts['rejected'] += 1
                    continue

                output_file.write(
                    json.dumps({'id': problem_id, 'theorem': theorem}) + '\n'
                )
                counts['output'] += 1

    return counts


def main() -> None:
    """Download or clean NuminaMath-LEAN."""
    parser = argparse.ArgumentParser(description='Prepare NuminaMath-LEAN.')
    parser.add_argument('action', choices=('download', 'clean'))
    args = parser.parse_args()

    if args.action == 'download':
        download_dataset()
        print(f'Downloaded NuminaMath-LEAN to {NUMINA_MATH_LEAN_PATH}')
    else:
        counts = clean_dataset()
        print(
            f"Wrote {counts['output']} theorems to {CLEANED_NUMINA_MATH_LEAN_PATH} "
            f"from {counts['input']} rows "
            f"({counts['duplicates']} duplicates, {counts['rejected']} rejected)."
        )


if __name__ == '__main__':
    main()
