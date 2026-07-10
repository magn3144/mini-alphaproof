import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from alphaproof.formalize.data_cleaning.filter_problems import FILTERED_NUMINA_MATH_1_5_PATH
from alphaproof.formalize.data_cleaning.model import load_cleaning_model
from alphaproof.formalize.data_cleaning.paths import CLEANED_NUMINA_MATH_1_5_PATH
from alphaproof.formalize.data_cleaning.pipeline import Timers, clean_records
from alphaproof.formalize.data_cleaning.records import (
        ensure_append_starts_on_new_line,
        raw_error_record,
        raw_row_summary,
        row_summary,
        source_id_is_cleaned,
        write_cleaned_rows,
        write_record,
)


def data_cleaning_summary(
        rows_read: int,
        missing_information_rows: int,
        errored_rows: int,
        output_rows: list[dict],
        row_summaries: list[dict],
        timers: dict[str, float],
) -> str:
    """Return a formatted summary of a data cleaning run."""
    timer_lines = [
            f'  {name}: {seconds:.3f}s'
            for name, seconds in sorted(timers.items())
    ]
    return '\n'.join(
            [
                    'Finished data cleaning:',
                    f'  Rows read: {rows_read}',
                    f'  Missing information rows: {missing_information_rows}',
                    f'  Errors: {errored_rows}',
                    f'  Rows written: {len(output_rows)}',
                    '',
                    'Row function outputs:',
                    json.dumps(row_summaries, ensure_ascii=False, indent=2),
                    '',
                    'Timings:',
                    *timer_lines,
            ]
    )


def clean_rows(
        rows_to_clean: int,
        input_path: Path = FILTERED_NUMINA_MATH_1_5_PATH,
        output_path: Path = CLEANED_NUMINA_MATH_1_5_PATH,
        print_summary: bool = False,
        batch_size: int = 4,
        device: str | None = None,
        torch_dtype: str = 'auto',
        quantization: str | None = None,
        enable_thinking: bool = False,
) -> None:
    """Run the data cleaning pipeline over the filtered dataset."""
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1.')

    qwen = load_cleaning_model(
            device,
            torch_dtype,
            quantization,
            enable_thinking,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_append_starts_on_new_line(output_path)

    rows_read = 0
    missing_information_rows = 0
    errored_rows = 0
    output_rows = []
    row_summaries = []
    timers = Timers()
    cleaned_rows = 0
    pending_records = []

    def timing_delta(before: dict[str, float]) -> dict[str, float]:
        return {
                name: seconds - before.get(name, 0.0)
                for name, seconds in timers.total.items()
                if seconds - before.get(name, 0.0) > 0
        }

    def process_pending_records(output_file: Any) -> None:
        nonlocal cleaned_rows
        nonlocal missing_information_rows
        nonlocal errored_rows

        if not pending_records:
            return

        before = dict(timers.total)
        results = clean_records(pending_records, qwen, timers)
        timings = timing_delta(before)

        for record, result in zip(pending_records, results):
            output_rows.extend(result.output_rows)
            row_summaries.append(row_summary(record, result, timings))
            write_cleaned_rows(
                    output_file,
                    result.output_rows,
                    result,
                    timings,
            )
            if result.missing_information:
                missing_information_rows += 1
            if result.errored:
                errored_rows += 1
            if result.output_rows:
                cleaned_rows += 1

        pending_records.clear()

    with input_path.open(encoding='utf-8') as input_file:
        with output_path.open('a', encoding='utf-8') as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                if cleaned_rows >= rows_to_clean:
                    break

                row_start = perf_counter()
                try:
                    record = json.loads(line)
                    if source_id_is_cleaned(record['id'], output_path):
                        continue
                except Exception as error:
                    row_timings = {'row_iteration': perf_counter() - row_start}
                    output_row = raw_error_record(line, error)
                    output_rows.append(output_row)
                    row_summaries.append(raw_row_summary(line, error, row_timings))
                    write_record(output_file, output_row)
                    output_file.flush()
                    rows_read += 1
                    errored_rows += 1
                    cleaned_rows += 1
                    timers.add('raw_row_error', row_timings['row_iteration'])
                    continue

                rows_read += 1
                pending_records.append(record)
                if (
                        len(pending_records) >= batch_size
                        or cleaned_rows + len(pending_records) >= rows_to_clean
                ):
                    process_pending_records(output_file)

            process_pending_records(output_file)

    if print_summary:
        print(
                data_cleaning_summary(
                        rows_read,
                        missing_information_rows,
                        errored_rows,
                        output_rows,
                        row_summaries,
                        timers.total,
                )
        )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for data cleaning."""
    parser = argparse.ArgumentParser()
    parser.add_argument('rows_to_clean', type=int)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--output-path', type=Path, default=CLEANED_NUMINA_MATH_1_5_PATH)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--quantization', choices=['4bit', '8bit'])
    parser.add_argument('--enable-thinking', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    clean_rows(
            args.rows_to_clean,
            output_path=args.output_path,
            print_summary=args.summary,
            batch_size=args.batch_size,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            enable_thinking=args.enable_thinking,
    )
