import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from alphaproof.formalize.data_cleaning.filter_problems import FILTERED_NUMINA_MATH_1_5_PATH
from alphaproof.formalize.data_cleaning.model import (
        CLEANING_MODEL_ALIASES,
        DEFAULT_CLEANING_MODEL,
        load_cleaning_model,
)
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


@dataclass
class CleaningRun:
    """Mutable counters and summaries for one data-cleaning run."""

    rows_read: int = 0
    missing_information_rows: int = 0
    errored_rows: int = 0
    cleaned_rows: int = 0
    output_rows: list[dict] = field(default_factory=list)
    row_summaries: list[dict] = field(default_factory=list)
    timers: Timers = field(default_factory=Timers)

    def timing_delta(self, before: dict[str, float]) -> dict[str, float]:
        """Return timer changes since before."""
        return {
                name: seconds - before.get(name, 0.0)
                for name, seconds in self.timers.total.items()
                if seconds - before.get(name, 0.0) > 0
        }

    def write_raw_error(
            self,
            line: str,
            error: Exception,
            row_timings: dict[str, float],
            output_file: Any,
    ) -> None:
        """Write and count one dataset line that could not be loaded."""
        output_row = raw_error_record(line, error)
        self.output_rows.append(output_row)
        self.row_summaries.append(raw_row_summary(line, error, row_timings))
        write_record(output_file, output_row)
        output_file.flush()
        self.rows_read += 1
        self.errored_rows += 1
        self.cleaned_rows += 1
        self.timers.add('raw_row_error', row_timings['row_iteration'])

    def write_results(
            self,
            records: list[dict],
            results: list[Any],
            output_file: Any,
            timings: dict[str, float],
    ) -> None:
        """Write and count cleaned results."""
        for record, result in zip(records, results):
            self.rows_read += 1
            self.output_rows.extend(result.output_rows)
            self.row_summaries.append(row_summary(record, result, timings))
            write_cleaned_rows(
                    output_file,
                    result.output_rows,
                    result,
                    timings,
            )
            if result.missing_information:
                self.missing_information_rows += 1
            if result.errored:
                self.errored_rows += 1
            if result.output_rows:
                self.cleaned_rows += 1


def take_matching_queued_records(
        queued_records: list[dict],
        question_type: str | None,
        records: list[dict],
        batch_size: int,
        rows_remaining: int,
        output_path: Path,
) -> None:
    """Move queued records with matching question_type into records."""
    queued_ix = 0
    while queued_ix < len(queued_records):
        if len(records) >= batch_size or len(records) >= rows_remaining:
            return

        record = queued_records[queued_ix]
        if source_id_is_cleaned(record['id'], output_path):
            del queued_records[queued_ix]
            continue

        if record.get('question_type') == question_type:
            records.append(record)
            del queued_records[queued_ix]
        else:
            queued_ix += 1


def next_question_type_batch(
        dataset_file: Any,
        queued_records: list[dict],
        output_file: Any,
        output_path: Path,
        batch_size: int,
        rows_remaining: int,
        run: CleaningRun,
) -> list[dict]:
    """Return the next batch whose records all have the same question_type."""
    records = []
    question_type = None
    start_cleaned_rows = run.cleaned_rows

    while (
            queued_records
            and len(records) + run.cleaned_rows - start_cleaned_rows < rows_remaining
    ):
        record = queued_records.pop(0)
        if source_id_is_cleaned(record['id'], output_path):
            continue

        records.append(record)
        question_type = record.get('question_type')
        break

    if records:
        take_matching_queued_records(
                queued_records,
                question_type,
                records,
                batch_size,
                rows_remaining,
                output_path,
        )

    while (
            len(records) < batch_size
            and len(records) + run.cleaned_rows - start_cleaned_rows < rows_remaining
    ):
        line = dataset_file.readline()
        if not line:
            break

        row_start = perf_counter()
        line = line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
            if source_id_is_cleaned(record['id'], output_path):
                continue
        except Exception as error:
            row_timings = {'row_iteration': perf_counter() - row_start}
            run.write_raw_error(
                    line,
                    error,
                    row_timings,
                    output_file,
            )
            if len(records) + run.cleaned_rows - start_cleaned_rows >= rows_remaining:
                break
            continue

        if not records:
            question_type = record.get('question_type')

        if record.get('question_type') == question_type:
            records.append(record)
        else:
            queued_records.append(record)

    return records


def data_cleaning_summary(run: CleaningRun) -> str:
    """Return a formatted summary of a data cleaning run."""
    timer_lines = [
            f'  {name}: {seconds:.3f}s'
            for name, seconds in sorted(run.timers.total.items())
    ]
    return '\n'.join(
            [
                    'Finished data cleaning:',
                    f'  Rows read: {run.rows_read}',
                    f'  Missing information rows: {run.missing_information_rows}',
                    f'  Errors: {run.errored_rows}',
                    f'  Rows written: {len(run.output_rows)}',
                    '',
                    'Row function outputs:',
                    json.dumps(run.row_summaries, ensure_ascii=False, indent=2),
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
        model: str = DEFAULT_CLEANING_MODEL,
        device: str | None = None,
        torch_dtype: str = 'auto',
        quantization: str | None = None,
) -> None:
    """Run the data cleaning pipeline over the filtered dataset."""
    if batch_size < 1:
        raise ValueError('batch_size must be at least 1.')

    qwen = load_cleaning_model(
            model,
            device,
            torch_dtype,
            quantization,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_append_starts_on_new_line(output_path)

    run = CleaningRun()
    queued_records = []

    with input_path.open(encoding='utf-8') as input_file:
        with output_path.open('a', encoding='utf-8') as output_file:
            while run.cleaned_rows < rows_to_clean:
                records = next_question_type_batch(
                        input_file,
                        queued_records,
                        output_file,
                        output_path,
                        batch_size,
                        rows_to_clean - run.cleaned_rows,
                        run,
                )

                if not records:
                    break

                before = dict(run.timers.total)
                results = clean_records(records, qwen, run.timers)
                run.write_results(
                        records,
                        results,
                        output_file,
                        run.timing_delta(before),
                )

    if print_summary:
        print(data_cleaning_summary(run))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for data cleaning."""
    parser = argparse.ArgumentParser()
    parser.add_argument('rows_to_clean', type=int)
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--output-path', type=Path, default=CLEANED_NUMINA_MATH_1_5_PATH)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument(
            '--model',
            choices=sorted(CLEANING_MODEL_ALIASES),
            default=DEFAULT_CLEANING_MODEL,
    )
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--torch-dtype', default='auto')
    parser.add_argument('--quantization', choices=['4bit', '8bit'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    clean_rows(
            args.rows_to_clean,
            output_path=args.output_path,
            print_summary=args.summary,
            batch_size=args.batch_size,
            model=args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
    )
