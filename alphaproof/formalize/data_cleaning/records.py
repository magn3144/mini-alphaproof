import json
from pathlib import Path
from typing import Any


def failed_record(record: dict, reason: str) -> dict:
    """Return a cleaned dataset row for a problem that cannot continue."""
    cleaned_record = dict(record)
    cleaned_record['source_id'] = record['id']
    cleaned_record['FAILED'] = reason
    cleaned_record['theorem'] = None
    return cleaned_record


def formalization_record(record: dict) -> dict:
    """Return a cleaned dataset row ready for later formalization."""
    cleaned_record = dict(record)
    cleaned_record.setdefault('source_id', record['id'])
    cleaned_record['FAILED'] = None
    cleaned_record['theorem'] = None
    return cleaned_record


def derived_record(
        record: dict,
        problem: str,
        answer: Any,
        suffix: str,
) -> dict:
    """Create a derived proof-problem row from an existing record."""
    derived = dict(record)
    derived['source_id'] = record['id']
    derived['id'] = f'{record["id"]}__{suffix}'
    derived['problem'] = problem
    derived['question_type'] = f'derived_from_{record["question_type"]}'
    derived['answer'] = answer
    return derived


def write_record(output_file: Any, record: dict) -> None:
    """Write one JSONL row."""
    output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def ensure_append_starts_on_new_line(output_path: Path) -> None:
    """Ensure appending a JSONL row starts on a fresh line."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return

    with output_path.open('rb+') as output_file:
        output_file.seek(-1, 2)
        if output_file.read(1) != b'\n':
            output_file.write(b'\n')


def cleaned_record_with_metadata(
        record: dict,
        boolean_outputs: dict,
        json_outputs: dict,
        row_timings: dict[str, float],
        error: str | None,
) -> dict:
    """Return a cleaned output row with data-cleaning metadata attached."""
    record_with_metadata = dict(record)
    record_with_metadata['boolean_outputs'] = dict(boolean_outputs)
    record_with_metadata['timings'] = dict(row_timings)
    if error is not None:
        record_with_metadata['error'] = error
    if error is not None and json_outputs:
        record_with_metadata['json_outputs'] = dict(json_outputs)
    return record_with_metadata


def exception_message(error: Exception) -> str:
    """Return a compact exception message for saved row metadata."""
    return f'{type(error).__name__}: {error}'


def raw_error_record(line: str, error: Exception) -> dict:
    """Return a failed row for an input line that could not be read."""
    return {
            'id': None,
            'source_id': None,
            'raw_input': line,
            'FAILED': 'error',
            'theorem': None,
            'error': exception_message(error),
    }


def source_id_is_cleaned(source_id: str, output_path: Path) -> bool:
    """Return whether source_id already exists in the cleaned dataset."""
    if not output_path.exists():
        return False

    with output_path.open(encoding='utf-8') as output_file:
        for line in output_file:
            line = line.strip()
            if not line:
                continue
            try:
                cleaned_record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if cleaned_record.get('source_id') == source_id:
                return True

    return False


def row_summary(record: dict, result: Any, row_timings: dict) -> dict:
    """Return the summary object for one cleaned input record."""
    return {
            'id': record.get('id'),
            'question_type': record.get('question_type'),
            'problem': record.get('problem'),
            'answer': record.get('answer'),
            'boolean_outputs': dict(result.boolean_outputs),
            'json_outputs': dict(result.json_outputs),
            'error': result.error,
            'timings': dict(row_timings),
    }


def raw_row_summary(line: str, error: Exception, row_timings: dict) -> dict:
    """Return the summary object for an unreadable input row."""
    return {
            'id': None,
            'question_type': None,
            'problem': None,
            'answer': None,
            'raw_input': line,
            'boolean_outputs': {},
            'json_outputs': {},
            'error': exception_message(error),
            'timings': dict(row_timings),
    }


def write_cleaned_rows(
        output_file: Any,
        rows: list[dict],
        result: Any,
        row_timings: dict[str, float],
) -> None:
    """Write cleaned rows with their data-cleaning metadata."""
    for output_row in rows:
        write_record(
                output_file,
                cleaned_record_with_metadata(
                        output_row,
                        result.boolean_outputs,
                        result.json_outputs,
                        row_timings,
                        result.error,
                ),
        )
    if rows:
        output_file.flush()
