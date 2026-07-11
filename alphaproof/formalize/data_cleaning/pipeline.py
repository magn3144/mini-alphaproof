from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from alphaproof.formalize.data_cleaning.model import (
        clear_accelerator_cache,
        is_out_of_memory_error,
)
from alphaproof.formalize.data_cleaning.model_calls import (
        InvalidJsonOutput,
        has_missing_information,
        has_several_statements,
        is_multi_part_problem,
        proof_problems_with_answer,
        proof_problems_without_answer,
        remove_options,
        split_into_several_problems,
)
from alphaproof.formalize.data_cleaning.records import (
        derived_record,
        exception_message,
        failed_record,
        formalization_record,
)
from alphaproof.formalize.qwen3 import Qwen3


@dataclass
class CleanResult:
    """Output rows and metadata produced while cleaning one source record."""

    output_rows: list[dict] = field(default_factory=list)
    boolean_outputs: dict = field(default_factory=dict)
    json_outputs: dict = field(default_factory=dict)
    error: str | None = None
    missing_information: bool = False
    errored: bool = False


class Timers:
    """Track total timings."""

    def __init__(self):
        self.total: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        """Add a timing sample to the totals."""
        self.total[name] = self.total.get(name, 0.0) + seconds

    def timed(self, name: str, function: Any, *args: Any) -> Any:
        """Call a function and record how long it takes."""
        start = perf_counter()
        try:
            return function(*args)
        finally:
            self.add(name, perf_counter() - start)


def record_error(result: CleanResult, error: Exception) -> None:
    """Attach exception details to a clean result."""
    result.error = exception_message(error)
    result.errored = True
    if isinstance(error, InvalidJsonOutput):
        result.json_outputs['invalid_json_output'] = error.output


def fail_record_result(record: dict, result: CleanResult, error: Exception) -> None:
    """Mark one source record as failed from an exception."""
    record_error(result, error)
    result.output_rows = [failed_record(record, 'error')]


def fail_indexed_records(
        indexed_records: list[tuple[int, dict]],
        results: list[CleanResult],
        error: Exception,
) -> None:
    """Mark each indexed source record as failed."""
    if is_out_of_memory_error(error):
        clear_accelerator_cache()

    seen = set()
    for record_ix, record in indexed_records:
        if record_ix in seen:
            continue
        seen.add(record_ix)
        fail_record_result(record, results[record_ix], error)


def proof_job(
        record_ix: int,
        record: dict,
        problem: str,
        answer: Any,
        suffix: str,
) -> dict:
    """Return the metadata needed to create proof-problem rows."""
    return {
            'record_ix': record_ix,
            'record': record,
            'problem': problem,
            'answer': answer,
            'suffix': suffix,
    }


def append_proof_row(
        job: dict,
        proof_problem: str,
        answer: Any,
        suffix: str,
        results: list[CleanResult],
) -> None:
    """Append one derived proof-problem row for a proof job."""
    result = results[job['record_ix']]
    if result.errored:
        return

    row = formalization_record(
            derived_record(
                    job['record'],
                    proof_problem,
                    answer,
                    suffix,
            )
    )
    result.output_rows.append(row)
    result.json_outputs.setdefault('add_proof_problem_rows', []).append(row)


def add_proof_rows(
        jobs: list[dict],
        model: Qwen3,
        results: list[CleanResult],
        timers: Timers,
) -> None:
    """Add answer-aware and answer-free proof rows for proof jobs."""
    jobs_with_answer = [
            job
            for job in jobs
            if job['answer'] is not None and str(job['answer']).strip() != ''
    ]
    if jobs_with_answer:
        try:
            proof_problems = timers.timed(
                    'proof_problems_with_answer',
                    proof_problems_with_answer,
                    jobs_with_answer,
                    model,
            )
        except Exception as error:
            fail_indexed_records(
                    [
                            (job['record_ix'], job['record'])
                            for job in jobs_with_answer
                    ],
                    results,
                    error,
            )
        else:
            for job, proof_problem in zip(jobs_with_answer, proof_problems):
                append_proof_row(
                        job,
                        proof_problem,
                        job['answer'],
                        f'{job["suffix"]}_with_answer',
                        results,
                )

    jobs_without_answer = [
            job
            for job in jobs
            if not results[job['record_ix']].errored
    ]
    if not jobs_without_answer:
        return

    try:
        proof_problems = timers.timed(
                'proof_problems_without_answer',
                proof_problems_without_answer,
                jobs_without_answer,
                model,
        )
    except Exception as error:
        fail_indexed_records(
                [
                        (job['record_ix'], job['record'])
                        for job in jobs_without_answer
                ],
                results,
                error,
        )
        return

    for job, proof_problem in zip(jobs_without_answer, proof_problems):
        append_proof_row(
                job,
                proof_problem,
                None,
                f'{job["suffix"]}_without_answer',
                results,
        )


def clean_math_word_problems(
        indexed_records: list[tuple[int, dict]],
        model: Qwen3,
        results: list[CleanResult],
        timers: Timers,
) -> None:
    """Clean math-word-problem records."""
    if not indexed_records:
        return

    records = [record for _, record in indexed_records]
    try:
        multi_part_outputs = timers.timed(
                'is_multi_part_problem',
                is_multi_part_problem,
                records,
                model,
        )
    except Exception as error:
        fail_indexed_records(indexed_records, results, error)
        return

    split_records = []
    jobs = []
    for (record_ix, record), (is_multi_part, answer) in zip(
            indexed_records,
            multi_part_outputs,
    ):
        result = results[record_ix]
        result.boolean_outputs['is_multi_part_problem'] = is_multi_part
        result.boolean_outputs['is_multi_part_problem_answer'] = answer
        if is_multi_part:
            split_records.append((record_ix, record))
        else:
            jobs.append(
                    proof_job(
                            record_ix,
                            record,
                            record['problem'],
                            record.get('answer'),
                            'part_0',
                    )
            )

    if split_records:
        try:
            split_outputs = timers.timed(
                    'split_into_several_problems',
                    split_into_several_problems,
                    [record for _, record in split_records],
                    model,
            )
        except Exception as error:
            fail_indexed_records(split_records, results, error)
        else:
            for (record_ix, record), problems in zip(split_records, split_outputs):
                results[record_ix].json_outputs[
                        'split_into_several_problems'
                ] = problems
                for problem_ix, problem in enumerate(problems):
                    jobs.append(
                            proof_job(
                                    record_ix,
                                    record,
                                    problem['problem'],
                                    problem.get('answer'),
                                    f'part_{problem_ix}',
                            )
                    )

    add_proof_rows(jobs, model, results, timers)


def clean_mcqs(
        indexed_records: list[tuple[int, dict]],
        model: Qwen3,
        results: list[CleanResult],
        timers: Timers,
) -> None:
    """Clean MCQ records."""
    if not indexed_records:
        return

    records = [record for _, record in indexed_records]
    try:
        statement_outputs = timers.timed(
                'has_several_statements',
                has_several_statements,
                records,
                model,
        )
    except Exception as error:
        fail_indexed_records(indexed_records, results, error)
        return

    split_records = []
    remove_records = []
    for (record_ix, record), (several_statements, answer) in zip(
            indexed_records,
            statement_outputs,
    ):
        result = results[record_ix]
        result.boolean_outputs['has_several_statements'] = several_statements
        result.boolean_outputs['has_several_statements_answer'] = answer
        if several_statements:
            split_records.append((record_ix, record))
        else:
            remove_records.append((record_ix, record))

    jobs = []
    if split_records:
        try:
            split_outputs = timers.timed(
                    'split_into_several_problems',
                    split_into_several_problems,
                    [record for _, record in split_records],
                    model,
            )
        except Exception as error:
            fail_indexed_records(split_records, results, error)
        else:
            for (record_ix, record), problems in zip(split_records, split_outputs):
                results[record_ix].json_outputs[
                        'split_into_several_problems'
                ] = problems
                for problem_ix, problem in enumerate(problems):
                    jobs.append(
                            proof_job(
                                    record_ix,
                                    record,
                                    problem['problem'],
                                    problem.get('answer'),
                                    f'mcq_{problem_ix}',
                            )
                    )

    if remove_records:
        try:
            removed_outputs = timers.timed(
                    'remove_options',
                    remove_options,
                    [record for _, record in remove_records],
                    model,
            )
        except Exception as error:
            fail_indexed_records(remove_records, results, error)
        else:
            for (record_ix, record), removed in zip(remove_records, removed_outputs):
                results[record_ix].json_outputs['remove_options'] = removed
                jobs.append(
                        proof_job(
                                record_ix,
                                record,
                                removed['problem'],
                                removed.get('answer'),
                                'mcq_0',
                        )
                )

    add_proof_rows(jobs, model, results, timers)


def clean_records(
        records: list[dict],
        model: Qwen3,
        timers: Timers,
) -> list[CleanResult]:
    """Clean a same-question-type batch using one model call per stage."""
    results = [CleanResult() for _ in records]
    if not records:
        return results

    indexed_records = list(enumerate(records))
    question_type = records[0].get('question_type')

    try:
        missing_outputs = timers.timed(
                'has_missing_information',
                has_missing_information,
                records,
                model,
        )
    except Exception as error:
        fail_indexed_records(indexed_records, results, error)
        return results

    non_missing_records = []
    for (record_ix, record), (missing_information, answer) in zip(
            indexed_records,
            missing_outputs,
    ):
        result = results[record_ix]
        result.boolean_outputs['has_missing_information'] = missing_information
        result.boolean_outputs['has_missing_information_answer'] = answer

        if missing_information:
            result.missing_information = True
            result.output_rows.append(failed_record(record, 'missing_information'))
        else:
            non_missing_records.append((record_ix, record))

    if question_type == 'proof':
        for record_ix, record in non_missing_records:
            results[record_ix].output_rows.append(formalization_record(record))
    elif question_type == 'math-word-problem':
        clean_math_word_problems(non_missing_records, model, results, timers)
    elif question_type == 'MCQ':
        clean_mcqs(non_missing_records, model, results, timers)
    else:
        for record_ix, record in non_missing_records:
            results[record_ix].output_rows.append(
                    failed_record(record, 'unsupported_type')
            )

    return results
