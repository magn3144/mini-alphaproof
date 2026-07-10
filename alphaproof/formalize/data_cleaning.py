import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from alphaproof.formalize.filter_problems import FILTERED_NUMINA_MATH_1_5_PATH
from alphaproof.formalize.qwen3 import Qwen3, Qwen3_8B


CLEANED_NUMINA_MATH_1_5_PATH = (
        FILTERED_NUMINA_MATH_1_5_PATH.parent / 'numina_math_1_5_cleaned.jsonl'
)
BOOLEAN_MAX_NEW_TOKENS = 8
JSON_MAX_NEW_TOKENS = 1024
SPLIT_MAX_NEW_TOKENS = 2048
THINKING_BOOLEAN_MAX_NEW_TOKENS = 4096
THINKING_JSON_MAX_NEW_TOKENS = 16384
THINKING_SPLIT_MAX_NEW_TOKENS = 32768

MISSING_INFORMATION_PROMPT = r"""<task>
Decide whether a math problem is missing information needed to solve or formalize it.
</task>

<instructions>
Answer only YES or NO.
Answer YES if the problem depends on a missing diagram, missing formula, missing table,
missing image, incomplete statement, or any other unavailable information.
Answer NO if the problem statement contains all information needed, even if it is hard.
</instructions>

<examples>
<example>
<problem>
Given are numbers $ a_1, a_2, \ldots, a_n $, each two of which are different. Find the
smallest value of the function defined by the formula Note: The formula or expression
that should follow is missing in the original text. If you have the complete formula,
please provide it for a full translation.
</problem>
<answer>YES</answer>
</example>

<example>
<problem>
Find all real numbers $a$ such that the roots of the polynomial
$$x^3 - 9x^2 + 42x + a$$ form an arithmetic progression and are not all real.
</problem>
<answer>NO</answer>
</example>
</examples>

<problem>
{problem}
</problem>
"""

MULTI_PART_PROMPT = """<task>
Decide whether a math problem contains multiple separate subproblems that should be
split before formalization.
</task>

<instructions>
Answer only YES or NO.
Answer YES when the problem asks for multiple numbered or lettered tasks, such as
(1), (2), (a), (b), or several separate questions.
Answer NO when the problem is one task, even if it asks to find all solutions.
</instructions>

<examples>
<example>
<problem>
(1) Find all integer pairs $(x, y)$ that satisfy $y^4 + 2x^4 + 1 = 4x^2y$;
(2) Find all positive integer solutions to $5(xy + yz + zx) = 4xyz$.
</problem>
<answer>YES</answer>
</example>

<example>
<problem>
Find all positive integers $n$ such that there exists a prime number $p$ such that
$p^n-(p-1)^n$ is a power of 3.
</problem>
<answer>NO</answer>
</example>
</examples>

<problem>
{problem}
</problem>
"""

SEVERAL_STATEMENTS_PROMPT = r"""<task>
Decide whether this multiple-choice math problem has options that are separate
mathematical statements or propositions.
</task>

<instructions>
Answer only YES or NO.
Answer YES if options A, B, C, D, etc. are separate true/false statements,
conclusions, or propositions.
Answer NO if the options are candidate numeric values, expressions, intervals,
points, equations, or other direct answers to one question.
</instructions>

<examples>
<example>
<problem>
Which of the following statements is incorrect?
A: Supplementary angles are congruent.
B: Vertical angles are equal.
C: The cube root of $-1$ is $-1$.
D: Two lines perpendicular to the same line in the same plane are parallel.
</problem>
<answer>YES</answer>
</example>

<example>
<problem>
The sequence $\{a_n\}$ satisfies $a_1=2$ and $a_{n+1}=4a_n-3$.
Then $a_{10}$ equals to ( )
A: $2^{18}-1$
B: $2^{18}+1$
C: $2^{20}+1$
D: $2^{20}-1$
</problem>
<answer>NO</answer>
</example>
</examples>

<problem>
{problem}
</problem>
"""

SPLIT_PROBLEM_PROMPT = r"""<task>
Split a math problem into separate standalone problems.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problems": [{"problem": "standalone problem text", "answer": null}]}

Each output problem must be understandable without the original combined problem.
If the input is a multi-part problem, split each part into one problem.
If the input is a multiple-choice problem whose options are separate statements,
turn each option into one true/false problem.
Use the provided answer only to assign each split problem's answer when possible.
For true/false statement options, use "true" or "false" as the answer.
</instructions>

<example>
<problem>
(1) Find $x+1$ if $x=2$. (2) Find $y+2$ if $y=3$.
</problem>
<answer>3, 5</answer>
<output>
{"problems": [{"problem": "Find $x+1$ if $x=2$.", "answer": "3"}, {"problem": "Find $y+2$ if $y=3$.", "answer": "5"}]}
</output>
</example>

<example>
<problem>
Which of the following statements is incorrect?
A: Supplementary angles are congruent.
B: Vertical angles are equal.
C: The cube root of $-1$ is $-1$.
D: Two lines perpendicular to the same line in the same plane are parallel.
</problem>
<answer>A</answer>
<output>
{"problems": [{"problem": "Supplementary angles are congruent.", "answer": "false"}, {"problem": "Vertical angles are equal.", "answer": "true"}, {"problem": "The cube root of $-1$ is $-1$.", "answer": "true"}, {"problem": "Two lines perpendicular to the same line in the same plane are parallel.", "answer": "true"}]}
</output>
</example>

<problem>
{problem}
</problem>
<answer>{answer}</answer>
"""

REMOVE_OPTIONS_PROMPT = r"""<task>
Convert a multiple-choice math problem into a direct math word problem.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problem": "self-contained problem text without option labels", "answer": "selected option value"}

Rephrase the MCQ problem statement as a math-word-problem such that the answer
option labels are removed. Make sure the resulting math-word-problem is self
contained. If information from the options is needed to make the problem self
contained, include that information in the rewritten problem without mentioning
which option is correct.
Return the answer as the text/value of the selected option, not as an option
label. If the provided answer is already a direct answer instead of an option
label, return it unchanged.
</instructions>

<example>
<problem>
Which of the following numbers is irrational?
A: $3.14$
B: $\frac{2}{7}$
C: $\sqrt{0.04}$
D: $\pi - 3.14$
</problem>
<answer>D</answer>
<output>
{"problem": "Determine which of the numbers $3.14$, $\\frac{2}{7}$, $\\sqrt{0.04}$, and $\\pi - 3.14$ is irrational.", "answer": "$\\pi - 3.14$"}
</output>
</example>

<problem>
{problem}
</problem>
<answer>{answer}</answer>
"""

ANSWER_PROOF_PROMPT = r"""<task>
Rewrite a math problem as a proof problem using the known answer.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problem": "natural language proof problem"}

The rewritten problem must ask to prove that the provided answer is the correct
answer to the original problem.
Do not include a solution or explanation.
</instructions>

<example>
<problem>
The sequence $\{a_n\}$ satisfies $a_1=2$ and $a_{n+1}=4a_n-3$. Find $a_{10}$.
</problem>
<answer>$2^{18}+1$</answer>
<output>
{"problem": "Prove that if the sequence $\\{a_n\\}$ satisfies $a_1=2$ and $a_{n+1}=4a_n-3$, then $a_{10}=2^{18}+1$."}
</output>
</example>

<problem>
{problem}
</problem>
<answer>{answer}</answer>
"""

ANSWERLESS_PROOF_PROMPT = r"""<task>
Rewrite a math problem as a proof problem without using or seeing its answer.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problem": "natural language proof problem"}

The rewritten problem must ask to prove that an answer exists for the original
problem. Do not guess or include the answer. Do not include a solution or explanation.
</instructions>

<example>
<problem>
The sequence $\{a_n\}$ satisfies $a_1=2$ and $a_{n+1}=4a_n-3$. Find $a_{10}$.
</problem>
<output>
{"problem": "Prove that there exists a value of $a_{10}$ for the sequence $\\{a_n\\}$ satisfying $a_1=2$ and $a_{n+1}=4a_n-3$."}
</output>
</example>

<problem>
{problem}
</problem>
"""


def fill_prompt(template: str, **values: str) -> str:
    """Fill simple prompt placeholders without interpreting JSON braces."""
    prompt = template
    for name, value in values.items():
        prompt = prompt.replace(f'{{{name}}}', value)
    return prompt


def json_prompt_text(value: Any) -> str:
    """Return value as JSON text for prompts that ask for JSON output."""
    return json.dumps(value, ensure_ascii=False)


def parse_yes_no(answer: str) -> bool:
    """Parse a YES/NO model response."""
    normalized_answer = answer.strip().upper()

    if normalized_answer.startswith('YES'):
        return True
    if normalized_answer.startswith('NO'):
        return False

    raise ValueError(f'Expected YES or NO, got: {answer!r}')


class InvalidJsonOutput(ValueError):
    """Raised when the model output cannot be parsed as JSON."""

    def __init__(self, output: str, error: json.JSONDecodeError):
        self.output = output
        message = (
                f'Expected valid JSON, got parse error at line {error.lineno} '
                f'column {error.colno}: {error.msg}'
        )
        super().__init__(message)


def parse_json_object(answer: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""
    try:
        return json.loads(answer.strip())
    except json.JSONDecodeError as error:
        raise InvalidJsonOutput(answer, error) from error


def max_tokens_for_model(
        model: Qwen3,
        max_new_tokens: int,
        thinking_max_new_tokens: int,
) -> int:
    """Return a larger generation budget when thinking is enabled."""
    if getattr(model, 'enable_thinking', False):
        return thinking_max_new_tokens
    return max_new_tokens


def sample_json_object(
        prompt: str,
        model: Qwen3,
        max_new_tokens: int = JSON_MAX_NEW_TOKENS,
        thinking_max_new_tokens: int = THINKING_JSON_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    """Sample and parse one JSON object from the model."""
    answer = model.sample(
            prompt,
            num_samples=1,
            max_new_tokens=max_tokens_for_model(
                    model,
                    max_new_tokens,
                    thinking_max_new_tokens,
            ),
            temperature=0.1,
            top_p=0.2,
    )[0]
    return parse_json_object(answer)


def has_missing_information(problem: str, model: Qwen3) -> tuple[bool, str]:
    """Return whether a problem is missing information and the raw answer."""
    prompt = fill_prompt(MISSING_INFORMATION_PROMPT, problem=problem.strip())
    answer = model.sample(
            prompt,
            num_samples=1,
            max_new_tokens=max_tokens_for_model(
                    model,
                    BOOLEAN_MAX_NEW_TOKENS,
                    THINKING_BOOLEAN_MAX_NEW_TOKENS,
            ),
            temperature=0.1,
            top_p=0.2,
    )[0]
    return parse_yes_no(answer), answer


def is_multi_part_problem(problem: str, model: Qwen3) -> tuple[bool, str]:
    """Return whether a problem should be split and the raw answer."""
    prompt = fill_prompt(MULTI_PART_PROMPT, problem=problem.strip())
    answer = model.sample(
            prompt,
            num_samples=1,
            max_new_tokens=max_tokens_for_model(
                    model,
                    BOOLEAN_MAX_NEW_TOKENS,
                    THINKING_BOOLEAN_MAX_NEW_TOKENS,
            ),
            temperature=0.1,
            top_p=0.2,
    )[0]
    return parse_yes_no(answer), answer


def has_several_statements(problem: str, model: Qwen3) -> tuple[bool, str]:
    """Return whether MCQ options are separate statements and the raw answer."""
    prompt = fill_prompt(SEVERAL_STATEMENTS_PROMPT, problem=problem.strip())
    answer = model.sample(
            prompt,
            num_samples=1,
            max_new_tokens=max_tokens_for_model(
                    model,
                    BOOLEAN_MAX_NEW_TOKENS,
                    THINKING_BOOLEAN_MAX_NEW_TOKENS,
            ),
            temperature=0.1,
            top_p=0.2,
    )[0]
    return parse_yes_no(answer), answer


def split_into_several_problems(record: dict, model: Qwen3) -> list[dict]:
    """Split a record into standalone problem/answer pairs."""
    prompt = fill_prompt(
            SPLIT_PROBLEM_PROMPT,
            problem=json_prompt_text(record['problem'].strip()),
            answer=json_prompt_text(record.get('answer')),
    )
    parsed = sample_json_object(
            prompt,
            model,
            max_new_tokens=SPLIT_MAX_NEW_TOKENS,
            thinking_max_new_tokens=THINKING_SPLIT_MAX_NEW_TOKENS,
    )
    problems = parsed.get('problems')
    if not isinstance(problems, list) or not problems:
        raise ValueError(f'Expected nonempty problems list, got: {parsed!r}')

    split_problems = []
    for problem in problems:
        if not isinstance(problem, dict):
            raise ValueError(f'Expected problem object, got: {problem!r}')
        text = problem.get('problem')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Expected problem text, got: {problem!r}')
        split_problems.append(
                {
                        'problem': text.strip(),
                        'answer': problem.get('answer'),
                }
        )

    return split_problems


def remove_options(record: dict, model: Qwen3) -> dict:
    """Convert a normal MCQ into a direct problem with a direct answer."""
    prompt = fill_prompt(
            REMOVE_OPTIONS_PROMPT,
            problem=json_prompt_text(record['problem'].strip()),
            answer=json_prompt_text(record.get('answer')),
    )
    parsed = sample_json_object(prompt, model)
    problem = parsed.get('problem')
    answer = parsed.get('answer')
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f'Expected problem text, got: {parsed!r}')
    if answer is None or str(answer).strip() == '':
        raise ValueError(f'Expected selected answer, got: {parsed!r}')
    return {'problem': problem.strip(), 'answer': answer}


def proof_problem_with_answer(
        problem: str,
        answer: Any,
        model: Qwen3,
) -> str:
    """Create a proof problem that includes the known answer."""
    if answer is None or str(answer).strip() == '':
        raise ValueError('Cannot create answer proof problem without an answer.')

    prompt = fill_prompt(
            ANSWER_PROOF_PROMPT,
            problem=json_prompt_text(problem.strip()),
            answer=json_prompt_text(str(answer).strip()),
    )
    parsed = sample_json_object(prompt, model)
    proof_problem = parsed.get('problem')
    if not isinstance(proof_problem, str) or not proof_problem.strip():
        raise ValueError(f'Expected proof problem text, got: {parsed!r}')
    return proof_problem.strip()


def proof_problem_without_answer(problem: str, model: Qwen3) -> str:
    """Create a proof problem without exposing the answer to the model."""
    prompt = fill_prompt(
            ANSWERLESS_PROOF_PROMPT,
            problem=json_prompt_text(problem.strip()),
    )
    parsed = sample_json_object(prompt, model)
    proof_problem = parsed.get('problem')
    if not isinstance(proof_problem, str) or not proof_problem.strip():
        raise ValueError(f'Expected proof problem text, got: {parsed!r}')
    return proof_problem.strip()


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
    """Track total timings and timings for the current input row."""

    def __init__(self):
        self.total: dict[str, float] = {}
        self.current_row: dict[str, float] = {}

    def add(self, name: str, seconds: float) -> None:
        """Add a timing sample to the total and current row timers."""
        self.total[name] = self.total.get(name, 0.0) + seconds
        self.current_row[name] = self.current_row.get(name, 0.0) + seconds

    def timed(self, name: str, function: Any, *args: Any) -> Any:
        """Call a function and record how long it takes."""
        start = perf_counter()
        try:
            return function(*args)
        finally:
            self.add(name, perf_counter() - start)


def add_proof_problem_rows(
        record: dict,
        problem: str,
        answer: Any,
        model: Qwen3,
        suffix: str,
) -> list[dict]:
    """Return answer-aware and answer-free proof problem rows."""
    rows = []
    if answer is not None and str(answer).strip() != '':
        with_answer = proof_problem_with_answer(problem, answer, model)
        rows.append(
                formalization_record(
                        derived_record(
                                record,
                                with_answer,
                                answer,
                                f'{suffix}_with_answer',
                        )
                ),
        )

    without_answer = proof_problem_without_answer(problem, model)
    rows.append(
            formalization_record(
                    derived_record(
                            record,
                            without_answer,
                            None,
                            f'{suffix}_without_answer',
                    )
            ),
    )
    return rows


def record_error(result: CleanResult, error: Exception) -> None:
    """Attach exception details to a clean result."""
    result.error = exception_message(error)
    result.errored = True
    if isinstance(error, InvalidJsonOutput):
        result.json_outputs['invalid_json_output'] = error.output


def proof_rows_for_problems(
        record: dict,
        problems: list[dict],
        model: Qwen3,
        prefix: str,
        result: CleanResult,
        timers: Timers,
) -> None:
    """Add proof-problem rows derived from standalone problems."""
    for problem_ix, split_problem in enumerate(problems):
        proof_rows = timers.timed(
                'add_proof_problem_rows',
                add_proof_problem_rows,
                record,
                split_problem['problem'],
                split_problem.get('answer'),
                model,
                f'{prefix}_{problem_ix}',
        )
        result.json_outputs.setdefault('add_proof_problem_rows', []).extend(
                proof_rows
        )
        result.output_rows.extend(proof_rows)


def clean_math_word_problem(
        record: dict,
        model: Qwen3,
        result: CleanResult,
        timers: Timers,
) -> None:
    """Clean a math-word-problem row into proof-problem rows."""
    is_multi_part, is_multi_part_answer = timers.timed(
            'is_multi_part_problem',
            is_multi_part_problem,
            record['problem'],
            model,
    )
    result.boolean_outputs['is_multi_part_problem'] = is_multi_part
    result.boolean_outputs['is_multi_part_problem_answer'] = is_multi_part_answer

    if is_multi_part:
        problems = timers.timed(
                'split_into_several_problems',
                split_into_several_problems,
                record,
                model,
        )
        result.json_outputs['split_into_several_problems'] = problems
    else:
        problems = [
                {
                        'problem': record['problem'],
                        'answer': record.get('answer'),
                }
        ]

    proof_rows_for_problems(
            record,
            problems,
            model,
            'part',
            result,
            timers,
    )


def clean_mcq(
        record: dict,
        model: Qwen3,
        result: CleanResult,
        timers: Timers,
) -> None:
    """Clean an MCQ row into proof-problem rows."""
    several_statements, several_statements_answer = timers.timed(
            'has_several_statements',
            has_several_statements,
            record['problem'],
            model,
    )
    result.boolean_outputs['has_several_statements'] = several_statements
    result.boolean_outputs['has_several_statements_answer'] = (
            several_statements_answer
    )

    if several_statements:
        problems = timers.timed(
                'split_into_several_problems',
                split_into_several_problems,
                record,
                model,
        )
        result.json_outputs['split_into_several_problems'] = problems
    else:
        removed_options = timers.timed(
                'remove_options',
                remove_options,
                record,
                model,
        )
        result.json_outputs['remove_options'] = removed_options
        problems = [removed_options]

    proof_rows_for_problems(
            record,
            problems,
            model,
            'mcq',
            result,
            timers,
    )


def clean_supported_record(
        record: dict,
        model: Qwen3,
        result: CleanResult,
        timers: Timers,
) -> None:
    """Clean a record after missing-information checks have passed."""
    question_type = record.get('question_type')

    if question_type == 'proof':
        result.output_rows.append(formalization_record(record))
        return

    if question_type == 'math-word-problem':
        clean_math_word_problem(record, model, result, timers)
        return

    if question_type == 'MCQ':
        clean_mcq(record, model, result, timers)
        return

    result.output_rows.append(failed_record(record, 'unsupported_type'))


def clean_record(record: dict, model: Qwen3, timers: Timers) -> CleanResult:
    """Clean one input record and return output rows plus metadata."""
    result = CleanResult()

    try:
        missing_information, missing_information_answer = timers.timed(
                'has_missing_information',
                has_missing_information,
                record['problem'],
                model,
        )
        result.boolean_outputs['has_missing_information'] = missing_information
        result.boolean_outputs['has_missing_information_answer'] = (
                missing_information_answer
        )
    except Exception as error:
        record_error(result, error)
        result.output_rows.append(failed_record(record, 'error'))
        return result

    if missing_information:
        result.missing_information = True
        result.output_rows.append(failed_record(record, 'missing_information'))
        return result

    try:
        clean_supported_record(record, model, result, timers)
    except Exception as error:
        record_error(result, error)
        result.output_rows.append(failed_record(record, 'error'))

    return result


def row_summary(record: dict, result: CleanResult, row_timings: dict) -> dict:
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
        result: CleanResult,
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


def load_cleaning_model(
        device: str | None,
        torch_dtype: str,
        quantization: str | None,
        enable_thinking: bool,
) -> Qwen3:
    """Load the model used for data cleaning."""
    if device is not None:
        model_device = torch.device(device)
    elif torch.cuda.is_available():
        model_device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        model_device = torch.device('mps')
    else:
        model_device = torch.device('cpu')

    if torch_dtype == 'auto' and model_device.type == 'mps':
        torch_dtype = 'float16'

    qwen = Qwen3_8B(
            device=model_device,
            torch_dtype=torch_dtype,
            enable_thinking=enable_thinking,
            quantization=quantization,
    )
    qwen.load()
    print(f'Loaded {qwen.model_name} from {qwen.model_dir} on {qwen.device}.')
    return qwen


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
        device: str | None = None,
        torch_dtype: str = 'auto',
        quantization: str | None = None,
        enable_thinking: bool = False,
) -> None:
    """Run the data cleaning pipeline over the filtered dataset."""
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

    with input_path.open(encoding='utf-8') as input_file:
        with output_path.open('a', encoding='utf-8') as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue
                if cleaned_rows >= rows_to_clean:
                    break

                timers.current_row = {}
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
                    timers.add('row_iteration', row_timings['row_iteration'])
                    continue

                rows_read += 1
                result = clean_record(record, qwen, timers)
                timers.add('row_iteration', perf_counter() - row_start)

                output_rows.extend(result.output_rows)
                row_summaries.append(
                        row_summary(record, result, timers.current_row)
                )
                write_cleaned_rows(
                        output_file,
                        result.output_rows,
                        result,
                        timers.current_row,
                )
                if result.missing_information:
                    missing_information_rows += 1
                if result.errored:
                    errored_rows += 1
                if result.output_rows:
                    cleaned_rows += 1

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
            device=args.device,
            torch_dtype=args.torch_dtype,
            quantization=args.quantization,
            enable_thinking=args.enable_thinking,
    )
