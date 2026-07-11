import json
from typing import Any

from alphaproof.formalize.data_cleaning.prompts import (
        ANSWER_PROOF_PROMPT,
        ANSWERLESS_PROOF_PROMPT,
        MATH_WORD_MULTI_PART_PROMPT,
        MATH_WORD_SPLIT_PROBLEM_PROMPT,
        MISSING_INFORMATION_PROMPT,
        MCQ_SPLIT_PROBLEM_PROMPT,
        PROOF_SEVERAL_QUESTIONS_PROMPT,
        PROOF_SPLIT_PROBLEM_PROMPT,
        REMOVE_OPTIONS_PROMPT,
        SEVERAL_STATEMENTS_PROMPT,
)
from alphaproof.formalize.qwen3 import Qwen3


BOOLEAN_MAX_NEW_TOKENS = 8
JSON_MAX_NEW_TOKENS = 1024
SPLIT_MAX_NEW_TOKENS = 2048
THINKING_BOOLEAN_MAX_NEW_TOKENS = 4096
THINKING_JSON_MAX_NEW_TOKENS = 16384
THINKING_SPLIT_MAX_NEW_TOKENS = 32768


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


def sample_json_objects(
        prompts: list[str],
        model: Qwen3,
        max_new_tokens: int = JSON_MAX_NEW_TOKENS,
        thinking_max_new_tokens: int = THINKING_JSON_MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    """Sample and parse one JSON object for each prompt."""
    answers = model.sample(
            prompts,
            max_new_tokens=max_tokens_for_model(
                    model,
                    max_new_tokens,
                    thinking_max_new_tokens,
            ),
            temperature=0.1,
            top_p=0.2,
    )
    return [parse_json_object(answer) for answer in answers]


def sample_yes_no(
        prompts: list[str],
        model: Qwen3,
) -> list[tuple[bool, str]]:
    """Sample and parse one YES/NO answer for each prompt."""
    answers = model.sample(
            prompts,
            max_new_tokens=max_tokens_for_model(
                    model,
                    BOOLEAN_MAX_NEW_TOKENS,
                    THINKING_BOOLEAN_MAX_NEW_TOKENS,
            ),
            temperature=0.1,
            top_p=0.2,
    )
    return [(parse_yes_no(answer), answer) for answer in answers]


def has_missing_information(
        records: list[dict],
        model: Qwen3,
) -> list[tuple[bool, str]]:
    """Return whether each problem is missing information and the raw answer."""
    prompts = [
            fill_prompt(MISSING_INFORMATION_PROMPT, problem=record['problem'].strip())
            for record in records
    ]
    return sample_yes_no(prompts, model)


def is_multi_part_problem(
        records: list[dict],
        model: Qwen3,
) -> list[tuple[bool, str]]:
    """Return whether each problem should be split and the raw answer."""
    prompts = [
            fill_prompt(
                    MATH_WORD_MULTI_PART_PROMPT,
                    problem=record['problem'].strip(),
            )
            for record in records
    ]
    return sample_yes_no(prompts, model)


def has_several_proof_questions(
        records: list[dict],
        model: Qwen3,
) -> list[tuple[bool, str]]:
    """Return whether each proof problem should be split and the raw answer."""
    prompts = [
            fill_prompt(
                    PROOF_SEVERAL_QUESTIONS_PROMPT,
                    problem=record['problem'].strip(),
            )
            for record in records
    ]
    return sample_yes_no(prompts, model)


def has_several_statements(
        records: list[dict],
        model: Qwen3,
) -> list[tuple[bool, str]]:
    """Return whether each MCQ has statement options and the raw answer."""
    prompts = [
            fill_prompt(SEVERAL_STATEMENTS_PROMPT, problem=record['problem'].strip())
            for record in records
    ]
    return sample_yes_no(prompts, model)


def parse_split_problems(
        parsed: dict[str, Any],
        include_answer: bool = True,
) -> list[dict]:
    """Parse split-problem JSON into standalone problem objects."""
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
        split_problem = {'problem': text.strip(), 'answer': None}
        if include_answer:
            split_problem['answer'] = problem.get('answer')
        split_problems.append(split_problem)

    return split_problems


def sample_split_problem_outputs(
        prompt_template: str,
        records: list[dict],
        model: Qwen3,
) -> list[list[dict]]:
    """Sample split-problem JSON outputs with the given prompt."""
    prompts = [
            fill_prompt(
                    prompt_template,
                    problem=json_prompt_text(record['problem'].strip()),
                    answer=json_prompt_text(record.get('answer')),
            )
            for record in records
    ]
    parsed_outputs = sample_json_objects(
            prompts,
            model,
            max_new_tokens=SPLIT_MAX_NEW_TOKENS,
            thinking_max_new_tokens=THINKING_SPLIT_MAX_NEW_TOKENS,
    )
    return [parse_split_problems(parsed) for parsed in parsed_outputs]


def split_math_word_problems(
        records: list[dict],
        model: Qwen3,
) -> list[list[dict]]:
    """Split each math word problem into standalone problem/answer pairs."""
    return sample_split_problem_outputs(
            MATH_WORD_SPLIT_PROBLEM_PROMPT,
            records,
            model,
    )


def split_mcq_statements(
        records: list[dict],
        model: Qwen3,
) -> list[list[dict]]:
    """Split each statement-option MCQ into true/false problems."""
    return sample_split_problem_outputs(MCQ_SPLIT_PROBLEM_PROMPT, records, model)


def split_proof_questions(
        records: list[dict],
        model: Qwen3,
) -> list[list[dict]]:
    """Split each proof problem into standalone proof problems."""
    prompts = [
            fill_prompt(
                    PROOF_SPLIT_PROBLEM_PROMPT,
                    problem=json_prompt_text(record['problem'].strip()),
            )
            for record in records
    ]
    parsed_outputs = sample_json_objects(
            prompts,
            model,
            max_new_tokens=SPLIT_MAX_NEW_TOKENS,
            thinking_max_new_tokens=THINKING_SPLIT_MAX_NEW_TOKENS,
    )
    return [
            parse_split_problems(parsed, include_answer=False)
            for parsed in parsed_outputs
    ]


def parse_removed_options(parsed: dict[str, Any]) -> dict:
    """Parse direct-problem JSON after removing MCQ options."""
    problem = parsed.get('problem')
    answer = parsed.get('answer')
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f'Expected problem text, got: {parsed!r}')
    if answer is None or str(answer).strip() == '':
        raise ValueError(f'Expected selected answer, got: {parsed!r}')
    return {'problem': problem.strip(), 'answer': answer}


def remove_options(
        records: list[dict],
        model: Qwen3,
) -> list[dict]:
    """Convert each normal MCQ into a direct problem with a direct answer."""
    prompts = [
            fill_prompt(
                    REMOVE_OPTIONS_PROMPT,
                    problem=json_prompt_text(record['problem'].strip()),
                    answer=json_prompt_text(record.get('answer')),
            )
            for record in records
    ]
    return [
            parse_removed_options(parsed)
            for parsed in sample_json_objects(prompts, model)
    ]


def proof_problems_with_answer(
        jobs: list[dict],
        model: Qwen3,
) -> list[str]:
    """Create proof problems that include known answers."""
    prompts = [
            fill_prompt(
                    ANSWER_PROOF_PROMPT,
                    problem=json_prompt_text(job['problem'].strip()),
                    answer=json_prompt_text(str(job['answer']).strip()),
            )
            for job in jobs
    ]
    return parse_proof_problems(sample_json_objects(prompts, model))


def proof_problems_without_answer(
        jobs: list[dict],
        model: Qwen3,
) -> list[str]:
    """Create proof problems without exposing answers to the model."""
    prompts = [
            fill_prompt(
                    ANSWERLESS_PROOF_PROMPT,
                    problem=json_prompt_text(job['problem'].strip()),
            )
            for job in jobs
    ]
    return parse_proof_problems(sample_json_objects(prompts, model))


def parse_proof_problems(parsed_outputs: list[dict[str, Any]]) -> list[str]:
    """Parse proof-problem JSON outputs."""
    proof_problems = []
    for parsed in parsed_outputs:
        proof_problem = parsed.get('problem')
        if not isinstance(proof_problem, str) or not proof_problem.strip():
            raise ValueError(f'Expected proof problem text, got: {parsed!r}')
        proof_problems.append(proof_problem.strip())
    return proof_problems
