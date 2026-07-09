import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from alphaproof.formalize.filter_problems import FILTERED_NUMINA_MATH_1_5_PATH
from alphaproof.formalize.qwen3 import Qwen3, Qwen3_8B


CLEANED_NUMINA_MATH_1_5_PATH = (
        FILTERED_NUMINA_MATH_1_5_PATH.parent / 'numina_math_1_5_cleaned.jsonl'
)

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
{"problem": "self-contained problem text without option labels"}

Rephrase the MCQ problem statement as a math-word-problem such that the answer
option labels are removed. Make sure the resulting math-word-problem is self
contained. If information from the options is needed to make the problem self
contained, include that information in the rewritten problem without mentioning
which option is correct.
</instructions>

<example>
<problem>
Which of the following numbers is irrational?
A: $3.14$
B: $\frac{2}{7}$
C: $\sqrt{0.04}$
D: $\pi - 3.14$
</problem>
<output>
{"problem": "Determine which of the numbers $3.14$, $\\frac{2}{7}$, $\\sqrt{0.04}$, and $\\pi - 3.14$ is irrational."}
</output>
</example>

<problem>
{problem}
</problem>
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


def parse_yes_no(answer: str) -> bool:
    """Parse a YES/NO model response."""
    normalized_answer = answer.strip().upper()

    if normalized_answer.startswith('YES'):
        return True
    if normalized_answer.startswith('NO'):
        return False

    raise ValueError(f'Expected YES or NO, got: {answer!r}')


def parse_json_object(answer: str) -> dict[str, Any]:
    """Parse a JSON object from a model response."""
    return json.loads(answer.strip())


def sample_json_object(
        prompt: str,
        model: Qwen3,
        max_new_tokens: int = 1024,
) -> dict[str, Any]:
    """Sample and parse one JSON object from the model."""
    answer = model.sample(
            prompt,
            num_samples=1,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            top_p=0.9,
    )[0]
    return parse_json_object(answer)


def has_missing_information(problem: str, model: Qwen3) -> bool:
    """Return whether a problem is missing information needed downstream."""
    prompt = fill_prompt(MISSING_INFORMATION_PROMPT, problem=problem.strip())
    answer = model.sample(prompt, num_samples=1, max_new_tokens=8)[0]
    return parse_yes_no(answer)


def is_multi_part_problem(problem: str, model: Qwen3) -> bool:
    """Return whether a problem should be split into multiple subproblems."""
    prompt = fill_prompt(MULTI_PART_PROMPT, problem=problem.strip())
    answer = model.sample(prompt, num_samples=1, max_new_tokens=8)[0]
    return parse_yes_no(answer)


def has_several_statements(problem: str, model: Qwen3) -> bool:
    """Return whether MCQ options are separate statements."""
    prompt = fill_prompt(SEVERAL_STATEMENTS_PROMPT, problem=problem.strip())
    answer = model.sample(prompt, num_samples=1, max_new_tokens=8)[0]
    return parse_yes_no(answer)


def split_into_several_problems(record: dict, model: Qwen3) -> list[dict]:
    """Split a record into standalone problem/answer pairs."""
    prompt = fill_prompt(
            SPLIT_PROBLEM_PROMPT,
            problem=record['problem'].strip(),
            answer=str(record.get('answer')),
    )
    parsed = sample_json_object(prompt, model, max_new_tokens=2048)
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


def selected_option_answer(record: dict) -> Any:
    """Return the selected MCQ option value when the answer is a simple label."""
    answer = record.get('answer')
    answer_label = str(answer).strip()

    if re.fullmatch(r'[A-F]', answer_label):
        option_pattern = re.compile(
                r'(?:^|\s)([A-F])\s*:\s*(.*?)(?=\s+[A-F]\s*:|\s*$)',
                re.DOTALL,
        )
        for label, option in option_pattern.findall(record['problem']):
            if label == answer_label:
                return option.strip()

        return answer

    if not re.fullmatch(r'[1-9]', answer_label):
        return answer

    option_pattern = re.compile(
            r'(?:^|\s)'
            r'(?:\(([1-9])\)|([1-9])\)|([1-9])\s*:|'
            r'(?:[Oo]ption|[Aa]nswer,\s*option)\s+([1-9])\s*[:.]?)'
            r'\s*(.*?)'
            r'(?=\s+(?:\([1-9]\)|[1-9]\)|[1-9]\s*:|'
            r'(?:[Oo]ption|[Aa]nswer,\s*option)\s+[1-9]\s*[:.]?)|\s*$)',
            re.DOTALL,
    )
    for parenthesized, closed, coloned, named, option in option_pattern.findall(
            record['problem'],
    ):
        label = parenthesized or closed or coloned or named
        if label == answer_label:
            return option.strip()

    return answer


def remove_options(record: dict, model: Qwen3) -> dict:
    """Convert a normal MCQ into a direct problem with a direct answer."""
    prompt = fill_prompt(
            REMOVE_OPTIONS_PROMPT,
            problem=record['problem'].strip(),
    )
    parsed = sample_json_object(prompt, model)
    problem = parsed.get('problem')
    answer = selected_option_answer(record)
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f'Expected problem text, got: {parsed!r}')
    if answer is None or str(answer).strip() == '':
        raise ValueError(f'Expected selected answer, got: {record!r}')
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
            problem=problem.strip(),
            answer=str(answer).strip(),
    )
    parsed = sample_json_object(prompt, model)
    proof_problem = parsed.get('problem')
    if not isinstance(proof_problem, str) or not proof_problem.strip():
        raise ValueError(f'Expected proof problem text, got: {parsed!r}')
    return proof_problem.strip()


def proof_problem_without_answer(problem: str, model: Qwen3) -> str:
    """Create a proof problem without exposing the answer to the model."""
    prompt = fill_prompt(ANSWERLESS_PROOF_PROMPT, problem=problem.strip())
    parsed = sample_json_object(prompt, model)
    proof_problem = parsed.get('problem')
    if not isinstance(proof_problem, str) or not proof_problem.strip():
        raise ValueError(f'Expected proof problem text, got: {parsed!r}')
    return proof_problem.strip()


def failed_record(record: dict, reason: str) -> dict:
    """Return a cleaned dataset row for a problem that cannot continue."""
    cleaned_record = dict(record)
    cleaned_record['FAILED'] = reason
    cleaned_record['theorem'] = None
    return cleaned_record


def formalization_record(record: dict) -> dict:
    """Return a cleaned dataset row ready for later formalization."""
    cleaned_record = dict(record)
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
    derived['id'] = f'{record["id"]}__{suffix}'
    derived['problem'] = problem
    derived['question_type'] = f'derived_from_{record["question_type"]}'
    derived['answer'] = answer
    return derived


def write_record(output_file: Any, record: dict) -> None:
    """Write one JSONL row."""
    output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def add_proof_problem_rows(
        record: dict,
        problem: str,
        answer: Any,
        model: Qwen3,
        output_file: Any,
        suffix: str,
) -> int:
    """Write answer-aware and answer-free proof problem rows."""
    rows_written = 0
    if answer is not None and str(answer).strip() != '':
        with_answer = proof_problem_with_answer(problem, answer, model)
        write_record(
                output_file,
                formalization_record(
                        derived_record(
                                record,
                                with_answer,
                                answer,
                                f'{suffix}_with_answer',
                        )
                ),
        )
        rows_written += 1

    without_answer = proof_problem_without_answer(problem, model)
    write_record(
            output_file,
            formalization_record(
                    derived_record(
                            record,
                            without_answer,
                            None,
                            f'{suffix}_without_answer',
                    )
            ),
    )
    rows_written += 1
    return rows_written

def main(
        input_path: Path = FILTERED_NUMINA_MATH_1_5_PATH,
        output_path: Path = CLEANED_NUMINA_MATH_1_5_PATH,
) -> None:
    """Run the data cleaning pipeline over the filtered dataset."""
    qwen = Qwen3_8B(quantization='8bit')
    qwen.load()
    print(f'Loaded {qwen.model_name} from {qwen.model_dir} on {qwen.device}.')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows_read = 0
    missing_information_rows = 0
    errored_rows = 0
    rows_written = 0
    timers: dict[str, float] = {}

    def add_timer(name: str, seconds: float) -> None:
        timers[name] = timers.get(name, 0.0) + seconds

    def timed(name: str, function: Any, *args: Any) -> Any:
        start = perf_counter()
        try:
            return function(*args)
        finally:
            add_timer(name, perf_counter() - start)

    with input_path.open(encoding='utf-8') as input_file:
        with output_path.open('w', encoding='utf-8') as output_file:
            for line in input_file:
                line = line.strip()
                if not line:
                    continue

                row_start = perf_counter()
                rows_read += 1
                record = json.loads(line)

                try:
                    missing_information = timed(
                            'has_missing_information',
                            has_missing_information,
                            record['problem'],
                            qwen,
                    )
                except Exception:
                    errored_rows += 1
                    add_timer('row_iteration', perf_counter() - row_start)
                    continue

                if missing_information:
                    write_record(
                            output_file,
                            failed_record(record, 'missing_information'),
                    )
                    missing_information_rows += 1
                    rows_written += 1
                    add_timer('row_iteration', perf_counter() - row_start)
                    continue

                try:
                    question_type = record.get('question_type')

                    if question_type == 'proof':
                        write_record(output_file, formalization_record(record))
                        rows_written += 1
                        continue

                    if question_type == 'math-word-problem':
                        if timed(
                                'is_multi_part_problem',
                                is_multi_part_problem,
                                record['problem'],
                                qwen,
                        ):
                            problems = timed(
                                    'split_into_several_problems',
                                    split_into_several_problems,
                                    record,
                                    qwen,
                            )
                        else:
                            problems = [
                                    {
                                            'problem': record['problem'],
                                            'answer': record.get('answer'),
                                    }
                            ]

                        for problem_ix, split_problem in enumerate(problems):
                            rows_written += timed(
                                    'add_proof_problem_rows',
                                    add_proof_problem_rows,
                                    record,
                                    split_problem['problem'],
                                    split_problem.get('answer'),
                                    qwen,
                                    output_file,
                                    f'part_{problem_ix}',
                            )
                        continue

                    if question_type == 'MCQ':
                        if timed(
                                'has_several_statements',
                                has_several_statements,
                                record['problem'],
                                qwen,
                        ):
                            problems = timed(
                                    'split_into_several_problems',
                                    split_into_several_problems,
                                    record,
                                    qwen,
                            )
                        else:
                            problems = [
                                    timed('remove_options', remove_options, record, qwen)
                            ]

                        for problem_ix, split_problem in enumerate(problems):
                            rows_written += timed(
                                    'add_proof_problem_rows',
                                    add_proof_problem_rows,
                                    record,
                                    split_problem['problem'],
                                    split_problem.get('answer'),
                                    qwen,
                                    output_file,
                                    f'mcq_{problem_ix}',
                            )
                        continue

                    write_record(output_file, failed_record(record, 'unsupported_type'))
                    rows_written += 1
                except Exception:
                    errored_rows += 1
                    continue
                finally:
                    add_timer('row_iteration', perf_counter() - row_start)

    print(
            'Finished data cleaning: '
            f'{rows_read} rows read, '
            f'{missing_information_rows} marked missing information, '
            f'{errored_rows} errors, '
            f'{rows_written} rows written.'
    )
    print(f'Timers: {timers}')


if __name__ == '__main__':
    main()
