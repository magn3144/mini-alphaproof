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

MATH_WORD_MULTI_PART_PROMPT = """<task>
Decide whether a math word problem contains multiple separate subproblems that
should be split before formalization.
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

PROOF_SEVERAL_QUESTIONS_PROMPT = r"""<task>
Decide whether a proof problem contains multiple separate proof questions that
should be split before formalization.
</task>

<instructions>
Answer only YES or NO.
Answer YES when the problem asks for multiple numbered or lettered proof tasks,
such as (1), (2), (a), (b), or several separate prove/show questions.
Answer NO when the problem is one proof task, even if the theorem has multiple
hypotheses, multiple cases, or several statements that naturally belong to one
conclusion.
</instructions>

<examples>
<example>
<problem>
(a) Prove that if $n$ is odd, then $n^2$ is odd. (b) Prove that if $n^2$ is
odd, then $n$ is odd.
</problem>
<answer>YES</answer>
</example>

<example>
<problem>
Prove that for every integer $n$, if $n$ is odd, then $n^2$ is odd and
$n^2 \equiv 1 \pmod 8$.
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

MATH_WORD_SPLIT_PROBLEM_PROMPT = r"""<task>
Split a math word problem into separate standalone problems.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problems": [{"problem": "standalone problem text", "answer": "extracted answer if available, otherwise null"}]}

Each output problem must be self contained.
Split each numbered, lettered, or otherwise separate part into one problem.
Use the provided answer only to assign each split problem's answer when possible.
If a split problem doesnt have an answer assigned, use null for its answer.
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

<problem>
{problem}
</problem>
<answer>{answer}</answer>
"""

MCQ_SPLIT_PROBLEM_PROMPT = r"""<task>
Split a multiple-choice math problem whose options are separate statements into
standalone true/false problems.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problems": [{"problem": "standalone statement text", "answer": "true"}]}

Each output problem must be self contained.
Turn each statement option into one true/false problem.
Use the provided answer to decide which statements are true or false.
</instructions>

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

PROOF_SPLIT_PROBLEM_PROMPT = r"""<task>
Split a proof problem into separate standalone proof problems.
</task>

<instructions>
Return only valid JSON.
Use this exact schema:
{"problems": [{"problem": "standalone proof problem text"}]}

Each output problem must be self contained.
Split each numbered, lettered, or otherwise separate proof task into one proof
problem. Keep each output as a proof problem. Do not include a solution or
explanation. Do not include an answer field.
</instructions>

<example>
<problem>
NT2 BUL

A positive integer is called a repunit, if it is written only by ones. The
repunit with $n$ digits will be denoted by $\underbrace{11 \ldots 1}_{n}$.
Prove that:

a) the repunit $\underbrace{11 \ldots 1}_{n}$ is divisible by 37 if and only
if $n$ is divisible by 3;

b) there exists a positive integer $k$ such that the repunit
$\underbrace{11 \ldots 1}_{n}$ is divisible by 41 if and only if $n$ is
divisible by $k$.
</problem>
<output>
{"problems": [{"problem": "A positive integer is called a repunit, if it is written only by ones. The repunit with $n$ digits will be denoted by $\\underbrace{11 \\ldots 1}_{n}$. Prove that the repunit $\\underbrace{11 \\ldots 1}_{n}$ is divisible by 37 if and only if $n$ is divisible by 3."}, {"problem": "A positive integer is called a repunit, if it is written only by ones. The repunit with $n$ digits will be denoted by $\\underbrace{11 \\ldots 1}_{n}$. Prove that there exists a positive integer $k$ such that the repunit $\\underbrace{11 \\ldots 1}_{n}$ is divisible by 41 if and only if $n$ is divisible by $k$."}]}
</output>
</example>

<problem>
{problem}
</problem>
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

TRIVIAL_EXISTENCE_THEOREM_PROMPT = r"""<task>
Decide whether an answerless existence proof problem is mathematically trivial.
</task>

<instructions>
Answer only YES or NO.
Answer YES if the requested existence follows immediately from the wording, without
solving any substantive part of the original problem. This includes merely claiming
that a well-defined expression has a value, that a finite set has a number of
elements, or that the original question has an answer.
Answer NO if a proof must construct an object satisfying mathematical constraints,
show that such an object exists, or show that an extremum is attained.
Judge the proof problem as written. Its existence claim can be trivial even when
computing the omitted answer would be difficult.
</instructions>

<examples>
<example>
<existence_theorem>
Prove that there exists a value of $(3x-4)^2$ when $x=-2$.
</existence_theorem>
<answer>YES</answer>
</example>

<example>
<existence_theorem>
Prove that there exists a number of distinct positive factors of 81.
</existence_theorem>
<answer>YES</answer>
</example>

<example>
<existence_theorem>
Prove that there exist positive rational numbers $a$ and $b$ such that
$\sqrt[b]{a}=ab$.
</existence_theorem>
<answer>NO</answer>
</example>

<example>
<existence_theorem>
Prove that there exists a least positive four-digit integer $x$ satisfying
$x \equiv 1 \pmod 3$, $2x+5 \equiv 11 \pmod 8$,
$-3x+2 \equiv 2x \pmod {13}$, and $5x-3 \equiv 12 \pmod 7$.
</existence_theorem>
<answer>NO</answer>
</example>
</examples>

<existence_theorem>
{existence_theorem}
</existence_theorem>
"""
