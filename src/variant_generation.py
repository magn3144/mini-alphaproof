"""Variant generation: LLM-based problem variant generation for AlphaProof."""

# pylint: disable=all

import random


def llm_sample(
    lean_problem: str, example_variants: list[tuple[str, str]],
    temperature: float, persona: str, prompting_strategy: str,
) -> str:
  """Samples formal Lean variants (as a string) via LLM for a Lean problem."""
  raise NotImplementedError()


def extract_lean_problems(sample: str) -> list[str]:
  """Extracts formal Lean problems from an LLM sample."""
  raise NotImplementedError()


def is_valid_syntax(lean_problem: str) -> bool:
  """Checks if a formal Lean problem is syntactically valid."""
  raise NotImplementedError()


def programmatic_augmentation(variants: list[str]) -> list[str]:
  """Generates programmatic variants for a set of formal Lean variants."""
  raise NotImplementedError()


def deduplicate_variants(variants: list[str]) -> list[str]:
  """Deduplicates a set of formal Lean variants."""
  return list(set(variants))


def get_most_interesting_variant(variants: list[str]) -> str:
  """Returns the most interesting variant."""
  raise NotImplementedError()


def example_variants_all() -> list[tuple[str, str]]:
  """Returns a set of example formal Lean problems and their corresponding variants."""
  raise NotImplementedError()


def vary_prompting_strategy() -> bool:
  """Returns whether to vary the prompting strategy."""
  raise NotImplementedError()


def sample_prompting_strategy() -> str:
  """Returns the problem strategy for sampling variants.

  Prompting strategy is one of the following:
  - REFORMULATION: semantically equivalent statement.
  - SIMPLIFICATION: simpler version of the statement.
  - GENERALIZATION: generalization of the statement.
  - SPECIALIZATION: special case of the statement.
  - LEMMA: statement that is useful to solve the original statement.
  - PROOFSTEP: statement that simulates a proof step.
  - PROOF_SIMULATION: variants that simulate a proof.
  - DEFINITION: statement that contains a new definition.
  - ANALOGY: statement that is analogous to the original statement.
  - PARTIALPOINTS: statement that is worth partial points.
  - HINDSIGHT: statement that empirically worked well.
  - PROBLEM_DECOMPOSITION: decomposing a statement into parts.
  - PART2PART: different part of the same underlying statement.
  - PROBLEM2PART: decomposing a statement into a part.
  """
  raise NotImplementedError()


def sample_variants(lean_problem: str) -> list[str]:
  """Samples a set of formal Lean variants for a formal Lean problem.

  For each variant, we call this function multiple times (possibly in parallel)
  to generate enough variants.

  Args:
    lean_problem: The Lean problem to sample variants for.
  Returns:
    A list of sampled Lean variants.
  """
  example_variants = example_variants_all()
  variants = []
  current_problem = lean_problem
  num_evolutions = random.choice([1, 3, 6, 10, 15])
  vary_prompting = vary_prompting_strategy()
  prompt_strategy = sample_prompting_strategy()
  for _ in range(num_evolutions):
    temperature = random.choice([0.5, 1.0, 1.5])
    persona = random.choice(["IMO winner", "Putnam winner"])
    if vary_prompting:
      prompt_strategy = sample_prompting_strategy()
    variant_sample = llm_sample(
        lean_problem=current_problem,
        example_variants=example_variants,
        temperature=temperature,
        persona=persona,
        prompting_strategy=prompt_strategy,
    )
    extracted_lean_problems = extract_lean_problems(variant_sample)
    current_lean_variants = [
        extracted_lean_problem
        for extracted_lean_problem in extracted_lean_problems
        if is_valid_syntax(extracted_lean_problem)
    ]
    if not current_lean_variants:
      break
    variants.extend(current_lean_variants)
    current_problem = get_most_interesting_variant(
        variants=current_lean_variants
    )
    programmatic_variants = programmatic_augmentation(current_lean_variants)
    variants.extend(
        programmatic_variant
        for programmatic_variant in programmatic_variants
        if is_valid_syntax(programmatic_variant)
    )
  variants = deduplicate_variants(variants)
  return variants
