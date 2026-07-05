# Notes for CODEX

- Write implementations as simple as possible, reusing exsisting functionality and avoiding ovverly complicated structures.
- The actual project is in "alphaproof/".
- "pseudocode.py" is not part of the actual project. It is used as a guide for how to structure the code. The code in "alphaproof/" should closely follow the structure of "pseudocode.py".
- When you want to use py_compile, dont place the cache files in this repo, its annoying.
- Always place imports at the top of files. Not inside functions.


# Design choices

These are the design choices we have made so far.
They might differ from the pseudocde, which is ok.

 - Used LeanTree for interacting with Lean 4.
 - Replay buffer samples uniformly.
 - Computes tactic prior by summing token logprobs. This is used as the prior in PUCT.
 - Value head uses mean pooled encoder output.
 - Value head is currently linear layer.
 - TODO: Data for each run should be stored like this, so runs can be resumed:
 runs/
   0/
    config.json
    matchmaker_stats.json
    results.jsonl
    replay_buffer.jsonl
    checkpoints/
      step_0001000.pt
      step_0002000.pt
      step_0003000.pt
 - Actors are run sequentially for now, for a specific amount of iterations each.
 - Encoder called again every time a node is expanded again.
 - The autoformalizer only generates one lean problem per natural language problem.
 - The dataset used is AI-MO/NuminaMath-1.5. Autoformalization is used in this project to convert problems to Lean.
 - Currently we are keeping only the easiest problems to formalize, but we might later improve the formalization pipeline to include more problems.