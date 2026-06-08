# AlphaProof Todo List

## 1. Core Architecture & Refactoring
- [x] Remove AP types.
- [x] Fix definition imports: `Node`, `Game`, `Player`, `Action`, `Observation`, `Config`, `Theorem`.
- [x] Fix broken imports across files (`src/training.py`, `src/mcts.py`, `src/network.py`).
- [ ] Implement proper tokenization for `Observation` and `Action` converting strings to tensors in the neural network or `ReplayBuffer`.
- [ ] Implement a full ReplayBuffer sampling (currently hardcoded to return `self.buffer[0]`).

## 2. Training Loop & Pipeline
- [ ] Implement the main Actor-Learner orchestrator loop in `scripts/train.py` (it currently halts at the `TODO: Implement training loop`).
- [ ] Implement evaluation logic in `scripts/evaluate.py`.
- [ ] Add parsing for the Mathlib SFT dataset.
- [ ] Update the training data sampler to mix batches at a 90% RL replay buffer and 10% Mathlib SFT distribution as described in the paper.
- [ ] Build distributed architecture (e.g., via multiprocessing or `ray`) connecting the Matchmaker, remote Actors scoring simulations, and the central Learner.

## 3. Auto-Formalization Pipeline
- [ ] Build asynchronous pipeline to generate massive scale self-play datasets (~80M problems).
- [ ] Implement iterative refinement (STaR-like) pipeline for the formalization models using `AlphaProof` itself to check logic equivalence.
- [ ] Expand the initial math problems dataset collection script.

## 4. Test-Time RL (TTRL) & Generalization
- [ ] Implement the `variant_generation.py` hooks for prompting models to create simplified/analogous variants around difficult proof statements.
- [ ] Introduce inference-time execution strategies where TTRL dedicates computation around singular problem variants.
