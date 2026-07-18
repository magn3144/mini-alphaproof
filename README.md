# AlphaProof Training Project (unofficial)
AlphaProof is a state-of-the-art reinforcement learning system developed by Google DeepMind, designed to formalize and solve complex, IMO-level mathematical problems. As the full source code has not been publicly released, I am implementing the training and inference pipelines for AlphaProof as part of my master's thesis.

This implementation is based on the official pseudocode provided by the DeepMind team. Notably, I have chosen to omit TTRL as it falls outside the scope of my research and requires computational resources beyond the scale of this project.

Disclaimers:
 - **Work in Progress**: This repository is currently under active development.
 - **Implementation Variance**: This version may differ from the original DeepMind implementation and is not guaranteed to achieve the same performance benchmarks.
 - **Academic Use**: This project is intended strictly for academic and research purposes.

## Training

Download and cheaply clean the already-formalized NuminaMath-LEAN statements:

```bash
python -m alphaproof.data.numina_math_lean download
python -m alphaproof.data.numina_math_lean clean
```

This writes the raw parquet and cleaned theorem JSONL under `data/dataset/`.
Lean compatibility is checked lazily when training initializes each theorem;
statements that do not elaborate are recorded as invalid and not sampled again.

Start a reinforcement-learning run from the default SFT checkpoint:

```bash
python -m alphaproof.training.train rl_run --wandb-mode online
```

The default single-actor profile runs 32 games with 800 simulations each,
then performs 10,000 learner updates with batch size 8. These defaults target
roughly a one-day run on a CUDA accelerator, but wall time depends on theorem
difficulty and hardware. Use `--wandb-mode disabled` to train without W&B.

## Inference

Run inference from either an SFT run or an RL run:

```bash
python -m alphaproof.inference.infer \
  --theorem 'theorem alphaproof_example : True := by sorry' \
  --num-simulations 800
```

Use `--theorem-file theorem.lean` instead of `--theorem` to read the theorem
from a file. The command prints a complete Lean declaration when it finds and
verifies a proof, and exits with status 1 when no verified proof is found.

## Interactive environment

Start the backend with the model used by the Agent tab:

```bash
python -m alphaproof.inference.interactive_env \
  --num-simulations 800 \
  --num-sampled-actions 3
```

Then start the React client in another terminal:

```bash
cd frontend
npm run dev
```

The Manual tab supports saved tactic branches and full AND-OR proof trees. The
Agent tab runs the normal MCTS search and polls the same tree view while the
search is active.
