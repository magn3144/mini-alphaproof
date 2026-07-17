# AlphaProof Training Project (unofficial)
AlphaProof is a state-of-the-art reinforcement learning system developed by Google DeepMind, designed to formalize and solve complex, IMO-level mathematical problems. As the full source code has not been publicly released, I am implementing the training and inference pipelines for AlphaProof as part of my master's thesis.

This implementation is based on the official pseudocode provided by the DeepMind team. Notably, I have chosen to omit TTRL as it falls outside the scope of my research and requires computational resources beyond the scale of this project.

Disclaimers:
 - **Work in Progress**: This repository is currently under active development.
 - **Implementation Variance**: This version may differ from the original DeepMind implementation and is not guaranteed to achieve the same performance benchmarks.
 - **Academic Use**: This project is intended strictly for academic and research purposes.

## Inference

Run inference from either an SFT run or an RL run:

```bash
python -m alphaproof.inference.infer \
  --run-dir data/runs/sft_mps_smoke \
  --theorem 'theorem alphaproof_example : True := by sorry' \
  --num-simulations 800
```

Use `--theorem-file theorem.lean` instead of `--theorem` to read the theorem
from a file. The command prints a complete Lean declaration when it finds and
verifies a proof, and exits with status 1 when no verified proof is found.
