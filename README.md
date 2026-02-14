# AlphaProof Training Project
AlphaProof is a state-of-the-art reinforcement learning system developed by Google DeepMind, designed to formalize and solve complex, IMO-level mathematical problems. As the full source code has not been publicly released, I am implementing the training and inference pipelines for AlphaProof as part of my master's thesis.

This implementation is based on the official pseudocode provided by the DeepMind team. Notably, I have chosen to omit TTRL (Thought-to-Reward Learning) as it falls outside the scope of my research and requires computational resources beyond the scale of this project.

Disclaimers:
 - **Work in Progress**: This repository is currently under active development.
 - **Implementation Variance**: This version may differ from the original DeepMind implementation and is not guaranteed to achieve the same performance benchmarks.
 - **Academic Use**: This project is intended strictly for academic and research purposes.

## Project Structure

```
AlphaProof/
├── src/                    # Source code
│   ├── models/            # Neural network architectures
│   ├── training/          # Training loops and utilities
│   ├── data/              # Data loading and preprocessing
│   ├── proof_search/      # Proof search algorithms
│   └── utils/             # Utility functions
├── data/                  # Datasets
│   ├── raw/              # Raw theorem data
│   ├── processed/        # Preprocessed data
│   └── theorems/         # Theorem databases
├── checkpoints/          # Model checkpoints
├── configs/              # Configuration files
├── scripts/              # Training and evaluation scripts
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit tests
├── logs/                 # Training logs
├── results/              # Evaluation results
└── docs/                 # Documentation

```
