# Lean 4 Environment

A Python interface for interactive theorem proving with Lean 4, designed for both human interaction and reinforcement learning agents.

## Features

- **Simple API**: Easy-to-use interface for applying tactics and checking proof state
- **RL-Ready**: Structured observations and actions for training RL agents
- **Error Handling**: Graceful handling of syntax errors, tactic failures, and timeouts
- **Backtracking**: Support for undoing tactics and exploring proof trees
- **Multiple Output Formats**: Human-readable, LLM-optimized, and JSON formats

## Quick Start

```python
from src.environment import Lean4Environment

# Create an environment with a theorem to prove
env = Lean4Environment(
    theorem_statement="theorem ex : ∀ n : Nat, n + 0 = n := by sorry",
    lean_version="v4.27.0-rc1",
    timeout=30,
    verbose=False
)

# View the current state
print(env.render())

# Apply a tactic
result = env.step("intro n")
if result.success:
    print("Tactic succeeded!")

# Continue proving
env.step("rfl")

# Check if proof is complete
if env.is_complete():
    print("Proof finished!")

# Clean up
env.close()
```

See the full README for more details and examples.
