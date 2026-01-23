# Lean 4 Environment

Interactive theorem proving environment for Lean 4, supporting both human interaction and RL agent training.

## Installation

```bash
pip install lean-interact
```

## Quick Start

### Basic Usage

```python
from src.environment import Lean4Environment

# Create environment with a theorem
env = Lean4Environment("theorem ex : 1 + 1 = 2 := by sorry")

# View current state
print(env.render())

# Apply a tactic
env.apply_tactic("norm_num")

# Check if proof is complete
if env.is_complete():
    print("Proof finished!")

# Clean up
env.close()
```

### Multi-Step Proofs

```python
env = Lean4Environment("theorem nat_add_zero : ∀ n : Nat, n + 0 = n := by sorry")

# Step 1: Introduce variable
env.apply_tactic("intro n")
print(env.render())

# Step 2: Apply reflexivity
env.apply_tactic("rfl")
print(env.render())

print(f"Complete: {env.is_complete()}")
```

### Context Manager

```python
with Lean4Environment("theorem ex : True := by sorry") as env:
    env.apply_tactic("trivial")
    print(env.render())
# Automatically cleaned up
```

## API for RL Agents

### Structured Step Interface

```python
env = Lean4Environment("theorem ex : 2 + 2 = 4 := by sorry")

# Apply tactic and get structured result
result = env.step("norm_num")

if result.success:
    print(f"Tactic succeeded!")
    print(f"Goals remaining: {result.new_state.num_goals()}")
    print(f"Proof complete: {result.proof_complete}")
else:
    print(f"Tactic failed: {result.error_message}")
```

### State Access

```python
# Get current state as string
state_str = env.get_state_string()

# Get structured state object
state = env.current_state

# Check number of goals
num_goals = state.num_goals()

# Access individual goals
for i, goal in enumerate(state.goals):
    print(f"Goal {i+1}:")
    print(f"  Hypotheses: {goal.hypotheses}")
    print(f"  Target: {goal.target}")
```

### Backtracking

```python
env = Lean4Environment("theorem ex : True := by sorry")

# Apply some tactics
env.apply_tactic("intro")  # Wrong tactic
env.apply_tactic("split")  # Another wrong tactic

# Backtrack 2 steps
env.backtrack(steps=2)

# Try correct tactic
env.apply_tactic("trivial")
```

### Statistics

```python
stats = env.get_stats()
print(f"Steps taken: {stats['steps_taken']}")
print(f"Tactics applied: {stats['tactics_applied']}")
print(f"Goals remaining: {stats['num_goals']}")
print(f"Proof complete: {stats['proof_complete']}")
```

## Configuration

Configure environment in `configs/train_config.yaml`:

```yaml
environment:
  lean_version: "v4.27.0-rc1"
  mathlib_version: null  # Use latest
  timeout: 30  # seconds
  verbose: false
  auto_recover: true
```

Or pass directly:

```python
env = Lean4Environment(
    theorem_statement="theorem ex : 1 + 1 = 2 := by sorry",
    lean_version="v4.27.0-rc1",
    timeout=30,
    verbose=True
)
```

## Output Formats

### Human-Readable (with colors)

```python
print(env.render(mode="human", use_color=True))
```

Output:
```
Theorem: ex
Steps taken: 0

Status: In progress (1 goal remaining)

Goal 1:
  Target:
    ⊢ 1 + 1 = 2
```

### LLM-Friendly

```python
print(env.current_state.to_string(format="llm"))
```

Output:
```
Goals: 1
Goal 1: Target: 1 + 1 = 2
```

### JSON

```python
print(env.current_state.to_string(format="json"))
```

## Error Handling

```python
from src.environment import (
    Lean4Exception,
    TacticFailedException,
    TheoremSyntaxException,
    ServerCrashException,
    TimeoutException
)

try:
    env = Lean4Environment("invalid theorem syntax")
except TheoremSyntaxException as e:
    print(f"Syntax error: {e.error_message}")

try:
    result = env.step("invalid_tactic")
    if not result.success:
        print(f"Tactic failed: {result.error_message}")
except TacticFailedException as e:
    print(f"Error: {e}")
```

## Architecture

### Components

- **Lean4Environment** - Main interface class
- **LeanInterface** - Low-level Lean server communication
- **ProofState** - Represents current proof state
- **Goal** - Individual proof goal
- **TacticResult** - Result of applying a tactic

### State Management

- Current state tracked automatically
- History stack for backtracking
- Automatic cleanup on close/exit

### Server Management

- Auto-recovery from crashes via `AutoLeanServer`
- Configurable timeouts
- Verbose logging option

## Testing

Run the test suite:

```bash
python test_lean_environment.py
```

## Future Enhancements

- Full Gymnasium API for RL training
- State serialization (pickle save/load)
- Batch theorem processing
- Tactic suggestion system
- Vectorized environments for parallel training
- Curriculum learning support
