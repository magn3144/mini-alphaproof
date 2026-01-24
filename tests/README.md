# Manual Testing Guide

## Interactive Testing Script

The `manual_test.py` script provides an interactive command-line interface for manually testing the Lean 4 environment.

### Running the Script

```bash
python tests/manual_test.py
# or
./tests/manual_test.py
```

### Usage

1. **Start the session**: The script will prompt you for a theorem to prove
   - Press Enter to use the default example: `theorem ex : 1 + 1 = 2 := by sorry`
   - Or enter your own theorem

2. **Navigate with tactics**: Enter tactics one at a time
   ```
   > intro n
   > norm_num
   > rfl
   ```

3. **Available commands**:
   - `<tactic>` - Apply any Lean tactic
   - `state` - Show the current proof state
   - `stats` - Show proof statistics
   - `back [n]` - Backtrack n steps (default: 1)
   - `reset [theorem]` - Reset with a new theorem
   - `help` - Show help message
   - `quit` - Exit the session

### Example Session

```
> intro n
✓ Tactic 'intro n' succeeded

======================================================================

Theorem: ex
Steps taken: 1

Status: In progress (1 goal remaining)

Goal 1:
  Hypotheses:
    n : Nat
  Target:
    ⊢ 1 + 1 = 2

======================================================================

> norm_num
✓ Tactic 'norm_num' succeeded

🎉 PROOF COMPLETE! 🎉
```

### Example Theorems to Try

1. Simple arithmetic:
   ```lean
   theorem ex : 1 + 1 = 2 := by sorry
   ```

2. Natural number addition:
   ```lean
   theorem add_comm : ∀ n m : Nat, n + m = m + n := by sorry
   ```

3. List operations:
   ```lean
   theorem list_append_nil : ∀ (α : Type) (l : List α), l ++ [] = l := by sorry
   ```

4. Logic:
   ```lean
   theorem and_comm : ∀ p q : Prop, p ∧ q → q ∧ p := by sorry
   ```

### Tips

- Use `state` frequently to see the current goals and hypotheses
- Use `back` to undo tactics that didn't work as expected
- Use `stats` to see how many steps you've taken
- The proof state shows all goals, hypotheses, and targets
- Press Ctrl+C or type `quit` to exit
