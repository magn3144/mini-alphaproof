"""Interactive console for the Lean environment."""

from src.environment import Environment, Theorem, State

def main():
    env = Environment()
    header = ""
    statement = "(n : Nat) → n = n ∧ n = n"
    theorem = Theorem(header=header, statement=statement)
    initial = env.initial_state(theorem)

    # Stack of states to prove (one per goal).
    stack: list[State] = [initial]

    while stack:
        state = stack[-1]
        print(f"\n--- State {state.id} (goals remaining: {len(stack)}) ---")
        print(f"Goal:\n{state.observation}\n")

        tactic = input("tactic> ")
        if tactic in ("quit", "exit"):
            return

        result = env.step(state.id, tactic)
        if not result:
            print("Error: invalid tactic\n")
            continue

        # Pop the current goal; push new subgoals.
        stack.pop()
        if not result[0].terminal:
            # Push in reverse so the first goal is on top.
            stack.extend(reversed(result))

    print("\nProof complete!")

if __name__ == "__main__":
    main()
