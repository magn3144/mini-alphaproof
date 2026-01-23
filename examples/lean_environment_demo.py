"""
Interactive demo of the Lean 4 environment.

This script demonstrates the key features of the Lean4Environment class
for interactive theorem proving.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import Lean4Environment


def demo_basic_usage():
    """Demonstrate basic environment usage."""
    print("=" * 70)
    print("DEMO 1: Basic Environment Usage")
    print("=" * 70)

    # Create environment with a simple theorem
    print("\n1. Creating environment with theorem: 1 + 1 = 2")
    env = Lean4Environment("theorem simple : 1 + 1 = 2 := by sorry", verbose=False)

    # Display initial state
    print("\n2. Initial proof state:")
    print(env.render(use_color=True))

    # Apply tactic
    print("\n3. Applying tactic: norm_num")
    success = env.apply_tactic("norm_num")

    if success:
        print("   ✓ Tactic succeeded!")
    else:
        print("   ✗ Tactic failed!")

    # Display final state
    print("\n4. Final proof state:")
    print(env.render(use_color=True))

    # Show statistics
    print("\n5. Proof statistics:")
    stats = env.get_stats()
    print(f"   Steps taken: {stats['steps_taken']}")
    print(f"   Proof complete: {stats['proof_complete']}")

    env.close()


def demo_interactive_proof():
    """Demonstrate multi-step interactive proof."""
    print("\n\n" + "=" * 70)
    print("DEMO 2: Interactive Multi-Step Proof")
    print("=" * 70)

    # More complex theorem
    print("\n1. Theorem: ∀ n : Nat, n + 0 = n")
    env = Lean4Environment(
        "theorem nat_add_zero : ∀ n : Nat, n + 0 = n := by sorry",
        verbose=False
    )

    print("\n2. Initial state:")
    print(env.render(use_color=True))

    # Interactive proof steps
    tactics = ["intro n", "rfl"]

    for i, tactic in enumerate(tactics, 1):
        print(f"\n3.{i} Applying: {tactic}")
        result = env.step(tactic)

        if result.success:
            print(f"   ✓ Success! {result.new_state.num_goals()} goals remaining")
            if result.proof_complete:
                print("   🎉 Proof complete!")
        else:
            print(f"   ✗ Failed: {result.error_message}")

        print("\n   Current state:")
        print(env.get_state_string())

    env.close()


def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n\n" + "=" * 70)
    print("DEMO 3: Error Handling and Recovery")
    print("=" * 70)

    env = Lean4Environment("theorem ex : True := by sorry", verbose=False)

    print("\n1. Trying invalid tactic: 'invalid_tactic'")
    result = env.step("invalid_tactic")

    if not result.success:
        print(f"   ✓ Correctly failed: {result.error_message[:100]}...")

    print("\n2. Environment still usable after error")
    print(f"   Current goals: {env.current_state.num_goals()}")

    print("\n3. Applying correct tactic: 'trivial'")
    result = env.step("trivial")

    if result.success and result.proof_complete:
        print("   ✓ Proof complete!")
        print(env.render(use_color=True))

    env.close()


def demo_backtracking():
    """Demonstrate backtracking functionality."""
    print("\n\n" + "=" * 70)
    print("DEMO 4: Backtracking")
    print("=" * 70)

    env = Lean4Environment(
        "theorem ex : ∀ n : Nat, n = n := by sorry",
        verbose=False
    )

    print("\n1. Initial state:")
    print(env.get_state_string())

    print("\n2. Applying tactics: 'intro n', 'intro m' (wrong)")
    env.apply_tactic("intro n")
    print(f"   After 'intro n': {env.current_state.num_goals()} goals")

    # Try wrong tactic
    result = env.step("intro m")
    if not result.success:
        print(f"   'intro m' failed (expected)")

    print("\n3. Applying correct tactic: 'rfl'")
    result = env.step("rfl")

    if result.success and result.proof_complete:
        print("   ✓ Proof complete!")

    print("\n4. Final statistics:")
    stats = env.get_stats()
    print(f"   Total steps: {stats['steps_taken']}")
    print(f"   Tactics used: {', '.join(stats['tactics_applied'])}")

    env.close()


def demo_different_formats():
    """Demonstrate different output formats."""
    print("\n\n" + "=" * 70)
    print("DEMO 5: Output Formats")
    print("=" * 70)

    env = Lean4Environment(
        "theorem ex : ∀ n m : Nat, n + m = m + n := by sorry",
        verbose=False
    )

    print("\n1. Human-readable format (with colors):")
    print(env.render(mode="human", use_color=True))

    print("\n2. LLM-friendly format:")
    print(env.current_state.to_string(format="llm"))

    print("\n3. JSON format:")
    print(env.current_state.to_string(format="json"))

    env.close()


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  Lean 4 Environment Interactive Demo".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")

    try:
        demo_basic_usage()
        demo_interactive_proof()
        demo_error_handling()
        demo_backtracking()
        demo_different_formats()

        print("\n\n" + "=" * 70)
        print("All demos completed successfully!")
        print("=" * 70)
        print("\nNext steps:")
        print("  - Try creating your own theorems")
        print("  - Explore different tactics (intro, rfl, norm_num, simp, etc.)")
        print("  - Build RL agents using the step() API")
        print("  - See src/environment/README.md for full documentation")
        print()

    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install required dependencies:")
        print("  pip install -r requirements.txt")
        print()

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nThis may occur if:")
        print("  - Lean 4 server fails to start")
        print("  - lean-interact is not properly installed")
        print("  - Network issues (lean-interact downloads Lean on first run)")
        print("\nTry:")
        print("  pip install --upgrade lean-interact")
        print()


if __name__ == "__main__":
    main()
