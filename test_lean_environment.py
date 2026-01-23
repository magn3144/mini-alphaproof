"""
Simple test script for Lean 4 environment.
Run this after installing lean-interact: pip install lean-interact
"""
from src.environment import Lean4Environment


def test_simple_proof():
    """Test a simple arithmetic proof."""
    print("=" * 60)
    print("Test 1: Simple arithmetic proof")
    print("=" * 60)

    env = Lean4Environment("theorem ex1 : 1 + 1 = 2 := by sorry")

    print("\nInitial state:")
    print(env.render())

    print("\n\nApplying tactic: norm_num")
    result = env.apply_tactic("norm_num")

    if result:
        print("✓ Tactic succeeded!")
        print("\nFinal state:")
        print(env.render())
        print(f"\nProof complete: {env.is_complete()}")
    else:
        print("✗ Tactic failed")

    env.close()


def test_multi_step_proof():
    """Test a multi-step proof."""
    print("\n\n" + "=" * 60)
    print("Test 2: Multi-step proof")
    print("=" * 60)

    env = Lean4Environment("theorem ex2 : ∀ n : Nat, n + 0 = n := by sorry")

    print("\nInitial state:")
    print(env.render())

    print("\n\nStep 1: intro n")
    if env.apply_tactic("intro n"):
        print("✓ Success")
        print(env.render())
    else:
        print("✗ Failed")

    print("\n\nStep 2: rfl")
    if env.apply_tactic("rfl"):
        print("✓ Success")
        print(env.render())
        print(f"\nProof complete: {env.is_complete()}")
    else:
        print("✗ Failed")

    env.close()


def test_backtracking():
    """Test backtracking functionality."""
    print("\n\n" + "=" * 60)
    print("Test 3: Backtracking")
    print("=" * 60)

    env = Lean4Environment("theorem ex3 : True := by sorry")

    print("\nInitial state:")
    print(env.render())

    print("\n\nApplying wrong tactic: intro")
    result = env.step("intro")
    if not result.success:
        print(f"✓ Tactic failed as expected: {result.error_message}")

    print("\n\nApplying correct tactic: trivial")
    if env.apply_tactic("trivial"):
        print("✓ Success")
        print(env.render())
        print(f"\nProof complete: {env.is_complete()}")
    else:
        print("✗ Failed")

    env.close()


def test_structured_api():
    """Test the structured API for RL agents."""
    print("\n\n" + "=" * 60)
    print("Test 4: Structured API (for RL)")
    print("=" * 60)

    env = Lean4Environment("theorem ex4 : 2 + 2 = 4 := by sorry")

    print("\nInitial state:")
    state = env.get_state_string()
    print(state)

    print("\n\nUsing step() method:")
    result = env.step("norm_num")

    print(f"Success: {result.success}")
    print(f"Proof complete: {result.proof_complete}")
    if result.success:
        print(f"New state has {result.new_state.num_goals()} goals")

    print("\n\nStatistics:")
    stats = env.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    env.close()


if __name__ == "__main__":
    print("Lean 4 Environment Test Suite")
    print("=" * 60)

    try:
        test_simple_proof()
        test_multi_step_proof()
        test_backtracking()
        test_structured_api()

        print("\n\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install dependencies:")
        print("  pip install lean-interact")

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("This may occur if Lean server fails to start.")
        print("Ensure Lean 4 is properly configured.")
