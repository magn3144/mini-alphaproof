"""
Comprehensive tests for Lean 4 environment.
Tests basic functionality without requiring Mathlib.
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment import (
    Lean4Environment,
    TacticFailedException,
    TheoremSyntaxException,
    Lean4Exception
)


class TestLean4Environment:
    """Test suite for Lean4Environment class."""

    def test_simple_proof_with_rfl(self):
        """Test a simple proof using rfl."""
        env = Lean4Environment("theorem ex1 : ∀ n : Nat, n + 0 = n := by intro n; sorry")

        # Check initial state
        assert not env.is_complete()
        assert env.current_state.num_goals() == 1

        # Apply rfl
        result = env.step("rfl")
        assert result.success
        assert result.proof_complete
        assert env.is_complete()

        env.close()

    def test_intro_tactic(self):
        """Test intro tactic on forall statement."""
        env = Lean4Environment("theorem ex2 : ∀ n m : Nat, n + 0 = m + 0 → n = m := by sorry")

        # Check initial state
        assert env.current_state.num_goals() == 1
        initial_target = env.current_state.goals[0].target
        assert "∀" in initial_target

        # Apply intro n
        result = env.step("intro n")
        assert result.success
        assert not result.proof_complete
        assert env.current_state.num_goals() == 1

        # Check that hypothesis was added
        assert len(env.current_state.goals[0].hypotheses) >= 1

        env.close()

    def test_trivial_tactic(self):
        """Test trivial tactic for True."""
        env = Lean4Environment("theorem ex3 : True := by sorry")

        # Apply trivial
        result = env.step("trivial")
        assert result.success
        assert result.proof_complete
        assert env.is_complete()

        env.close()

    def test_failed_tactic(self):
        """Test that invalid tactics fail gracefully."""
        env = Lean4Environment("theorem ex4 : True := by sorry")

        # Try invalid tactic
        result = env.step("intro")
        assert not result.success
        assert result.error_message is not None
        assert "intro" in result.error_message.lower() or "failed" in result.error_message.lower()

        env.close()

    def test_invalid_theorem_syntax(self):
        """Test that invalid theorem syntax raises exception."""
        with pytest.raises(TheoremSyntaxException):
            env = Lean4Environment("this is not a valid theorem")

    def test_backtracking(self):
        """Test backtracking functionality."""
        env = Lean4Environment("theorem ex5 : ∀ n : Nat, n = n := by sorry")

        # Apply intro
        env.apply_tactic("intro n")
        assert env.steps_taken == 1

        # Apply another intro (should fail, but let's try)
        env.apply_tactic("rfl")
        assert env.steps_taken == 2

        # Backtrack one step
        state = env.backtrack(1)
        assert env.steps_taken == 1
        assert state.num_goals() == 1

        # Backtrack to beginning
        state = env.backtrack(1)
        assert env.steps_taken == 0

        env.close()

    def test_reset(self):
        """Test environment reset."""
        env = Lean4Environment("theorem ex6 : True := by sorry")

        # Apply tactic
        env.apply_tactic("trivial")
        assert env.is_complete()

        # Reset
        env.reset()
        assert not env.is_complete()
        assert env.steps_taken == 0
        assert len(env.tactics_applied) == 0

        env.close()

    def test_statistics(self):
        """Test statistics tracking."""
        env = Lean4Environment("theorem ex7 : ∀ n : Nat, n = n := by intro n; sorry")

        # Apply tactics
        env.apply_tactic("rfl")

        # Check stats
        stats = env.get_stats()
        assert stats["steps_taken"] == 1
        assert len(stats["tactics_applied"]) == 1
        assert stats["tactics_applied"][0] == "rfl"
        assert stats["proof_complete"] == True

        env.close()

    def test_context_manager(self):
        """Test using environment as context manager."""
        with Lean4Environment("theorem ex8 : True := by sorry") as env:
            result = env.step("trivial")
            assert result.success
            assert env.is_complete()

    def test_multi_goal_theorem(self):
        """Test theorem with multiple goals."""
        # Use constructor to create multiple goals
        env = Lean4Environment("theorem ex9 : True ∧ True := by sorry")

        # Apply constructor to split into two goals
        result = env.step("constructor")
        if result.success:
            # Should have 2 goals now
            assert env.current_state.num_goals() == 2

            # Solve first goal
            result = env.step("trivial")
            if result.success and env.current_state.num_goals() > 0:
                # Solve second goal
                result = env.step("trivial")
                if result.success:
                    assert env.is_complete()

        env.close()

    def test_hypothesis_parsing(self):
        """Test that hypotheses are parsed correctly."""
        env = Lean4Environment("theorem ex10 : ∀ n : Nat, n = n := by intro n; sorry")

        # After intro, should have hypothesis
        assert env.current_state.num_goals() == 1
        goal = env.current_state.goals[0]

        # Check hypothesis
        assert len(goal.hypotheses) >= 1
        assert "n" in goal.hypotheses[0]
        assert "Nat" in goal.hypotheses[0]

        # Check target
        assert "=" in goal.target

        env.close()

    def test_render_methods(self):
        """Test different rendering methods."""
        env = Lean4Environment("theorem ex11 : True := by sorry")

        # Test human format
        human_str = env.render(mode="human")
        assert "Status:" in human_str
        assert "Goal" in human_str

        # Test LLM format
        llm_str = env.render(mode="llm")
        assert "Goals:" in llm_str or "Goal" in llm_str

        # Test JSON format
        json_str = env.render(mode="json")
        assert "{" in json_str and "}" in json_str

        env.close()


def test_basic_flow():
    """Integration test of basic proof flow."""
    print("\n" + "="*60)
    print("Integration Test: Basic Proof Flow")
    print("="*60)

    env = Lean4Environment("theorem add_zero : ∀ n : Nat, n + 0 = n := by intro n; sorry")

    print("\nInitial state:")
    print(env.render())

    print("\nApplying tactic: rfl")
    result = env.step("rfl")

    if result.success:
        print("✓ Tactic succeeded!")
        print(f"Proof complete: {result.proof_complete}")
        print("\nFinal state:")
        print(env.render())
    else:
        print(f"✗ Tactic failed: {result.error_message}")

    env.close()
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run basic flow test
    test_basic_flow()

    # Run pytest tests
    pytest.main([__file__, "-v"])
