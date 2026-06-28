"""Small debugger-friendly exercise for alphaproof.environment.

Run with:
    uv run python scripts/debug_environment.py

Set breakpoints inside this file or in alphaproof/environment.py to step
through Environment.initial_state and Environment.step.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaproof.environment import Environment
from leantree import LeanProject


def main() -> None:
    project = LeanProject('lean_project')

    with Environment(project, imports=()) as environment:
        simple_theorem = 'theorem alpha_debug_simple : True := by sorry'
        simple_initial = environment.initial_state(simple_theorem)
        assert simple_initial.id == 0
        assert not simple_initial.terminal
        assert simple_initial.num_goals == 1

        simple_done = environment.step(simple_initial.id, 'trivial')
        assert simple_done.terminal
        assert simple_done.num_goals == 0

        multi_theorem = 'theorem alpha_debug_multi : True /\\ True := by sorry'
        multi_initial = environment.initial_state(multi_theorem)
        assert not multi_initial.terminal
        assert multi_initial.num_goals == 1

        split_state = environment.step(multi_initial.id, 'constructor')
        assert not split_state.terminal
        assert split_state.num_goals == 2

        left_goal = environment.step(split_state.id, 'focus_goal 0')
        assert not left_goal.terminal
        assert left_goal.num_goals == 1

        left_done = environment.step(left_goal.id, 'trivial')
        assert left_done.terminal

        right_goal = environment.step(split_state.id, 'focus_goal 1')
        assert not right_goal.terminal
        assert right_goal.num_goals == 1

        right_done = environment.step(right_goal.id, 'trivial')
        assert right_done.terminal

    print('Environment debug scenario passed.')


if __name__ == '__main__':
    main()
