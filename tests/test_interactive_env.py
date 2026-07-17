import threading
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

from alphaproof.core.config import Config
from alphaproof.core.environment import Action, Observation, State
from alphaproof.core.network import Network, NetworkSamplingOutput
from alphaproof.inference.interactive_env import AgentSession, ManualSession


THEOREM = 'theorem interactive_test : True := by sorry'


class FakeManualEnvironment:
    """Provide deterministic branches for manual tree tests."""

    def __init__(self):
        self.steps: list[tuple[int, str]] = []
        self.closed = False

    def initial_state(self, theorem: str) -> State:
        del theorem
        return State(0, 0.0, Observation([]), False, 1)

    def step(
        self,
        state_id: int,
        action: Action,
        tactic_timeout: float = 1.0,
    ) -> State:
        del tactic_timeout
        tactic = str(action)
        self.steps.append((state_id, tactic))
        if tactic == 'bad':
            raise ValueError('Invalid tactic')
        if tactic == 'split':
            return State(10, 0.0, Observation([]), False, 2)
        if tactic == 'focus_goal 0':
            return State(11, 0.0, Observation([]), False, 1)
        if tactic == 'focus_goal 1':
            return State(12, 0.0, Observation([]), False, 1)
        if tactic in ('finish', 'alternative'):
            return State(20 + len(self.steps), 0.0, Observation([]), True, 0)
        return State(30 + len(self.steps), 0.0, Observation([]), False, 1)

    def close(self) -> None:
        self.closed = True


class FakeAgentEnvironment:
    """Reject one generated tactic and solve with another."""

    def __init__(self, *args: Any, **kwargs: Any):
        del args, kwargs

    def initial_state(self, theorem: str) -> State:
        del theorem
        return State(0, 0.0, Observation([]), False, 1)

    def step(
        self,
        state_id: int,
        action: Action,
        tactic_timeout: float = 1.0,
    ) -> State:
        del state_id, tactic_timeout
        if str(action) == 'bad':
            raise ValueError('Invalid tactic')
        return State(1, 0.0, Observation([]), True, 0)

    def close(self) -> None:
        pass


class FakeAgentNetwork:
    def __init__(self, actions: dict[str, float]):
        self.actions = actions

    def sample(self, observation: str) -> NetworkSamplingOutput:
        del observation
        return NetworkSamplingOutput(cast(Any, self.actions), -1.0)


def search_config(num_simulations: int = 1) -> Config:
    return cast(
        Config,
        SimpleNamespace(
            num_simulations=num_simulations,
            prior_temperature=200,
            tactic_timeout=1.0,
            no_legal_actions_value=-40,
            ps_c=0.01,
            ps_alpha=0.6,
            pb_c_base=3200,
            pb_c_init=0.001,
            value_discount=0.99,
        ),
    )


def create_manual() -> tuple[ManualSession, FakeManualEnvironment]:
    environment = FakeManualEnvironment()
    session = ManualSession(
        THEOREM,
        environment=cast(Any, environment),
    )
    return session, environment


def add_action(
    session: ManualSession,
    node_id: str,
    tactic: str,
) -> str:
    session.create_action(node_id)
    snapshot = session.snapshot('manual')

    def find_pending(node: dict[str, Any]) -> dict[str, Any] | None:
        if node['kind'] == 'action' and not node['action']:
            return node
        for child in node['children']:
            found = find_pending(child)
            if found is not None:
                return found
        return None

    record = find_pending(snapshot['root'])
    assert record is not None
    session.update_action(record['id'], tactic)
    return cast(str, record['id'])


class ManualSessionTest(unittest.TestCase):
    def test_pending_action_does_not_call_environment(self):
        session, environment = create_manual()
        root_id = session.snapshot('manual')['root']['id']

        action_id = add_action(session, root_id, 'first')
        action = session.snapshot('manual')['root']['children'][0]

        self.assertEqual(environment.steps, [])
        self.assertEqual(action['id'], action_id)
        self.assertEqual(action['status'], 'pending')

    def test_multiple_sibling_actions_are_kept(self):
        session, _ = create_manual()
        root_id = session.snapshot('manual')['root']['id']
        first_id = add_action(session, root_id, 'first')
        session.execute_action(first_id)
        second_id = add_action(session, root_id, 'alternative')
        session.execute_action(second_id)

        snapshot = session.snapshot('manual')

        self.assertEqual(len(snapshot['root']['children']), 2)
        self.assertEqual(
            [child['status'] for child in snapshot['root']['children']],
            ['succeeded', 'succeeded'],
        )
        self.assertTrue(snapshot['complete'])

    def test_invalid_action_remains_visible_and_does_not_block_solution(self):
        session, _ = create_manual()
        root_id = session.snapshot('manual')['root']['id']
        bad_id = add_action(session, root_id, 'bad')
        session.execute_action(bad_id)
        good_id = add_action(session, root_id, 'finish')
        session.execute_action(good_id)

        snapshot = session.snapshot('manual')

        self.assertEqual(snapshot['root']['children'][0]['status'], 'invalid')
        self.assertIn('Invalid tactic', snapshot['root']['children'][0]['error'])
        self.assertEqual(snapshot['root']['children'][1]['status'], 'succeeded')
        self.assertTrue(snapshot['complete'])

    def test_and_node_requires_every_focused_goal(self):
        session, _ = create_manual()
        root_id = session.snapshot('manual')['root']['id']
        split_id = add_action(session, root_id, 'split')
        session.execute_action(split_id)
        snapshot = session.snapshot('manual')
        and_node = snapshot['root']['children'][0]['children'][0]

        self.assertEqual(and_node['nodeType'], 'AND')
        self.assertEqual(len(and_node['children']), 2)

        first_goal = and_node['children'][0]
        first_id = add_action(session, first_goal['id'], 'finish')
        session.execute_action(first_id)
        self.assertFalse(session.snapshot('manual')['complete'])

        snapshot = session.snapshot('manual')
        second_goal = snapshot['root']['children'][0]['children'][0]['children'][1]
        second_id = add_action(session, second_goal['id'], 'finish')
        session.execute_action(second_id)

        solved = session.snapshot('manual')
        self.assertTrue(solved['complete'])
        self.assertIn('split', solved['proofScript'])


class AgentSessionTest(unittest.TestCase):
    def run_session(
        self,
        actions: dict[str, float],
        cancel: bool = False,
        final_result: bool = True,
    ) -> dict[str, Any]:
        with (
            patch(
                'alphaproof.inference.interactive_env.Environment',
                FakeAgentEnvironment,
            ),
            patch(
                'alphaproof.inference.interactive_env.final_check',
                return_value=final_result,
            ),
        ):
            session = AgentSession(
                THEOREM,
                search_config(),
                cast(Network, FakeAgentNetwork(actions)),
                seed=0,
            )
            if cancel:
                session.cancel()
            finished = threading.Event()
            session.start(finished.set)
            self.assertTrue(finished.wait(2))
            assert session.thread is not None
            session.thread.join()
            return session.snapshot('agent')

    def test_invalid_agent_tactic_is_red_and_search_continues(self):
        snapshot = self.run_session({'bad': 0.0, 'finish': 0.0})

        self.assertEqual(snapshot['status'], 'solved')
        self.assertEqual(
            [child['status'] for child in snapshot['root']['children']],
            ['invalid', 'succeeded'],
        )

    def test_agent_reports_exhaustion(self):
        snapshot = self.run_session({'bad': 0.0})

        self.assertEqual(snapshot['status'], 'exhausted')
        self.assertEqual(snapshot['root']['children'][0]['status'], 'invalid')

    def test_agent_cancels_before_next_evaluation(self):
        snapshot = self.run_session({'finish': 0.0}, cancel=True)

        self.assertEqual(snapshot['status'], 'cancelled')
        self.assertEqual(snapshot['root']['children'], [])

    def test_agent_reports_failed_final_verification(self):
        snapshot = self.run_session({'finish': 0.0}, final_result=False)

        self.assertEqual(snapshot['status'], 'error')
        self.assertFalse(snapshot['complete'])
        self.assertIn('final verification', snapshot['error'])


if __name__ == '__main__':
    unittest.main()
