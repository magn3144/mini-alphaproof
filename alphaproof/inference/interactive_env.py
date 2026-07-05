from __future__ import annotations

import argparse
import json
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ''):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from alphaproof.core.environment import Environment, NodeType, State
from alphaproof.core.game import Game, Node, action_to_tactic, extract_proof_script
from alphaproof.core.helper import replace_sorry_proof, theorem_for_game
from alphaproof.core.paths import LEAN_PROJECT_DIR
from leantree import LeanProject


class InteractiveSession:
    """A user-driven proof game backed by the normal Lean environment."""

    def __init__(self, theorem: str, imports: tuple[str, ...] = ('Mathlib',)):
        self.theorem = theorem
        self.imports = imports
        self.env = Environment(LeanProject(str(LEAN_PROJECT_DIR)), imports=imports)
        self.game = Game(theorem=theorem, disprove=False, num_simulations=0)
        self.nodes: dict[str, Node] = {}
        self.node_ids: dict[int, str] = {}

        try:
            initial_state = self.env.initial_state(theorem)
        except Exception:
            self.env.close()
            raise
        self.game.root = Node(
            action=None,
            observation=initial_state.observation,
            prior=1.0,
            state_id=initial_state.id,
            node_type=NodeType.OR,
            reward=initial_state.reward,
            is_optimal=True,
            is_terminal=initial_state.terminal,
        )
        self._register_node(self.game.root)

    def close(self) -> None:
        self.env.close()

    def apply_action(self, node_id: str, action: str) -> dict[str, Any]:
        """Apply a tactic to a visible leaf node and return the new tree."""
        action = action.strip()
        if not action:
            raise ValueError('Enter a tactic.')

        node = self._node(node_id)
        if node.node_type != NodeType.OR:
            raise ValueError('Actions can only be applied to proof goals.')
        if node.is_terminal:
            raise ValueError('This goal is already proved.')
        if node.expanded():
            raise ValueError('This goal has already been expanded.')

        state = self.env.step(node.state_id, action)
        self._attach_state(node, action, state)
        return self.snapshot()

    def snapshot(self, session_id: str | None = None) -> dict[str, Any]:
        complete = self.is_complete()
        data = {
            'root': self._serialize_node(self.game.root),
            'complete': complete,
            'proofScript': self.proof_script() if complete else None,
        }
        if session_id is not None:
            data['sessionId'] = session_id
        return data

    def is_complete(self) -> bool:
        return self._is_proven(self.game.root)

    def proof_script(self) -> str:
        proof_lines = extract_proof_script(self.game.root)
        theorem = theorem_for_game(self.game.theorem, self.game.disprove)
        declaration = replace_sorry_proof(theorem, proof_lines)
        imports = '\n'.join(f'import {module}' for module in self.imports)
        prefix = f'{imports}\n\n' if imports else ''
        return f'{prefix}{declaration}\n'

    def _attach_state(self, node: Node, action: str, state: State) -> None:
        if state.terminal or state.num_goals <= 1:
            child = self._new_node(
                action=action,
                state=state,
                node_type=NodeType.OR,
            )
            node.children[action] = child
            return

        and_node = self._new_node(
            action=action,
            state=state,
            node_type=NodeType.AND,
        )
        node.children[action] = and_node

        for index in range(state.num_goals):
            focus_action = f'focus_goal {index}'
            focus_state = self.env.step(state.id, focus_action)
            child = self._new_node(
                action=focus_action,
                state=focus_state,
                node_type=NodeType.OR,
            )
            and_node.children[focus_action] = child

    def _new_node(self, action: str, state: State, node_type: NodeType) -> Node:
        node = Node(
            action=action,
            observation=state.observation,
            prior=1.0,
            state_id=state.id,
            node_type=node_type,
            reward=state.reward,
            is_optimal=True,
            is_terminal=state.terminal,
        )
        self._register_node(node)
        return node

    def _register_node(self, node: Node) -> str:
        node_id = uuid.uuid4().hex
        self.nodes[node_id] = node
        self.node_ids[id(node)] = node_id
        return node_id

    def _node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise ValueError(f'Unknown node id: {node_id}') from exc

    def _serialize_node(
        self,
        node: Node,
        edge_action: str | None = None,
    ) -> dict[str, Any]:
        children = []
        for action, child in node.children.items():
            tactic = action_to_tactic(action)
            if child.node_type == NodeType.AND:
                for grandchild in child.children.values():
                    children.append(self._serialize_node(grandchild, tactic))
            else:
                children.append(self._serialize_node(child, tactic))

        return {
            'id': self.node_ids[id(node)],
            'edgeAction': edge_action,
            'observation': self._observation_text(node),
            'goals': self._serialize_goals(node),
            'terminal': node.is_terminal,
            'expanded': node.expanded(),
            'proven': self._is_proven(node),
            'children': children,
        }

    def _serialize_goals(self, node: Node) -> list[dict[str, Any]]:
        goals = getattr(node.observation, 'goals', [])
        return [
            {
                'tag': goal.tag,
                'type': goal.type,
                'hypotheses': [
                    {
                        'name': hypothesis.user_name,
                        'type': hypothesis.type,
                        'value': hypothesis.value,
                    }
                    for hypothesis in goal.hypotheses
                ],
            }
            for goal in goals
        ]

    def _observation_text(self, node: Node) -> str:
        if node.is_terminal:
            return 'Proof closed.'
        return str(node.observation)

    def _is_proven(self, node: Node) -> bool:
        if node.is_terminal:
            return True
        if node.node_type == NodeType.OR:
            return any(
                child.is_optimal and self._is_proven(child)
                for child in node.children.values()
            )
        if node.node_type == NodeType.AND:
            return bool(node.children) and all(
                self._is_proven(child)
                for child in node.children.values()
            )
        raise ValueError(f'Unknown node type: {node.node_type}')


class SessionStore:
    """Thread-safe storage for local interactive proof sessions."""

    def __init__(self, imports: tuple[str, ...] = ('Mathlib',)):
        self.imports = imports
        self._sessions: dict[str, InteractiveSession] = {}
        self._lock = threading.Lock()

    def create(self, theorem: str) -> dict[str, Any]:
        session = InteractiveSession(theorem, imports=self.imports)
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = session
        return session.snapshot(session_id)

    def get(self, session_id: str) -> InteractiveSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError(f'Unknown session id: {session_id}')
        return session

    def step(self, session_id: str, node_id: str, action: str) -> dict[str, Any]:
        session = self.get(session_id)
        with self._lock:
            data = session.apply_action(node_id, action)
        data['sessionId'] = session_id
        return data

    def snapshot(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        return session.snapshot(session_id)

    def reset(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()
        return {'ok': True}

    def close_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()


def make_handler(store: SessionStore) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = 'AlphaProofInteractive/0.1'

        def do_OPTIONS(self) -> None:
            self._send_json(200, {'ok': True})

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            try:
                if parsed.path == '/api/health':
                    self._send_json(200, {'ok': True})
                elif parsed.path == '/api/state':
                    query = parse_qs(parsed.query)
                    session_id = query.get('sessionId', [''])[0]
                    self._send_json(200, store.snapshot(session_id))
                else:
                    self._send_json(404, {'error': 'Not found'})
            except Exception as exc:
                self._send_json(400, {'error': str(exc)})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                data = self._read_json()
                if parsed.path == '/api/start':
                    theorem = str(data.get('theorem', '')).strip()
                    if not theorem:
                        raise ValueError('Enter a theorem.')
                    self._send_json(200, store.create(theorem))
                elif parsed.path == '/api/step':
                    self._send_json(
                        200,
                        store.step(
                            str(data.get('sessionId', '')),
                            str(data.get('nodeId', '')),
                            str(data.get('action', '')),
                        ),
                    )
                elif parsed.path == '/api/reset':
                    self._send_json(200, store.reset(str(data.get('sessionId', ''))))
                else:
                    self._send_json(404, {'error': 'Not found'})
            except Exception as exc:
                self._send_json(400, {'error': str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write(f'{self.address_string()} - {fmt % args}\n')

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                return {}
            body = self.rfile.read(content_length)
            return json.loads(body.decode('utf-8'))

        def _send_json(self, status: int, data: dict[str, Any]) -> None:
            payload = json.dumps(data).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.end_headers()
            self.wfile.write(payload)

    return Handler


def parse_imports(imports: str) -> tuple[str, ...]:
    return tuple(
        module.strip()
        for module in imports.split(',')
        if module.strip()
    )


def run_server(
    host: str = '127.0.0.1',
    port: int = 8000,
    imports: tuple[str, ...] = ('Mathlib',),
) -> None:
    store = SessionStore(imports=imports)
    server = HTTPServer((host, port), make_handler(store))
    print(f'Interactive AlphaProof backend running at http://{host}:{port}')
    try:
        server.serve_forever()
    finally:
        store.close_all()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the interactive AlphaProof backend.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument(
        '--imports',
        default='Mathlib',
        help='Comma-separated Lean modules to import before each theorem.',
    )
    args = parser.parse_args()
    run_server(args.host, args.port, imports=parse_imports(args.imports))


if __name__ == '__main__':
    main()
