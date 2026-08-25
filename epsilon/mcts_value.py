# mcts_value.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch

from linithenv import LinithEnv, GameState
from epsilon.legacy.model import LinithNet  # we will only use its value output


Action = Tuple  # same convention as in linith_env


@dataclass
class Node:
    to_move: int
    visit_count: int = 0
    value_sum: float = 0.0
    children: Dict[Action, "Node"] = field(default_factory=dict)
    legal_actions: Optional[List[Action]] = None
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def clone_env(env: LinithEnv) -> LinithEnv:
    """Deep-ish copy of the env so each simulation can mutate independently."""
    new_env = LinithEnv(max_moves=env.max_moves)
    if env.state is None:
        new_env.state = None
        return new_env

    s = env.state
    new_env.state = GameState(
        board=s.board.copy(),
        current_player=s.current_player,
        actions_left=s.actions_left,
        done=s.done,
        winner=s.winner,
        move_count=s.move_count,
    )
    return new_env


class ValueMCTS:
    def __init__(self, net: LinithNet, c_uct: float = 1.5, device: str = "cpu"):
        self.net = net
        self.c_uct = c_uct
        self.device = torch.device(device)

    def search(self, root_env: LinithEnv, num_simulations: int) -> Dict[Action, int]:
        """
        Run MCTS from root_env and return visit counts per action for the root.
        """
        if root_env.state is None:
            raise RuntimeError("Env not reset before MCTS.search().")

        root = Node(to_move=root_env.state.current_player)
        self._expand(root, root_env)

        for _ in range(num_simulations):
            self._simulate(root, root_env)

        # Collect visit counts for root children
        visit_counts: Dict[Action, int] = {}
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count

        return visit_counts

    # ---------- internals ----------

    def _simulate(self, root: Node, root_env: LinithEnv) -> None:
        env = clone_env(root_env)
        node = root
        path: List[Node] = [node]

        # 1) Selection
        while node.is_expanded and not env.state.done:  # type: ignore[union-attr]
            if not node.legal_actions:
                break

            best_score = -1e9
            best_action: Optional[Action] = None

            for action in node.legal_actions:
                child = node.children.get(action)
                if child is None:
                    # unvisited child: pretend N=0, Q=0
                    N_sa = 0
                    q_sa = 0.0
                else:
                    N_sa = child.visit_count
                    q_sa = child.q_value

                N_s = max(1, node.visit_count)
                u = self.c_uct * math.sqrt(math.log(N_s + 1) / (N_sa + 1))
                score = q_sa + u

                if score > best_score:
                    best_score = score
                    best_action = action

            if best_action is None:
                break

            # step env
            obs, reward, done, info = env.step(best_action)
            # descend / create child
            child = node.children.get(best_action)
            if child is None:
                child = Node(to_move=env.state.current_player)  # type: ignore[union-attr]
                node.children[best_action] = child
            node = child
            path.append(node)

        # 2) Evaluate leaf
        if env.state.done:  # type: ignore[union-attr]
            # Terminal: value from the perspective of the player to move at *this* node
            winner = env.state.winner  # type: ignore[union-attr]
            if winner is None:
                value = 0.0
            else:
                # if winner == node.to_move → +1, else -1
                value = 1.0 if winner == node.to_move else -1.0
        else:
            # Non-terminal: expand and evaluate with network
            value = self._expand(node, env)

        # 3) Backup
        self._backpropagate(path, value)

    def _expand(self, node: Node, env: LinithEnv) -> float:
        if env.state is None:
            raise RuntimeError("Env not reset before expand.")

        legal = env.legal_actions()
        node.legal_actions = legal
        node.is_expanded = True

        if not legal or env.state.done:  # type: ignore[union-attr]
            # no moves: treat as draw from here
            return 0.0

        # Evaluate board with value net
        state_tensor = torch.from_numpy(env.encode_state()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.net(state_tensor)  # we ignore policy head, only value
        return float(value.squeeze().item())

    def _backpropagate(self, path: List[Node], value: float) -> None:
        """
        Backup value along the path. path[0] is root, path[-1] is leaf node.
        Value is from the perspective of the leaf node's to_move; we flip sign
        at each step when going up.
        """
        v = value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += v
            v = -v  # flip for the other side
