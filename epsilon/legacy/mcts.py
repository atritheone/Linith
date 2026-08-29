# mcts.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import torch

from linithenv import LinithEnv, GameState, SUN, MOON
from action_space import ACTION_SIZE, decode_action, legal_action_indices
from model import LinithNet


@dataclass
class Node:
    to_move: int                          # SUN or MOON at this node
    prior: float = 0.0                    # P(s,a)
    visit_count: int = 0                  # N(s,a)
    value_sum: float = 0.0                # W(s,a)
    children: Dict[int, "Node"] = field(default_factory=dict)
    legal_actions: Optional[List[int]] = None
    is_expanded: bool = False

    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, action_idx: int, parent_N: int, c_puct: float) -> float:
        child = self.children.get(action_idx)
        if child is None:
            # no stats yet, assume Q=0
            N_sa = 0
            q = 0.0
        else:
            N_sa = child.visit_count
            q = child.q_value() if child.to_move == self.to_move else -child.q_value()

        p = 0.0
        if self.legal_actions is not None and action_idx in self.legal_actions:
            # we store prior in each child node
            if child is not None:
                p = child.prior

        u = c_puct * p * math.sqrt(max(1, parent_N)) / (1 + N_sa)
        return q + u


def clone_env(env: LinithEnv) -> LinithEnv:
    """
    Deep-ish copy of the environment so each simulation can mutate freely.
    """
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
        max_moves=s.max_moves,
    )
    return new_env


class MCTS:
    def __init__(
        self,
        net: LinithNet,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        device: str = "cpu",
    ):
        self.net = net
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.device = torch.device(device)

    def run(self, env: LinithEnv, num_simulations: int) -> np.ndarray:
        """
        Run MCTS from the given root env state.

        Returns:
          visit_counts: np.ndarray of shape (ACTION_SIZE,), counts for each action index.
        """
        if env.state is None:
            raise RuntimeError("Env not reset before MCTS.run()")

        root = Node(to_move=env.state.current_player, prior=1.0)
        # expand root once to get priors and legal actions
        self._expand_node(root, env, add_dirichlet_noise=True)

        for _ in range(num_simulations):
            self._simulate(root, env)

        # build visit count vector from root's children
        visit_counts = np.zeros(ACTION_SIZE, dtype=np.float32)
        for a_idx, child in root.children.items():
            visit_counts[a_idx] = child.visit_count

        return visit_counts

    # ---------- internals ----------

    def _simulate(self, root: Node, root_env: LinithEnv, idx=None) -> None:
        """
        One simulation: selection → expansion → backup.
        """
        env = clone_env(root_env)
        node = root
        search_path: List[Tuple[Node, Optional[int]]] = [(node, None)]

        # 1) Selection
        while node.is_expanded and not env.state.done:  # type: ignore[union-attr]
            if not node.legal_actions:
                break

            parent_N = max(1, node.visit_count)
            # Pick action with max PUCT
            best_score = -1e9
            best_action_idx = None

            for a_idx in node.legal_actions:
                score = node.puct_score(a_idx, parent_N, self.c_puct)
                if score > best_score:
                    best_score = score
                    best_action_idx = a_idx

            if best_action_idx is None:
                break

            # step environment
            action = decode_action(best_action_idx, env)
            obs, reward, done, info = env.step(action)

            # move to child node (create if needed)
            child = node.children.get(best_action_idx)
            if child is None:
                child = Node(
                    to_move=env.state.current_player,  # type: ignore[union-attr]
                    prior=node.children.get(best_action_idx, Node(to_move=node.to_move)).prior,
                )
                node.children[best_action_idx] = child
            child.to_move = env.state.current_player  # type: ignore[union-attr]

            node = child
            search_path.append((node, best_action_idx))

        # 2) Evaluate leaf
        if env.state.done:  # type: ignore[union-attr]
            # Terminal: value from the perspective of node.to_move
            winner = env.state.winner  # type: ignore[union-attr]
            if winner is None or winner == 0:
                value = 0.0
            else:
                value = 1.0 if winner == node.to_move else -1.0
        else:
            # Non-terminal: expand with network evaluation
            value = self._expand_node(node, env, add_dirichlet_noise=False)

        # 3) Backup
        self._backpropagate(search_path, value)

    def _expand_node(self, node: Node, env: LinithEnv, add_dirichlet_noise: bool) -> float:
        """
        Expand node using network. Returns value (from node.to_move perspective).
        """
        if env.state is None:
            raise RuntimeError("Env not reset before expand_node")
        node.to_move = env.state.current_player

        # legal moves at this state
        legal_idxs = legal_action_indices(env)
        node.legal_actions = legal_idxs

        if not legal_idxs or env.state.done:  # type: ignore[union-attr]
            # No legal moves / terminal; treat as draw evaluation
            node.is_expanded = True
            return 0.0

        # network evaluation
        state_tensor = torch.from_numpy(env.encode_state()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.net(state_tensor)
        value = value.squeeze().item()  # scalar in [-1,1]

        # convert logits to softmax over action space, but mask illegal actions
        logits = policy_logits.squeeze(0).cpu().numpy()  # (ACTION_SIZE,)
        # mask illegal
        mask = np.full_like(logits, -1e9, dtype=np.float32)
        mask[legal_idxs] = logits[legal_idxs]
        # softmax
        max_logit = np.max(mask)
        exp = np.exp(mask - max_logit)
        probs = exp / np.sum(exp)

        # Dirichlet noise (only for root)
        if add_dirichlet_noise:
            alpha = self.dirichlet_alpha
            noise = np.random.dirichlet([alpha] * len(legal_idxs)).astype(np.float32)
            eps = self.dirichlet_eps
            for i, a_idx in enumerate(legal_idxs):
                probs[a_idx] = (1 - eps) * probs[a_idx] + eps * noise[i]

        # create children with priors
        for a_idx in legal_idxs:
            p = float(probs[a_idx])
            if a_idx not in node.children:
                node.children[a_idx] = Node(to_move=env.state.current_player, prior=p)  # type: ignore[union-attr]
            else:
                node.children[a_idx].prior = p

        node.is_expanded = True
        return value

    def _backpropagate(self, path: List[Tuple[Node, Optional[int]]], value: float) -> None:
        """
        Backup value along the path. The value is from the perspective of the
        *last node's to_move*. Actions do not necessarily alternate players.
        """
        leaf_player = path[-1][0].to_move
        # walk from leaf to root
        for node, _ in reversed(path):
            node.visit_count += 1
            node.value_sum += value if node.to_move == leaf_player else -value
