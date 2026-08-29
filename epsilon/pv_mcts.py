import numpy as np
import torch
import torch.nn.functional as F

try:
    from .action_space import ACTION_SIZE, encode_action
except ImportError:
    from action_space import ACTION_SIZE, encode_action

NUM_ACTIONS = ACTION_SIZE


def _hashable_action(value):
    """Normalise pybind list-backed actions for use as tree dictionary keys."""
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_action(item) for item in value)
    return value


class PVNode:
    __slots__ = ("parent", "action_from_parent", "player",
                 "children", "P", "W", "N", "Q")

    def __init__(self, parent, action_from_parent, player):
        self.parent = parent
        self.action_from_parent = action_from_parent  # env-native action
        self.player = player  # player to move at this node

        self.children = {}    # action -> PVNode

        self.P = 0.0          # prior
        self.W = 0.0          # total value
        self.N = 0.0          # visit count
        self.Q = 0.0          # mean value


class PV_MCTS:
    """
    AlphaZero-style Monte Carlo Tree Search using a policy/value net.
    Compatible with both Python LinithEnv and C++ LinithEnv exposed via pybind11.
    """

    def __init__(self, net, device: str = "cpu", c_puct: float = 1.5):
        self.net = net
        self.device = torch.device(device)
        self.c_puct = c_puct

    # ---------------------------------------------------------
    #  PUBLIC INTERFACE
    # ---------------------------------------------------------
    def search(
        self,
        env,
        num_simulations: int,
        *,
        add_root_noise: bool = False,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
    ):
        """
        Run MCTS from the current env state.

        Args:
            env: LinithEnv instance (Python or C++ backed)
            num_simulations: number of simulations
            add_root_noise: whether to add Dirichlet noise at root
            dirichlet_alpha: α parameter for Dirichlet noise
            dirichlet_eps: mixing weight for Dirichlet noise

        Returns:
            dict[action -> visit_count]
        """
        root_player = env.state.current_player
        root = PVNode(parent=None, action_from_parent=None, player=root_player)

        # ----- Root expansion -----
        policy, value = self._evaluate(env)
        legal_actions = env.legal_actions()
        self._expand(root, env, legal_actions, policy)

        # Optional Dirichlet noise (AlphaZero style)
        if add_root_noise:
            self._add_root_dirichlet_noise(
                root,
                alpha=dirichlet_alpha,
                eps=dirichlet_eps,
            )

        # Bootstrap root stats
        root.W += float(value)
        root.N += 1.0
        root.Q = root.W / root.N

        # ----- Main simulation loop -----
        for _ in range(max(0, num_simulations - 1)):
            node, sim_env = self._select(env, root, root_player)

            if sim_env.state.done:
                v = self._terminal_value(sim_env, root_player)
            else:
                p, v = self._evaluate(sim_env)
                if sim_env.state.current_player != root_player:
                    v = -v
                acts = sim_env.legal_actions()
                self._expand(node, sim_env, acts, p)

            self._backpropagate(node, v)

        return {a: child.N for a, child in root.children.items()}

    # ---------------------------------------------------------
    #  EVALUATION
    # ---------------------------------------------------------
    def _evaluate(self, env):
        """
        Run the net on the current board.
        Works with C++ env because GameState.to_tensor() is bound.
        """
        # Both Python and C++ LinithEnv expose env.state.to_tensor()
        obs = env.state.to_tensor()                 # (8, 10, 10)
        x = torch.from_numpy(obs).unsqueeze(0).to(self.device)  # (1, 8, 10, 10)

        with torch.no_grad():
            logits, value = self.net(x)

        policy = F.softmax(logits, dim=1)[0].cpu().numpy()  # (NUM_ACTIONS,)
        value_scalar = float(value.item())
        return policy, value_scalar

    # ---------------------------------------------------------
    #  EXPAND
    # ---------------------------------------------------------
    def _expand(self, node: PVNode, env, legal_actions, policy_vector: np.ndarray):
        """
        Expand node with child nodes for each legal action.
        `policy_vector` is the NN's policy over full ACTION_SIZE.
        """
        if not legal_actions:
            return
        node.player = env.state.current_player

        # Build children with priors from policy vector
        total_p = 0.0
        tmp = []
        for a in legal_actions:
            try:
                idx = encode_action(env, a)
            except Exception:
                # Skip actions that can't be encoded (e.g. weird >6-swan subsets)
                continue
            p = float(policy_vector[idx])
            tmp.append((_hashable_action(a), p))
            total_p += max(p, 0.0)

        # Fallback: if nothing encodable, make uniform over legal actions
        if not tmp:
            p_uniform = 1.0 / len(legal_actions)
            for raw_action in legal_actions:
                a = _hashable_action(raw_action)
                child = node.children.get(a)
                if child is None:
                    child = PVNode(parent=node, action_from_parent=a, player=None)
                    node.children[a] = child
                child.P = p_uniform
            return

        # Normalise over encodable subset; allow zero or near-zero priors, but keep them
        if total_p <= 0.0:
            # If all probs are non-positive, fall back to uniform over encodable set
            p_uniform = 1.0 / len(tmp)
            for a, _ in tmp:
                child = node.children.get(a)
                if child is None:
                    child = PVNode(parent=node, action_from_parent=a, player=None)
                    node.children[a] = child
                child.P = p_uniform
        else:
            for a, p in tmp:
                child = node.children.get(a)
                if child is None:
                    child = PVNode(parent=node, action_from_parent=a, player=None)
                    node.children[a] = child
                child.P = max(p, 0.0) / total_p

    # ---------------------------------------------------------
    #  SELECT
    # ---------------------------------------------------------
    def _select(self, env, root: PVNode, root_player: int):
        """
        Traverse the tree from root using PUCT, return (leaf_node, sim_env).
        """
        node = root
        sim_env = env.clone()

        while node.children:
            best_action = None
            best_child = None
            best_score = -1e30

            sqrt_N = np.sqrt(max(1e-6, node.N))

            for a, child in node.children.items():
                # Standard AlphaZero PUCT
                exploitation = child.Q if node.player == root_player else -child.Q
                U = exploitation + self.c_puct * child.P * (sqrt_N / (1.0 + child.N))
                if U > best_score:
                    best_score = U
                    best_action = a
                    best_child = child

            # Step environment
            obs, reward, done, info = sim_env.step(best_action)
            node = best_child
            node.player = sim_env.state.current_player

            if node.N == 0 or done:
                break

        return node, sim_env

    # ---------------------------------------------------------
    #  BACKPROP
    # ---------------------------------------------------------
    def _backpropagate(self, node: PVNode, value: float):
        """
        Backpropagate value up the path.
        value is from root_player's perspective.
        """
        cur = node
        while cur is not None:
            cur.N += 1.0
            cur.W += value
            cur.Q = cur.W / cur.N

            cur = cur.parent

    # ---------------------------------------------------------
    #  TERMINAL VALUE
    # ---------------------------------------------------------
    def _terminal_value(self, env, root_player: int) -> float:
        winner = env.state.winner
        if winner is None or winner == 0:
            return 0.0
        return 1.0 if winner == root_player else -1.0

    # ---------------------------------------------------------
    #  DIRICHLET ROOT NOISE
    # ---------------------------------------------------------
    def _add_root_dirichlet_noise(self, root: PVNode, alpha: float, eps: float):
        """
        AlphaZero-style Dirichlet noise injection at the root:
            P' = (1 - eps) * P + eps * Dir(alpha)
        """
        if not root.children:
            return

        actions = list(root.children.keys())
        K = len(actions)
        if K == 0:
            return

        noise = np.random.dirichlet([alpha] * K)
        for a, n in zip(actions, noise):
            child = root.children[a]
            p_old = float(child.P)
            p_new = (1.0 - eps) * p_old + eps * float(n)
            child.P = p_new
