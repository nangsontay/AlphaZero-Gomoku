# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes
"""

import numpy as np

from tactic import apply_tactical_prior_bonus


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Child statistics are stored in parallel NumPy arrays on the parent node so
    PUCT selection can be vectorized. Each child node also mirrors its own
    (N, W) totals so subtree reuse via update_with_move preserves statistics
    after the child is detached and becomes the new root.
    """

    __slots__ = ('_parent', '_parent_idx', '_prior', '_children_actions',
                 '_priors', '_N_arr', '_W_arr', '_child_nodes', '_N', '_W')

    def __init__(self, parent, prior_p, parent_idx=None):
        self._parent = parent
        self._parent_idx = parent_idx
        self._prior = float(prior_p)
        self._children_actions = None
        self._priors = None
        self._N_arr = None
        self._W_arr = None
        self._child_nodes = None
        self._N = 0.0
        self._W = 0.0

    @property
    def _n_visits(self):
        """Backward-compatible read-only visit count alias."""
        return self._N

    @property
    def _Q(self):
        """Backward-compatible read-only mean value alias."""
        return self.Q()

    @property
    def _P(self):
        """Backward-compatible read-only prior alias."""
        return self._prior

    def Q(self):
        return self._W / self._N if self._N > 0 else 0.0

    def expand(self, action_priors):
        """Expand this node using legal action priors from the policy."""
        action_priors = list(action_priors)
        if not action_priors:
            return
        actions, priors = zip(*action_priors)
        self._children_actions = np.asarray(actions, dtype=np.int32)
        self._priors = np.asarray(priors, dtype=np.float32)
        n = len(self._children_actions)
        self._N_arr = np.zeros(n, dtype=np.float32)
        self._W_arr = np.zeros(n, dtype=np.float32)
        self._child_nodes = [None] * n

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        q = np.divide(
            self._W_arr,
            self._N_arr,
            out=np.zeros_like(self._W_arr, dtype=np.float32),
            where=self._N_arr > 0,
        )
        u = (c_puct * self._priors * np.sqrt(max(self._N, 1e-8)) /
             (1.0 + self._N_arr))
        idx = int(np.argmax(q + u))
        action = int(self._children_actions[idx])
        child = self._child_nodes[idx]
        if child is None:
            child = TreeNode(self, float(self._priors[idx]), idx)
            self._child_nodes[idx] = child
        return action, child

    def get_child(self, action, create=False):
        """Return the child for action, optionally instantiating its node."""
        if self.is_leaf():
            return None
        matches = np.flatnonzero(self._children_actions == int(action))
        if len(matches) == 0:
            return None
        idx = int(matches[0])
        child = self._child_nodes[idx]
        if child is None and create:
            child = TreeNode(self, float(self._priors[idx]), idx)
            self._child_nodes[idx] = child
        return child

    def iter_child_visits(self):
        """Yield (action, visit_count) for root probability extraction."""
        if self.is_leaf():
            return []
        return zip(self._children_actions.tolist(), self._N_arr.tolist())

    def add_virtual_loss(self, n_vl):
        if n_vl <= 0:
            return
        self._N += n_vl
        self._W -= n_vl
        if self._parent is not None:
            self._parent._N_arr[self._parent_idx] += n_vl
            self._parent._W_arr[self._parent_idx] -= n_vl

    def revert_virtual_loss(self, n_vl):
        if n_vl <= 0:
            return
        self._N -= n_vl
        self._W += n_vl
        if self._parent is not None:
            self._parent._N_arr[self._parent_idx] -= n_vl
            self._parent._W_arr[self._parent_idx] += n_vl

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        self._N += 1.0
        self._W += float(leaf_value)
        if self._parent is not None:
            self._parent._N_arr[self._parent_idx] += 1.0
            self._parent._W_arr[self._parent_idx] += float(leaf_value)

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        parent_visits = self._parent._N if self._parent is not None else self._N
        u = c_puct * self._prior * np.sqrt(max(parent_visits, 1e-8)) / (1 + self._N)
        return self.Q() + u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children_actions is None

    def is_root(self):
        return self._parent is None

    def apply_dirichlet_noise(self, dirichlet_alpha, noise_eps):
        """Blend Dirichlet exploration noise into this node's child priors."""
        if self.is_leaf() or len(self._priors) == 0:
            return
        noise = np.random.dirichlet(
            float(dirichlet_alpha) * np.ones(len(self._priors), dtype=np.float32)
        ).astype(np.float32)
        self._priors = ((1.0 - float(noise_eps)) * self._priors +
                        float(noise_eps) * noise).astype(np.float32, copy=False)


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, policy_value_batch_fn=None,
                  c_puct=3.0, n_playout=10000, vl_k=4, n_vl=1.0,
                  max_oversample=3, tactic_prior_weight=0.35):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        policy_value_batch_fn: optional function that takes a batch of state
            arrays and returns full-board priors plus values.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._policy_batch = policy_value_batch_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._vl_k = max(1, int(vl_k))
        self._n_vl = float(n_vl)
        self._max_oversample = max(1, int(max_oversample))
        self._tactic_prior_weight = max(0.0, float(tactic_prior_weight))
        self._root_noise_applied = False

    def _playout(self, state):
        """Run one compatibility playout through the batched implementation."""
        old_n_playout = self._n_playout
        old_vl_k = self._vl_k
        old_n_vl = self._n_vl
        try:
            self._n_playout = 1
            self._vl_k = 1
            self._n_vl = 0.0
            self.get_move_probs(state, temp=1.0)
        finally:
            self._n_playout = old_n_playout
            self._vl_k = old_vl_k
            self._n_vl = old_n_vl

    def get_move_probs(self, state, temp=1e-3,
                       dirichlet_alpha=None, noise_eps=0.0):
        """Run leaf-parallel playouts and return actions and probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        if (not self._root.is_leaf() and dirichlet_alpha is not None and
                noise_eps > 0 and not self._root_noise_applied):
            self._root.apply_dirichlet_noise(dirichlet_alpha, noise_eps)
            self._root_noise_applied = True

        completed = 0
        while completed < self._n_playout:
            target_nn = min(self._vl_k, self._n_playout - completed)
            nn_leaves = []
            terminals_done = 0
            attempts = 0
            attempt_cap = max(target_nn * self._max_oversample, target_nn + 4)

            while len(nn_leaves) < target_nn and attempts < attempt_cap:
                attempts += 1
                node = self._root
                sim_state = state.copy_fast()
                path = []

                while not node.is_leaf():
                    action, node = node.select(self._c_puct)
                    sim_state.do_move(action)
                    node.add_virtual_loss(self._n_vl)
                    path.append(node)

                end, winner = sim_state.game_end()
                if end:
                    if winner == -1:
                        leaf_value = 0.0
                    else:
                        leaf_value = (
                            1.0 if winner == sim_state.get_current_player()
                            else -1.0
                        )
                    self._backup_and_revert(path, -leaf_value)
                    terminals_done += 1
                else:
                    nn_leaves.append((node, sim_state, path))
                    if node.is_root():
                        break

            if nn_leaves:
                eval_results = self._evaluate_leaves(nn_leaves)
                for (leaf, sim, path), (action_priors, value) in zip(
                        nn_leaves, eval_results):
                    if leaf.is_leaf():
                        leaf.expand(self._with_tactical_prior(action_priors, sim))
                        if (leaf.is_root() and dirichlet_alpha is not None and
                                noise_eps > 0 and not self._root_noise_applied):
                            leaf.apply_dirichlet_noise(dirichlet_alpha, noise_eps)
                            self._root_noise_applied = True
                    self._backup_and_revert(path, -float(value))

            completed += len(nn_leaves) + terminals_done

            if attempts >= attempt_cap and not nn_leaves and terminals_done == 0:
                break

        # calc the move probabilities based on visit counts at the root node
        act_visits = list(self._root.iter_child_visits())
        if not act_visits:
            return [], []
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def _evaluate_leaves(self, nn_leaves):
        """Evaluate selected non-terminal leaves, using batch API if present."""
        if self._policy_batch is None:
            results = []
            for _, sim, _ in nn_leaves:
                action_probs, value = self._policy(sim)
                results.append((list(action_probs), float(value)))
            return results

        states_np = np.stack([
            np.ascontiguousarray(sim.current_state().astype(np.float32))
            for _, sim, _ in nn_leaves
        ])
        priors_batch, values_batch = self._policy_batch(states_np)
        values_batch = np.asarray(values_batch).reshape(-1)
        results = []
        for (_, sim, _), priors, value in zip(nn_leaves, priors_batch, values_batch):
            legal = sim.availables
            results.append(([(a, float(priors[a])) for a in legal], float(value)))
        return results

    def _with_tactical_prior(self, action_priors, board):
        """Apply a soft tactical prior bonus before node expansion."""
        return apply_tactical_prior_bonus(
            action_priors, board, bonus_weight=self._tactic_prior_weight)

    def _backup_and_revert(self, path, leaf_value):
        """Revert virtual loss and back up values along a selected path."""
        if not path:
            self._root.update(leaf_value)
            return
        sign = 1.0
        for node in reversed(path):
            node.revert_virtual_loss(self._n_vl)
            node.update(sign * leaf_value)
            sign = -sign
        self._root.update(sign * leaf_value)

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        child = self._root.get_child(last_move, create=True)
        if child is not None:
            self._root = child
            self._root._parent = None
            self._root._parent_idx = None
        else:
            self._root = TreeNode(None, 1.0)
        self._root_noise_applied = False

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function, policy_value_batch_function=None,
                  c_puct=3.0, n_playout=2000, is_selfplay=0,
                  dirichlet_alpha=0.05, noise_eps=0.25,
                  vl_k=4, n_vl=1.0, max_oversample=3,
                  tactic_prior_weight=0.35):
        self.mcts = MCTS(policy_value_function,
                         policy_value_batch_fn=policy_value_batch_function,
                         c_puct=c_puct,
                         n_playout=n_playout,
                          vl_k=vl_k,
                          n_vl=n_vl,
                          max_oversample=max_oversample,
                          tactic_prior_weight=tactic_prior_weight)
        self._is_selfplay = is_selfplay
        self._dirichlet_alpha = float(dirichlet_alpha)
        self._noise_eps = float(noise_eps)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            if self._is_selfplay:
                acts, probs = self.mcts.get_move_probs(
                    board, temp,
                    dirichlet_alpha=self._dirichlet_alpha,
                    noise_eps=self._noise_eps)
            else:
                acts, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(acts)] = probs
            if self._is_selfplay:
                move = np.random.choice(acts, p=probs)
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = acts[np.argmax(probs)]
                # Reuse subtree: advance tree by the move we chose.
                self.mcts.update_with_move(move)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def notify_opponent_move(self, move):
        """Call this after the opponent plays so the tree can be reused."""
        self.mcts.update_with_move(move)

    def __str__(self):
        return "MCTS {}".format(self.player)
