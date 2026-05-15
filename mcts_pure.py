# -*- coding: utf-8 -*-
"""
A truly pure Monte Carlo Tree Search (MCTS) opponent for Gomoku.

This module is the canonical *baseline* opponent used by:
  - ``play.py``                    (UI evaluation vs. AlphaZero agent)
  - ``train_gpu_evaluator.py``     (benchmark games during training)

Design rules (textbook MCTS — no tactical knowledge whatsoever):
  - Expansion priors are uniform over every legal move.
  - Leaf value comes from a random rollout to terminal (or
    ``rollout_limit`` plies — treated as draw on truncation).
  - No pattern scoring, fork detection, heuristic leaf evaluation,
    tactical priors, or proximity / radius move filtering.
  - No dependency on :mod:`tactic` or :mod:`tactic_patterns`.

The pure MCTS behaviour lives entirely here so callers can rely on it
as a stable evaluation baseline.  Training-side tactical helpers live
in :mod:`tactic` / :mod:`tactic_patterns`.

@author: Junxiao Song (original), reduced to a pure baseline.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Policy / rollout helpers used by the pure MCTS search
# ---------------------------------------------------------------------------

def _pure_policy_value_fn(board):
    """Uniform priors over every legal move; zero placeholder value.

    The actual leaf value comes from a random rollout, not from this
    function.  Kept as a function (rather than inlined) so the search
    loop mirrors the textbook "expand with a policy, then evaluate"
    pattern.
    """
    n = len(board.availables)
    if n == 0:
        return [], 0.0
    prob = 1.0 / n
    return [(m, prob) for m in board.availables], 0.0


def _rollout_policy_fn(board):
    """Coarse random policy used in the rollout phase."""
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


# ---------------------------------------------------------------------------
# MCTS tree node (textbook PUCT)
# ---------------------------------------------------------------------------

class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0 * (leaf_value - self._Q) / self._n_visits

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
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


# ---------------------------------------------------------------------------
# Pure MCTS — uniform priors + random rollouts for leaf evaluation.
# ---------------------------------------------------------------------------

class MCTS(object):
    """Canonical pure Monte Carlo Tree Search (no tactical heuristics).

    - Expansion priors are uniform over all legal moves.
    - Leaf value is obtained by a random rollout to terminal (or
      ``rollout_limit`` plies, whichever comes first; truncation is
      scored as a draw).
    - No pattern scoring, fork detection, proximity filtering, or
      heuristic value shaping is applied anywhere in the search.
    """

    def __init__(self, c_puct=5, n_playout=2000, rollout_limit=1000):
        """
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        n_playout: number of simulations to run from the root.
        rollout_limit: hard cap on plies per rollout (truncated rollouts
            are treated as draws).
        """
        self._root = TreeNode(None, 1.0)
        self._c_puct = c_puct
        self._n_playout = n_playout
        self._rollout_limit = rollout_limit

    def _playout(self, state):
        """Single playout: select → expand (uniform) → rollout → backprop.

        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        end, winner = state.game_end()
        if not end:
            action_probs, _ = _pure_policy_value_fn(state)
            node.expand(action_probs)

        # Random rollout for leaf evaluation — no heuristic shortcut.
        leaf_value = self._evaluate_rollout(state)
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state):
        """Run a random rollout to terminal.

        Returns +1 / -1 / 0 from the perspective of the player whose turn
        it is at the leaf (i.e. ``state.get_current_player()`` BEFORE any
        rollout move is played).
        """
        player = state.get_current_player()
        for _ in range(self._rollout_limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = _rollout_policy_fn(state)
            max_action = max(action_probs, key=lambda x: x[1])[0]
            state.do_move(max_action)
        else:
            # Rollout limit hit without terminal — treat as draw.
            return 0.0

        if winner == -1:
            return 0.0
        return 1.0 if winner == player else -1.0

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited
        action.

        state: the current game state

        Return: the selected action
        """
        for _ in range(self._n_playout):
            state_copy = state.copy_fast()
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


# ---------------------------------------------------------------------------
# Public AI player wrapper — drives the search and exposes the game API.
# ---------------------------------------------------------------------------

class MCTSPlayer(object):
    """AI player based on truly pure MCTS.

    Public surface is backward-compatible with all existing callers
    (``play.py``, ``train_gpu_evaluator.py``):

        MCTSPlayer(c_puct=5, n_playout=2000)
        MCTSPlayer(c_puct=5, n_playout=2000, pure=True)

    The ``pure`` keyword is accepted for backward compatibility — this
    module is now pure-only, so the flag has no effect on behaviour.
    """

    def __init__(self, c_puct=5, n_playout=2000, pure=True, **_kwargs):
        # ``pure`` is intentionally ignored: this module no longer
        # offers a non-pure mode.  The flag is preserved so existing
        # callers that pass ``pure=True`` continue to work unchanged.
        del pure
        self.mcts = MCTS(c_puct=c_puct, n_playout=n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            # Reuse subtree: advance tree by the move we chose.
            self.mcts.update_with_move(move)
            return move
        else:
            print("WARNING: the board is full")

    def notify_opponent_move(self, move):
        """Call this after the opponent plays so the tree can be reused."""
        self.mcts.update_with_move(move)

    def __str__(self):
        return "Pure MCTS {}".format(getattr(self, "player", "?"))
