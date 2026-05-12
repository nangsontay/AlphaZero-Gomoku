# -*- coding: utf-8 -*-
"""
A pure implementation of the Monte Carlo Tree Search (MCTS)
Enhanced with tactical heuristics from strategy.md:
  - Open Three / Open Four / Blocked Three / Gap pattern recognition
  - Fork (Double Threat) detection
  - Instant heuristic leaf evaluation (no rollout needed)
  - Heuristic expansion priors (not uniform)
  - Search tree reuse between moves
  - Proximity-based move filtering on large boards

@author: Junxiao Song (original), enhanced for tactical play
"""

import numpy as np


# ---------------------------------------------------------------------------
# Pattern scoring engine — the heart of the tactical improvements.
#
# For each empty cell we score how good it is for attack (current player)
# and defence (opponent).  The scoring recognises:
#   - Win-in-1          (5 in a row)
#   - Open Four          (4 in a row, both ends open)
#   - Half-open Four     (4 in a row, one end open — "blocked four")
#   - Open Three         (3 in a row, both ends open)
#   - Blocked Three      (3 in a row, one end blocked)
#   - Gap patterns       (e.g. XX_X, X_XX)
#   - Open Two           (2 in a row, both ends open)
#
# A fork is detected when a single move creates ≥2 "open three or better"
# threats simultaneously.
# ---------------------------------------------------------------------------

# Score weights (tuned for Gomoku heuristic strength)
_SCORE_WIN         = 1000000   # 5-in-a-row — instant win
_SCORE_OPEN_FOUR   = 100000    # ○●●●●○  — unstoppable unless opponent has win
_SCORE_HALF_FOUR   = 5000      # ×●●●●○  — must block
_SCORE_OPEN_THREE  = 3000      # ○●●●○   — very dangerous
_SCORE_BLOCKED_THREE = 400     # ×●●●○   — moderate threat
_SCORE_GAP_THREE   = 2500      # ●●_●  or  ●_●●  — sneaky, forms open four
_SCORE_OPEN_TWO    = 200       # ○●●○
_SCORE_BLOCKED_TWO = 30        # ×●●○
_SCORE_FORK_BONUS  = 80000     # creating ≥2 open-threes / gap-threes at once

# Defence multiplier: blocking an opponent threat is almost as good as
# creating your own, but slightly less to prefer offence when equal.
_DEFENCE_MULTIPLIER = 0.95


def _line_score(count, open_ends, has_gap=False):
    """Return a heuristic score for a line pattern.

    count     : number of own stones (including the candidate move)
    open_ends : 0, 1, or 2 open ends around the line
    has_gap   : True if the line contains a gap (e.g. XX_X)
    """
    if open_ends == 0 and not has_gap:
        return 0  # fully blocked, useless

    if count >= 5:
        return _SCORE_WIN

    if count == 4:
        if open_ends == 2:
            return _SCORE_OPEN_FOUR
        elif open_ends >= 1 or has_gap:
            return _SCORE_HALF_FOUR
        return 0

    if count == 3:
        if has_gap:
            return _SCORE_GAP_THREE  # e.g. XX_X — can become open four
        if open_ends == 2:
            return _SCORE_OPEN_THREE
        elif open_ends == 1:
            return _SCORE_BLOCKED_THREE
        return 0

    if count == 2:
        if open_ends == 2:
            return _SCORE_OPEN_TWO
        elif open_ends == 1:
            return _SCORE_BLOCKED_TWO
        return 0

    return 0


def _scan_direction(board, move, player, dh, dw):
    """Scan one direction (and its reverse) from `move` for `player`.

    Returns (count, open_ends, has_gap) where count includes the candidate
    move itself.
    """
    h = move // board.width
    w = move % board.width
    states = board.states
    n_in_row = board.n_in_row

    count = 1
    open_ends = 0
    has_gap = False

    # --- forward direction ---
    gap_used_fwd = False
    r, c = h + dh, w + dw
    while 0 <= r < board.height and 0 <= c < board.width:
        pos = r * board.width + c
        if states.get(pos) == player:
            count += 1
            r += dh
            c += dw
        elif not gap_used_fwd and states.get(pos) is None:
            # check if there's a stone after the gap
            nr, nc = r + dh, c + dw
            if (0 <= nr < board.height and 0 <= nc < board.width and
                    states.get(nr * board.width + nc) == player):
                gap_used_fwd = True
                has_gap = True
                count += 1  # count the stone after gap
                r, c = nr + dh, nc + dw
                continue
            else:
                open_ends += 1
                break
        else:
            if states.get(pos) is None:
                open_ends += 1
            break
    else:
        pass  # hit board edge => not open

    # --- backward direction ---
    gap_used_bwd = False
    r, c = h - dh, w - dw
    while 0 <= r < board.height and 0 <= c < board.width:
        pos = r * board.width + c
        if states.get(pos) == player:
            count += 1
            r -= dh
            c -= dw
        elif not gap_used_bwd and states.get(pos) is None:
            nr, nc = r - dh, c - dw
            if (0 <= nr < board.height and 0 <= nc < board.width and
                    states.get(nr * board.width + nc) == player):
                gap_used_bwd = True
                has_gap = True
                count += 1
                r, c = nr - dh, nc - dw
                continue
            else:
                open_ends += 1
                break
        else:
            if states.get(pos) is None:
                open_ends += 1
            break
    else:
        pass  # hit board edge

    # Cap count at n_in_row (e.g. 5) — more doesn't help.
    count = min(count, n_in_row)
    open_ends = min(open_ends, 2)

    return count, open_ends, has_gap


_DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def _evaluate_move(board, move, player):
    """Return (total_score, n_strong_threats) for `player` placing at `move`.

    n_strong_threats counts how many directions produce an open-three-or-better
    pattern; ≥2 means a fork.
    """
    total = 0
    strong_threats = 0  # count of open-three-or-better per direction
    for dh, dw in _DIRECTIONS:
        count, open_ends, has_gap = _scan_direction(board, move, player, dh, dw)
        s = _line_score(count, open_ends, has_gap)
        total += s
        if s >= _SCORE_GAP_THREE:  # open three, gap three, or better
            strong_threats += 1
    # Fork bonus: two or more strong threats at once
    if strong_threats >= 2:
        total += _SCORE_FORK_BONUS
    return total, strong_threats


def _move_heuristic_score(board, move):
    """Combined attack + defence score for a single move."""
    curr = board.current_player
    opp = board.players[0] if curr == board.players[1] else board.players[1]

    atk_score, _ = _evaluate_move(board, move, curr)
    def_score, _ = _evaluate_move(board, move, opp)

    return atk_score + _DEFENCE_MULTIPLIER * def_score


# ---------------------------------------------------------------------------
# Proximity filter: on a 15×15 board with few stones, restrict candidate
# moves to cells within `radius` of any existing stone.  This massively
# reduces the branching factor early on.
# ---------------------------------------------------------------------------

def _get_candidate_moves(board, radius=2):
    """Return a subset of board.availables near existing stones."""
    if not board.states:
        # Empty board — play centre.
        centre = (board.height // 2) * board.width + (board.width // 2)
        return [centre]

    neighbours = set()
    for pos in board.states:
        h = pos // board.width
        w = pos % board.width
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                nh, nw = h + dh, w + dw
                if 0 <= nh < board.height and 0 <= nw < board.width:
                    nb = nh * board.width + nw
                    if nb in board._available_set:
                        neighbours.add(nb)

    # Safety: if filter is too aggressive, fall back to all availables.
    if not neighbours:
        return list(board.availables)
    return list(neighbours)


# ---------------------------------------------------------------------------
# Heuristic policy-value function (replacing random rollout + uniform prior)
# ---------------------------------------------------------------------------

def heuristic_policy_value_fn(board):
    """Heuristic expansion prior AND leaf evaluation.

    Scores each candidate move with the pattern engine, converts to
    a probability distribution (prior), and returns a positional value
    in [-1, 1].  This replaces the expensive full-game rollout with a
    single O(candidates × 4-directions) scan — typically ~50-100× faster.
    """
    candidates = _get_candidate_moves(board, radius=2)
    if not candidates:
        candidates = list(board.availables)

    curr = board.current_player
    opp = board.players[0] if curr == board.players[1] else board.players[1]

    scores = []
    best_atk = 0
    best_def = 0
    for m in candidates:
        atk, _ = _evaluate_move(board, m, curr)
        dfn, _ = _evaluate_move(board, m, opp)
        combined = atk + _DEFENCE_MULTIPLIER * dfn
        scores.append(combined)
        best_atk = max(best_atk, atk)
        best_def = max(best_def, dfn)

    scores = np.array(scores, dtype=np.float64)
    scores = scores + 1.0  # floor so zero-score moves still get explored

    # Softmax-like conversion to probabilities
    log_scores = np.log(scores + 1e-10)
    log_scores -= np.max(log_scores)
    probs = np.exp(log_scores)
    probs /= probs.sum()

    action_probs = list(zip(candidates, probs))

    # Positional value estimate from the pattern landscape.
    value = 0.0
    if best_atk >= _SCORE_WIN:
        value = 1.0
    elif best_def >= _SCORE_WIN:
        value = -1.0
    elif best_atk >= _SCORE_OPEN_FOUR:
        value = 0.8
    elif best_def >= _SCORE_OPEN_FOUR:
        value = -0.8
    elif best_atk >= _SCORE_FORK_BONUS:
        value = 0.6
    elif best_def >= _SCORE_FORK_BONUS:
        value = -0.6
    elif best_atk >= _SCORE_OPEN_THREE:
        value = 0.3
    elif best_def >= _SCORE_OPEN_THREE:
        value = -0.3
    else:
        # Subtle advantage from material balance of threats.
        atk_sum = sum(1 for s in scores if s > _SCORE_OPEN_TWO)
        def_total = sum(1 for m in candidates
                        if _evaluate_move(board, m, opp)[0] >= _SCORE_OPEN_TWO)
        if atk_sum + def_total > 0:
            value = 0.1 * (atk_sum - def_total) / (atk_sum + def_total + 1)

    return action_probs, value


# ---------------------------------------------------------------------------
# MCTS tree node (unchanged from original)
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
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

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
# MCTS with instant heuristic evaluation (no rollout)
# ---------------------------------------------------------------------------

class MCTS(object):
    """Monte Carlo Tree Search with heuristic leaf evaluation.

    Instead of playing random/heuristic moves to the end of the game
    (rollout), each leaf is evaluated instantly by the pattern scoring
    engine.  This is ~50-100× faster per playout than a full rollout
    on a 15×15 board.
    """

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Check terminal leaves before expanding the node.
        end, winner = state.game_end()
        if not end:
            # Expand AND evaluate in one call — no rollout needed.
            action_probs, leaf_value = self._policy(state)
            node.expand(action_probs)
        else:
            # Terminal node: exact outcome.
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )
        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
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


class MCTSPlayer(object):
    """AI player based on MCTS with tactical heuristics."""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(heuristic_policy_value_fn, c_puct, n_playout)

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
        return "MCTS {}".format(self.player)
