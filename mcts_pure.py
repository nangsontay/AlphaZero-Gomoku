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
_SCORE_HALF_FOUR   = 8000      # ×●●●●○  — must block
_SCORE_OPEN_THREE  = 4000      # ○●●●○   — very dangerous
_SCORE_BLOCKED_THREE = 800     # ×●●●○   — moderate threat
_SCORE_GAP_THREE   = 3500      # ●●_●  or  ●_●●  — sneaky, forms open four
_SCORE_OPEN_TWO    = 600       # ○●●○    — building block for future threats
_SCORE_BLOCKED_TWO = 80        # ×●●○
_SCORE_OPEN_ONE    = 20        # ○●○     — small positional value

# Combination bonuses — rewarding multi-directional threats
_SCORE_FORK_BONUS       = 80000   # ≥2 strong threats (open three+) at once
_SCORE_COMBO_BONUS      = 25000   # 1 strong + 1 moderate (open3 + blocked3)
_SCORE_MODERATE_FORK    = 10000   # ≥2 moderate threats (blocked3 + blocked3)
_SCORE_PREFORK_SETUP    = 1500    # ≥2 open twos converging — future fork
_SCORE_SETUP_COMBO      = 900     # 1 open two + 1 moderate threat

# Defence multiplier: blocking an opponent threat is almost as good as
# creating your own, but slightly less to prefer offence when equal.
_DEFENCE_MULTIPLIER = 0.9


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
        if has_gap:
            # ●_● pattern — can become gap three with one move
            if open_ends >= 1:
                return _SCORE_BLOCKED_TWO * 3  # 240: more dangerous than simple blocked two
            return _SCORE_BLOCKED_TWO
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
    """Return (total_score, n_strong_threats, detail) for `player` at `move`.

    Threat classification per direction:
      strong   : open three, gap three, open four, win  (score ≥ GAP_THREE)
      moderate : blocked three, half-open four           (score ≥ BLOCKED_THREE)
      building : open two                                (score ≥ OPEN_TWO)

    Combination bonuses:
      ≥2 strong             → fork bonus      (unstoppable)
      1 strong + 1 moderate → combo bonus      (very hard to defend)
      ≥2 moderate           → moderate fork    (hard to defend)
      ≥2 building           → pre-fork setup   (future fork potential)
      1 building + 1 mod    → setup combo      (growing advantage)
    """
    total = 0
    strong_threats = 0
    moderate_threats = 0
    building_moves = 0

    for dh, dw in _DIRECTIONS:
        count, open_ends, has_gap = _scan_direction(board, move, player, dh, dw)
        s = _line_score(count, open_ends, has_gap)
        total += s
        if s >= _SCORE_GAP_THREE:
            strong_threats += 1
        elif s >= _SCORE_BLOCKED_THREE:
            moderate_threats += 1
        elif s >= _SCORE_OPEN_TWO:
            building_moves += 1

    # --- Multi-directional combination bonuses (most important first) ---
    if strong_threats >= 2:
        total += _SCORE_FORK_BONUS
    elif strong_threats >= 1 and moderate_threats >= 1:
        total += _SCORE_COMBO_BONUS
    elif moderate_threats >= 2:
        total += _SCORE_MODERATE_FORK
    elif building_moves >= 2:
        total += _SCORE_PREFORK_SETUP
    elif building_moves >= 1 and moderate_threats >= 1:
        total += _SCORE_SETUP_COMBO

    return total, strong_threats


def _move_heuristic_score(board, move):
    """Combined attack + defence score for a single move."""
    curr = board.current_player
    opp = board.players[0] if curr == board.players[1] else board.players[1]

    atk_score, _ = _evaluate_move(board, move, curr)
    def_score, _ = _evaluate_move(board, move, opp)

    return atk_score + _DEFENCE_MULTIPLIER * def_score


# ---------------------------------------------------------------------------
# Board-level threat scanning: what threats already exist on the board?
# Used by the value function to assess initiative and chain potential.
# ---------------------------------------------------------------------------

def _count_board_threats(board, player):
    """Scan the board for existing threat patterns of `player`.

    Returns a dict with counts of each threat type found on the board.
    Now detects BOTH contiguous patterns AND gap patterns (e.g. ●_●●).
    """
    states = board.states
    width, height = board.width, board.height
    threats = {'open_four': 0, 'half_four': 0, 'open_three': 0,
               'gap_three': 0, 'blocked_three': 0, 'open_two': 0}

    visited_contig = set()
    visited_gap = set()

    for pos, p in states.items():
        if p != player:
            continue
        h = pos // width
        w = pos % width
        for dh, dw in _DIRECTIONS:
            # --- 1) Contiguous scan (forward only to avoid double-count) ---
            key_c = (pos, dh, dw)
            if key_c not in visited_contig:
                count = 1
                r, c = h + dh, w + dw
                while (0 <= r < height and 0 <= c < width and
                       states.get(r * width + c) == player):
                    visited_contig.add((r * width + c, dh, dw))
                    count += 1
                    r += dh
                    c += dw

                if count >= 2:
                    open_ends = 0
                    if (0 <= r < height and 0 <= c < width and
                            states.get(r * width + c) is None):
                        open_ends += 1
                    br, bc = h - dh, w - dw
                    if (0 <= br < height and 0 <= bc < width and
                            states.get(br * width + bc) is None):
                        open_ends += 1

                    if count >= 4:
                        if open_ends >= 2:
                            threats['open_four'] += 1
                        elif open_ends >= 1:
                            threats['half_four'] += 1
                    elif count == 3:
                        if open_ends >= 2:
                            threats['open_three'] += 1
                        elif open_ends >= 1:
                            threats['blocked_three'] += 1
                    elif count == 2:
                        if open_ends >= 2:
                            threats['open_two'] += 1

            # --- 2) Gap pattern scan: ●_●, ●_●●, ●●_● ---
            # Look for a gap (empty cell) immediately forward, then stones.
            gr, gc = h + dh, w + dw
            if not (0 <= gr < height and 0 <= gc < width):
                continue
            gap_pos = gr * width + gc
            if states.get(gap_pos) is not None:
                continue  # not a gap
            # Check for stones after the gap
            gr2, gc2 = gr + dh, gc + dw
            gap_key = (pos, gap_pos, dh, dw)
            if gap_key in visited_gap:
                continue
            visited_gap.add(gap_key)

            gap_count = 1  # the starting stone
            # Count stones after the gap
            sr, sc = gr2, gc2
            while (0 <= sr < height and 0 <= sc < width and
                   states.get(sr * width + sc) == player):
                gap_count += 1
                sr += dh
                sc += dw
            # Count stones before the starting stone (backward)
            br, bc = h - dh, w - dw
            while (0 <= br < height and 0 <= bc < width and
                   states.get(br * width + bc) == player):
                gap_count += 1
                br -= dh
                bc -= dw

            if gap_count < 3:
                continue  # need at least 3 stones with gap to be dangerous

            # Check open ends around the full gap pattern
            gap_open = 0
            if (0 <= sr < height and 0 <= sc < width and
                    states.get(sr * width + sc) is None):
                gap_open += 1
            if (0 <= br < height and 0 <= bc < width and
                    states.get(br * width + bc) is None):
                gap_open += 1

            if gap_count >= 4:
                threats['half_four'] += 1
            elif gap_count == 3:
                threats['gap_three'] += 1

    return threats


def _best_opponent_move_score(board):
    """1-ply urgency check: find the opponent's strongest available move.

    This detects imminent threats like ○○●○●●○○ where the opponent
    can create an open four on their next turn.  Returns the highest
    score any single opponent move would achieve.
    """
    opp = board.players[0] if board.current_player == board.players[1] \
        else board.players[1]
    candidates = _get_candidate_moves(board, radius=2)
    best = 0
    for m in candidates:
        score, _ = _evaluate_move(board, m, opp)
        if score > best:
            best = score
            if best >= _SCORE_WIN:  # can't get worse, stop early
                break
    return best


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
    in [-1, 1].

    Enhanced with:
      - Multi-level combination bonus (fork, combo, pre-fork setup)
      - Board-level threat context (existing open/gap threes boost urgency)
      - 1-ply opponent urgency check (detects imminent open-four threats)
      - Sharper priors via power-law scaling
      - Initiative-aware value estimation
    """
    candidates = _get_candidate_moves(board, radius=2)
    if not candidates:
        candidates = list(board.availables)

    curr = board.current_player
    opp = board.players[0] if curr == board.players[1] else board.players[1]

    # --- Scan existing board threats for context ---
    curr_threats = _count_board_threats(board, curr)
    opp_threats = _count_board_threats(board, opp)
    has_existing_strong = (curr_threats['open_three'] > 0 or
                           curr_threats['gap_three'] > 0)
    opp_has_strong = (opp_threats['open_three'] > 0 or
                      opp_threats['gap_three'] > 0)

    # --- 1-ply urgency check: what can the opponent do next turn? ---
    opp_best = _best_opponent_move_score(board)

    scores = []
    atk_scores = []
    def_scores = []
    best_atk = 0
    best_def = 0
    for m in candidates:
        atk, atk_st = _evaluate_move(board, m, curr)
        dfn, def_st = _evaluate_move(board, m, opp)

        # Context bonus: if we already have an open/gap three elsewhere,
        # ANY move that creates another threat is extremely valuable
        # because the opponent can't block two threats at once.
        if has_existing_strong and atk >= _SCORE_OPEN_THREE:
            atk += _SCORE_COMBO_BONUS
        elif has_existing_strong and atk >= _SCORE_BLOCKED_THREE:
            atk += _SCORE_MODERATE_FORK

        # If opponent has strong threats, urgently boost blocking moves.
        if opp_has_strong and dfn >= _SCORE_OPEN_THREE:
            dfn += _SCORE_HALF_FOUR
        elif opp_has_strong and dfn >= _SCORE_BLOCKED_THREE:
            dfn += _SCORE_BLOCKED_THREE  # moderate urgency

        # 1-ply urgency: if opponent can create open four next turn,
        # any blocking move (that reduces their best threat) gets a
        # big defense boost.
        if opp_best >= _SCORE_OPEN_FOUR and dfn >= _SCORE_HALF_FOUR:
            dfn += _SCORE_COMBO_BONUS  # MUST block or lose

        combined = atk + _DEFENCE_MULTIPLIER * dfn
        scores.append(combined)
        atk_scores.append(atk)
        def_scores.append(dfn)
        best_atk = max(best_atk, atk)
        best_def = max(best_def, dfn)

    scores = np.array(scores, dtype=np.float64)
    scores = scores + 1.0  # floor so zero-score moves still get explored

    # Sharper priors via power-law scaling.
    log_scores = np.log(scores + 1e-10)
    log_scores -= np.max(log_scores)
    sharpened = log_scores * 1.5
    probs = np.exp(sharpened)
    probs /= probs.sum()

    action_probs = list(zip(candidates, probs))

    # --- Value estimation: aggressive and urgency-aware ---
    value = 0.0
    if best_atk >= _SCORE_WIN:
        value = 1.0
    elif best_def >= _SCORE_WIN:
        value = -1.0
    elif best_atk >= _SCORE_OPEN_FOUR:
        value = 0.9
    elif best_def >= _SCORE_OPEN_FOUR:
        value = -0.85
    elif best_atk >= _SCORE_FORK_BONUS:
        value = 0.75
    elif best_def >= _SCORE_FORK_BONUS:
        value = -0.65
    elif best_atk >= _SCORE_COMBO_BONUS:
        value = 0.55
    elif best_def >= _SCORE_COMBO_BONUS:
        value = -0.45
    elif best_atk >= _SCORE_OPEN_THREE:
        value = 0.35
    elif best_def >= _SCORE_OPEN_THREE:
        value = -0.3
    elif best_atk >= _SCORE_PREFORK_SETUP:
        value = 0.15
    elif best_def >= _SCORE_PREFORK_SETUP:
        value = -0.1
    else:
        n_atk = sum(1 for s in atk_scores if s >= _SCORE_OPEN_TWO)
        n_def = sum(1 for s in def_scores if s >= _SCORE_OPEN_TWO)
        if n_atk + n_def > 0:
            value = 0.08 * (n_atk - n_def) / (n_atk + n_def + 1)

    # 1-ply urgency override: if opponent can create open four or fork
    # next turn, the position is dire regardless of our own threats.
    if opp_best >= _SCORE_OPEN_FOUR:
        value = min(value, -0.85)
    elif opp_best >= _SCORE_FORK_BONUS:
        value = min(value, -0.7)
    elif opp_best >= _SCORE_OPEN_THREE:
        value = min(value, -0.35)

    # Initiative bonus from existing board threats.
    initiative = (curr_threats['open_three'] * 0.15
                  + curr_threats['gap_three'] * 0.12
                  + curr_threats['open_two'] * 0.03
                  - opp_threats['open_three'] * 0.12
                  - opp_threats['gap_three'] * 0.10
                  - opp_threats['open_two'] * 0.02)
    value = max(-1.0, min(1.0, value + initiative))

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
