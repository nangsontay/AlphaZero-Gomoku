# -*- coding: utf-8 -*-
"""
Tactical analysis module for Gomoku.

Provides rich tactical labels for training the Neural Net's tactic head,
and soft MCTS prior bonuses.  Labels now cover the full spectrum of
Gomoku tactics (from strategy.md):

  - Win-in-1 / Block win-in-1           (1.0 / 0.95)
  - Fork (≥2 strong threats)            (0.90 / 0.85 block)
  - Open Four creation                  (0.85 / 0.80 block)
  - Combo (strong + moderate threat)    (0.70 / 0.65 block)
  - Open Three / Gap Three creation     (0.50 / 0.45 block)
  - Moderate Fork (2× blocked three)    (0.40 / 0.35 block)
  - Pre-fork setup (≥2 open twos)       (0.30 / 0.25 block)
  - Blocked Three / Half-Four           (0.20 / 0.15 block)
  - Open Two                            (0.10)

Uses the pattern scoring engine from :mod:`tactic_patterns` for direction
scanning.  This module is independent of :mod:`mcts_pure` so the pure
MCTS opponent can be kept truly heuristic-free.
"""

import numpy as np

# Import the pattern-scoring primitives from tactic_patterns.
# tactic_patterns does not import from tactic, so no circular dependency.
from tactic_patterns import (
    _scan_direction, _line_score, _DIRECTIONS,
    _SCORE_WIN, _SCORE_OPEN_FOUR, _SCORE_HALF_FOUR,
    _SCORE_OPEN_THREE, _SCORE_BLOCKED_THREE, _SCORE_GAP_THREE,
    _SCORE_OPEN_TWO,
)


# -----------------------------------------------------------------------
# Core helpers (kept for backward compat — used by get_tactic_forced_move)
# -----------------------------------------------------------------------

def is_winning_move(board, move, player):
    """
    Kiểm tra cực nhanh xem nếu đánh 'move', 'player' có đạt 5 quân liên tiếp không.
    """
    h = move // board.width
    w = move % board.width
    n = board.n_in_row
    states = board.states

    # 4 Hướng: Ngang, Dọc, Chéo chính, Chéo phụ
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dh, dw in directions:
        count = 1
        # Đếm tới
        r, c = h + dh, w + dw
        while 0 <= r < board.height and 0 <= c < board.width and states.get(r * board.width + c) == player:
            count += 1
            r += dh
            c += dw
        # Đếm lùi
        r, c = h - dh, w - dw
        while 0 <= r < board.height and 0 <= c < board.width and states.get(r * board.width + c) == player:
            count += 1
            r -= dh
            c -= dw

        if count >= n:
            return True
    return False


def count_line_after_move(board, move, player):
    """Return the longest contiguous line length after `player` plays `move`."""
    h = move // board.width
    w = move % board.width
    states = board.states
    best = 1
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    for dh, dw in directions:
        count = 1
        r, c = h + dh, w + dw
        while 0 <= r < board.height and 0 <= c < board.width and states.get(r * board.width + c) == player:
            count += 1
            r += dh
            c += dw
        r, c = h - dh, w - dw
        while 0 <= r < board.height and 0 <= c < board.width and states.get(r * board.width + c) == player:
            count += 1
            r -= dh
            c -= dw
        best = max(best, count)
    return best


def is_threat_move(board, move, player):
    """Soft tactical signal: move creates at least an n-1 contiguous threat."""
    return count_line_after_move(board, move, player) >= max(2, board.n_in_row - 1)


# -----------------------------------------------------------------------
# Enhanced pattern-based tactical evaluation (uses tactic_patterns engine)
# -----------------------------------------------------------------------

def _evaluate_move_tactical(board, move, player):
    """Evaluate one move for `player` and return a detailed threat breakdown.

    Returns:
        max_score      : highest single-direction score
        strong_threats : count of directions with open three / gap three / better
        moderate_threats : count of directions with blocked three / half four
        building_moves : count of directions with open two
    """
    strong_threats = 0
    moderate_threats = 0
    building_moves = 0
    max_score = 0

    for dh, dw in _DIRECTIONS:
        count, open_ends, has_gap = _scan_direction(board, move, player, dh, dw)
        s = _line_score(count, open_ends, has_gap)
        if s > max_score:
            max_score = s
        if s >= _SCORE_GAP_THREE:
            strong_threats += 1
        elif s >= _SCORE_BLOCKED_THREE:
            moderate_threats += 1
        elif s >= _SCORE_OPEN_TWO:
            building_moves += 1

    return max_score, strong_threats, moderate_threats, building_moves


def _move_tactic_label(board, move, curr_player, opp_player):
    """Compute a tactic label in [0, 1] for a single move.

    The label reflects the BEST tactical purpose of the move, considering
    both attack (for curr_player) and defense (blocking opp_player).

    Label scale (higher = more tactically important):
        1.00  win-in-1
        0.95  block win-in-1
        0.90  fork (≥2 strong attack threats)
        0.85  block fork / create open four
        0.80  block open four
        0.70  combo (strong + moderate attack)
        0.65  block combo
        0.50  create open three / gap three
        0.45  block open three / gap three
        0.40  moderate fork (≥2 moderate threats)
        0.35  block moderate fork
        0.30  pre-fork setup (≥2 open twos)
        0.25  block pre-fork
        0.20  create blocked three / half four
        0.15  block blocked three
        0.10  create open two
    """
    label = 0.0

    # --- Attack evaluation ---
    atk_max, atk_strong, atk_mod, atk_build = \
        _evaluate_move_tactical(board, move, curr_player)

    if atk_max >= _SCORE_WIN:
        return 1.0  # absolute priority

    if atk_strong >= 2:                         # fork
        label = max(label, 0.90)
    elif atk_max >= _SCORE_OPEN_FOUR:           # open four
        label = max(label, 0.85)
    elif atk_strong >= 1 and atk_mod >= 1:      # combo (open3 + blocked3)
        label = max(label, 0.70)
    elif atk_strong >= 1:                       # open three or gap three
        label = max(label, 0.50)
    elif atk_mod >= 2:                          # moderate fork (2× blocked 3)
        label = max(label, 0.40)
    elif atk_build >= 2:                        # pre-fork (2+ open twos)
        label = max(label, 0.30)
    elif atk_mod >= 1:                          # blocked three / half four
        label = max(label, 0.20)
    elif atk_build >= 1 and atk_mod >= 1:       # open two + moderate
        label = max(label, 0.25)
    elif atk_build >= 1:                        # open two
        label = max(label, 0.10)

    # --- Defense evaluation ---
    def_max, def_strong, def_mod, def_build = \
        _evaluate_move_tactical(board, move, opp_player)

    if def_max >= _SCORE_WIN:
        label = max(label, 0.95)    # block win-in-1

    if def_strong >= 2:                         # block fork
        label = max(label, 0.85)
    elif def_max >= _SCORE_OPEN_FOUR:           # block open four
        label = max(label, 0.80)
    elif def_strong >= 1 and def_mod >= 1:      # block combo
        label = max(label, 0.65)
    elif def_strong >= 1:                       # block open/gap three
        label = max(label, 0.45)
    elif def_mod >= 2:                          # block moderate fork
        label = max(label, 0.35)
    elif def_build >= 2:                        # block pre-fork
        label = max(label, 0.25)
    elif def_mod >= 1:                          # block blocked three
        label = max(label, 0.15)

    return label


# -----------------------------------------------------------------------
# Public API (backward-compatible signatures)
# -----------------------------------------------------------------------

def get_tactic_scores(board, win_score=1.0, block_score=0.9,
                      threat_score=0.35, block_threat_score=0.25):
    """Return per-move tactical scores using the enhanced pattern engine.

    Scores are soft labels in [0, 1].  They are used both as auxiliary
    training targets for the tactic head AND as MCTS prior multipliers.

    The enhanced version detects:
      - Win / block win
      - Fork / block fork  (≥2 open threes at once)
      - Open four / block open four
      - Combo / block combo (open three + blocked three)
      - Open three, gap three / block
      - Moderate fork / block (2× blocked three)
      - Pre-fork setup  (≥2 open twos converging)
      - Blocked three, half four / block
      - Open two
    """
    curr_player = board.current_player
    opp_player = (board.players[0] if curr_player == board.players[1]
                  else board.players[1])

    scores = {}
    for move in board.availables:
        label = _move_tactic_label(board, move, curr_player, opp_player)
        if label > 0.0:
            scores[move] = label

    return scores


def get_tactic_label_vector(board):
    """Build a board-size soft label vector for tactical moves."""
    labels = np.zeros(board.width * board.height, dtype=np.float32)
    for move, score in get_tactic_scores(board).items():
        labels[int(move)] = float(score)
    return labels


def apply_tactical_prior_bonus(action_priors, board, bonus_weight=0.35,
                               epsilon=1e-8):
    """Blend tactical scores into policy priors with a soft multiplicative bonus.

    The relative ordering from the neural policy is preserved when no tactical
    signal exists. With signals, priors are multiplied by
    `1 + bonus_weight * score` and renormalized over legal actions.
    """
    action_priors = list(action_priors)
    if not action_priors or bonus_weight <= 0:
        return action_priors

    scores = get_tactic_scores(board)
    if not scores:
        return action_priors

    adjusted = []
    total = 0.0
    for action, prior in action_priors:
        prior = max(float(prior), float(epsilon))
        multiplier = 1.0 + float(bonus_weight) * float(scores.get(int(action), 0.0))
        value = prior * multiplier
        adjusted.append((int(action), value))
        total += value

    if total <= 0.0:
        uniform = 1.0 / len(adjusted)
        return [(action, uniform) for action, _ in adjusted]
    return [(action, value / total) for action, value in adjusted]


def get_tactic_forced_move(board):
    """
    Dựa trên strategy.md: Quét tìm các nước đi bắt buộc (Win hoặc Defense Block).
    Trả về: (tactic_move, is_win)
    """
    curr_player = board.current_player
    opp_player = board.players[0] if curr_player == board.players[1] else board.players[1]

    # Ưu tiên 1: Mình có thể thắng ngay (Win-in-1)
    for move in board.availables:
        if is_winning_move(board, move, curr_player):
            return move, True

    # Ưu tiên 2: Đối thủ có thể thắng ngay -> Bắt buộc phải chặn (Block Win-in-1)
    for move in board.availables:
        if is_winning_move(board, move, opp_player):
            return move, False

    return None, None
