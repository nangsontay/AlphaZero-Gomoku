# -*- coding: utf-8 -*-
"""
Tactical pattern primitives for Gomoku.

This module owns the low-level line/direction scanning engine used by
:mod:`tactic` to label moves and shape MCTS priors.  It was extracted
out of ``mcts_pure`` so the training/tactics pipeline does not depend
on the pure MCTS opponent module — keeping ``mcts_pure`` free to be a
truly pure (textbook) MCTS implementation.

Public symbols (consumed by :mod:`tactic`):
    _SCORE_WIN, _SCORE_OPEN_FOUR, _SCORE_HALF_FOUR,
    _SCORE_OPEN_THREE, _SCORE_BLOCKED_THREE, _SCORE_GAP_THREE,
    _SCORE_OPEN_TWO,
    _DIRECTIONS,
    _scan_direction(board, move, player, dh, dw),
    _line_score(count, open_ends, has_gap=False).
"""

# ---------------------------------------------------------------------------
# Pattern scoring engine — the heart of the tactical labels.
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
