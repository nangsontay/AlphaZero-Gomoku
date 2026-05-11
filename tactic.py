import numpy as np


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


def get_tactic_scores(board, win_score=1.0, block_score=0.9,
                      threat_score=0.35, block_threat_score=0.25):
    """Return per-move tactical scores for immediate wins, blocks, and threats.

    Scores are intentionally soft. They are used both as auxiliary training
    labels and as MCTS prior multipliers, not as hard move overrides.
    """
    curr_player = board.current_player
    opp_player = board.players[0] if curr_player == board.players[1] else board.players[1]
    scores = {}
    for move in board.availables:
        if is_winning_move(board, move, curr_player):
            scores[move] = max(scores.get(move, 0.0), float(win_score))
        if is_winning_move(board, move, opp_player):
            scores[move] = max(scores.get(move, 0.0), float(block_score))

    if scores:
        return scores

    for move in board.availables:
        if is_threat_move(board, move, curr_player):
            scores[move] = max(scores.get(move, 0.0), float(threat_score))
        if is_threat_move(board, move, opp_player):
            scores[move] = max(scores.get(move, 0.0), float(block_threat_score))
    return scores


def get_tactic_label_vector(board):
    """Build a board-size multi-hot/soft label vector for tactical moves."""
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

    # Bạn có thể phát triển thêm Heuristic cho Open Four / Double Threat ở đây sau.
    return None, None
