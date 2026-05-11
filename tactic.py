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
