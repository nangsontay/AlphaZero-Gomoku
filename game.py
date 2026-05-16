# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np

from tactic import (get_tactic_label_vector, is_winning_move,
                    get_tactic_forced_move)


class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 15))
        self.height = int(kwargs.get('height', 15))
        self.in_channels = int(kwargs.get('in_channels', 4))
        self.states = {}
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]
        self.availables = []
        self._available_set = set()
        self._available_pos = {}

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # Keep available moves list-compatible for policy indexing, while
        # maintaining a set and position map for O(1) remove/restore.
        self.availables = list(range(self.width * self.height))
        self._available_set = set(self.availables)
        self._available_pos = {move: idx for idx, move in enumerate(self.availables)}
        self.states = {}
        self.last_move = -1

    def copy_fast(self):
        """Return a fast Board copy for MCTS simulations.

        Rule constants are copied by reference/value, while the mutable move
        tracking containers that do_move() mutates are cloned explicitly.
        """
        new = Board.__new__(Board)
        new.width = self.width
        new.height = self.height
        new.in_channels = self.in_channels
        new.n_in_row = self.n_in_row
        new.players = self.players
        new.states = dict(self.states)
        new.availables = list(self.availables)
        new._available_set = set(self._available_set)
        new._available_pos = dict(self._available_pos)
        new.current_player = self.current_player
        new.last_move = self.last_move
        return new

    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

    def current_state(self, in_channels=None):
        """return the board state from the perspective of the current player.

        Channels 0-3 preserve the legacy AlphaZero representation:
        current-player stones, opponent stones, last move, and side-to-play.
        When ``in_channels >= 5``, channel 4 is ``opp_win_here``: legal moves
        where the opponent would win immediately if the current player does not
        block.
        """

        channels = self.in_channels if in_channels is None else int(in_channels)
        if channels < 4:
            raise ValueError('current_state requires at least 4 channels')
        square_state = np.zeros((channels, self.width, self.height), dtype=np.float32)
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width,
                            move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width,
                            move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width,
                            self.last_move % self.height] = 1.0
        # Channel 3 encodes the side to move. Use the actual current_player so
        # the encoding is correct even when start_player != 0 (i.e. when move
        # parity doesn't line up with the canonical "first player to move"
        # convention).
        if self.current_player == self.players[0]:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        if channels >= 5 and self.availables:
            opp_player = (self.players[0] if self.current_player == self.players[1]
                          else self.players[1])
            for move in self.availables:
                if is_winning_move(self, move, opp_player):
                    square_state[4][move // self.width,
                                    move % self.height] = 1.0
        return square_state

    def do_move(self, move):
        if move not in self._available_set:
            raise ValueError('move {} is not available'.format(move))

        removed_index = self._available_pos[move]
        last_available = self.availables[-1]

        self.availables[removed_index] = last_available
        self._available_pos[last_available] = removed_index
        self.availables.pop()
        del self._available_pos[move]
        self._available_set.remove(move)

        self.states[move] = self.current_player
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(states.keys())
        if len(moved) < self.n_in_row *2-1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1) and
                    len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player


class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                loc = i * width + j
                p = board.states.get(loc, -1)
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1,
                   move_log_prefix=None):
        """start a game between two players

        If move_log_prefix is a non-empty string, print a per-move progress line
        (with elapsed time per move and cumulative time) prefixed with it. This
        is useful for showing that long-running evaluation games are making
        progress and not deadlocked.
        """
        import time as _time
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        # Drop any stale MCTS tree carried over from a previous game.
        for _pl in (player1, player2):
            if hasattr(_pl, 'reset_player'):
                _pl.reset_player()
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        log_moves = bool(move_log_prefix)
        game_start_t = _time.time() if log_moves else 0.0
        move_idx = 0
        while True:
            current_player = self.board.get_current_player()
            player_in_turn = players[current_player]
            move_start_t = _time.time() if log_moves else 0.0
            move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            # Keep the opponent's MCTS tree in sync with the actual move.
            other_player = players[p2 if current_player == p1 else p1]
            if hasattr(other_player, 'notify_opponent_move'):
                other_player.notify_opponent_move(move)
            if log_moves:
                move_idx += 1
                move_elapsed = _time.time() - move_start_t
                total_elapsed = _time.time() - game_start_t
                player_name = type(player_in_turn).__module__ + "." + \
                    type(player_in_turn).__name__
                print(
                    "{} move {:>3} by {} (player {}): action={}, "
                    "move_elapsed={:.1f}s, total_elapsed={:.1f}s".format(
                        move_log_prefix, move_idx, player_name,
                        current_player, move, move_elapsed, total_elapsed),
                    flush=True,
                )
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3,
                        temperature_moves=None, temp_high=1.0,
                        temp_low=1e-3, return_tactic_labels=False):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training.
        If return_tactic_labels is true, return
        (state, mcts_probs, z, tactic_label) samples.

        temp remains as the backward-compatible fixed-temperature path when
        temperature_moves is None. Pass temperature_moves=0 to use temp_low
        for the whole game, or a positive value to use temp_high for the first
        temperature_moves plies and temp_low afterwards.
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players, tactic_labels = [], [], [], []
        move_idx = 0
        while True:
            cur_temp = temp
            if temperature_moves is not None:
                cur_temp = temp_high if move_idx < temperature_moves else temp_low
            move, move_probs = player.get_action(self.board,
                                                 temp=cur_temp,
                                                 return_prob=1)
            # Patch (a): expert-iteration label shaping. When a forced move
            # exists at THIS state (own win-in-1 or block opponent win-in-1,
            # symmetric via get_tactic_forced_move), sharpen the POLICY TARGET
            # to one-hot on it. The move actually played stays the pure-MCTS
            # `move`, and the value target `winners_z` is untouched (the game
            # is played out for real), so this is supervised label shaping,
            # NOT a heuristic in the play/search loop.
            forced_move, _is_win = get_tactic_forced_move(self.board)
            if forced_move is not None:
                shaped = np.zeros_like(move_probs)
                shaped[int(forced_move)] = 1.0
                move_probs = shaped
            # store the data
            states.append(self.board.current_state().copy())
            mcts_probs.append(move_probs)
            current_players.append(self.board.current_player)
            if return_tactic_labels:
                tactic_labels.append(get_tactic_label_vector(self.board))
            # perform a move
            self.board.do_move(move)
            move_idx += 1
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.game_end()
            if end:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                if return_tactic_labels:
                    return winner, zip(states, mcts_probs, winners_z,
                                       tactic_labels)
                return winner, zip(states, mcts_probs, winners_z)
