# -*- coding: utf-8 -*-
"""
ĐẠI CHIẾN: MCTS-AlphaZero (policy-value net) VS MCTS-Pure (random rollout)

Chạy một giải đấu luân phiên màu giữa:
  - MCTS-AlphaZero: tìm kiếm PUCT dẫn dắt bởi mạng policy-value
                    (current_policy.model), giá trị lá lấy từ value head.
  - MCTS-Pure     : MCTS cổ điển, prior đồng đều + rollout ngẫu nhiên,
                    không dùng mạng nơ-ron.

Cách dùng:
    python eval_mcts_vs_pure.py [n_games] [az_playout] [pure_playout]

Mặc định: 10 ván, AlphaZero 400 playout, Pure 2000 playout.
"""

import os
import sys
import json
import time

import numpy as np
import torch

from game import Board
from mcts_alphaZero import MCTSPlayer as AlphaZeroPlayer
from mcts_pure import MCTSPlayer as PureMCTSPlayer
from policy_value_net_mlp import PolicyValueNet


def load_policy(model_file="current_policy.model"):
    """Nạp checkpoint pure-MLP/Mixer giống hệt logic trong play.py."""
    width = height = 15
    if not os.path.exists(model_file):
        raise RuntimeError(f"Không tìm thấy model tại {model_file}")

    state_dict = torch.load(model_file, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"'{model_file}' không phải checkpoint hợp lệ.")

    is_mlp = ("policy_head.weight" in state_dict and
              "embed.proj.weight" in state_dict)
    is_mixer = ("policy_head.weight" in state_dict and
                "embed.weight" in state_dict and
                any(k.startswith("blocks.") for k in state_dict))
    if not (is_mlp or is_mixer):
        raise RuntimeError(
            f"Từ chối nạp '{model_file}': không phải checkpoint pure-MLP.")

    backbone = "mixer" if is_mixer else "mlp"
    if backbone == "mixer":
        policy_out = int(state_dict["blocks.0.token_mix.0.weight"].shape[1])
    else:
        policy_out = int(state_dict["policy_head.weight"].shape[0])
    inferred = int(round(policy_out ** 0.5))
    if inferred * inferred != policy_out:
        raise RuntimeError("Bàn cờ không vuông — không được hỗ trợ.")
    width = height = inferred

    kwargs = {"backbone": backbone}
    if backbone == "mixer":
        kwargs["in_channels"] = int(state_dict["embed.weight"].shape[1])
        sidecar = model_file + ".json"
        if not os.path.exists(sidecar):
            raise RuntimeError(
                f"Checkpoint mixer cần sidecar '{sidecar}'.")
        with open(sidecar) as f:
            meta = json.load(f)
        mixer = meta.get("mixer") or {}
        kwargs.update({
            "mixer_dim": mixer.get("dim", 128),
            "mixer_depth": mixer.get("depth", 6),
            "mixer_token_hidden": mixer.get("token_hidden", 256),
            "mixer_ch_hidden": mixer.get("ch_hidden", 384),
            "mixer_value_hidden": mixer.get("value_hidden", 128),
            "mixer_dropout": mixer.get("dropout", 0.1),
        })
    else:
        kwargs["in_channels"] = int(state_dict["embed.proj.weight"].shape[1])

    policy = PolicyValueNet(
        width, height, model_file=model_file, use_gpu=True,
        search_d4_random=False,  # eval determinism
        **kwargs,
    )
    return policy, width, height, kwargs["in_channels"]


def play_one_game(board, az_player, pure_player, az_is_black,
                   take_center=True):
    """Đánh một ván headless. Trả về (winner_tag, n_moves).

    winner_tag: 'AZ', 'PURE', hoặc 'DRAW'.
    az_is_black: True nếu AlphaZero cầm Đen (đi trước).
    """
    board.init_board(0)  # players[0] luôn đi trước
    p1, p2 = board.players  # p1 = Đen (đi trước)

    if az_is_black:
        az_player.set_player_ind(p1)
        pure_player.set_player_ind(p2)
    else:
        az_player.set_player_ind(p2)
        pure_player.set_player_ind(p1)
    players = {az_player.player: az_player, pure_player.player: pure_player}

    n_moves = 0
    first_move_done = False
    while True:
        cur = board.get_current_player()
        player_in_turn = players[cur]

        # Khai cuộc: chiếm ô trung tâm nếu còn trống (giống ảnh mẫu).
        center = (board.height // 2) * board.width + (board.width // 2)
        if (take_center and not first_move_done and
                center in board.availables):
            move = center
            who = "AZ" if player_in_turn is az_player else "PURE"
            print(f"  \U0001F52E Trực giác: Ô trung tâm còn trống, "
                  f"{who} chiếm luôn!")
        else:
            move = player_in_turn.get_action(board)
        first_move_done = True

        board.do_move(move)
        n_moves += 1

        # Báo nước đi cho đối thủ để tái sử dụng cây tìm kiếm (nếu có).
        other = pure_player if player_in_turn is az_player else az_player
        if hasattr(other, "notify_opponent_move"):
            other.notify_opponent_move(move)

        end, winner = board.game_end()
        if end:
            if winner == -1:
                return "DRAW", n_moves
            return ("AZ" if winner == az_player.player else "PURE"), n_moves


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    az_playout = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    pure_playout = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    model_file = "current_policy.model"

    print("\U0001F4E6 Đang nạp các đấu thủ vào RAM...")
    policy, width, height, in_ch = load_policy(model_file)
    print(f"   Model: {model_file} | backbone in_channels={in_ch} "
          f"| bàn {width}x{height}")
    print()
    print(f"\U0001F680 BẮT ĐẦU ĐẠI CHIẾN: "
          f"MCTS-AlphaZero ({az_playout} playout) VS "
          f"MCTS-Pure ({pure_playout} playout)")
    print("=" * 78)

    az_wins = pure_wins = draws = 0
    for g in range(1, n_games + 1):
        # Đấu thủ mới mỗi ván để cây tìm kiếm không bị lẫn trạng thái cũ.
        az_player = AlphaZeroPlayer(
            policy.policy_value_fn,
            policy_value_batch_function=policy.policy_value,
            c_puct=5, n_playout=az_playout)
        pure_player = PureMCTSPlayer(c_puct=5, n_playout=pure_playout)
        board = Board(width=width, height=height, n_in_row=5,
                      in_channels=in_ch)

        az_is_black = (g % 2 == 1)  # luân phiên màu mỗi ván
        t0 = time.time()
        tag, n_moves = play_one_game(board, az_player, pure_player,
                                     az_is_black)
        dt = time.time() - t0

        az_color = "Đen" if az_is_black else "Trắng"
        pure_color = "Trắng" if az_is_black else "Đen"
        if tag == "AZ":
            result = f"AlphaZero ({az_color}) THẮNG \U0001F3C6"
            az_wins += 1
        elif tag == "PURE":
            result = f"Pure ({pure_color}) THẮNG \U0001F3C6"
            pure_wins += 1
        else:
            result = "HÒA"
            draws += 1

        print(f"Trận {g:02d} | AlphaZero cầm {az_color:5s} | {result:32s}"
              f" | Số bước: {n_moves:3d} | {dt:7.2f}s")

    print()
    print("=" * 78)
    print("\U0001F4CA TỔNG KẾT LUÂN PHIÊN")
    print("=" * 78)
    print(f"  - AlphaZero thắng : {az_wins}/{n_games}")
    print(f"  - Pure MCTS thắng : {pure_wins}/{n_games}")
    print(f"  - Hòa             : {draws}/{n_games}")


if __name__ == "__main__":
    main()
