# -*- coding: utf-8 -*-
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
import torch
import json
import os
from game import Board
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net_mlp import PolicyValueNet

class HumanPlayer(object):
    """
    Người chơi (Human) thông qua click giao diện.
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def __str__(self):
        return "Human"

class RandomPlayer(object):
    """
    Agent chơi ngẫu nhiên, tuân thủ luật.
    """
    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        return random.choice(board.availables)

    def __str__(self):
        return "Random Player {}".format(self.player)

class GomokuGUI:
    """
    Giao diện Tkinter đấu 1 ván giữa MCTS AlphaZero và đối thủ tùy chọn.
    """
    def __init__(self, root, width, height, n_in_row, policy_value_fn):
        self.root = root
        self.width = width
        self.height = height
        self.n_in_row = n_in_row
        
        self.board = Board(width=self.width, height=self.height, n_in_row=self.n_in_row)
        self.policy_value_fn = policy_value_fn
        self.ai_player = None
        
        # Biến lưu trữ đối thủ và lượt đánh
        self.ai_level = tk.StringVar(value="Level 9 (1000 Playouts)")
        self.opponent_type = tk.StringVar(value="Human")
        self.first_player = tk.StringVar(value="AI (Black)")
        self.last_move_marker = None
        self.is_playing = False
        
        self._setup_ui()
        self.draw_board()

    def _setup_ui(self):
        # Control Panel
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        tk.Label(control_frame, text="AI Level:", font=("Arial", 12)).grid(row=0, column=0, padx=5)
        level_menu = ttk.Combobox(control_frame, textvariable=self.ai_level, values=["Level 9 (1000 Playouts)", "Level 10 (10000 Playouts)"], state="readonly", width=22)
        level_menu.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        tk.Label(control_frame, text="Opponent:", font=("Arial", 12)).grid(row=1, column=0, padx=5)
        opponent_menu = ttk.Combobox(control_frame, textvariable=self.opponent_type, values=["Human", "Random", "MCTS Pure"], state="readonly", width=12)
        opponent_menu.grid(row=1, column=1, padx=5)

        tk.Label(control_frame, text="First Move:", font=("Arial", 12)).grid(row=1, column=2, padx=5)
        first_move_menu = ttk.Combobox(control_frame, textvariable=self.first_player, values=["AI (Black)", "Opponent (Black)"], state="readonly", width=15)
        first_move_menu.grid(row=1, column=3, padx=5)

        self.start_btn = tk.Button(control_frame, text="Start Game", command=self.start_game, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
        self.start_btn.grid(row=1, column=4, padx=15)

        self.info_label = tk.Label(self.root, text="Select settings and press Start Game", font=("Arial", 14, "bold"))
        self.info_label.pack(pady=5)

        # Canvas cho bàn cờ
        self.cell_size = 40
        self.canvas = tk.Canvas(self.root, width=self.width * self.cell_size, height=self.height * self.cell_size, bg='#F5DEB3')
        self.canvas.pack(padx=20, pady=10)
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.width):
            self.canvas.create_line(self.cell_size/2 + i*self.cell_size, self.cell_size/2, 
                                    self.cell_size/2 + i*self.cell_size, self.height*self.cell_size - self.cell_size/2)
        for i in range(self.height):
            self.canvas.create_line(self.cell_size/2, self.cell_size/2 + i*self.cell_size, 
                                    self.width*self.cell_size - self.cell_size/2, self.cell_size/2 + i*self.cell_size)

    def draw_piece(self, move, player):
        h = move // self.width
        w = move % self.width
        x = self.cell_size/2 + w * self.cell_size
        y = self.cell_size/2 + h * self.cell_size
        
        # Vẽ quân cờ
        color = "black" if player == 1 else "white"
        self.canvas.create_oval(x - 15, y - 15, x + 15, y + 15, fill=color, tags="piece")

        # Xóa highlight cũ và vẽ highlight mới (viền đỏ quanh quân cờ vừa đánh)
        self.canvas.delete("highlight")
        self.canvas.create_oval(x - 17, y - 17, x + 17, y + 17, outline="red", width=2, tags="highlight")
        # Hoặc dùng chấm đỏ ở giữa quân cờ: self.canvas.create_oval(x - 4, y - 4, x + 4, y + 4, fill="red", outline="red", tags="highlight")

    def start_game(self):
        if self.is_playing:
            return
        
        self.is_playing = True
        self.start_btn.config(state="disabled")
        self.draw_board()

        # Khởi tạo AI dựa trên Level
        if self.ai_level.get() == "Level 9 (1000 Playouts)":
            self.ai_player = MCTSPlayer(self.policy_value_fn, c_puct=5, n_playout=1000)
        else:
            self.ai_player = MCTSPlayer(self.policy_value_fn, c_puct=5, n_playout=10000)

        # Khởi tạo đối thủ dựa trên lựa chọn UI
        if self.opponent_type.get() == "Human":
            self.opponent = HumanPlayer()
        elif self.opponent_type.get() == "Random":
            self.opponent = RandomPlayer()
        else:
            # Use truly pure MCTS (uniform priors + random rollout, no
            # tactical heuristics) as the canonical evaluation opponent.
            self.opponent = MCTS_Pure(c_puct=5, n_playout=2000)

        # Xác định ai đi trước (Black luôn đi trước trong Gomoku)
        self.board.init_board(0) # 0 nghĩa là p1 luôn đi trước
        p1, p2 = self.board.players
        
        if self.first_player.get() == "AI (Black)":
            self.ai_player.set_player_ind(p1)
            self.opponent.set_player_ind(p2)
            self.players = {p1: self.ai_player, p2: self.opponent}
        else:
            self.ai_player.set_player_ind(p2)
            self.opponent.set_player_ind(p1)
            self.players = {p2: self.ai_player, p1: self.opponent}
            
        self.info_label.config(text="Game Started! AI vs " + self.opponent_type.get())
        
        # Bắt đầu vòng lặp đánh cờ
        self.root.after(100, self.play_turn)

    def check_game_end(self):
        end, winner = self.board.game_end()
        if end:
            self.is_playing = False
            self.start_btn.config(state="normal")
            
            if winner == self.ai_player.player:
                self.info_label.config(text="GAME OVER: AI MCTS AlphaZero WINS!")
            elif winner == -1:
                self.info_label.config(text="GAME OVER: TIE!")
            else:
                self.info_label.config(text=f"GAME OVER: {self.opponent_type.get()} WINS!")
            return True
        return False

    def play_turn(self):
        if not self.is_playing:
            return
            
        current_player = self.board.get_current_player()
        player_in_turn = self.players[current_player]
        
        if isinstance(player_in_turn, HumanPlayer):
            self.info_label.config(text="Your turn. Please click on the board.")
            return # Đợi human click
        
        # Cập nhật thông báo trạng thái
        player_name = "AI" if player_in_turn == self.ai_player else self.opponent_type.get()
        self.info_label.config(text=f"{player_name} is thinking...")
        self.root.update() # Bắt buộc update UI ngay lập tức trước khi gọi hàm get_action nặng

        # Lấy nước đi
        move = player_in_turn.get_action(self.board)
        self.board.do_move(move)

        # Notify the OTHER player about this move so it can reuse its tree.
        other_player = (
            self.opponent if player_in_turn == self.ai_player
            else self.ai_player
        )
        if hasattr(other_player, 'notify_opponent_move'):
            other_player.notify_opponent_move(move)
        
        # Cập nhật UI
        self.draw_piece(move, current_player)
        
        if not self.check_game_end():
            # Nghỉ 0.1s rồi cho người kia đánh
            self.root.after(100, self.play_turn)

    def on_canvas_click(self, event):
        if not self.is_playing:
            return
            
        current_player = self.board.get_current_player()
        player_in_turn = self.players[current_player]
        
        if isinstance(player_in_turn, HumanPlayer):
            # Tính toán vị trí click
            w = int(round((event.x - self.cell_size/2) / self.cell_size))
            h = int(round((event.y - self.cell_size/2) / self.cell_size))
            
            # Kiểm tra click có trong bàn cờ không
            if 0 <= w < self.width and 0 <= h < self.height:
                move = h * self.width + w
                if move in self.board.availables:
                    self.board.do_move(move)
                    self.draw_piece(move, current_player)

                    # Notify other players about the human move for tree reuse.
                    for p in self.players.values():
                        if p is not player_in_turn and hasattr(p, 'notify_opponent_move'):
                            p.notify_opponent_move(move)
                    
                    if not self.check_game_end():
                        # Chuyển lượt cho AI/máy
                        self.root.after(100, self.play_turn)

def run():
    n = 5
    width, height = 15, 15
    model_file = 'current_policy.model'

    best_policy = None
    try:
        state_dict = torch.load(model_file, map_location="cpu")
        if isinstance(state_dict, dict):
            is_mlp_checkpoint = (
                "policy_head.weight" in state_dict and
                "embed.proj.weight" in state_dict
            )
            is_mixer_checkpoint = (
                "policy_head.weight" in state_dict and
                "embed.weight" in state_dict and
                any(k.startswith("blocks.") for k in state_dict)
            )
            if is_mlp_checkpoint or is_mixer_checkpoint:
                backbone = "mixer" if is_mixer_checkpoint else "mlp"
                if backbone == "mixer":
                    policy_out = int(state_dict["blocks.0.token_mix.0.weight"].shape[1])
                else:
                    policy_out = int(state_dict["policy_head.weight"].shape[0])
                inferred = int(round(policy_out ** 0.5))
                # E08 guard: refuse non-square boards.
                if inferred * inferred != policy_out:
                    raise RuntimeError(
                        f"Refusing to load '{model_file}': policy_head.weight "
                        f"shape [{policy_out}, ...] is not a square; non-square "
                        f"boards are not supported in this MLP build."
                    )
                # A12 consistency check: warn (and adopt) if checkpoint disagrees.
                if inferred != width or inferred != height:
                    print(
                        f"WARNING: checkpoint board size {inferred}x{inferred} "
                        f"does not match defaults {width}x{height}; adopting "
                        f"checkpoint dimensions."
                    )
                    width = height = inferred
                kwargs = {"backbone": backbone}
                if backbone == "mixer":
                    inferred_in_channels = int(state_dict["embed.weight"].shape[1])
                else:
                    inferred_in_channels = int(state_dict["embed.proj.weight"].shape[1])
                kwargs["in_channels"] = inferred_in_channels
                if backbone == "mixer":
                    sidecar_path = model_file + ".json"
                    if not os.path.exists(sidecar_path):
                        raise RuntimeError(
                            f"Refusing to load '{model_file}': mixer checkpoint "
                            f"requires sidecar '{sidecar_path}' for architecture "
                            f"parameters."
                        )
                    with open(sidecar_path) as f:
                        meta = json.load(f)
                    if meta.get("backbone") != "mixer" or \
                       meta.get("MLP_ARCH_VERSION") != "2.0.0-mixer":
                        raise RuntimeError(
                            f"Refusing to load '{model_file}': mixer checkpoint "
                            f"sidecar does not declare backbone='mixer' and "
                            f"MLP_ARCH_VERSION='2.0.0-mixer'."
                        )
                    mixer = meta.get("mixer") or {}
                    kwargs.update({
                        "mixer_dim": mixer.get("dim", 128),
                        "mixer_depth": mixer.get("depth", 6),
                        "mixer_token_hidden": mixer.get("token_hidden", 256),
                        "mixer_ch_hidden": mixer.get("ch_hidden", 384),
                        "mixer_value_hidden": mixer.get("value_hidden", 128),
                        "mixer_dropout": mixer.get("dropout", 0.1),
                    })
                best_policy = PolicyValueNet(
                    width, height, model_file=model_file, use_gpu=False,
                    search_d4_random=False,  # eval determinism (Cluster F / A16)
                    **kwargs,
                )
            else:
                print(
                    "Refusing to load '{}': not a pure-MLP checkpoint. "
                    "The no-CNN constraint forbids loading legacy CNN files."
                    .format(model_file)
                )
    except Exception as exc:
        print("Failed to inspect '{}': {}".format(model_file, exc))

    if best_policy is None:
        # GATE-2 closed (2026-05-07): numpy fallback removed.
        # E09 guard: explicit controlled failure instead of legacy pickle path.
        raise RuntimeError(
            f"Failed to load '{model_file}': not a pure-MLP checkpoint. "
            f"The numpy CNN-shaped fallback was removed under the strict "
            f"no-CNN reading (GATE-1=strict-everywhere, GATE-2=remove)."
        )

    root = tk.Tk()
    root.title("Gomoku MCTS AlphaZero Evaluation")
    
    app = GomokuGUI(root, width, height, n, best_policy.policy_value_fn)
    
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()

if __name__ == '__main__':
    run()
