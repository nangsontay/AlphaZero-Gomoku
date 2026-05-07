# -*- coding: utf-8 -*-
"""
Multiprocess training pipeline for AlphaZero Gomoku with live worker progress.

Main process:
  - keeps PolicyValueNet on GPU when --no-gpu is not used
  - performs policy_update()

Worker processes:
  - run self-play on CPU
  - each worker loads a CPU snapshot of the current model
  - return augmented play data to the main process

Run:
  env -u LD_LIBRARY_PATH python3 train_mp_progress.py --num-workers 6 --games-per-worker 1 --threads-per-worker 1
"""

from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import random
import time
import traceback
from collections import defaultdict, deque

# Set thread limits before importing numpy/torch where possible.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_mlp import PolicyValueNet


def set_worker_thread_limits(num_threads=1):
    """Avoid oversubscribing CPU threads inside each worker process."""
    num_threads = max(1, int(num_threads))

    os.environ["OMP_NUM_THREADS"] = str(num_threads)
    os.environ["MKL_NUM_THREADS"] = str(num_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    try:
        torch.set_num_threads(num_threads)
    except Exception:
        pass


def get_equi_data(play_data, board_width, board_height):
    """Augment self-play data by rotations and flips."""
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                np.flipud(mcts_prob.reshape(board_height, board_width)),
                i
            )
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten(),
                winner
            ))

            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten(),
                winner
            ))

    return extend_data


def selfplay_worker(args):
    """Worker entrypoint. Keep top-level for multiprocessing spawn."""
    worker_id = int(args["worker_id"])

    try:
        set_worker_thread_limits(args.get("threads_per_worker", 1))

        seed = int(args.get("seed", 0)) + worker_id
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)

        board_width = int(args["board_width"])
        board_height = int(args["board_height"])
        n_in_row = int(args["n_in_row"])
        n_playout = int(args["n_playout"])
        c_puct = float(args["c_puct"])
        temp = float(args["temp"])
        temperature_moves = args.get("temperature_moves", None)
        if temperature_moves is not None:
            temperature_moves = int(temperature_moves)
        temp_high = float(args.get("temp_high", 1.0))
        temp_low = float(args.get("temp_low", 1e-3))
        n_games = int(args["n_games"])
        model_file = args["model_file"]
        dirichlet_alpha = float(args.get("dirichlet_alpha", 0.05))
        noise_eps = float(args.get("noise_eps", 0.25))

        print(
            "[worker {}] start: games={}, n_playout={}, pid={}".format(
                worker_id,
                n_games,
                n_playout,
                os.getpid()
            ),
            flush=True
        )

        board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
        game = Game(board)

        # Workers use CPU. Main process trains on GPU.
        policy_value_net = PolicyValueNet(
            board_width,
            board_height,
            model_file=model_file,
            use_gpu=False
        )

        mcts_player = MCTSPlayer(
            policy_value_net.policy_value_fn,
            c_puct=c_puct,
            n_playout=n_playout,
            is_selfplay=1,
            dirichlet_alpha=dirichlet_alpha,
            noise_eps=noise_eps
        )

        all_data = []
        episode_lens = []
        start = time.time()

        for game_idx in range(n_games):
            game_start = time.time()
            winner, play_data = game.start_self_play(
                mcts_player,
                temp=temp,
                temperature_moves=temperature_moves,
                temp_high=temp_high,
                temp_low=temp_low)
            play_data = list(play_data)
            episode_lens.append(len(play_data))
            all_data.extend(get_equi_data(play_data, board_width, board_height))

            print(
                "[worker {}] game {}/{} done: winner={}, episode_len={}, {:.1f}s".format(
                    worker_id,
                    game_idx + 1,
                    n_games,
                    winner,
                    len(play_data),
                    time.time() - game_start
                ),
                flush=True
            )

        elapsed = time.time() - start
        print(
            "[worker {}] done: games={}, augmented_positions={}, {:.1f}s".format(
                worker_id,
                len(episode_lens),
                len(all_data),
                elapsed
            ),
            flush=True
        )

        return {
            "ok": True,
            "worker_id": worker_id,
            "data": all_data,
            "episode_lens": episode_lens,
            "elapsed": elapsed,
        }

    except Exception as exc:
        print(
            "[worker {}] ERROR: {}\n{}".format(
                worker_id,
                exc,
                traceback.format_exc()
            ),
            flush=True
        )
        return {
            "ok": False,
            "worker_id": worker_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "data": [],
            "episode_lens": [],
            "elapsed": 0.0,
        }


class TrainPipeline(object):
    def __init__(
        self,
        init_model=None,
        use_gpu=True,
        num_workers=6,
        games_per_worker=1,
        threads_per_worker=1,
        n_playout=400,
        batch_size=1024,
        game_batch_num=1500,
        check_freq=50,
        eval_games=10,
        dirichlet_alpha=0.05,
        noise_eps=0.25,
        temperature_moves=8,
        temp_high=1.0,
        temp_low=1e-3,
        worker_model_file="./_tmp_selfplay_policy.model",
    ):
        self.use_gpu = bool(use_gpu)
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA không khả dụng. Kiểm tra NVIDIA driver, nvidia-smi và bản PyTorch CUDA."
            )

        print("Using GPU:", self.use_gpu, flush=True)
        if self.use_gpu:
            print("GPU name:", torch.cuda.get_device_name(0), flush=True)

        self.num_workers = max(1, int(num_workers))
        self.games_per_worker = max(1, int(games_per_worker))
        self.threads_per_worker = max(1, int(threads_per_worker))
        self.worker_model_file = worker_model_file

        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(
            width=self.board_width,
            height=self.board_height,
            n_in_row=self.n_in_row
        )
        self.game = Game(self.board)

        self.learn_rate = 5e-4
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = int(n_playout)
        self.c_puct = 5
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.noise_eps = float(noise_eps)
        self.temperature_moves = int(temperature_moves) if temperature_moves is not None else None
        self.temp_high = float(temp_high)
        self.temp_low = float(temp_low)
        self.buffer_size = 10000
        self.batch_size = int(batch_size)
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = self.num_workers * self.games_per_worker

        self.epochs = 5
        self.kl_targ = 0.03
        self.global_update_count = 0
        self.check_freq = int(check_freq)
        self.game_batch_num = int(game_batch_num)
        self.eval_games = int(eval_games)
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

        if init_model:
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                model_file=init_model,
                use_gpu=self.use_gpu,
                sym_loss_weight=0.05
            )
        else:
            self.policy_value_net = PolicyValueNet(
                self.board_width,
                self.board_height,
                use_gpu=self.use_gpu,
                sym_loss_weight=0.05
            )

        self.mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout,
            is_selfplay=1,
            dirichlet_alpha=self.dirichlet_alpha,
            noise_eps=self.noise_eps
        )

    def save_cpu_model_for_workers(self):
        state_dict = self.policy_value_net.get_policy_param()
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if hasattr(value, "detach"):
                cpu_state_dict[key] = value.detach().cpu()
            else:
                cpu_state_dict[key] = value
        torch.save(cpu_state_dict, self.worker_model_file)

    def build_worker_tasks(self):
        base_seed = int(time.time() * 1000000) % (2 ** 31 - 1)
        return [
            {
                "worker_id": worker_id,
                "seed": base_seed,
                "model_file": self.worker_model_file,
                "n_games": self.games_per_worker,
                "board_width": self.board_width,
                "board_height": self.board_height,
                "n_in_row": self.n_in_row,
                "n_playout": self.n_playout,
                "c_puct": self.c_puct,
                "temp": self.temp,
                "dirichlet_alpha": self.dirichlet_alpha,
                "noise_eps": self.noise_eps,
                "temperature_moves": self.temperature_moves,
                "temp_high": self.temp_high,
                "temp_low": self.temp_low,
                "threads_per_worker": self.threads_per_worker,
            }
            for worker_id in range(self.num_workers)
        ]

    def collect_selfplay_data_parallel(self, pool=None):
        self.save_cpu_model_for_workers()
        tasks = self.build_worker_tasks()

        print(
            "self-play batch start: workers={}, games_per_worker={}, n_playout={}".format(
                self.num_workers,
                self.games_per_worker,
                self.n_playout
            ),
            flush=True
        )

        if pool is None:
            results_iter = map(selfplay_worker, tasks)
        else:
            # imap_unordered lets the main process print progress as each worker finishes.
            results_iter = pool.imap_unordered(selfplay_worker, tasks)

        episode_lens = []
        total_games = 0
        total_augmented_positions = 0
        worker_times = []
        completed = 0

        for result in results_iter:
            completed += 1

            if not result.get("ok", False):
                raise RuntimeError(
                    "Worker {} failed: {}\n{}".format(
                        result.get("worker_id"),
                        result.get("error"),
                        result.get("traceback")
                    )
                )

            data = result["data"]
            lens = result["episode_lens"]

            self.data_buffer.extend(data)
            episode_lens.extend(lens)
            total_games += len(lens)
            total_augmented_positions += len(data)
            worker_times.append(float(result["elapsed"]))

            print(
                "self-play progress: {}/{} worker(s) returned, data_buffer={}".format(
                    completed,
                    self.num_workers,
                    len(self.data_buffer)
                ),
                flush=True
            )

        self.episode_len = float(np.mean(episode_lens)) if episode_lens else 0.0

        print(
            "self-play batch done: workers={}, games={}, avg_episode_len={:.1f}, "
            "augmented_positions={}, slowest_worker={:.1f}s".format(
                self.num_workers,
                total_games,
                self.episode_len,
                total_augmented_positions,
                max(worker_times) if worker_times else 0.0
            ),
            flush=True
        )

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]

        # Counter normalisation rule: warmup is anchored on
        # `global_update_count` defined as "number of `train_step` invocations
        # on the main trainer's network". This counter is a property of the
        # trainer, NOT the worker count. Same convention applies in train.py
        # and train_gpu_evaluator.py.
        warmup_steps = 500
        if self.global_update_count < warmup_steps:
            warmup_lr = self.learn_rate * (self.global_update_count + 1) / warmup_steps
        else:
            warmup_lr = self.learn_rate

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)

        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch,
                mcts_probs_batch,
                winner_batch,
                warmup_lr * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(
                old_probs * (
                    np.log(old_probs + 1e-10) -
                    np.log(new_probs + 1e-10)
                ),
                axis=1
            ))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        self.global_update_count += 1

        winner_np = np.array(winner_batch)
        winner_var = np.var(winner_np)

        if winner_var > 1e-12:
            explained_var_old = (
                1 - np.var(winner_np - old_v.flatten()) / winner_var
            )
            explained_var_new = (
                1 - np.var(winner_np - new_v.flatten()) / winner_var
            )
        else:
            explained_var_old = 0.0
            explained_var_new = 0.0

        print((
            "kl:{:.5f},"
            "lr_multiplier:{:.3f},"
            "loss:{},"
            "entropy:{},"
            "explained_var_old:{:.3f},"
            "explained_var_new:{:.3f}"
        ).format(
            kl,
            self.lr_multiplier,
            loss,
            entropy,
            explained_var_old,
            explained_var_new
        ), flush=True)
        return loss, entropy

    def policy_evaluate(self, n_games=None):
        if n_games is None:
            n_games = self.eval_games

        if n_games <= 0:
            return self.best_win_ratio

        current_mcts_player = MCTSPlayer(
            self.policy_value_net.policy_value_fn,
            c_puct=self.c_puct,
            n_playout=self.n_playout
        )
        pure_mcts_player = MCTS_Pure(
            c_puct=5,
            n_playout=self.pure_mcts_playout_num
        )

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(
                current_mcts_player,
                pure_mcts_player,
                start_player=i % 2,
                is_shown=0
            )
            win_cnt[winner] += 1

        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num,
            win_cnt[1],
            win_cnt[2],
            win_cnt[-1]
        ), flush=True)
        return win_ratio

    def run(self):
        pool = None
        try:
            if self.num_workers > 1:
                ctx = mp.get_context("spawn")
                pool = ctx.Pool(processes=self.num_workers, maxtasksperchild=1)
                print(
                    "Multiprocess self-play: {} CPU workers, {} game(s)/worker".format(
                        self.num_workers,
                        self.games_per_worker
                    ),
                    flush=True
                )
            else:
                print("Sequential self-play: 1 worker", flush=True)

            for i in range(self.game_batch_num):
                self.collect_selfplay_data_parallel(pool=pool)

                print("batch i:{}, data_buffer:{}".format(
                    i + 1,
                    len(self.data_buffer)
                ), flush=True)

                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                    self.policy_value_net.save_model("./current_policy.model")

                if (i + 1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i + 1), flush=True)
                    win_ratio = self.policy_evaluate(self.eval_games)
                    self.policy_value_net.save_model("./current_policy.model")

                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!", flush=True)
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model("./best_policy.model")
                        if (
                            self.best_win_ratio == 1.0 and
                            self.pure_mcts_playout_num < 5000
                        ):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

        except KeyboardInterrupt:
            print("\nInterrupted. Saving checkpoint...", flush=True)
            self.policy_value_net.save_model("./interrupt_policy.model")
            self.policy_value_net.save_model("./current_policy.model")
            print("Saved: interrupt_policy.model and current_policy.model", flush=True)
        finally:
            if pool is not None:
                pool.close()
                pool.join()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AlphaZero Gomoku with multiprocess CPU self-play"
    )
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--games-per-worker", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--n-playout", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--game-batch-num", type=int, default=1500)
    parser.add_argument("--check-freq", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.05)
    parser.add_argument("--noise-eps", type=float, default=0.25)
    parser.add_argument("--temperature-moves", type=int, default=8)
    parser.add_argument("--temp-high", type=float, default=1.0)
    parser.add_argument("--temp-low", type=float, default=1e-3)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()

    training_pipeline = TrainPipeline(
        init_model=args.init_model,
        use_gpu=not args.no_gpu,
        num_workers=args.num_workers,
        games_per_worker=args.games_per_worker,
        threads_per_worker=args.threads_per_worker,
        n_playout=args.n_playout,
        batch_size=args.batch_size,
        game_batch_num=args.game_batch_num,
        check_freq=args.check_freq,
        eval_games=args.eval_games,
        dirichlet_alpha=args.dirichlet_alpha,
        noise_eps=args.noise_eps,
        temperature_moves=args.temperature_moves,
        temp_high=args.temp_high,
        temp_low=args.temp_low,
    )
    training_pipeline.run()
