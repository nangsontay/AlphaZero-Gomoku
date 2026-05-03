# -*- coding: utf-8 -*-
"""
AlphaZero Gomoku training with a central batched GPU evaluator.

Main process trains the model. CPU worker processes run MCTS/self-play.
A dedicated GPU evaluator process batches leaf-state inference requests from
all workers and calls PolicyValueNet.policy_value(batch).
"""
from __future__ import print_function

import argparse
import multiprocessing as mp
import os
import queue
import random
import time
import traceback
from collections import defaultdict, deque

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import torch

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet


def set_cpu_threads(n=1):
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def get_equi_data(play_data, board_width, board_height):
    extend_data = []
    for state, mcts_prob, winner in play_data:
        for i in [1, 2, 3, 4]:
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                np.flipud(mcts_prob.reshape(board_height, board_width)), i
            )
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten(),
                winner,
            ))
            equi_state = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob = np.fliplr(equi_mcts_prob)
            extend_data.append((
                equi_state,
                np.flipud(equi_mcts_prob).flatten(),
                winner,
            ))
    return extend_data


class RemotePolicyValueClient(object):
    """Worker-side proxy used as MCTS policy_value_fn(board)."""

    def __init__(self, worker_id, board_width, board_height,
                 request_queue, response_queue, response_timeout=180.0,
                 log_every=2000):
        self.worker_id = int(worker_id)
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.response_timeout = float(response_timeout)
        self.log_every = int(log_every)
        self.request_id = 0

    def policy_value_fn(self, board):
        legal_positions = board.availables
        state = np.ascontiguousarray(
            board.current_state().reshape(
                4, self.board_width, self.board_height
            ).astype(np.float32)
        )
        self.request_id += 1
        rid = self.request_id
        if self.log_every > 0 and rid % self.log_every == 0:
            print("[worker {}] eval requests sent: {}".format(
                self.worker_id, rid), flush=True)

        self.request_queue.put({
            "type": "eval",
            "worker_id": self.worker_id,
            "request_id": rid,
            "state": state,
        })

        while True:
            try:
                resp = self.response_queue.get(timeout=self.response_timeout)
            except queue.Empty:
                raise RuntimeError(
                    "worker {} timed out waiting for GPU evaluator rid={}".format(
                        self.worker_id, rid
                    )
                )
            if resp.get("type") == "error":
                raise RuntimeError("GPU evaluator error: {}".format(
                    resp.get("error")))
            if resp.get("request_id") != rid:
                continue
            act_probs = resp["act_probs"]
            value = float(resp["value"])
            return zip(legal_positions, act_probs[legal_positions]), value


def gpu_evaluator_loop(model_file, board_width, board_height, request_queue,
                       response_queues, stats_queue, eval_batch_size=64,
                       eval_timeout_ms=5, use_gpu=True, threads=1,
                       log_every_batches=200):
    set_cpu_threads(threads)
    total_requests = 0
    total_batches = 0
    max_batch = 0
    start = time.time()
    try:
        print("[gpu-evaluator] loading {} use_gpu={}".format(
            model_file, use_gpu), flush=True)
        net = PolicyValueNet(board_width, board_height,
                             model_file=model_file, use_gpu=use_gpu)
        if use_gpu and torch.cuda.is_available():
            print("[gpu-evaluator] GPU: {}".format(
                torch.cuda.get_device_name(0)), flush=True)

        pending = []
        stopping = False
        timeout_sec = max(0.001, float(eval_timeout_ms) / 1000.0)

        while True:
            if not pending and not stopping:
                msg = request_queue.get()
                if msg.get("type") == "stop":
                    break
                pending.append(msg)

            deadline = time.time() + timeout_sec
            while (not stopping) and len(pending) < int(eval_batch_size):
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    msg = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if msg.get("type") == "stop":
                    stopping = True
                    break
                pending.append(msg)

            if pending:
                states = np.asarray([p["state"] for p in pending], dtype=np.float32)
                act_probs_batch, value_batch = net.policy_value(states)
                bsz = len(pending)
                total_requests += bsz
                total_batches += 1
                max_batch = max(max_batch, bsz)

                for item, act_probs, value in zip(pending, act_probs_batch, value_batch):
                    wid = int(item["worker_id"])
                    response_queues[wid].put({
                        "type": "eval_result",
                        "request_id": item["request_id"],
                        "act_probs": act_probs.astype(np.float32, copy=False),
                        "value": float(np.asarray(value).reshape(-1)[0]),
                    })
                pending = []

                if log_every_batches > 0 and total_batches % int(log_every_batches) == 0:
                    avg = float(total_requests) / max(1, total_batches)
                    print("[gpu-evaluator] batches={}, requests={}, avg_batch={:.2f}, max_batch={}".format(
                        total_batches, total_requests, avg, max_batch), flush=True)

            if stopping and not pending:
                break

        elapsed = time.time() - start
        stats_queue.put({
            "ok": True,
            "requests": total_requests,
            "batches": total_batches,
            "avg_batch": float(total_requests) / max(1, total_batches),
            "max_batch": max_batch,
            "elapsed": elapsed,
        })
        print("[gpu-evaluator] stopped: requests={}, batches={}, avg_batch={:.2f}, max_batch={}, elapsed={:.1f}s".format(
            total_requests, total_batches,
            float(total_requests) / max(1, total_batches), max_batch, elapsed),
            flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        err = "{}\n{}".format(exc, tb)
        print("[gpu-evaluator] ERROR:\n{}".format(err), flush=True)
        for q in response_queues:
            try:
                q.put({"type": "error", "error": err})
            except Exception:
                pass
        try:
            stats_queue.put({
                "ok": False,
                "error": str(exc),
                "traceback": tb,
                "requests": total_requests,
                "batches": total_batches,
            })
        except Exception:
            pass


def selfplay_worker_remote(args, request_queue, response_queue, output_queue):
    wid = int(args["worker_id"])
    try:
        set_cpu_threads(args.get("threads_per_worker", 1))
        seed = int(args.get("seed", 0)) + wid
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)

        bw = int(args["board_width"])
        bh = int(args["board_height"])
        n_games = int(args["n_games"])
        n_playout = int(args["n_playout"])
        c_puct = float(args["c_puct"])
        temp = float(args["temp"])

        print("[worker {}] start: games={}, n_playout={}, pid={}".format(
            wid, n_games, n_playout, os.getpid()), flush=True)

        board = Board(width=bw, height=bh, n_in_row=int(args["n_in_row"]))
        game = Game(board)
        client = RemotePolicyValueClient(
            worker_id=wid,
            board_width=bw,
            board_height=bh,
            request_queue=request_queue,
            response_queue=response_queue,
            response_timeout=float(args["response_timeout"]),
            log_every=int(args.get("worker_log_every", 2000)),
        )
        mcts_player = MCTSPlayer(client.policy_value_fn, c_puct=c_puct,
                                 n_playout=n_playout, is_selfplay=1)

        all_data = []
        episode_lens = []
        start = time.time()
        for game_idx in range(n_games):
            t0 = time.time()
            winner, play_data = game.start_self_play(mcts_player, temp=temp)
            play_data = list(play_data)
            episode_lens.append(len(play_data))
            all_data.extend(get_equi_data(play_data, bw, bh))
            print("[worker {}] game {}/{} done: winner={}, episode_len={}, eval_requests={}, {:.1f}s".format(
                wid, game_idx + 1, n_games, winner, len(play_data),
                client.request_id, time.time() - t0), flush=True)

        output_queue.put({
            "ok": True,
            "worker_id": wid,
            "data": all_data,
            "episode_lens": episode_lens,
            "eval_requests": client.request_id,
            "elapsed": time.time() - start,
        })
    except Exception as exc:
        tb = traceback.format_exc()
        print("[worker {}] ERROR: {}\n{}".format(wid, exc, tb), flush=True)
        output_queue.put({
            "ok": False,
            "worker_id": wid,
            "error": str(exc),
            "traceback": tb,
            "data": [],
            "episode_lens": [],
            "eval_requests": 0,
            "elapsed": 0.0,
        })


class TrainPipeline(object):
    def __init__(self, init_model=None, use_gpu=True, num_workers=6,
                 games_per_worker=1, threads_per_worker=1, n_playout=400,
                 batch_size=512, game_batch_num=1500, check_freq=50,
                 eval_games=10, eval_batch_size=64, eval_timeout_ms=5,
                 response_timeout=180.0,
                 worker_model_file="./_tmp_gpu_evaluator_policy.model"):
        self.use_gpu = bool(use_gpu)
        if self.use_gpu and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        print("Using GPU: {}".format(self.use_gpu), flush=True)
        if self.use_gpu:
            print("GPU name: {}".format(torch.cuda.get_device_name(0)), flush=True)

        self.num_workers = max(1, int(num_workers))
        self.games_per_worker = max(1, int(games_per_worker))
        self.threads_per_worker = max(1, int(threads_per_worker))
        self.eval_batch_size = max(1, int(eval_batch_size))
        self.eval_timeout_ms = max(1, int(eval_timeout_ms))
        self.response_timeout = float(response_timeout)
        self.worker_model_file = worker_model_file

        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = int(n_playout)
        self.c_puct = 5
        self.buffer_size = 50000
        self.batch_size = int(batch_size)
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = int(check_freq)
        self.game_batch_num = int(game_batch_num)
        self.eval_games = int(eval_games)
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 1000

        self.policy_value_net = PolicyValueNet(
            self.board_width, self.board_height,
            model_file=init_model, use_gpu=self.use_gpu)

    def save_cpu_model_for_evaluator(self):
        sd = self.policy_value_net.get_policy_param()
        cpu_sd = {}
        for k, v in sd.items():
            cpu_sd[k] = v.detach().cpu() if hasattr(v, "detach") else v
        torch.save(cpu_sd, self.worker_model_file)

    def build_worker_args(self):
        base_seed = int(time.time() * 1000000) % (2 ** 31 - 1)
        tasks = []
        for wid in range(self.num_workers):
            tasks.append({
                "worker_id": wid,
                "seed": base_seed,
                "n_games": self.games_per_worker,
                "board_width": self.board_width,
                "board_height": self.board_height,
                "n_in_row": self.n_in_row,
                "n_playout": self.n_playout,
                "c_puct": self.c_puct,
                "temp": self.temp,
                "threads_per_worker": self.threads_per_worker,
                "response_timeout": self.response_timeout,
                "worker_log_every": 2000,
            })
        return tasks

    def collect_selfplay_data_remote_gpu(self):
        self.save_cpu_model_for_evaluator()
        ctx = mp.get_context("spawn")
        request_queue = ctx.Queue(maxsize=max(16, self.num_workers * 4))
        response_queues = [ctx.Queue(maxsize=16) for _ in range(self.num_workers)]
        output_queue = ctx.Queue()
        stats_queue = ctx.Queue()

        evaluator = ctx.Process(
            target=gpu_evaluator_loop,
            args=(
                self.worker_model_file,
                self.board_width,
                self.board_height,
                request_queue,
                response_queues,
                stats_queue,
                self.eval_batch_size,
                self.eval_timeout_ms,
                self.use_gpu,
                1,
                200,
            ),
            name="gpu-evaluator")

        workers = []
        print("remote-GPU self-play start: workers={}, games_per_worker={}, n_playout={}, eval_batch_size={}, eval_timeout_ms={}".format(
            self.num_workers, self.games_per_worker, self.n_playout,
            self.eval_batch_size, self.eval_timeout_ms), flush=True)
        evaluator.start()

        for args in self.build_worker_args():
            wid = int(args["worker_id"])
            p = ctx.Process(
                target=selfplay_worker_remote,
                args=(args, request_queue, response_queues[wid], output_queue),
                name="selfplay-worker-{}".format(wid))
            p.start()
            workers.append(p)

        results = []
        completed = 0
        last_log = time.time()
        try:
            while completed < self.num_workers:
                try:
                    result = output_queue.get(timeout=10.0)
                except queue.Empty:
                    now = time.time()
                    if now - last_log >= 30:
                        alive = [p.name for p in workers if p.is_alive()]
                        print("waiting for workers: completed={}/{}, alive={}".format(
                            completed, self.num_workers, alive), flush=True)
                        last_log = now
                    continue

                completed += 1
                results.append(result)
                if not result.get("ok", False):
                    raise RuntimeError("worker {} failed: {}\n{}".format(
                        result.get("worker_id"), result.get("error"),
                        result.get("traceback")))
                print("self-play progress: {}/{} worker(s), worker={}, games={}, eval_requests={}, elapsed={:.1f}s".format(
                    completed, self.num_workers, result.get("worker_id"),
                    len(result.get("episode_lens", [])),
                    result.get("eval_requests", 0),
                    float(result.get("elapsed", 0.0))), flush=True)

            request_queue.put({"type": "stop"})
            for p in workers:
                p.join(timeout=30)
                if p.is_alive():
                    print("terminating stuck worker {}".format(p.name), flush=True)
                    p.terminate()
                    p.join(timeout=10)
            evaluator.join(timeout=60)
            if evaluator.is_alive():
                print("terminating stuck GPU evaluator", flush=True)
                evaluator.terminate()
                evaluator.join(timeout=10)
        except BaseException:
            for p in workers:
                if p.is_alive():
                    p.terminate()
            try:
                request_queue.put({"type": "stop"})
            except Exception:
                pass
            if evaluator.is_alive():
                evaluator.terminate()
            raise

        episode_lens = []
        total_games = 0
        total_positions = 0
        total_eval_requests = 0
        worker_times = []
        for r in results:
            self.data_buffer.extend(r["data"])
            episode_lens.extend(r["episode_lens"])
            total_games += len(r["episode_lens"])
            total_positions += len(r["data"])
            total_eval_requests += int(r.get("eval_requests", 0))
            worker_times.append(float(r.get("elapsed", 0.0)))
        self.episode_len = float(np.mean(episode_lens)) if episode_lens else 0.0

        try:
            st = stats_queue.get_nowait()
        except queue.Empty:
            st = None
        if st:
            if not st.get("ok", False):
                raise RuntimeError("GPU evaluator failed: {}\n{}".format(
                    st.get("error"), st.get("traceback")))
            print("gpu-evaluator stats: requests={}, batches={}, avg_batch={:.2f}, max_batch={}, elapsed={:.1f}s".format(
                st.get("requests", 0), st.get("batches", 0),
                st.get("avg_batch", 0.0), st.get("max_batch", 0),
                st.get("elapsed", 0.0)), flush=True)

        print("remote-GPU self-play done: games={}, avg_episode_len={:.1f}, augmented_positions={}, eval_requests={}, slowest_worker={:.1f}s, data_buffer={}".format(
            total_games, self.episode_len, total_positions, total_eval_requests,
            max(worker_times) if worker_times else 0.0, len(self.data_buffer)),
            flush=True)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [d[0] for d in mini_batch]
        mcts_probs_batch = [d[1] for d in mini_batch]
        winner_batch = [d[2] for d in mini_batch]

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = 0.0
        loss = 0.0
        entropy = 0.0
        new_v = old_v
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch,
                self.learn_rate * self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        winner_np = np.array(winner_batch)
        winner_var = np.var(winner_np)
        if winner_var > 1e-12:
            ev_old = 1 - np.var(winner_np - old_v.flatten()) / winner_var
            ev_new = 1 - np.var(winner_np - new_v.flatten()) / winner_var
        else:
            ev_old = 0.0
            ev_new = 0.0
        print("kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
            kl, self.lr_multiplier, loss, entropy, ev_old, ev_new), flush=True)
        return loss, entropy

    def policy_evaluate(self, n_games=None):
        if n_games is None:
            n_games = self.eval_games
        if n_games <= 0:
            return self.best_win_ratio
        current = MCTSPlayer(self.policy_value_net.policy_value_fn,
                             c_puct=self.c_puct, n_playout=self.n_playout)
        pure = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current, pure, start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]),
            flush=True)
        return win_ratio

    def run(self):
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data_remote_gpu()
                print("batch i:{}, data_buffer:{}".format(
                    i + 1, len(self.data_buffer)), flush=True)
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
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
        except KeyboardInterrupt:
            print("\nInterrupted. Saving checkpoint...", flush=True)
            self.policy_value_net.save_model("./interrupt_policy.model")
            self.policy_value_net.save_model("./current_policy.model")
            print("Saved: interrupt_policy.model and current_policy.model", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train AlphaZero Gomoku with central batched GPU evaluator")
    p.add_argument("--init-model", default=None)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--games-per-worker", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--n-playout", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--game-batch-num", type=int, default=1500)
    p.add_argument("--check-freq", type=int, default=50)
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-timeout-ms", type=int, default=5)
    p.add_argument("--response-timeout", type=float, default=180.0)
    return p.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    pipeline = TrainPipeline(
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
        eval_batch_size=args.eval_batch_size,
        eval_timeout_ms=args.eval_timeout_ms,
        response_timeout=args.response_timeout,
    )
    pipeline.run()
