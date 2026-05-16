# -*- coding: utf-8 -*-
"""
AlphaZero Gomoku training with a central batched GPU evaluator.

Main process trains the model. CPU worker processes run MCTS/self-play.
A dedicated GPU evaluator process batches leaf-state inference requests from
all workers and calls PolicyValueNet.policy_value(batch).
"""
from __future__ import print_function

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import os
import queue
import random
import signal
import time
import traceback
from collections import defaultdict, deque
from multiprocessing import resource_tracker, shared_memory
import numpy as np
import torch

from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_mlp import PolicyValueNet
from tactic import get_tactic_forced_move, get_tactic_label_vector


class _ShuttingDown(Exception):
    """Internal signal: child process should exit cleanly because the parent
    asked for shutdown (e.g. via Ctrl+C in main)."""
    pass


def _ignore_sigint_in_child():
    """Make child process ignore SIGINT so Ctrl+C in the terminal does not
    raise KeyboardInterrupt mid-operation in workers/evaluator. Only the main
    process handles Ctrl+C; it coordinates shutdown via shutdown_event."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except (ValueError, OSError):
        # Not on main thread of child? best-effort.
        pass


def _unregister_shm_from_tracker(shm):
    """Workaround for https://bugs.python.org/issue38119 — when a child opens
    an existing SharedMemory by name, multiprocessing.resource_tracker
    over-registers it and emits a spurious "leaked shared_memory" warning at
    interpreter shutdown. The parent owns the lifetime; tell the tracker to
    forget about it in the child."""
    if shm is None:
        return
    name = getattr(shm, "_name", None) or getattr(shm, "name", None)
    if not name:
        return
    try:
        resource_tracker.unregister(name, "shared_memory")
    except Exception:
        pass


def _attach_parent_owned_shm(name):
    """Attach to parent-owned SharedMemory without registering it in this child.

    The old workaround opened SharedMemory normally, then called
    resource_tracker.unregister(). With multiple children attached to the same
    parent-owned block, those duplicate UNREGISTER messages can make the shared
    resource_tracker process remove a name that is no longer in its cache and
    print a shutdown KeyError. Avoid that race entirely by suppressing only the
    child-side shared_memory REGISTER that SharedMemory(name=...) performs.
    """
    original_register = resource_tracker.register

    def register(resource_name, rtype):
        if rtype == "shared_memory":
            return
        return original_register(resource_name, rtype)

    resource_tracker.register = register
    try:
        return shared_memory.SharedMemory(name=name)
    finally:
        resource_tracker.register = original_register

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")




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
    """D4-augment training samples.

    Supports three input tuple shapes (backward-compatible):
      - 3-tuple: (state, mcts_prob, winner)
      - 4-tuple: (state, mcts_prob, winner, tactic_label)        # self-play B1
      - 5-tuple: (state, mcts_prob, winner, value_mask, tactic)  # pretraining

    The scalar `winner` / `value_mask` carry over unchanged through each D4
    rotation+flip; only board-shaped fields (state, mcts_prob, tactic_label)
    are transformed.
    """
    extend_data = []
    for sample in play_data:
        state, mcts_prob, winner = sample[:3]
        if len(sample) >= 5:
            value_mask = sample[3]
            tactic_label = sample[4]
        elif len(sample) == 4:
            value_mask = None
            tactic_label = sample[3]
        else:
            value_mask = None
            tactic_label = None
        for i in range(4):
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                mcts_prob.reshape(board_height, board_width), i
            )
            if tactic_label is not None:
                equi_tactic = np.rot90(
                    tactic_label.reshape(board_height, board_width), i
                )
            else:
                equi_tactic = None

            def _emit(s_arr, p_arr, t_arr):
                if t_arr is None:
                    extend_data.append((s_arr, p_arr, winner))
                elif value_mask is None:
                    extend_data.append((s_arr, p_arr, winner, t_arr))
                else:
                    extend_data.append((
                        s_arr, p_arr, winner, value_mask, t_arr))

            _emit(
                equi_state,
                equi_mcts_prob.flatten(),
                equi_tactic.flatten() if equi_tactic is not None else None,
            )
            equi_state_flip = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob_flip = np.fliplr(equi_mcts_prob)
            equi_tactic_flip = (np.fliplr(equi_tactic)
                                if equi_tactic is not None else None)
            _emit(
                equi_state_flip,
                equi_mcts_prob_flip.flatten(),
                equi_tactic_flip.flatten() if equi_tactic_flip is not None else None,
            )
    return extend_data


def _generate_tactical_samples_raw(board_width=15, board_height=15, n_in_row=5,
                                   num_samples=2048, max_random_moves=36,
                                   seed=None, forced_ratio=0.6,
                                   block_value_target=0.3,
                                   softmax_temperature=1.0,
                                   in_channels=4):
    """Generate un-augmented tactical 5-tuples
    (state, policy, value, value_mask, tactic).

    `value_mask` is 1.0 for rows whose `value` target is a real game-theoretic
    signal (forced win / forced block with a known value target) and 0.0 for
    non-forced rows whose tactic-softmax policy target is meaningful but whose
    value target is unknown. Trainers must gate the value loss by `value_mask`
    so unknown-value rows do not pull the value head toward 0 on every
    tactical position.

    This is the picklable, CPU-only core used by both the single-process path
    and the process-pool workers. It must not import or touch CUDA/torch.
    """
    rng = random.Random(seed)
    samples = []
    forced_count = 0
    non_forced_count = 0
    attempts = 0
    num_samples = int(num_samples)
    max_attempts = max(num_samples * 20, 100)
    board_size = int(board_width) * int(board_height)
    forced_ratio = min(1.0, max(0.0, float(forced_ratio)))
    target_forced = int(round(num_samples * forced_ratio))
    target_non_forced = num_samples - target_forced
    softmax_temperature = max(1e-6, float(softmax_temperature))
    while len(samples) < num_samples and attempts < max_attempts:
        attempts += 1
        board = Board(width=board_width, height=board_height, n_in_row=n_in_row,
                      in_channels=in_channels)
        board.init_board(start_player=rng.randrange(2))
        move_count = rng.randint(max(0, n_in_row - 2), max(0, int(max_random_moves)))
        for _ in range(move_count):
            if not board.availables:
                break
            move = rng.choice(board.availables)
            board.do_move(move)
            end, _ = board.game_end()
            if end:
                break
        end, _ = board.game_end()
        if end or not board.availables:
            continue
        tactic_label = get_tactic_label_vector(board)
        if float(tactic_label.max()) <= 0.0:
            continue
        state = board.current_state(in_channels).copy().astype(np.float32)
        forced_move, is_win = get_tactic_forced_move(board)
        policy_target = np.zeros(board_size, dtype=np.float32)
        if forced_move is not None:
            if target_forced > 0 and forced_count >= target_forced and non_forced_count < target_non_forced:
                continue
            policy_target[int(forced_move)] = 1.0
            value_target = 1.0 if bool(is_win) else float(block_value_target)
            # Forced rows have a defensible value target: 1.0 for win-in-1
            # and `block_value_target` for forced blocks. Train value head.
            value_mask = 1.0
            forced_count += 1
        else:
            if target_non_forced > 0 and non_forced_count >= target_non_forced and forced_count < target_forced:
                continue
            legal = np.zeros(board_size, dtype=np.float32)
            legal[np.asarray(board.availables, dtype=np.int64)] = 1.0
            legal_scores = tactic_label.astype(np.float32) * legal
            max_score = float(legal_scores.max())
            if max_score <= 0.0:
                continue
            logits = legal_scores / softmax_temperature
            logits[legal <= 0.0] = -np.inf
            logits = logits - np.nanmax(logits)
            probs = np.exp(logits).astype(np.float32)
            probs[legal <= 0.0] = 0.0
            denom = float(probs.sum())
            if denom <= 0.0 or not np.isfinite(denom):
                continue
            policy_target = probs / denom
            # Non-forced tactical positions have a meaningful policy target
            # (softmax over tactic scores) but the game-theoretic value is
            # unknown. Use 0.0 as a placeholder and mask the value loss off
            # via `value_mask=0` so the value head is not pinned to 0 on
            # every non-forced tactical sample.
            value_target = 0.0
            value_mask = 0.0
            non_forced_count += 1
        samples.append((state, policy_target, value_target, value_mask,
                        tactic_label))
    return samples, forced_count, non_forced_count, attempts


def _tactical_samples_worker(arg_tuple):
    """Top-level picklable worker entry point for the process pool.

    Must remain importable under spawn semantics — accepts a single tuple so it
    can be dispatched via ProcessPoolExecutor.map(). Returns the same shape as
    _generate_tactical_samples_raw().
    """
    (board_width, board_height, n_in_row, num_samples, max_random_moves,
     seed, forced_ratio, block_value_target, softmax_temperature,
     in_channels) = arg_tuple
    # Pin BLAS/OMP to 1 thread per worker so CPU oversubscription doesn't
    # destroy wall-clock when the parent uses many workers.
    try:
        set_cpu_threads(1)
    except Exception:
        pass
    return _generate_tactical_samples_raw(
        board_width=board_width,
        board_height=board_height,
        n_in_row=n_in_row,
        num_samples=num_samples,
        max_random_moves=max_random_moves,
        seed=seed,
        forced_ratio=forced_ratio,
        block_value_target=block_value_target,
        softmax_temperature=softmax_temperature,
        in_channels=in_channels,
    )


def generate_tactical_samples(board_width=15, board_height=15, n_in_row=5,
                              num_samples=2048, max_random_moves=36,
                              seed=None, forced_ratio=0.6,
                              block_value_target=0.3,
                              softmax_temperature=1.0,
                              workers=1, in_channels=4):
    """Generate tactical states with policy, value, and tactic targets.

    When ``workers <= 1`` (default) the existing single-process generator is
    used. When ``workers >= 2`` a spawn-based ``ProcessPoolExecutor`` splits
    ``num_samples`` across worker processes; each worker only produces
    CPU/numpy data and the parent merges results and applies D4 augmentation.
    """
    num_samples = int(num_samples)
    workers = max(1, int(workers or 1))

    if num_samples <= 0:
        return []

    if workers <= 1:
        t0 = time.time()
        raw, forced, non_forced, attempts = _generate_tactical_samples_raw(
            board_width=board_width,
            board_height=board_height,
            n_in_row=n_in_row,
            num_samples=num_samples,
            max_random_moves=max_random_moves,
            seed=seed,
            forced_ratio=forced_ratio,
            block_value_target=block_value_target,
            softmax_temperature=softmax_temperature,
            in_channels=in_channels,
        )
        augmented = get_equi_data(raw, board_width, board_height)
        print(
            "tactical sample gen: mode=single, workers=1, raw={} (forced={}, non_forced={}, attempts={}), augmented={}, elapsed={:.2f}s".format(
                len(raw), forced, non_forced, attempts, len(augmented),
                time.time() - t0,
            ),
            flush=True,
        )
        return augmented

    # Multi-process path. Split samples across workers as evenly as possible.
    effective_workers = min(workers, num_samples)
    base = num_samples // effective_workers
    remainder = num_samples % effective_workers
    chunks = []
    base_seed = (int(seed) if seed is not None else int(time.time())) & 0x7FFFFFFF
    for wid in range(effective_workers):
        chunk_n = base + (1 if wid < remainder else 0)
        if chunk_n <= 0:
            continue
        # Deterministic-but-distinct per-worker seed.
        worker_seed = (base_seed + wid * 2654435761) & 0x7FFFFFFF
        chunks.append((
            int(board_width),
            int(board_height),
            int(n_in_row),
            int(chunk_n),
            int(max_random_moves),
            int(worker_seed),
            float(forced_ratio),
            float(block_value_target),
            float(softmax_temperature),
            int(in_channels),
        ))

    t0 = time.time()
    ctx = mp.get_context("spawn")
    raw_total = []
    total_forced = 0
    total_non_forced = 0
    total_attempts = 0
    per_worker_summary = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=effective_workers, mp_context=ctx
    ) as executor:
        for wid, result in enumerate(
            executor.map(_tactical_samples_worker, chunks)
        ):
            raw, forced, non_forced, attempts = result
            raw_total.extend(raw)
            total_forced += forced
            total_non_forced += non_forced
            total_attempts += attempts
            per_worker_summary.append(
                "w{}={}(f{}/n{}/a{})".format(
                    wid, len(raw), forced, non_forced, attempts
                )
            )

    augmented = get_equi_data(raw_total, board_width, board_height)
    print(
        "tactical sample gen: mode=pool, workers={}, raw={} (forced={}, non_forced={}, attempts={}), augmented={}, elapsed={:.2f}s, per_worker=[{}]".format(
            effective_workers, len(raw_total), total_forced, total_non_forced,
            total_attempts, len(augmented), time.time() - t0,
            ", ".join(per_worker_summary),
        ),
        flush=True,
    )
    return augmented


def generate_forced_block_probe_positions(board_width=15, board_height=15,
                                          n_in_row=5, num_positions=200,
                                          max_random_moves=60, seed=None,
                                          max_attempts=None,
                                          in_channels=4):
    """Generate held-out forced-block positions for tactical probe accuracy.

    Returns tuples ``(state, forced_move, legal_mask)``.  The generator accepts
    only positions where the side to move has no immediate win and must block
    the opponent's immediate win, as determined by ``get_tactic_forced_move``.
    """
    rng = random.Random(seed)
    num_positions = max(0, int(num_positions))
    board_size = int(board_width) * int(board_height)
    if max_attempts is None:
        max_attempts = max(num_positions * 1000, 1000)
    else:
        max_attempts = max(int(max_attempts), num_positions)

    positions = []
    attempts = 0
    while len(positions) < num_positions and attempts < max_attempts:
        attempts += 1
        board = Board(width=board_width, height=board_height, n_in_row=n_in_row,
                      in_channels=in_channels)
        board.init_board(start_player=rng.randrange(2))
        move_count = rng.randint(max(0, n_in_row - 2), max(0, int(max_random_moves)))
        for _ in range(move_count):
            if not board.availables:
                break
            move = rng.choice(board.availables)
            board.do_move(move)
            end, _ = board.game_end()
            if end:
                break
        end, _ = board.game_end()
        if end or not board.availables:
            continue
        forced_move, is_win = get_tactic_forced_move(board)
        if forced_move is None or bool(is_win):
            continue
        legal_mask = np.zeros(board_size, dtype=np.float32)
        legal_mask[np.asarray(board.availables, dtype=np.int64)] = 1.0
        positions.append((
            board.current_state(in_channels).copy().astype(np.float32),
            int(forced_move),
            legal_mask,
        ))

    print(
        "forced-block probe gen: positions={}, attempts={}, target={}".format(
            len(positions), attempts, num_positions),
        flush=True,
    )
    return positions


class RemotePolicyValueClient(object):
    """Worker-side proxy used as MCTS policy_value_fn(board)."""

    def __init__(self, worker_id, board_width, board_height,
                 request_queue, response_queue, response_timeout=180.0,
                 log_every=2000, slot_pool_queue=None, shm_in_name=None,
                 shm_out_name=None, shm_slots=0, in_channels=4,
                 shutdown_event=None, poll_interval=0.5):
        self.worker_id = int(worker_id)
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.board_size = self.board_width * self.board_height
        self.in_channels = int(in_channels)
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.response_timeout = float(response_timeout)
        self.log_every = int(log_every)
        self.request_id = 0
        self.slot_pool_queue = slot_pool_queue
        self.shm_slots = int(shm_slots or 0)
        self.shm_in = None
        self.shm_out = None
        self.shm_in_view = None
        self.shm_out_view = None
        self.shutdown_event = shutdown_event
        self.poll_interval = max(0.05, float(poll_interval))
        if shm_in_name and shm_out_name and slot_pool_queue is not None:
            # Parent owns the lifetime; do not register these attachments in
            # this child, otherwise duplicate unregisters can trip the shared
            # resource_tracker at process shutdown.
            self.shm_in = _attach_parent_owned_shm(shm_in_name)
            self.shm_out = _attach_parent_owned_shm(shm_out_name)
            self.shm_in_view = np.ndarray(
                (self.shm_slots, self.in_channels,
                 self.board_width, self.board_height),
                dtype=np.float32,
                buffer=self.shm_in.buf,
            )
            self.shm_out_view = np.ndarray(
                (self.shm_slots, self.board_size + 1),
                dtype=np.float32,
                buffer=self.shm_out.buf,
            )

    def _shutdown_requested(self):
        return self.shutdown_event is not None and self.shutdown_event.is_set()

    def _get_response_cooperative(self):
        """Wait for a response while remaining responsive to shutdown_event.

        Polls response_queue with a short timeout; bails out promptly if the
        parent signaled shutdown. Raises queue.Empty after response_timeout
        seconds of no response, or _ShuttingDown when the parent asked us to
        stop."""
        deadline = time.time() + self.response_timeout
        while True:
            if self._shutdown_requested():
                raise _ShuttingDown()
            remaining = deadline - time.time()
            if remaining <= 0:
                raise queue.Empty()
            try:
                return self.response_queue.get(
                    timeout=min(self.poll_interval, remaining))
            except queue.Empty:
                continue

    def _get_slot_cooperative(self):
        """Acquire a shared-memory slot while remaining responsive to shutdown."""
        deadline = time.time() + self.response_timeout
        while True:
            if self._shutdown_requested():
                raise _ShuttingDown()
            remaining = deadline - time.time()
            if remaining <= 0:
                raise queue.Empty()
            try:
                return self.slot_pool_queue.get(
                    timeout=min(self.poll_interval, remaining))
            except queue.Empty:
                continue

    def close(self):
        self.shm_in_view = None
        self.shm_out_view = None
        if self.shm_in is not None:
            self.shm_in.close()
            self.shm_in = None
        if self.shm_out is not None:
            self.shm_out.close()
            self.shm_out = None

    def _has_shared_memory(self):
        return (self.slot_pool_queue is not None and
                self.shm_in_view is not None and
                self.shm_out_view is not None)

    def _send_shared_request(self, state, rid):
        slot = self._get_slot_cooperative()
        released = False
        try:
            self.shm_in_view[slot] = np.ascontiguousarray(state, dtype=np.float32)
            self.request_queue.put((int(slot), self.worker_id, int(rid)))
            while True:
                try:
                    resp = self._get_response_cooperative()
                except queue.Empty:
                    raise RuntimeError(
                        "worker {} timed out waiting for GPU evaluator rid={}".format(
                            self.worker_id, rid
                        )
                    )
                if isinstance(resp, dict) and resp.get("type") == "error":
                    raise RuntimeError("GPU evaluator error: {}".format(
                        resp.get("error")))
                if isinstance(resp, dict):
                    resp_slot = resp.get("slot")
                    resp_rid = resp.get("request_id")
                else:
                    resp_slot, resp_rid = resp[0], resp[1]
                if int(resp_rid) != int(rid):
                    continue
                if int(resp_slot) != int(slot):
                    continue
                priors = self.shm_out_view[slot, :self.board_size].copy()
                value = float(self.shm_out_view[slot, self.board_size])
                self.slot_pool_queue.put(slot)
                released = True
                return priors, value
        finally:
            if not released:
                try:
                    self.slot_pool_queue.put(slot)
                except Exception:
                    pass

    def policy_value_fn(self, board):
        legal_positions = board.availables
        state = np.ascontiguousarray(
            board.current_state(self.in_channels).reshape(
                self.in_channels, self.board_width, self.board_height
            ).astype(np.float32)
        )
        self.request_id += 1
        rid = self.request_id
        if self.log_every > 0 and rid % self.log_every == 0:
            print("[worker {}] eval requests sent: {}".format(
                self.worker_id, rid), flush=True)

        if self._has_shared_memory():
            act_probs, value = self._send_shared_request(state, rid)
            return zip(legal_positions, act_probs[legal_positions]), value

        self.request_queue.put({
            "type": "eval",
            "worker_id": self.worker_id,
            "request_id": rid,
            "state": state,
        })

        while True:
            try:
                resp = self._get_response_cooperative()
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

    def policy_value_batch_fn(self, states_np):
        """Evaluate a batch of already-built board state tensors remotely.

        states_np must be shaped (B, C, board_width, board_height). Full-board
        priors are returned so MCTS can slice them by each leaf's legal moves.
        """
        states_np = np.ascontiguousarray(states_np, dtype=np.float32)
        if states_np.ndim != 4:
            raise ValueError("states_np must have shape (B, C, H, W)")
        batch_size = int(states_np.shape[0])
        if batch_size == 0:
            return (np.empty((0, self.board_width * self.board_height), dtype=np.float32),
                    np.empty((0,), dtype=np.float32))

        rids = []
        slots_by_rid = {}
        if self._has_shared_memory():
            try:
                for i in range(batch_size):
                    self.request_id += 1
                    rid = self.request_id
                    rids.append(rid)
                    if self.log_every > 0 and rid % self.log_every == 0:
                        print("[worker {}] eval requests sent: {}".format(
                            self.worker_id, rid), flush=True)
                    slot = self._get_slot_cooperative()
                    slots_by_rid[rid] = slot
                    self.shm_in_view[slot] = states_np[i]
                    self.request_queue.put((int(slot), self.worker_id, int(rid)))
            except Exception:
                for slot in list(slots_by_rid.values()):
                    try:
                        self.slot_pool_queue.put(slot)
                    except Exception:
                        pass
                raise
        else:
            for i in range(batch_size):
                self.request_id += 1
                rid = self.request_id
                rids.append(rid)
                if self.log_every > 0 and rid % self.log_every == 0:
                    print("[worker {}] eval requests sent: {}".format(
                        self.worker_id, rid), flush=True)
                self.request_queue.put({
                    "type": "eval",
                    "worker_id": self.worker_id,
                    "request_id": rid,
                    "state": np.ascontiguousarray(states_np[i]),
                })

        rid_to_idx = {rid: i for i, rid in enumerate(rids)}
        priors_out = [None] * batch_size
        values_out = np.empty(batch_size, dtype=np.float32)
        received = 0
        try:
            while received < batch_size:
                try:
                    resp = self._get_response_cooperative()
                except queue.Empty:
                    raise RuntimeError(
                        "worker {} timed out waiting for GPU evaluator batch={} received={}".format(
                            self.worker_id, batch_size, received
                        )
                    )
                if isinstance(resp, dict) and resp.get("type") == "error":
                    raise RuntimeError("GPU evaluator error: {}".format(
                        resp.get("error")))
                if isinstance(resp, dict):
                    rid = resp.get("request_id")
                    slot = resp.get("slot")
                else:
                    slot, rid = resp[0], resp[1]
                if rid not in rid_to_idx:
                    continue
                idx = rid_to_idx[rid]
                if priors_out[idx] is not None:
                    continue
                if self._has_shared_memory():
                    expected_slot = slots_by_rid[rid]
                    if int(slot) != int(expected_slot):
                        continue
                    priors_out[idx] = self.shm_out_view[expected_slot, :self.board_size].copy()
                    values_out[idx] = float(self.shm_out_view[expected_slot, self.board_size])
                    self.slot_pool_queue.put(expected_slot)
                    del slots_by_rid[rid]
                else:
                    priors_out[idx] = np.asarray(resp["act_probs"], dtype=np.float32)
                    values_out[idx] = float(resp["value"])
                received += 1
        finally:
            for slot in list(slots_by_rid.values()):
                try:
                    self.slot_pool_queue.put(slot)
                except Exception:
                    pass

        return np.stack(priors_out), values_out


class CudaGraphInferenceWrapper(object):
    """Fixed-batch CUDA Graph inference wrapper for evaluator-only use."""

    def __init__(self, model, batch_size, in_shape, device,
                 dtype=torch.float16):
        self.model = model
        self.batch_size = int(batch_size)
        self.in_shape = tuple(int(x) for x in in_shape)
        self.device = device
        self.dtype = dtype
        self.static_in = torch.zeros(
            (self.batch_size,) + self.in_shape,
            device=self.device,
            dtype=self.dtype,
        ).to(memory_format=torch.channels_last)
        self.graph = None
        self.static_log_p = None
        self.static_v = None
        self.capture()

    def capture(self):
        self.model.eval()
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            with torch.no_grad():
                for _ in range(3):
                    self.model(self.static_in)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(self.graph):
                self.static_log_p, self.static_v, _ = self.model(self.static_in)

    def run(self, batch_np):
        batch_np = np.ascontiguousarray(batch_np, dtype=np.float32)
        n = int(batch_np.shape[0])
        if n > self.batch_size:
            raise ValueError("batch {} > captured size {}".format(
                n, self.batch_size))
        if n == 0:
            board_size = int(self.static_log_p.shape[1])
            return (np.empty((0, board_size), dtype=np.float32),
                    np.empty((0, 1), dtype=np.float32))

        host = torch.from_numpy(batch_np)
        if self.dtype == torch.float16:
            host = host.half()
        host = host.to(memory_format=torch.channels_last)
        self.static_in[:n].copy_(host.to(self.device, non_blocking=True),
                                 non_blocking=True)
        if n < self.batch_size:
            self.static_in[n:].zero_()
        self.graph.replay()
        torch.cuda.synchronize(self.device)
        priors = torch.exp(self.static_log_p[:n].float()).detach().cpu().numpy()
        values = self.static_v[:n].float().detach().cpu().numpy()
        return priors, values


def gpu_evaluator_loop(model_file, board_width, board_height, request_queue,
                       response_queues, stats_queue, eval_batch_size=256,
                       eval_timeout_ms=5, use_gpu=True, threads=1,
                       log_every_batches=200, weight_event=None,
                       shutdown_event=None, shm_in_name=None, shm_out_name=None,
                       shm_slots=0, in_channels=4, use_cuda_graphs=True,
                       inference_fp16=True, backbone="mlp",
                       mixer_dim=128, mixer_depth=6,
                       mixer_token_hidden=256, mixer_ch_hidden=384,
                       mixer_value_hidden=128, mixer_dropout=0.1):
    _ignore_sigint_in_child()
    set_cpu_threads(threads)
    total_requests = 0
    total_batches = 0
    max_batch = 0
    start = time.time()
    shm_in = None
    shm_out = None
    shm_in_view = None
    shm_out_view = None
    board_size = int(board_width) * int(board_height)
    try:
        print("[gpu-evaluator] loading {} use_gpu={}".format(
            model_file, use_gpu), flush=True)
        net = PolicyValueNet(board_width, board_height,
                             model_file=model_file, use_gpu=use_gpu,
                             in_channels=in_channels,
                             use_amp=False,
                             backbone=backbone,
                             mixer_dim=mixer_dim,
                             mixer_depth=mixer_depth,
                             mixer_token_hidden=mixer_token_hidden,
                             mixer_ch_hidden=mixer_ch_hidden,
                             mixer_value_hidden=mixer_value_hidden,
                             mixer_dropout=mixer_dropout)
        evaluator_use_gpu = bool(use_gpu and torch.cuda.is_available())
        evaluator_fp16 = bool(evaluator_use_gpu and inference_fp16)
        cuda_graph = None

        def optimize_evaluator_model():
            net.policy_value_net = net.policy_value_net.to(
                memory_format=torch.channels_last)
            if evaluator_fp16:
                net.policy_value_net = net.policy_value_net.half()
            net.policy_value_net.eval()

        def capture_cuda_graph_or_none():
            if not (evaluator_use_gpu and use_cuda_graphs):
                return None
            try:
                graph_dtype = torch.float16 if evaluator_fp16 else torch.float32
                graph = CudaGraphInferenceWrapper(
                    net.policy_value_net,
                    int(eval_batch_size),
                    (int(in_channels), int(board_width), int(board_height)),
                    net.device,
                    dtype=graph_dtype,
                )
                print("[gpu-evaluator] CUDA Graph captured: batch_size={}, dtype={}".format(
                    int(eval_batch_size), graph_dtype), flush=True)
                return graph
            except Exception as exc:
                print("[gpu-evaluator] CUDA Graph capture failed; using eager inference: {}".format(
                    exc), flush=True)
                return None

        def first_param_data_ptr():
            try:
                return next(net.policy_value_net.parameters()).data_ptr()
            except StopIteration:
                return None

        optimize_evaluator_model()
        if evaluator_use_gpu:
            print("[gpu-evaluator] GPU: {}".format(
                torch.cuda.get_device_name(0)), flush=True)
            print("[gpu-evaluator] inference optimizations: fp16={}, channels_last=True, cuda_graphs={}".format(
                evaluator_fp16, bool(use_cuda_graphs)), flush=True)
            cuda_graph = capture_cuda_graph_or_none()

        if shm_in_name and shm_out_name:
            # Parent owns the shm lifetime; avoid child-side tracking instead
            # of registering and then manually unregistering, which can send a
            # duplicate UNREGISTER and trigger resource_tracker KeyError.
            shm_in = _attach_parent_owned_shm(shm_in_name)
            shm_out = _attach_parent_owned_shm(shm_out_name)
            shm_in_view = np.ndarray(
                (int(shm_slots), int(in_channels), int(board_width), int(board_height)),
                dtype=np.float32,
                buffer=shm_in.buf,
            )
            shm_out_view = np.ndarray(
                (int(shm_slots), board_size + 1),
                dtype=np.float32,
                buffer=shm_out.buf,
            )

        pending = []
        stopping = False
        timeout_sec = max(0.001, float(eval_timeout_ms) / 1000.0)

        while not stopping and not (shutdown_event is not None and shutdown_event.is_set()):
            if weight_event is not None and weight_event.is_set():
                try:
                    old_ptr = first_param_data_ptr()
                    state = torch.load(model_file, map_location=net.device)
                    net.policy_value_net.load_state_dict(state)
                    optimize_evaluator_model()
                    new_ptr = first_param_data_ptr()
                    if cuda_graph is not None and old_ptr != new_ptr:
                        print("[gpu-evaluator] parameter data_ptr changed on reload; recapturing CUDA Graph", flush=True)
                        cuda_graph = capture_cuda_graph_or_none()
                    print("[gpu-evaluator] hot-reloaded weights from {}".format(
                        model_file), flush=True)
                except Exception as exc:
                    print("[gpu-evaluator] reload failed: {}".format(exc), flush=True)
                finally:
                    weight_event.clear()

            if not pending and not stopping:
                try:
                    msg = request_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(msg, dict) and msg.get("type") in ("stop", "shutdown"):
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
                if isinstance(msg, dict) and msg.get("type") in ("stop", "shutdown"):
                    stopping = True
                    break
                pending.append(msg)

            if pending:
                if shm_in_view is not None:
                    states = np.asarray([shm_in_view[int(p[0])] for p in pending],
                                        dtype=np.float32)
                else:
                    states = np.asarray([p["state"] for p in pending], dtype=np.float32)
                if cuda_graph is not None:
                    act_probs_batch, value_batch = cuda_graph.run(states)
                else:
                    act_probs_batch, value_batch = net.policy_value_inference(
                        states,
                        fp16=evaluator_fp16,
                        channels_last=evaluator_use_gpu,
                    )
                bsz = len(pending)
                total_requests += bsz
                total_batches += 1
                max_batch = max(max_batch, bsz)

                for item, act_probs, value in zip(pending, act_probs_batch, value_batch):
                    if shm_out_view is not None:
                        slot, wid, rid = int(item[0]), int(item[1]), int(item[2])
                        shm_out_view[slot, :board_size] = act_probs.astype(np.float32, copy=False)
                        shm_out_view[slot, board_size] = float(np.asarray(value).reshape(-1)[0])
                        response_queues[wid].put((slot, rid))
                    else:
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

    except BaseException as exc:
        tb = traceback.format_exc()
        err = "{}\n{}".format(exc, tb)
        # KeyboardInterrupt should not normally reach here because we install
        # SIG_IGN at startup, but keep the handler defensive.
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
    finally:
        shm_in_view = None
        shm_out_view = None
        if shm_in is not None:
            try:
                shm_in.close()
            except Exception:
                pass
        if shm_out is not None:
            try:
                shm_out.close()
            except Exception:
                pass
        # Don't block waiting for queue feeder threads at exit if the parent
        # has stopped consuming.
        for q in [request_queue, stats_queue] + list(response_queues or []):
            if q is None:
                continue
            try:
                q.cancel_join_thread()
            except Exception:
                pass


def selfplay_worker_remote(args, request_queue, response_queue, replay_queue,
                           shutdown_event=None, slot_pool_queue=None,
                           shm_in_name=None, shm_out_name=None, shm_slots=0):
    _ignore_sigint_in_child()
    wid = int(args["worker_id"])
    client = None
    try:
        set_cpu_threads(args.get("threads_per_worker", 1))
        seed = int(args.get("seed", 0)) + wid
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed)

        bw = int(args["board_width"])
        bh = int(args["board_height"])
        n_games = int(args.get("n_games", 1))
        persistent = bool(args.get("persistent", False))
        n_playout = int(args["n_playout"])
        c_puct = float(args["c_puct"])
        temp = float(args["temp"])
        temperature_moves = args.get("temperature_moves", None)
        if temperature_moves is not None:
            temperature_moves = int(temperature_moves)
        temp_high = float(args.get("temp_high", 1.0))
        temp_low = float(args.get("temp_low", 1e-3))
        dirichlet_alpha = float(args.get("dirichlet_alpha", 0.05))
        noise_eps = float(args.get("noise_eps", 0.25))
        vl_k = int(args.get("vl_k", 4))
        n_vl = float(args.get("n_vl", 1.0))
        max_oversample = int(args.get("max_oversample", 3))

        print("[worker {}] start: games={}, n_playout={}, vl_k={}, n_vl={}, max_oversample={}, pid={}".format(
            wid, n_games, n_playout, vl_k, n_vl, max_oversample,
            os.getpid()), flush=True)

        in_channels = int(args.get("in_channels", 4))
        board = Board(width=bw, height=bh, n_in_row=int(args["n_in_row"]),
                      in_channels=in_channels)
        game = Game(board)
        client = RemotePolicyValueClient(
            worker_id=wid,
            board_width=bw,
            board_height=bh,
            request_queue=request_queue,
            response_queue=response_queue,
            response_timeout=float(args["response_timeout"]),
            log_every=int(args.get("worker_log_every", 2000)),
            slot_pool_queue=slot_pool_queue,
            shm_in_name=shm_in_name,
            shm_out_name=shm_out_name,
            shm_slots=shm_slots,
            in_channels=in_channels,
            shutdown_event=shutdown_event,
        )
        mcts_player = MCTSPlayer(client.policy_value_fn,
                                 client.policy_value_batch_fn,
                                 c_puct=c_puct,
                                 n_playout=n_playout, is_selfplay=1,
                                 dirichlet_alpha=dirichlet_alpha,
                                 noise_eps=noise_eps,
                                 vl_k=vl_k,
                                 n_vl=n_vl,
                                 max_oversample=max_oversample)

        episode_lens = []
        all_data = []
        start = time.time()
        games_done = 0
        while True:
            if shutdown_event is not None and shutdown_event.is_set():
                break
            if not persistent and games_done >= n_games:
                break
            t0 = time.time()
            winner, play_data = game.start_self_play(
                mcts_player,
                temp=temp,
                temperature_moves=temperature_moves,
                temp_high=temp_high,
                temp_low=temp_low,
                return_tactic_labels=True)
            play_data = list(play_data)
            augmented = get_equi_data(play_data, bw, bh)
            episode_lens.append(len(play_data))
            games_done += 1
            if persistent:
                replay_item = {
                    "ok": True,
                    "worker_id": wid,
                    "data": augmented,
                    "episode_lens": [len(play_data)],
                    "eval_requests": client.request_id,
                    "elapsed": time.time() - start,
                }
                # Backpressure: keep retrying with short waits instead of
                # silently dropping completed games. The trainer drains the
                # replay queue every iteration of `TrainPipeline.run`, so a
                # full queue means the trainer is currently busy training;
                # waiting is the correct behaviour, not data loss. We poll
                # `shutdown_event` between retries so Ctrl+C still terminates
                # workers promptly. Stalls log periodically so they are
                # visible if the trainer is permanently stuck.
                put_attempt = 0
                while True:
                    if shutdown_event is not None and shutdown_event.is_set():
                        # Caller already asked us to stop; drop this final
                        # game intentionally rather than blocking forever.
                        break
                    try:
                        replay_queue.put(replay_item, timeout=5.0)
                        break
                    except queue.Full:
                        put_attempt += 1
                        # Log every ~30s of backpressure so operators can see
                        # that workers are blocked on the trainer.
                        if put_attempt % 6 == 1:
                            print(
                                "[worker {}] replay queue full; backpressuring "
                                "completed game (attempt {})".format(
                                    wid, put_attempt),
                                flush=True,
                            )
                        continue
            else:
                all_data.extend(augmented)
            game_label = str(games_done) if persistent else "{}/{}".format(games_done, n_games)
            print("[worker {}] game {} done: winner={}, episode_len={}, augmented_positions={}, eval_requests={}, {:.1f}s".format(
                wid, game_label, winner, len(play_data), len(augmented),
                client.request_id, time.time() - t0), flush=True)
        if not persistent:
            replay_queue.put({
                "ok": True,
                "worker_id": wid,
                "data": all_data,
                "episode_lens": episode_lens,
                "eval_requests": client.request_id,
                "elapsed": time.time() - start,
            })
    except _ShuttingDown:
        # Parent asked for shutdown; exit cleanly without noise.
        print("[worker {}] shutdown requested; exiting".format(wid), flush=True)
    except BaseException as exc:
        tb = traceback.format_exc()
        print("[worker {}] ERROR: {}\n{}".format(wid, exc, tb), flush=True)
        if shutdown_event is None or not shutdown_event.is_set():
            try:
                replay_queue.put({
                    "ok": False,
                    "worker_id": wid,
                    "error": str(exc),
                    "traceback": tb,
                    "data": [],
                    "episode_lens": [],
                    "eval_requests": getattr(client, "request_id", 0),
                    "elapsed": 0.0,
                }, timeout=5.0)
            except Exception:
                pass
    finally:
        if client is not None:
            try:
                client.close()
            except Exception:
                pass
        # Avoid hanging Python's queue feeder threads at process exit if the
        # parent has already left the consumer side.
        for q in (request_queue, response_queue, replay_queue, slot_pool_queue):
            if q is None:
                continue
            try:
                q.cancel_join_thread()
            except Exception:
                pass


class TrainPipeline(object):
    def __init__(self, init_model=None, use_gpu=True, num_workers=10,
                 games_per_worker=1, threads_per_worker=1, n_playout=800,
                 batch_size=512, game_batch_num=1500, check_freq=50,
                 eval_games=10, eval_batch_size=256, eval_timeout_ms=8,
                 response_timeout=180.0, c_puct=3.0, eval_n_playout=400,
                 dirichlet_alpha=0.05, noise_eps=0.25,
                 vl_k=4, n_vl=1.0, max_oversample=3,
                 temperature_moves=8, temp_high=1.0, temp_low=1e-3,
                  buffer_size=500000, recent_sample_window=200000,
                  worker_model_file="./_tmp_gpu_evaluator_policy.model",
                  batch_log_file="training_batches.log",
                  use_cuda_graphs=True, inference_fp16=True,
                  tactic_pretrain_steps=10000,
                  tactic_pretrain_samples=80000,
                  tactic_pretrain_batch_size=512,
                  tactic_pretrain_lr=1e-3,
                  tactic_pretrain_workers=1,
                  tactic_loss_weight=0.25,
                  pretrain_tactic_loss_weight=0.5,
                  forced_ratio=0.6,
                  block_value_target=0.3,
                  tactic_sample_weight=1.5,
                  tactic_probe=False,
                  tactic_probe_steps=1000,
                  tactic_probe_samples=5000,
                  tactic_probe_positions=200,
                  tactic_probe_threshold=0.85,
                  tactic_probe_bug_threshold=0.50,
                  in_channels=4,
                  backbone="mlp",
                  mixer_dim=128,
                  mixer_depth=6,
                  mixer_token_hidden=256,
                  mixer_ch_hidden=384,
                  mixer_value_hidden=128,
                  mixer_dropout=0.1):
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
        self.batch_log_file = batch_log_file
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self.inference_fp16 = bool(inference_fp16)
        self.tactic_pretrain_steps = max(0, int(tactic_pretrain_steps))
        self.tactic_pretrain_samples = max(0, int(tactic_pretrain_samples))
        self.tactic_pretrain_batch_size = max(1, int(tactic_pretrain_batch_size))
        self.tactic_pretrain_lr = float(tactic_pretrain_lr)
        self.tactic_pretrain_workers = max(1, int(tactic_pretrain_workers))
        self.tactic_loss_weight = max(0.0, float(tactic_loss_weight))
        self.pretrain_tactic_loss_weight = max(0.0, float(pretrain_tactic_loss_weight))
        self.forced_ratio = min(1.0, max(0.0, float(forced_ratio)))
        self.block_value_target = float(block_value_target)
        self.tactic_sample_weight = max(0.0, float(tactic_sample_weight))
        self.tactic_probe = bool(tactic_probe)
        self.tactic_probe_steps = max(0, int(tactic_probe_steps))
        self.tactic_probe_samples = max(0, int(tactic_probe_samples))
        self.tactic_probe_positions = max(1, int(tactic_probe_positions))
        self.tactic_probe_threshold = float(tactic_probe_threshold)
        self.tactic_probe_bug_threshold = float(tactic_probe_bug_threshold)
        self.backbone = str(backbone).lower()
        self.in_channels = int(in_channels)
        self.mixer_dim = int(mixer_dim)
        self.mixer_depth = int(mixer_depth)
        self.mixer_token_hidden = int(mixer_token_hidden)
        self.mixer_ch_hidden = int(mixer_ch_hidden)
        self.mixer_value_hidden = int(mixer_value_hidden)
        self.mixer_dropout = float(mixer_dropout)
        self.last_update_metrics = None
        self.ctx = None
        self.request_queue = None
        self.replay_queue = None
        self.response_queues = None
        self.stats_queue = None
        self.weight_event = None
        self.shutdown_event = None
        self.slot_pool_queue = None
        self.shm_in = None
        self.shm_out = None
        self.shm_slots = 0
        self.shm_in_shape = None
        self.shm_out_shape = None
        self.evaluator_proc = None
        self.worker_procs = []
        self.pipeline_started = False

        self.board_width = 15
        self.board_height = 15
        self.n_in_row = 5
        self.board = Board(width=self.board_width, height=self.board_height,
                           n_in_row=self.n_in_row,
                           in_channels=self.in_channels)
        self.game = Game(self.board)
        self.learn_rate = 1e-3
        self.lr_multiplier = 1.0
        self.temp = 1.0
        self.n_playout = int(n_playout)
        self.eval_n_playout = int(eval_n_playout)
        self.c_puct = float(c_puct)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.noise_eps = float(noise_eps)
        self.vl_k = max(1, int(vl_k))
        self.n_vl = float(n_vl)
        self.max_oversample = max(1, int(max_oversample))
        self.temperature_moves = int(temperature_moves) if temperature_moves is not None else None
        self.temp_high = float(temp_high)
        self.temp_low = float(temp_low)
        self.buffer_size = int(buffer_size)
        self.recent_sample_window = max(1, int(recent_sample_window))
        self.batch_size = int(batch_size)
        self.check_freq = max(1, int(check_freq))
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = 2
        self.kl_targ = 0.03
        self.global_update_count = 0
        self.weight_push_every = 4
        self.lr_schedule = [
            (1500, 1e-3),
            (8000, 4e-4),
            (30000, 1e-4),
            (float("inf"), 2e-5),
        ]
        self.game_batch_num = int(game_batch_num)
        self.eval_games = int(eval_games)
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 400

        self.policy_value_net = PolicyValueNet(
            self.board_width, self.board_height,
            model_file=init_model, use_gpu=self.use_gpu,
            in_channels=self.in_channels,
            sym_loss_weight=0.0,
            tactic_loss_weight=self.tactic_loss_weight,
            tactic_sample_weight=self.tactic_sample_weight,
            backbone=self.backbone,
            mixer_dim=self.mixer_dim,
            mixer_depth=self.mixer_depth,
            mixer_token_hidden=self.mixer_token_hidden,
            mixer_ch_hidden=self.mixer_ch_hidden,
            mixer_value_hidden=self.mixer_value_hidden,
            mixer_dropout=self.mixer_dropout)

    def tactical_pretrain(self, steps=None, samples=None, seed=None):
        """Run optional supervised tactical pretraining through full train_step."""
        steps = self.tactic_pretrain_steps if steps is None else max(0, int(steps))
        samples = self.tactic_pretrain_samples if samples is None else max(0, int(samples))
        if steps <= 0 or samples <= 0:
            print("tactical pretraining disabled", flush=True)
            return
        gen_mode = "pool" if self.tactic_pretrain_workers > 1 else "single"
        print("tactical pretraining: samples={}, steps={}, batch_size={}, lr={:.6g}, forced_ratio={:.2f}, block_value={:.3f}, gen_mode={}, gen_workers={}".format(
            samples, steps,
            self.tactic_pretrain_batch_size, self.tactic_pretrain_lr,
            self.forced_ratio, self.block_value_target,
            gen_mode, self.tactic_pretrain_workers), flush=True)
        data = generate_tactical_samples(
            self.board_width, self.board_height, self.n_in_row,
            num_samples=samples,
            max_random_moves=60,
            forced_ratio=self.forced_ratio,
            block_value_target=self.block_value_target,
            seed=(int(time.time()) % (2 ** 31 - 1) if seed is None else int(seed)),
            workers=self.tactic_pretrain_workers,
            in_channels=self.policy_value_net.in_channels)
        if not data:
            print("tactical pretraining skipped: no tactical samples generated", flush=True)
            return []
        losses = []
        old_tactic_loss_weight = self.policy_value_net.tactic_loss_weight
        self.policy_value_net.tactic_loss_weight = self.pretrain_tactic_loss_weight
        try:
            for step in range(steps):
                batch_size = min(self.tactic_pretrain_batch_size, len(data))
                mini_batch = random.sample(data, batch_size)
                state_batch = [d[0] for d in mini_batch]
                policy_batch = [d[1] for d in mini_batch]
                value_batch = [d[2] for d in mini_batch]
                # 5-tuple shape after F2: d[3] = value_mask, d[4] = tactic.
                # 4-tuple fallback (older callers): assume forced/labeled.
                if len(mini_batch[0]) >= 5:
                    value_mask = [float(d[3]) for d in mini_batch]
                    tactic_batch = [d[4] for d in mini_batch]
                else:
                    value_mask = [1.0 for _ in mini_batch]
                    tactic_batch = [d[3] for d in mini_batch]
                tactic_mask = [1.0 for _ in mini_batch]
                loss, entropy = self.policy_value_net.train_step(
                    state_batch, policy_batch, value_batch,
                    self.tactic_pretrain_lr,
                    tactic_batch=tactic_batch,
                    tactic_mask=tactic_mask,
                    value_mask=value_mask)
                losses.append(float(loss))
                if (step + 1) % max(1, steps // 5) == 0:
                    print("tactical pretrain step {}/{}: loss={:.6f}, entropy={:.6f}".format(
                        step + 1, steps, float(loss), float(entropy)), flush=True)
        finally:
            self.policy_value_net.tactic_loss_weight = old_tactic_loss_weight
        print("tactical pretraining done: generated_positions={}, mean_loss={:.6f}".format(
            len(data), float(np.nanmean(losses)) if losses else 0.0), flush=True)
        return data

    def evaluate_forced_block_probe(self, num_positions=None, seed=None):
        """Evaluate top-1 policy accuracy on held-out forced-block positions."""
        num_positions = (self.tactic_probe_positions if num_positions is None
                         else max(1, int(num_positions)))
        probe = generate_forced_block_probe_positions(
            self.board_width, self.board_height, self.n_in_row,
            num_positions=num_positions,
            max_random_moves=60,
            seed=(int(time.time()) % (2 ** 31 - 1) if seed is None else int(seed)),
            in_channels=self.policy_value_net.in_channels,
        )
        if not probe:
            metrics = {
                "positions": 0,
                "correct": 0,
                "accuracy": 0.0,
                "target": float(self.tactic_probe_threshold),
                "likely_generation_bug": True,
                "scale_up_recommended": False,
            }
            print("forced-block probe: no positions generated; likely probe/sample generation bug", flush=True)
            return metrics

        states = [p[0] for p in probe]
        forced_moves = [p[1] for p in probe]
        legal_masks = [p[2] for p in probe]
        act_probs, _ = self.policy_value_net.policy_value(states)
        correct = 0
        for probs, forced_move, legal_mask in zip(act_probs, forced_moves, legal_masks):
            masked = np.asarray(probs, dtype=np.float64).copy()
            masked[np.asarray(legal_mask) <= 0.0] = -np.inf
            pred = int(np.argmax(masked))
            if pred == int(forced_move):
                correct += 1
        accuracy = float(correct) / float(len(probe))
        metrics = {
            "positions": int(len(probe)),
            "correct": int(correct),
            "accuracy": float(accuracy),
            "target": float(self.tactic_probe_threshold),
            "likely_generation_bug": bool(accuracy < self.tactic_probe_bug_threshold),
            "scale_up_recommended": bool(accuracy >= self.tactic_probe_threshold),
        }
        print(
            "forced-block probe: accuracy={:.2%} ({}/{}), target={:.2%}".format(
                accuracy, correct, len(probe), self.tactic_probe_threshold),
            flush=True,
        )
        if accuracy < self.tactic_probe_bug_threshold:
            print(
                "forced-block probe: accuracy below {:.0%}; likely bug in tactical sample generation. Do not recommend scale-up.".format(
                    self.tactic_probe_bug_threshold),
                flush=True,
            )
        elif accuracy < self.tactic_probe_threshold:
            print(
                "forced-block probe: below acceptance target; do not recommend scale-up yet.",
                flush=True,
            )
        else:
            print("forced-block probe: acceptance target met; scale-up is allowed.", flush=True)
        return metrics

    def run_tactical_probe(self):
        """Run the small 5k/1000-step probe before any full training run."""
        print(
            "tactical probe: pretrain_steps={}, samples={}, eval_forced_blocks={}".format(
                self.tactic_probe_steps, self.tactic_probe_samples,
                self.tactic_probe_positions),
            flush=True,
        )
        self.tactical_pretrain(
            steps=self.tactic_probe_steps,
            samples=self.tactic_probe_samples,
            seed=12345,
        )
        metrics = self.evaluate_forced_block_probe(
            num_positions=self.tactic_probe_positions,
            seed=54321,
        )
        print("tactical probe metrics: {}".format(json.dumps(metrics, sort_keys=True)), flush=True)
        return metrics

    def save_cpu_model_for_evaluator(self):
        """Atomically publish a CPU copy of the worker model checkpoint.

        The GPU evaluator hot-reloads `self.worker_model_file` whenever
        `self.weight_event` is set. Writing in place via `torch.save` is not
        atomic: the evaluator's `torch.load` can race a partial write and
        crash with a truncated-tensor error or load corrupted weights. To
        avoid this we write to a temp file in the same directory, flush + fsync
        the bytes, then `os.replace` onto the target path. `os.replace` is
        atomic on POSIX and Windows for files on the same filesystem, so the
        evaluator either sees the old or the new file, never a partial one.
        """
        sd = self.policy_value_net.get_policy_param()
        cpu_sd = {}
        for k, v in sd.items():
            cpu_sd[k] = v.detach().cpu() if hasattr(v, "detach") else v
        target = self.worker_model_file
        target_dir = os.path.dirname(os.path.abspath(target)) or "."
        os.makedirs(target_dir, exist_ok=True)
        # NamedTemporaryFile manages cleanup on failure; we rename on success.
        # Use a custom tmp name in the same directory so os.replace is atomic.
        tmp_path = "{}.tmp.{}.{}".format(target, os.getpid(), int(time.time() * 1000))
        try:
            with open(tmp_path, "wb") as f:
                torch.save(cpu_sd, f)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    # Some filesystems (e.g. tmpfs) reject fsync; the rename is
                    # still atomic, so swallow this and continue.
                    pass
            os.replace(tmp_path, target)
        except BaseException:
            # Best-effort cleanup of the temp file if we failed before replace.
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def setup_shared_memory(self):
        board_size = self.board_width * self.board_height
        in_channels = self.policy_value_net.in_channels
        self.shm_slots = max(1024, self.num_workers * self.vl_k * 4)
        self.shm_in_shape = (self.shm_slots, in_channels,
                             self.board_width, self.board_height)
        self.shm_out_shape = (self.shm_slots, board_size + 1)
        in_bytes = int(np.prod(self.shm_in_shape) * np.dtype(np.float32).itemsize)
        out_bytes = int(np.prod(self.shm_out_shape) * np.dtype(np.float32).itemsize)
        self.shm_in = shared_memory.SharedMemory(create=True, size=in_bytes)
        self.shm_out = shared_memory.SharedMemory(create=True, size=out_bytes)
        shm_in_view = np.ndarray(self.shm_in_shape, dtype=np.float32,
                                 buffer=self.shm_in.buf)
        shm_out_view = np.ndarray(self.shm_out_shape, dtype=np.float32,
                                  buffer=self.shm_out.buf)
        shm_in_view.fill(0.0)
        shm_out_view.fill(0.0)
        for slot in range(self.shm_slots):
            self.slot_pool_queue.put(slot)
        print("shared-memory IPC: slots={}, shm_in={} bytes, shm_out={} bytes".format(
            self.shm_slots, in_bytes, out_bytes), flush=True)

    def cleanup_shared_memory(self):
        for attr in ("shm_in", "shm_out"):
            shm = getattr(self, attr, None)
            if shm is None:
                continue
            try:
                shm.close()
            except Exception:
                pass
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                print("shared-memory cleanup warning for {}: {}".format(
                    attr, exc), flush=True)
            setattr(self, attr, None)

    def start_async_pipeline(self):
        if self.pipeline_started:
            return
        self.save_cpu_model_for_evaluator()
        self.ctx = mp.get_context("spawn")
        self.request_queue = self.ctx.Queue(maxsize=max(4096, self.num_workers * self.vl_k * 16))
        self.response_queues = [self.ctx.Queue(maxsize=64) for _ in range(self.num_workers)]
        self.replay_queue = self.ctx.Queue(maxsize=max(64, self.num_workers * 8))
        self.stats_queue = self.ctx.Queue()
        self.weight_event = self.ctx.Event()
        self.shutdown_event = self.ctx.Event()
        self.slot_pool_queue = self.ctx.Queue(maxsize=max(1024, self.num_workers * self.vl_k * 4))
        self.setup_shared_memory()

        self.evaluator_proc = self.ctx.Process(
            target=gpu_evaluator_loop,
            args=(
                self.worker_model_file,
                self.board_width,
                self.board_height,
                self.request_queue,
                self.response_queues,
                self.stats_queue,
                self.eval_batch_size,
                self.eval_timeout_ms,
                self.use_gpu,
                1,
                200,
                self.weight_event,
                self.shutdown_event,
                self.shm_in.name,
                self.shm_out.name,
                self.shm_slots,
                self.policy_value_net.in_channels,
                self.use_cuda_graphs,
                self.inference_fp16,
                self.backbone,
                self.mixer_dim,
                self.mixer_depth,
                self.mixer_token_hidden,
                self.mixer_ch_hidden,
                self.mixer_value_hidden,
                self.mixer_dropout,
            ),
            name="gpu-evaluator")
        self.evaluator_proc.start()

        self.worker_procs = []
        for args in self.build_worker_args():
            args = dict(args)
            args["persistent"] = True
            wid = int(args["worker_id"])
            p = self.ctx.Process(
                target=selfplay_worker_remote,
                args=(
                    args,
                    self.request_queue,
                    self.response_queues[wid],
                    self.replay_queue,
                    self.shutdown_event,
                    self.slot_pool_queue,
                    self.shm_in.name,
                    self.shm_out.name,
                    self.shm_slots,
                ),
                name="selfplay-worker-{}".format(wid))
            p.start()
            self.worker_procs.append(p)
        self.pipeline_started = True
        print("persistent remote-GPU self-play started: workers={}, n_playout={}, c_puct={}, dirichlet_alpha={}, noise_eps={}, vl_k={}, n_vl={}, max_oversample={}, temperature_moves={}, temp_high={}, temp_low={}, eval_batch_size={}, eval_timeout_ms={}".format(
            self.num_workers, self.n_playout, self.c_puct,
            self.dirichlet_alpha, self.noise_eps, self.vl_k, self.n_vl,
            self.max_oversample, self.temperature_moves, self.temp_high,
            self.temp_low, self.eval_batch_size, self.eval_timeout_ms),
            flush=True)

    def _drain_queue(self, q, max_items=10000):
        """Best-effort drain of a multiprocessing queue so the producer's
        feeder thread can flush and the join doesn't deadlock."""
        if q is None:
            return 0
        drained = 0
        while drained < max_items:
            try:
                q.get_nowait()
                drained += 1
            except queue.Empty:
                break
            except (OSError, ValueError, EOFError):
                break
            except Exception:
                break
        return drained

    def stop_async_pipeline(self):
        if not self.pipeline_started and not self.worker_procs and self.evaluator_proc is None:
            self.cleanup_shared_memory()
            return

        # 1. Tell every child to stop.
        if self.shutdown_event is not None:
            try:
                self.shutdown_event.set()
            except Exception:
                pass
        if self.request_queue is not None:
            # Send one sentinel per child so any worker still polling
            # request_queue (none currently do, but the evaluator does) wakes
            # up promptly.
            try:
                self.request_queue.put({"type": "shutdown"})
            except Exception:
                pass

        worker_join_timeout = 5.0
        evaluator_join_timeout = 10.0

        # 2. Continuously drain response/replay/stats queues while waiting on
        # workers. This unblocks any worker still trying to put a final replay
        # item, and prevents feeder threads from holding locks.
        deadline = time.time() + worker_join_timeout
        while time.time() < deadline:
            self._drain_queue(self.replay_queue)
            self._drain_queue(self.stats_queue)
            for rq in (self.response_queues or []):
                self._drain_queue(rq)
            if all(not p.is_alive() for p in self.worker_procs):
                break
            time.sleep(0.1)

        for p in list(self.worker_procs):
            p.join(timeout=1.0)
            if p.is_alive():
                print("terminating stuck worker {}".format(p.name), flush=True)
                try:
                    p.terminate()
                except Exception:
                    pass
                p.join(timeout=3.0)
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
                    p.join(timeout=1.0)

        # 3. Wait for evaluator, draining its inputs/outputs in parallel.
        if self.evaluator_proc is not None:
            ev_deadline = time.time() + evaluator_join_timeout
            while time.time() < ev_deadline and self.evaluator_proc.is_alive():
                self._drain_queue(self.request_queue)
                self._drain_queue(self.stats_queue)
                for rq in (self.response_queues or []):
                    self._drain_queue(rq)
                self.evaluator_proc.join(timeout=0.2)
            if self.evaluator_proc.is_alive():
                print("terminating stuck GPU evaluator", flush=True)
                try:
                    self.evaluator_proc.terminate()
                except Exception:
                    pass
                self.evaluator_proc.join(timeout=3.0)
                if self.evaluator_proc.is_alive():
                    try:
                        self.evaluator_proc.kill()
                    except Exception:
                        pass
                    self.evaluator_proc.join(timeout=1.0)

        # 4. Final drain so our own feeder threads don't block at process
        # exit, then explicitly cancel join threads on every queue.
        self._drain_queue(self.request_queue)
        self._drain_queue(self.replay_queue)
        self._drain_queue(self.stats_queue)
        self._drain_queue(self.slot_pool_queue)
        for rq in (self.response_queues or []):
            self._drain_queue(rq)

        all_queues = [
            self.request_queue, self.replay_queue, self.stats_queue,
            self.slot_pool_queue,
        ] + list(self.response_queues or [])
        for q in all_queues:
            if q is None:
                continue
            try:
                q.cancel_join_thread()
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

        # 5. Release shared memory last; children are gone so unlink is safe.
        self.cleanup_shared_memory()

        self.worker_procs = []
        self.evaluator_proc = None
        self.request_queue = None
        self.response_queues = None
        self.replay_queue = None
        self.stats_queue = None
        self.slot_pool_queue = None
        self.pipeline_started = False

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
                "dirichlet_alpha": self.dirichlet_alpha,
                "noise_eps": self.noise_eps,
                "vl_k": self.vl_k,
                "n_vl": self.n_vl,
                "max_oversample": self.max_oversample,
                "temperature_moves": self.temperature_moves,
                "temp_high": self.temp_high,
                "temp_low": self.temp_low,
                "temp": self.temp,
                "threads_per_worker": self.threads_per_worker,
                "response_timeout": self.response_timeout,
                "worker_log_every": 2000,
                "in_channels": self.policy_value_net.in_channels,
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
                None,
                None,
                None,
                None,
                0,
                self.policy_value_net.in_channels,
                self.use_cuda_graphs,
                self.inference_fp16,
                self.backbone,
                self.mixer_dim,
                self.mixer_depth,
                self.mixer_token_hidden,
                self.mixer_ch_hidden,
                self.mixer_value_hidden,
                self.mixer_dropout,
            ),
            name="gpu-evaluator")

        workers = []
        print("remote-GPU self-play start: workers={}, games_per_worker={}, n_playout={}, c_puct={}, dirichlet_alpha={}, noise_eps={}, vl_k={}, n_vl={}, max_oversample={}, temperature_moves={}, temp_high={}, temp_low={}, eval_batch_size={}, eval_timeout_ms={}".format(
            self.num_workers, self.games_per_worker, self.n_playout,
            self.c_puct, self.dirichlet_alpha, self.noise_eps,
            self.vl_k, self.n_vl, self.max_oversample,
            self.temperature_moves, self.temp_high, self.temp_low,
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

    def drain_replay_queue(self, max_games=None, timeout=1.0):
        max_games = max_games or max(1, self.num_workers * max(1, self.games_per_worker))
        drained_games = 0
        total_positions = 0
        episode_lens = []
        total_eval_requests = 0
        worker_times = []
        deadline = time.time() + float(timeout)
        while drained_games < max_games:
            remaining = max(0.0, deadline - time.time())
            if drained_games > 0 and remaining <= 0:
                break
            try:
                result = self.replay_queue.get(timeout=remaining if drained_games == 0 else 0.0)
            except queue.Empty:
                break
            if not result.get("ok", False):
                raise RuntimeError("worker {} failed: {}\n{}".format(
                    result.get("worker_id"), result.get("error"),
                    result.get("traceback")))
            data = result.get("data", [])
            self.data_buffer.extend(data)
            episode_lens.extend(result.get("episode_lens", []))
            total_positions += len(data)
            total_eval_requests += int(result.get("eval_requests", 0))
            worker_times.append(float(result.get("elapsed", 0.0)))
            drained_games += len(result.get("episode_lens", [])) or 1
        if episode_lens:
            self.episode_len = float(np.mean(episode_lens))
        return {
            "games": int(drained_games),
            "positions": int(total_positions),
            "episode_len": float(getattr(self, "episode_len", 0.0)),
            "eval_requests": int(total_eval_requests),
            "slowest_worker_elapsed": max(worker_times) if worker_times else 0.0,
        }

    def check_async_processes(self):
        if self.evaluator_proc is not None and not self.evaluator_proc.is_alive():
            raise RuntimeError("GPU evaluator exited unexpectedly with exitcode={}".format(
                self.evaluator_proc.exitcode))
        dead = [p for p in self.worker_procs if not p.is_alive()]
        if dead:
            raise RuntimeError("self-play worker(s) exited unexpectedly: {}".format(
                [(p.name, p.exitcode) for p in dead]))

    def get_scheduled_lr(self):
        for boundary, lr in self.lr_schedule:
            if self.global_update_count < boundary:
                return lr
        return self.lr_schedule[-1][1]

    def policy_update(self):
        sample_window = min(len(self.data_buffer), self.recent_sample_window)
        recent_buffer = list(self.data_buffer)[-sample_window:]
        mini_batch = random.sample(recent_buffer, self.batch_size)
        state_batch = [d[0] for d in mini_batch]
        mcts_probs_batch = [d[1] for d in mini_batch]
        winner_batch = [d[2] for d in mini_batch]
        # Replay buffers can contain mixed tuple shapes:
        #   - 3-tuple: (state, mcts_prob, winner)                     no tactic
        #   - 4-tuple: (state, mcts_prob, winner, tactic)             B1 self-play
        #   - 5-tuple: (state, mcts_prob, winner, value_mask, tactic) pretraining
        # Build per-row `tactic_batch`, `tactic_mask` (for tactic-aux loss and
        # sample-weight gating) plus `value_mask` (so unknown-value rows don't
        # train the value head against a placeholder 0). Self-play 3/4-tuples
        # are real game outcomes, so their value_mask defaults to 1.
        zero_label = np.zeros(
            self.board_width * self.board_height, dtype=np.float32)
        tactic_batch = []
        tactic_mask = []
        value_mask = []
        any_value_mask = False
        for d in mini_batch:
            if len(d) >= 5:
                tactic_batch.append(d[4])
                tactic_mask.append(1.0)
                value_mask.append(float(d[3]))
                any_value_mask = True
            elif len(d) == 4:
                tactic_batch.append(d[3])
                tactic_mask.append(1.0)
                value_mask.append(1.0)
            else:
                tactic_batch.append(zero_label)
                tactic_mask.append(0.0)
                value_mask.append(1.0)
        # Preserve backward compat with the train_step value-loss path: only
        # forward value_mask if at least one row is actually masked off; this
        # keeps the dense-mean path active in the common all-self-play case.
        value_mask_arg = value_mask if any_value_mask else None

        self.learn_rate = self.get_scheduled_lr()

        # Counter normalisation rule: warmup is anchored on
        # `global_update_count` defined as "number of `train_step` invocations
        # on the main trainer's network". This counter is a property of the
        # trainer, NOT the worker count. Same convention applies in train.py
        # and train_mp.py.
        warmup_steps = 500
        if self.global_update_count < warmup_steps:
            warmup_lr = self.learn_rate * (self.global_update_count + 1) / warmup_steps
        else:
            warmup_lr = self.learn_rate

        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        kl = 0.0
        loss = 0.0
        entropy = 0.0
        new_v = old_v
        for _ in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                state_batch, mcts_probs_batch, winner_batch,
                warmup_lr * self.lr_multiplier,
                tactic_batch=tactic_batch,
                tactic_mask=tactic_mask,
                value_mask=value_mask_arg)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 5.0:
            self.lr_multiplier = min(5.0, self.lr_multiplier * 1.5)

        winner_np = np.array(winner_batch)
        winner_var = np.var(winner_np)
        n_pos = int((winner_np > 0).sum())
        n_neg = int((winner_np < 0).sum())
        n_zero = int((winner_np == 0).sum())
        old_v_flat = old_v.flatten()
        new_v_flat = new_v.flatten()
        if winner_var > 1e-12:
            ev_old = 1 - np.var(winner_np - old_v_flat) / winner_var
            ev_new = 1 - np.var(winner_np - new_v_flat) / winner_var
        else:
            ev_old = 0.0
            ev_new = 0.0
        self.global_update_count += 1
        print(f"z_dist: pos={n_pos}, neg={n_neg}, zero={n_zero}, winner_var={winner_var:.6f} | "
              f"v_old: mean={old_v_flat.mean():.4f} std={old_v_flat.std():.4f} | "
              f"v_new: mean={new_v_flat.mean():.4f} std={new_v_flat.std():.4f} | "
              f"ev_old={ev_old:.6f} ev_new={ev_new:.6f}", flush=True)
        effective_lr = self.learn_rate * self.lr_multiplier
        print("update:{},base_lr:{:.6g},effective_lr:{:.6g},sample_window:{},kl:{:.5f},lr_multiplier:{:.3f},loss:{},entropy:{},explained_var_old:{:.3f},explained_var_new:{:.3f}".format(
            self.global_update_count, self.learn_rate,
            effective_lr, sample_window,
            kl, self.lr_multiplier, loss, entropy, ev_old, ev_new), flush=True)
        self.last_update_metrics = {
            "update": self.global_update_count,
            "base_lr": float(self.learn_rate),
            "effective_lr": float(effective_lr),
            "sample_window": int(sample_window),
            "kl": float(kl),
            "lr_multiplier": float(self.lr_multiplier),
            "loss": float(loss),
            "entropy": float(entropy),
            "explained_var_old": float(ev_old),
            "explained_var_new": float(ev_new),
            "z_pos": int(n_pos),
            "z_neg": int(n_neg),
            "z_zero": int(n_zero),
            "winner_var": float(winner_var),
            "v_old_mean": float(old_v_flat.mean()),
            "v_old_std": float(old_v_flat.std()),
            "v_new_mean": float(new_v_flat.mean()),
            "v_new_std": float(new_v_flat.std()),
        }
        train_metrics = getattr(self.policy_value_net, "last_train_metrics", {}) or {}
        self.last_update_metrics.update({
            "mean_sample_w": float(train_metrics.get("mean_sample_w", 1.0)),
            "frac_high_weight": float(train_metrics.get("frac_high_weight", 0.0)),
        })
        return loss, entropy

    def append_batch_log(self, batch_result):
        if not self.batch_log_file:
            return
        log_dir = os.path.dirname(os.path.abspath(self.batch_log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(self.batch_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(batch_result, sort_keys=True) + "\n")

    def policy_evaluate(self, n_games=None):
        if n_games is None:
            n_games = self.eval_games
        if n_games <= 0:
            return self.best_win_ratio

        def policy_value_batch_fn(states_np):
            return self.policy_value_net.policy_value_inference(states_np)

        print(
            "Starting policy evaluation against Pure MCTS for {} games...".format(
                n_games
            ),
            flush=True,
        )
        current = MCTSPlayer(self.policy_value_net.policy_value_fn,
                             policy_value_batch_fn,
                             c_puct=self.c_puct,
                             n_playout=self.eval_n_playout,
                             vl_k=self.vl_k,
                             n_vl=self.n_vl,
                             max_oversample=self.max_oversample)
        pure = MCTS_Pure(c_puct=5, n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        eval_start = time.time()
        for i in range(n_games):
            game_start = time.time()
            move_log_prefix = "[eval game {}/{}]".format(i + 1, n_games)
            winner = self.game.start_play(current, pure, start_player=i % 2,
                                          is_shown=0,
                                          move_log_prefix=move_log_prefix)
            win_cnt[winner] += 1
            elapsed = time.time() - game_start
            total_elapsed = time.time() - eval_start
            print(
                "Evaluation game {}/{} finished: winner={}, win={}, lose={}, tie={}, elapsed={:.1f}s, total_elapsed={:.1f}s".format(
                    i + 1,
                    n_games,
                    winner,
                    win_cnt[1],
                    win_cnt[2],
                    win_cnt[-1],
                    elapsed,
                    total_elapsed,
                ),
                flush=True,
            )
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]),
            flush=True)
        return win_ratio

    def run(self):
        self.tactical_pretrain()
        self.start_async_pipeline()
        try:
            target_updates = self.game_batch_num
            update_count = 0
            loop_count = 0
            last_save_update = 0
            while update_count < target_updates:
                self.check_async_processes()
                loop_count += 1
                replay_metrics = self.drain_replay_queue(
                    max_games=max(1, self.num_workers * self.games_per_worker),
                    timeout=10.0,
                )
                print("async loop:{}, updates:{}/{}, drained_games:{}, positions:{}, data_buffer:{}".format(
                    loop_count, update_count, target_updates,
                    replay_metrics["games"], replay_metrics["positions"],
                    len(self.data_buffer)), flush=True)
                update_metrics = None
                if len(self.data_buffer) > self.batch_size:
                    self.policy_update()
                    update_count += 1
                    update_metrics = self.last_update_metrics
                    if update_count % self.weight_push_every == 0:
                        self.save_cpu_model_for_evaluator()
                        self.weight_event.set()
                        print("signaled GPU evaluator weight reload at update {}".format(
                            update_count), flush=True)
                    self.policy_value_net.save_model("./current_policy.model")
                    last_save_update = update_count
                self.append_batch_log({
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "batch": int(loop_count),
                    "update": int(update_count),
                    "data_buffer": int(len(self.data_buffer)),
                    "episode_len": float(replay_metrics["episode_len"]),
                    "drained_games": int(replay_metrics["games"]),
                    "drained_positions": int(replay_metrics["positions"]),
                    "eval_requests": int(replay_metrics["eval_requests"]),
                    "updated": update_metrics is not None,
                    "update_metrics": update_metrics,
                })
                if (update_count > 0 and update_metrics is not None and
                        update_count % self.check_freq == 0):
                    print("current training update: {}".format(update_count), flush=True)
                    win_ratio = self.policy_evaluate(self.eval_games)
                    self.policy_value_net.save_model("./current_policy.model")
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!", flush=True)
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model("./best_policy.model")
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
            if last_save_update != update_count:
                self.policy_value_net.save_model("./current_policy.model")
        except KeyboardInterrupt:
            # Save the model FIRST, before any cleanup that could block, so a
            # second Ctrl+C cannot prevent us from persisting the latest
            # weights.
            print("\nInterrupted. Saving checkpoint immediately...", flush=True)
            saved_paths = []
            for path in ("./interrupt_policy.model", "./current_policy.model"):
                try:
                    self.policy_value_net.save_model(path)
                    saved_paths.append(path)
                except BaseException as save_exc:
                    print("Save FAILED for {}: {}".format(path, save_exc),
                          flush=True)
            if saved_paths:
                print("Saved: {}".format(", ".join(saved_paths)), flush=True)
            else:
                print("WARNING: no checkpoint files were saved.", flush=True)
            # Signal children to exit ASAP so the join in `finally` is fast.
            if self.shutdown_event is not None:
                try:
                    self.shutdown_event.set()
                except Exception:
                    pass
        finally:
            try:
                self.stop_async_pipeline()
            except BaseException as stop_exc:
                print("stop_async_pipeline error: {}".format(stop_exc),
                      flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train AlphaZero Gomoku with central batched GPU evaluator")
    p.add_argument("--init-model", default=None)
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--games-per-worker", type=int, default=1)
    p.add_argument("--threads-per-worker", type=int, default=1)
    p.add_argument("--n-playout", type=int, default=800)
    p.add_argument("--eval-n-playout", type=int, default=400)
    p.add_argument("--c-puct", type=float, default=3.0)
    p.add_argument("--dirichlet-alpha", type=float, default=0.05)
    p.add_argument("--noise-eps", type=float, default=0.25)
    p.add_argument("--vl-k", type=int, default=4,
                   help="Non-terminal leaves to collect per MCTS neural-net batch.")
    p.add_argument("--n-vl", type=float, default=1.0,
                   help="Virtual loss magnitude applied during leaf selection.")
    p.add_argument("--max-oversample", type=int, default=3,
                   help="Terminal oversampling cap multiplier for leaf collection.")
    p.add_argument("--temperature-moves", type=int, default=8)
    p.add_argument("--temp-high", type=float, default=1.0)
    p.add_argument("--temp-low", type=float, default=1e-3)
    p.add_argument("--buffer-size", type=int, default=500000)
    p.add_argument("--recent-sample-window", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--game-batch-num", type=int, default=1500)
    p.add_argument("--check-freq", type=int, default=50,
                   help="Run policy evaluation every N training updates.")
    p.add_argument("--eval-games", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-timeout-ms", type=int, default=8)
    p.add_argument("--response-timeout", type=float, default=180.0)
    p.add_argument("--disable-cuda-graphs", action="store_true",
                   help="Disable evaluator CUDA Graph capture/replay.")
    p.add_argument("--disable-inference-fp16", action="store_true",
                   help="Disable evaluator FP16 + channels_last inference path.")
    p.add_argument("--tactic-pretrain-steps", type=int, default=10000,
                   help="Full policy/value/tactic tactical pretraining steps before self-play. Use 0 to disable.")
    p.add_argument("--tactic-pretrain-samples", type=int, default=80000,
                   help="Number of generated tactical base samples before D4 augmentation.")
    p.add_argument("--tactic-pretrain-batch-size", type=int, default=512)
    p.add_argument("--tactic-pretrain-lr", type=float, default=1e-3)
    p.add_argument("--tactic-pretrain-workers", type=int, default=1,
                   help="Number of CPU worker processes for tactical sample generation. 1 keeps the existing single-process behavior; >=2 enables a spawn-based ProcessPoolExecutor (CPU/numpy only, no CUDA in workers).")
    p.add_argument("--tactic-loss-weight", type=float, default=0.25,
                   help="Weight for auxiliary tactic BCE loss during policy updates. New self-play data carries B1 tactic labels; old 3-tuple replay still falls back to zero labels. Use 0 to disable.")
    p.add_argument("--pretrain-tactic-loss-weight", type=float, default=0.5,
                   help="Auxiliary tactic BCE loss weight used only during tactical pretraining.")
    p.add_argument("--tactic-sample-weight", type=float, default=1.5,
                   help="Per-sample policy/value loss boost from tactic labels. 0 disables Step 2 reweighting.")
    p.add_argument("--forced-ratio", type=float, default=0.6,
                   help="Target ratio of forced win/block samples in tactical pretraining data.")
    p.add_argument("--block-value-target", type=float, default=0.3,
                   help="Value target for forced block tactical pretraining samples.")
    p.add_argument("--tactic-probe", action="store_true",
                   help="Run only the 5k-sample/1000-step forced-block tactical probe, then exit without self-play or checkpoint writes.")
    p.add_argument("--tactic-probe-steps", type=int, default=1000)
    p.add_argument("--tactic-probe-samples", type=int, default=5000)
    p.add_argument("--tactic-probe-positions", type=int, default=200)
    p.add_argument("--tactic-probe-threshold", type=float, default=0.85,
                   help="Acceptance target for forced-block top-1 probe accuracy.")
    p.add_argument("--tactic-probe-bug-threshold", type=float, default=0.50,
                   help="Below this forced-block top-1 accuracy, report likely tactical sample generation bug.")
    p.add_argument("--in-channels", type=int, default=4,
                   help="Neural-network input channels. 4 preserves legacy checkpoints; 5 enables the opp_win_here must-block tactical plane.")
    p.add_argument("--backbone", choices=("mlp", "mixer"), default="mlp",
                   help="Policy-value backbone. Default keeps the existing MLP; mixer is opt-in.")
    p.add_argument("--mixer-dim", type=int, default=128)
    p.add_argument("--mixer-depth", type=int, default=6)
    p.add_argument("--mixer-token-hidden", type=int, default=256)
    p.add_argument("--mixer-ch-hidden", type=int, default=384)
    p.add_argument("--mixer-value-hidden", type=int, default=128)
    p.add_argument("--mixer-dropout", type=float, default=0.1)
    p.add_argument("--batch-log-file", default="training_batches.log",
                   help="Path to append one JSON training summary per game batch. Use an empty string to disable.")
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
        eval_n_playout=args.eval_n_playout,
        c_puct=args.c_puct,
        dirichlet_alpha=args.dirichlet_alpha,
        noise_eps=args.noise_eps,
        vl_k=args.vl_k,
        n_vl=args.n_vl,
        max_oversample=args.max_oversample,
        temperature_moves=args.temperature_moves,
        temp_high=args.temp_high,
        temp_low=args.temp_low,
        buffer_size=args.buffer_size,
        recent_sample_window=args.recent_sample_window,
        batch_size=args.batch_size,
        game_batch_num=args.game_batch_num,
        check_freq=args.check_freq,
        eval_games=args.eval_games,
        eval_batch_size=args.eval_batch_size,
        eval_timeout_ms=args.eval_timeout_ms,
        response_timeout=args.response_timeout,
        batch_log_file=args.batch_log_file,
        use_cuda_graphs=not args.disable_cuda_graphs,
        inference_fp16=not args.disable_inference_fp16,
        tactic_pretrain_steps=args.tactic_pretrain_steps,
        tactic_pretrain_samples=args.tactic_pretrain_samples,
        tactic_pretrain_batch_size=args.tactic_pretrain_batch_size,
        tactic_pretrain_lr=args.tactic_pretrain_lr,
        tactic_pretrain_workers=args.tactic_pretrain_workers,
        tactic_loss_weight=args.tactic_loss_weight,
        pretrain_tactic_loss_weight=args.pretrain_tactic_loss_weight,
        forced_ratio=args.forced_ratio,
        block_value_target=args.block_value_target,
        tactic_sample_weight=args.tactic_sample_weight,
        tactic_probe=args.tactic_probe,
        tactic_probe_steps=args.tactic_probe_steps,
        tactic_probe_samples=args.tactic_probe_samples,
        tactic_probe_positions=args.tactic_probe_positions,
        tactic_probe_threshold=args.tactic_probe_threshold,
        tactic_probe_bug_threshold=args.tactic_probe_bug_threshold,
        in_channels=args.in_channels,
        backbone=args.backbone,
        mixer_dim=args.mixer_dim,
        mixer_depth=args.mixer_depth,
        mixer_token_hidden=args.mixer_token_hidden,
        mixer_ch_hidden=args.mixer_ch_hidden,
        mixer_value_hidden=args.mixer_value_hidden,
        mixer_dropout=args.mixer_dropout,
    )
    if args.tactic_probe:
        pipeline.run_tactical_probe()
    else:
        pipeline.run()
