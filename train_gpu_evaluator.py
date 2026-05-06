# -*- coding: utf-8 -*-
"""
AlphaZero Gomoku training with a central batched GPU evaluator.

Main process trains the model. CPU worker processes run MCTS/self-play.
A dedicated GPU evaluator process batches leaf-state inference requests from
all workers and calls PolicyValueNet.policy_value(batch).
"""
from __future__ import print_function

import argparse
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
        for i in range(4):
            equi_state = np.array([np.rot90(s, i) for s in state])
            equi_mcts_prob = np.rot90(
                mcts_prob.reshape(board_height, board_width), i
            )
            extend_data.append((
                equi_state,
                equi_mcts_prob.flatten(),
                winner,
            ))
            equi_state_flip = np.array([np.fliplr(s) for s in equi_state])
            equi_mcts_prob_flip = np.fliplr(equi_mcts_prob)
            extend_data.append((
                equi_state_flip,
                equi_mcts_prob_flip.flatten(),
                winner,
            ))
    return extend_data


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
            board.current_state().reshape(
                4, self.board_width, self.board_height
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

        states_np must be shaped (B, 4, board_width, board_height). Full-board
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
                self.static_log_p, self.static_v = self.model(self.static_in)

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
                       inference_fp16=True):
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
                             use_amp=False)
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
            slot_pool_queue=slot_pool_queue,
            shm_in_name=shm_in_name,
            shm_out_name=shm_out_name,
            shm_slots=shm_slots,
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
                temp_low=temp_low)
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
                try:
                    replay_queue.put(replay_item, timeout=10.0)
                except queue.Full:
                    print("[worker {}] replay queue full; dropping completed game".format(
                        wid), flush=True)
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
                 response_timeout=180.0, c_puct=3.0, eval_n_playout=1600,
                 dirichlet_alpha=0.05, noise_eps=0.25,
                 vl_k=4, n_vl=1.0, max_oversample=3,
                 temperature_moves=8, temp_high=1.0, temp_low=1e-3,
                 buffer_size=500000, recent_sample_window=200000,
                 worker_model_file="./_tmp_gpu_evaluator_policy.model",
                 batch_log_file="training_batches.log",
                 use_cuda_graphs=True, inference_fp16=True):
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
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        self.learn_rate = 2e-3
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
        self.epochs = 5
        self.kl_targ = 0.02
        self.global_update_count = 0
        self.weight_push_every = 4
        self.lr_schedule = [
            (3000, 2e-3),
            (15000, 5e-4),
            (40000, 1e-4),
            (float("inf"), 2e-5),
        ]
        self.game_batch_num = int(game_batch_num)
        self.eval_games = int(eval_games)
        self.best_win_ratio = 0.0
        self.pure_mcts_playout_num = 2000

        self.policy_value_net = PolicyValueNet(
            self.board_width, self.board_height,
            model_file=init_model, use_gpu=self.use_gpu)

    def save_cpu_model_for_evaluator(self):
        sd = self.policy_value_net.get_policy_param()
        cpu_sd = {}
        for k, v in sd.items():
            cpu_sd[k] = v.detach().cpu() if hasattr(v, "detach") else v
        torch.save(cpu_sd, self.worker_model_file)

    def setup_shared_memory(self):
        board_size = self.board_width * self.board_height
        in_channels = 4
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
                4,
                self.use_cuda_graphs,
                self.inference_fp16,
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
                4,
                self.use_cuda_graphs,
                self.inference_fp16,
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

        self.learn_rate = self.get_scheduled_lr()
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
            winner = self.game.start_play(current, pure, start_player=i % 2,
                                          is_shown=0)
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
    p.add_argument("--eval-n-playout", type=int, default=1600)
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
    p.add_argument("--eval-games", type=int, default=10)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--eval-timeout-ms", type=int, default=8)
    p.add_argument("--response-timeout", type=float, default=180.0)
    p.add_argument("--disable-cuda-graphs", action="store_true",
                   help="Disable evaluator CUDA Graph capture/replay.")
    p.add_argument("--disable-inference-fp16", action="store_true",
                   help="Disable evaluator FP16 + channels_last inference path.")
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
    )
    pipeline.run()
