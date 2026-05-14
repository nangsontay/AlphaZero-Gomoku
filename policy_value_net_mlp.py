# -*- coding: utf-8 -*-
"""
Pure-MLP policy-value network for AlphaZero Gomoku.

Drop-in replacement for policy_value_net_pytorch.PolicyValueNet under the
"no nn.Conv*d anywhere on the runtime import path" constraint.

Public API matches policy_value_net_pytorch.PolicyValueNet exactly so that
train.py, train_mp.py, train_gpu_evaluator.py, and play.py can switch by a
single import line.
"""

import json
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from contextlib import nullcontext

# v2 addition (§3.6): module-level architecture version.
MLP_ARCH_VERSION = "1.1.0-tactic"


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def _apply_d4(state, k_rot, do_flip):
    """state: (4, 15, 15) numpy. Returns transformed state in same shape."""
    s = np.rot90(state, k_rot, axes=(1, 2))
    if do_flip:
        s = s[:, :, ::-1]
    return np.ascontiguousarray(s)


def _invert_d4_policy(probs225, k_rot, do_flip, board_w=15, board_h=15):
    """probs225: (225,) numpy of policy on the transformed board.
    Returns probs in the original board's index ordering."""
    # E06 guard: refuse mismatched policy lengths.
    if probs225.size != board_w * board_h:
        raise ValueError(
            f"_invert_d4_policy expected length {board_w*board_h}; "
            f"got {probs225.size}."
        )
    p = probs225.reshape(board_h, board_w)
    if do_flip:
        p = p[:, ::-1]
    p = np.rot90(p, -k_rot)
    return np.ascontiguousarray(p).reshape(-1)


class PerCellEmbed(nn.Module):
    """Per-cell shared linear projection.

    Each of the 225 cells' 4-channel feature vector is projected through the
    SAME nn.Linear(in_channels -> embed_dim).
    """

    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.proj = nn.Linear(self.in_channels, self.embed_dim)

    def forward(self, x):
        # E01 guard: surface shape mismatches as actionable errors.
        if x.dim() != 4 or x.size(1) != self.in_channels:
            raise ValueError(
                f"PerCellEmbed expected (N, {self.in_channels}, H, W); "
                f"got tuple({tuple(x.shape)})."
            )
        n = x.size(0)
        x = x.permute(0, 2, 3, 1).reshape(n, -1, self.in_channels)
        x = self.proj(x)                          # (N, 225, embed_dim)
        return x.reshape(n, -1)                   # (N, 225*embed_dim)


class MLPResBlock(nn.Module):
    """Bottleneck pre-norm residual block.

    x + Linear(inner→dim, GELU(LN(Linear(dim→inner, GELU(LN(x))))))
    With bottleneck_ratio=4: dim→dim//4→dim, reducing params ~4× per block.
    """

    def __init__(self, dim, dropout=0.1, norm="ln", act="gelu",
                 bottleneck_ratio=4):
        super().__init__()
        inner = max(64, dim // bottleneck_ratio)
        Norm = nn.LayerNorm if norm == "ln" else nn.BatchNorm1d
        Act = nn.GELU if act == "gelu" else nn.ReLU
        self.n1 = Norm(dim)
        self.n2 = Norm(inner)
        self.fc1 = nn.Linear(dim, inner)
        self.fc2 = nn.Linear(inner, dim)
        self.act = Act()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.fc1(self.act(self.n1(x)))
        h = self.drop(h)
        h = self.fc2(self.act(self.n2(h)))
        return x + h


class MLPNet(nn.Module):
    """Pure-MLP policy-value net for 15x15 Gomoku.

    Approximate parameter count: ~5.5M (bottleneck config, optimised for 15x15).
    Uses bottleneck ResBlocks (dim→dim//4→dim) to cut per-block params ~4×.
    """

    def __init__(self, board_width, board_height, in_channels=4,
                 embed_dim=16, hidden_dim=256, num_blocks=5,
                 value_hidden=128, dropout=0.1, norm="ln", act="gelu",
                 bottleneck_ratio=4):
        super().__init__()
        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_blocks = int(num_blocks)
        self.value_hidden = int(value_hidden)
        self.norm = str(norm)
        self.act = str(act)
        self.board_size = self.board_width * self.board_height
        self.bottleneck_ratio = int(bottleneck_ratio)

        self.embed = PerCellEmbed(self.in_channels, self.embed_dim)

        Norm = nn.LayerNorm if self.norm == "ln" else nn.BatchNorm1d
        Act = nn.GELU if self.act == "gelu" else nn.ReLU

        self.stem = nn.Sequential(
            nn.Linear(self.board_size * self.embed_dim, self.hidden_dim),
            Norm(self.hidden_dim),
            Act(),
        )
        self.trunk = nn.Sequential(*[
            MLPResBlock(self.hidden_dim, dropout, norm, act,
                        bottleneck_ratio=self.bottleneck_ratio)
            for _ in range(self.num_blocks)
        ])
        self.head_norm = Norm(self.hidden_dim)

        self.policy_head = nn.Linear(self.hidden_dim, self.board_size)
        self.tactic_head = nn.Linear(self.hidden_dim, self.board_size)

        self.value_fc1 = nn.Linear(self.hidden_dim, self.value_hidden)
        self.value_act = Act()
        self.value_fc2 = nn.Linear(self.value_hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # E02 guard: surface board-size mismatches.
        if x.size(2) != self.board_height or x.size(3) != self.board_width:
            raise ValueError(
                f"MLPNet expected H={self.board_height}, W={self.board_width}; "
                f"got H={x.size(2)}, W={x.size(3)}."
            )
        x = self.embed(x)                                  # (N, 225*embed)
        x = self.stem(x)                                   # (N, hidden)
        x = self.trunk(x)                                  # (N, hidden)
        x = self.head_norm(x)                              # (N, hidden)
        # Cast logits to float32 BEFORE log_softmax so AMP/FP16 paths stay
        # numerically stable.
        log_p = F.log_softmax(self.policy_head(x).float(), dim=1)   # (N, 225)
        tactic_logits = self.tactic_head(x).float()                 # (N, 225)
        v = torch.tanh(self.value_fc2(self.value_act(self.value_fc1(x))))
        return log_p, v, tactic_logits


class PolicyValueNet:
    """policy-value network wrapper. Public API mirrors the CNN wrapper exactly."""

    def __init__(self, board_width, board_height, model_file=None,
                 use_gpu=False, in_channels=4,
                 # MLP-specific knobs (callers may ignore them):
                 embed_dim=24, hidden_dim=768, num_blocks=3,
                 value_hidden=128, dropout=0.1, norm="ln", act="gelu",
                  use_amp=None, sym_loss_weight=0.0,
                  tactic_loss_weight=0.25,
                 # v2 addition (§5): control random D4 in policy_value_fn.
                 search_d4_random=True,
                 # CNN knobs accepted-and-ignored for backward-compat:
                 num_blocks_cnn=None, channels=None):
        self.use_gpu = bool(use_gpu) and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.in_channels = int(in_channels)
        self.l2_const = 1e-4
        self.learn_rate = 5e-4
        self.grad_clip_norm = 1.0
        self.use_amp = bool(self.use_gpu if use_amp is None else use_amp)
        self.sym_loss_weight = float(sym_loss_weight)
        self.tactic_loss_weight = float(tactic_loss_weight)
        self.search_d4_random = bool(search_d4_random)

        self.policy_value_net = MLPNet(
            self.board_width, self.board_height,
            in_channels=self.in_channels,
            embed_dim=embed_dim, hidden_dim=hidden_dim,
            num_blocks=num_blocks, value_hidden=value_hidden,
            dropout=dropout, norm=norm, act=act,
        ).to(self.device)

        self.optimizer = optim.SGD(
            self.policy_value_net.parameters(),
            lr=self.learn_rate, momentum=0.9,
            weight_decay=self.l2_const, nesterov=True,
        )

        # GradScaler for AMP. Mirrors the CNN wrapper's compatibility shim.
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
            except TypeError:
                self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            legacy_grad_scaler = getattr(torch.cuda.amp, "GradScaler")
            self.scaler = legacy_grad_scaler(enabled=self.use_amp)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            # E03 guard: accept the common "{'state_dict': ...}" wrapping.
            if isinstance(net_params, dict) and "state_dict" in net_params \
                    and not any(k.endswith(".weight") for k in net_params.keys()
                                if k != "state_dict"):
                net_params = net_params["state_dict"]
            try:
                missing, unexpected = self.policy_value_net.load_state_dict(
                    net_params, strict=False)
                allowed_missing = {"tactic_head.weight", "tactic_head.bias"}
                real_missing = [k for k in missing if k not in allowed_missing]
                if real_missing or unexpected:
                    raise RuntimeError(
                        "missing keys: {}, unexpected keys: {}".format(
                            real_missing, list(unexpected)))
                if missing:
                    print("[mlp] initialized new tactic head while loading legacy checkpoint: {}".format(
                        ", ".join(missing)), flush=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Incompatible MLP checkpoint. The file at '{}' does not "
                    "match the pure-MLP architecture defined in "
                    "policy_value_net_mlp.MLPNet. If this is a legacy CNN "
                    "checkpoint (e.g. saved by policy_value_net_pytorch.py "
                    "before the no-CNN refactor), it cannot be reused: the "
                    "no-CNN constraint forbids both runtime CNN inference "
                    "and any offline CNN-instantiating conversion tool."
                    .format(model_file)
                ) from exc

            # v2 addition (§3.6): version-check sidecar if present.
            self._maybe_check_sidecar(model_file)

    # v2 addition (§3.6) — sidecar plumbing.
    def _sidecar_path(self, model_file):
        return model_file + ".json"

    def _build_sidecar(self):
        return {
            "MLP_ARCH_VERSION": MLP_ARCH_VERSION,
            "board_width": self.board_width,
            "board_height": self.board_height,
            "in_channels": self.in_channels,
            "embed_dim": self.policy_value_net.embed_dim,
            "hidden_dim": self.policy_value_net.hidden_dim,
            "num_blocks": self.policy_value_net.num_blocks,
            "value_hidden": self.policy_value_net.value_hidden,
            "norm": self.policy_value_net.norm,
            "act": self.policy_value_net.act,
            "d4_randomisation": self.search_d4_random,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "git_sha": os.environ.get("GIT_SHA", "unknown"),
        }

    def _maybe_check_sidecar(self, model_file):
        path = self._sidecar_path(model_file)
        if not os.path.exists(path):
            print(f"[mlp] WARNING: no sidecar for '{model_file}'; "
                  f"loaded weights without version check.")
            return
        with open(path) as f:
            meta = json.load(f)
        ver = meta.get("MLP_ARCH_VERSION")
        if ver != MLP_ARCH_VERSION:
            if ver == "1.0.0":
                print(f"[mlp] WARNING: loading legacy MLP sidecar version '{ver}' "
                      f"with newly initialized tactic head.", flush=True)
                return
            raise RuntimeError(
                f"MLP_ARCH_VERSION mismatch: checkpoint='{ver}' "
                f"vs running='{MLP_ARCH_VERSION}'. Refusing to load."
            )

    def save_model(self, model_file):
        """save model params to file plus sidecar metadata (v2 §3.6)."""
        torch.save(self.policy_value_net.state_dict(), model_file)
        sidecar = self._build_sidecar()
        with open(self._sidecar_path(model_file), "w") as f:
            json.dump(sidecar, f, indent=2)

    # --- verbatim port from policy_value_net_pytorch.PolicyValueNet ---

    def _to_tensor(self, array_like):
        arr = np.asarray(array_like, dtype=np.float32)
        return torch.from_numpy(arr).to(self.device)

    def _autocast_context(self):
        if not self.use_amp:
            return nullcontext()
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda")
        legacy_autocast = getattr(torch.cuda.amp, "autocast")
        return legacy_autocast()

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        self.policy_value_net.eval()

        state_batch = self._to_tensor(state_batch)

        with torch.no_grad():
            with self._autocast_context():
                log_act_probs, value, _ = self.policy_value_net(state_batch)

        act_probs = torch.exp(log_act_probs).detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        return act_probs, value

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    # --- end verbatim port ---

    def policy_value_inference(self, state_batch, fp16=False,
                               channels_last=False):
        """Evaluator-only inference path. NO D4 randomisation here (§5
        boundary contract: GATE-3 closed = accept asymmetric)."""
        self.policy_value_net.eval()
        arr = np.asarray(state_batch, dtype=np.float32)
        state_tensor = torch.from_numpy(arr).to(self.device, non_blocking=True)
        if fp16 and self.use_gpu:
            state_tensor = state_tensor.half()
        if channels_last:
            # NHWC layout is fine on the 4D input; the MLP forward will permute
            # away to (N, 225, C) before any matmul, so the layout becomes a
            # documented no-op for an all-Linear backbone. Kept for API parity
            # with policy_value_net_pytorch.PolicyValueNet so the GPU
            # evaluator's call site does not need to change.
            state_tensor = state_tensor.to(memory_format=torch.channels_last)
        with torch.no_grad():
            log_act_probs, value, _ = self.policy_value_net(state_tensor)
        act_probs = torch.exp(log_act_probs.float()).detach().cpu().numpy()
        value = value.float().detach().cpu().numpy()
        return act_probs, value

    def policy_value_fn(self, board):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        self.policy_value_net.eval()
        legal_positions = board.availables

        state = np.ascontiguousarray(
            board.current_state().reshape(
                -1, self.in_channels, self.board_width, self.board_height
            ).astype(np.float32)
        )

        # E07 guard: terminal board with no legal positions.
        if not legal_positions:
            state_tensor = torch.from_numpy(state).to(self.device)
            with torch.no_grad():
                with self._autocast_context():
                    _, v, _ = self.policy_value_net(state_tensor)
            return iter([]), float(v.detach().cpu().numpy()[0][0])

        if self.search_d4_random:
            k_rot = random.randint(0, 3)
            do_flip = bool(random.getrandbits(1))
            state[0] = _apply_d4(state[0], k_rot, do_flip)
        else:
            k_rot, do_flip = 0, False

        state_tensor = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            with self._autocast_context():
                log_p, v, _ = self.policy_value_net(state_tensor)
        act_probs = torch.exp(log_p).detach().cpu().numpy().flatten()
        value = float(v.detach().cpu().numpy()[0][0])

        if self.search_d4_random:
            act_probs = _invert_d4_policy(
                act_probs, k_rot, do_flip,
                board_w=self.board_width, board_h=self.board_height,
            )

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr,
                   tactic_batch=None, tactic_mask=None):
        """perform a training step (v2 §3.4 with E04 finite-loss guard)."""
        self.policy_value_net.train()
        state_batch = self._to_tensor(state_batch)
        mcts_probs = self._to_tensor(mcts_probs)
        winner_batch = self._to_tensor(winner_batch)
        tactic_targets = None
        tactic_mask_tensor = None
        if tactic_batch is not None:
            tactic_targets = self._to_tensor(tactic_batch)
            if tactic_mask is not None:
                tactic_mask_tensor = self._to_tensor(tactic_mask).view(-1, 1)

        self.optimizer.zero_grad(set_to_none=True)
        set_learning_rate(self.optimizer, lr)

        with self._autocast_context():
            log_p, v, tactic_logits = self.policy_value_net(state_batch)
            value_loss = F.mse_loss(v.view(-1), winner_batch)
            policy_loss = -torch.mean(torch.sum(mcts_probs * log_p, dim=1))
            loss = value_loss + policy_loss
            if tactic_targets is not None and self.tactic_loss_weight > 0.0:
                tactic_loss_raw = F.binary_cross_entropy_with_logits(
                    tactic_logits, tactic_targets, reduction='none')
                if tactic_mask_tensor is not None:
                    tactic_loss_raw = tactic_loss_raw * tactic_mask_tensor
                    valid_elements = tactic_mask_tensor.sum() * tactic_loss_raw.size(1)
                    tactic_loss = tactic_loss_raw.sum() / valid_elements.clamp_min(1.0)
                else:
                    tactic_loss = tactic_loss_raw.mean()
                loss = loss + self.tactic_loss_weight * tactic_loss

            # Optional symmetry regularisation (see §6).
            if self.sym_loss_weight > 0.0:
                loss = loss + self.sym_loss_weight * self._sym_loss(
                    state_batch, log_p
                )

        # E04 guard: never let NaN/Inf reach .backward().
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return float("nan"), float("nan")

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.policy_value_net.parameters(), self.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy_value_net.parameters(), self.grad_clip_norm
            )
            self.optimizer.step()

        log_p_f32 = log_p.float()
        entropy = -torch.mean(
            torch.sum(torch.exp(log_p_f32) * log_p_f32, dim=1)
        )
        return loss.item(), entropy.item()

    def tactic_train_step(self, state_batch, tactic_batch, lr):
        """Train only the auxiliary tactic objective on generated tactic labels."""
        self.policy_value_net.train()
        state_batch = self._to_tensor(state_batch)
        tactic_targets = self._to_tensor(tactic_batch)

        self.optimizer.zero_grad(set_to_none=True)
        set_learning_rate(self.optimizer, lr)

        with self._autocast_context():
            _, _, tactic_logits = self.policy_value_net(state_batch)
            loss = F.binary_cross_entropy_with_logits(
                tactic_logits, tactic_targets)

        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            return float("nan")

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.policy_value_net.parameters(), self.grad_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.policy_value_net.parameters(), self.grad_clip_norm
            )
            self.optimizer.step()
        return loss.item()

    def _sym_loss(self, state_batch, log_p_orig):
        """Symmetry regularisation. state_batch: (N, 4, 15, 15) on self.device.

        v2 §6: one-sided cross-entropy with a stop-gradient teacher derived
        from the same network on the original state.
        """
        # E05 guard: empty batch produces NaN under mean(); short-circuit.
        if state_batch.size(0) == 0:
            return torch.zeros((), device=self.device)

        # Pick one of the 7 non-identity D4 transforms uniformly (with
        # short-circuit on identity).
        k_rot = random.randint(0, 3)
        do_flip = bool(random.getrandbits(1))
        if k_rot == 0 and not do_flip:
            return torch.zeros((), device=self.device)

        s = state_batch
        if k_rot:
            s = torch.rot90(s, k_rot, dims=(2, 3))
        if do_flip:
            s = torch.flip(s, dims=(3,))

        log_p_d4, _, _ = self.policy_value_net(s)

        p_orig = torch.exp(log_p_orig).detach().view(
            -1, self.board_height, self.board_width
        )
        if k_rot:
            p_orig = torch.rot90(p_orig, k_rot, dims=(1, 2))
        if do_flip:
            p_orig = torch.flip(p_orig, dims=(2,))
        p_orig = p_orig.reshape(-1, self.board_height * self.board_width)

        return -(p_orig * log_p_d4).sum(dim=1).mean()
