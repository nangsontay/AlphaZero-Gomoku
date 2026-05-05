# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in PyTorch
Updated for modern PyTorch + CUDA.

Original author: Junxiao Song
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from contextlib import nullcontext


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class ResBlock(nn.Module):
    """Two-convolution residual block with BatchNorm."""

    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual, inplace=True)
        return out


class ResNet(nn.Module):
    """10x128 AlphaZero-style policy-value ResNet for Gomoku."""

    def __init__(self, board_width, board_height, in_channels=4,
                 channels=128, num_blocks=10):
        super(ResNet, self).__init__()

        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.in_channels = int(in_channels)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.board_size = self.board_width * self.board_height

        self.stem = nn.Sequential(
            nn.Conv2d(
                self.in_channels, self.channels,
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[
            ResBlock(self.channels) for _ in range(self.num_blocks)
        ])

        # Fully convolutional policy head: one logit per board cell.
        self.policy_head = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # Value head keeps an FC collapse to scalar but widens the bottleneck.
        self.value_conv = nn.Conv2d(
            self.channels, 4, kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(4)
        self.value_fc1 = nn.Linear(4 * self.board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, state_input):
        x = self.stem(state_input)
        x = self.res_blocks(x)

        policy_logits = self.policy_head(x).reshape(-1, self.board_size)
        log_act_probs = F.log_softmax(policy_logits.float(), dim=1)

        value = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        value = value.reshape(-1, 4 * self.board_size)
        value = F.relu(self.value_fc1(value), inplace=True)
        value = torch.tanh(self.value_fc2(value))

        return log_act_probs, value


# Backward-compatible name for any external code importing Net directly.
Net = ResNet


class PolicyValueNet:
    """policy-value network"""
    def __init__(self, board_width, board_height, model_file=None,
                 use_gpu=False, in_channels=4, num_blocks=10, channels=128,
                 use_amp=None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.board_width = int(board_width)
        self.board_height = int(board_height)
        self.in_channels = int(in_channels)
        self.l2_const = 1e-4
        self.learn_rate = 2e-3
        self.grad_clip_norm = 1.0
        self.use_amp = bool(self.use_gpu if use_amp is None else use_amp)

        self.policy_value_net = ResNet(
            self.board_width,
            self.board_height,
            in_channels=self.in_channels,
            channels=channels,
            num_blocks=num_blocks,
        ).to(self.device)
        self.optimizer = optim.SGD(
            self.policy_value_net.parameters(),
            lr=self.learn_rate,
            momentum=0.9,
            weight_decay=self.l2_const,
            nesterov=True,
        )
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                self.scaler = torch.amp.GradScaler(
                    "cuda", enabled=self.use_amp
                )
            except TypeError:
                self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        else:
            legacy_grad_scaler = getattr(torch.cuda.amp, "GradScaler")
            self.scaler = legacy_grad_scaler(enabled=self.use_amp)

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            try:
                self.policy_value_net.load_state_dict(net_params)
            except RuntimeError as exc:
                raise RuntimeError(
                    "Incompatible model checkpoint for the Phase-B ResNet "
                    "architecture. Start with init_model=None or provide a "
                    "checkpoint saved from this architecture."
                ) from exc

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
                log_act_probs, value = self.policy_value_net(state_batch)

        act_probs = torch.exp(log_act_probs).detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        return act_probs, value

    def policy_value_inference(self, state_batch, fp16=False,
                               channels_last=False):
        """Evaluator-only inference path for persistent FP16/channels_last nets."""
        self.policy_value_net.eval()

        arr = np.asarray(state_batch, dtype=np.float32)
        state_tensor = torch.from_numpy(arr).to(self.device, non_blocking=True)
        if fp16 and self.use_gpu:
            state_tensor = state_tensor.half()
        if channels_last:
            state_tensor = state_tensor.to(memory_format=torch.channels_last)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_tensor)

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
        current_state = np.ascontiguousarray(
            board.current_state().reshape(
                -1, self.in_channels, self.board_width, self.board_height
            ).astype(np.float32)
        )

        state_tensor = torch.from_numpy(current_state).to(self.device)

        with torch.no_grad():
            with self._autocast_context():
                log_act_probs, value = self.policy_value_net(state_tensor)

        act_probs = torch.exp(log_act_probs).detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy()[0][0]

        act_probs = zip(legal_positions, act_probs[legal_positions])
        return act_probs, value

    def train_step(self, state_batch, mcts_probs, winner_batch, lr):
        """perform a training step"""
        self.policy_value_net.train()

        state_batch = self._to_tensor(state_batch)
        mcts_probs = self._to_tensor(mcts_probs)
        winner_batch = self._to_tensor(winner_batch)

        # zero the parameter gradients
        self.optimizer.zero_grad(set_to_none=True)

        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        with self._autocast_context():
            log_act_probs, value = self.policy_value_net(state_batch)

            # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
            # L2 penalty is incorporated in optimizer
            value_loss = F.mse_loss(value.view(-1), winner_batch)
            policy_loss = -torch.mean(
                torch.sum(mcts_probs * log_act_probs, dim=1)
            )
            loss = value_loss + policy_loss

        # backward and optimize
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

        # calc policy entropy, for monitoring only
        log_act_probs_for_entropy = log_act_probs.float()
        entropy = -torch.mean(
            torch.sum(
                torch.exp(log_act_probs_for_entropy) * log_act_probs_for_entropy,
                dim=1,
            )
        )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        """save model params to file"""
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)
