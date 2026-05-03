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


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class Net(nn.Module):
    """policy-value network module"""
    def __init__(self, board_width, board_height):
        super(Net, self).__init__()

        self.board_width = board_width
        self.board_height = board_height

        # common layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # action policy layers
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_fc1 = nn.Linear(
            4 * board_width * board_height,
            board_width * board_height
        )

        # state value layers
        self.val_conv1 = nn.Conv2d(128, 2, kernel_size=1)
        self.val_fc1 = nn.Linear(2 * board_width * board_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, state_input):
        # common layers
        x = F.relu(self.conv1(state_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # action policy layers
        x_act = F.relu(self.act_conv1(x))
        x_act = x_act.view(-1, 4 * self.board_width * self.board_height)
        x_act = F.log_softmax(self.act_fc1(x_act), dim=1)

        # state value layers
        x_val = F.relu(self.val_conv1(x))
        x_val = x_val.view(-1, 2 * self.board_width * self.board_height)
        x_val = F.relu(self.val_fc1(x_val))
        x_val = torch.tanh(self.val_fc2(x_val))

        return x_act, x_val


class PolicyValueNet:
    """policy-value network"""
    def __init__(self, board_width, board_height, model_file=None, use_gpu=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")

        self.board_width = board_width
        self.board_height = board_height
        self.l2_const = 1e-4

        self.policy_value_net = Net(board_width, board_height).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            weight_decay=self.l2_const
        )

        if model_file:
            net_params = torch.load(model_file, map_location=self.device)
            self.policy_value_net.load_state_dict(net_params)

    def _to_tensor(self, array_like):
        arr = np.asarray(array_like, dtype=np.float32)
        return torch.from_numpy(arr).to(self.device)

    def policy_value(self, state_batch):
        """
        input: a batch of states
        output: a batch of action probabilities and state values
        """
        self.policy_value_net.eval()

        state_batch = self._to_tensor(state_batch)

        with torch.no_grad():
            log_act_probs, value = self.policy_value_net(state_batch)

        act_probs = torch.exp(log_act_probs).detach().cpu().numpy()
        value = value.detach().cpu().numpy()
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
                -1, 4, self.board_width, self.board_height
            ).astype(np.float32)
        )

        state_tensor = torch.from_numpy(current_state).to(self.device)

        with torch.no_grad():
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
        self.optimizer.zero_grad()

        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(state_batch)

        # loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, dim=1))
        loss = value_loss + policy_loss

        # backward and optimize
        loss.backward()
        self.optimizer.step()

        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs) * log_act_probs, dim=1)
        )

        return loss.item(), entropy.item()

    def get_policy_param(self):
        return self.policy_value_net.state_dict()

    def save_model(self, model_file):
        """save model params to file"""
        net_params = self.get_policy_param()
        torch.save(net_params, model_file)
