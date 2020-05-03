#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import abc
from pdb import set_trace as bp
import numpy as np
import torch
import torch.nn as nn

from a2c_ppo_acktr.distributions import Gaussian, Categorical
from a2c_ppo_acktr.models.rnn_state_encoder import RNNStateEncoder


class Policy(nn.Module):
    def __init__(self, batch_size, input_size=1, hidden_size=1, output_size=1, mean_range=[0, 1]):
        super().__init__()
        self.net = BasicNet(batch_size, input_size, hidden_size)
        # self.action_distribution = CategoricalNet(self.net.output_size, self.dim_actions)
        self.action_distribution = Gaussian(batch_size * hidden_size, num_outputs=output_size, mean_range=mean_range, std_epsilon=0.001)
        self.critic = CriticHead(batch_size * hidden_size)
        self.recurrent_hidden_state_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size

    def forward(self, *x):
        raise NotImplementedError

    def act(self, observations, rnn_hidden_states, deterministic=False):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states)
        features = features.view(-1, self.batch_size * self.recurrent_hidden_state_size)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, distribution

    def get_value(self, observations, rnn_hidden_states):
        features, _ = self.net(observations, rnn_hidden_states)
        features = features.view(-1, self.batch_size * self.recurrent_hidden_state_size)
        return self.critic(features)

    def evaluate_actions(self, observations, rnn_hidden_states, action):
        features, rnn_hidden_states = self.net(observations, rnn_hidden_states)
        features = features.view(-1, self.batch_size * self.recurrent_hidden_state_size)
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states#, distribution



class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass


class BasicNet(Net):
    def __init__(self, batch_size, input_size=1, hidden_size=1):
        super().__init__()
        self._batch_size = batch_size
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.state_encoder = RNNStateEncoder(self._batch_size, input_size=self._input_size, hidden_size=self._hidden_size)
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states):
        output, rnn_hidden_states = self.state_encoder(observations, rnn_hidden_states)
        return output, rnn_hidden_states
