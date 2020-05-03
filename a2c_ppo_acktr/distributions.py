import math
from pdb import set_trace as bp
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super.entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_channels = 1):
        # num_inputs: #features for each coord
        # num_outputs: action_space
        super(Categorical, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_channels = num_channels
        print(num_channels)
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))
            

    def forward(self, x):
        # x: (coord, batch, *features)
        # will coordinate-wisely return distributions
        distributions = []
        print(self.num_channels)
        for channel in range(self.num_channels):
            dist = FixedCategorical(logits = self.linear(x[channel]))
            distributions.append(dist)
        return MultiCategorical(distributions)


class MultiCategorical(nn.Module):
    def __init__(self, distributions):
        super(MultiCategorical, self).__init__()
        # coordinate-wise distributions
        self.distributions = distributions

    def sample(self):
        actions = []
        for dist in self.distributions:
            actions.append(dist.sample())
        return torch.cat(actions, dim=1)
    
    def log_probs(self, actions, is_sum=True):
        # actions: (batch, coord)
        log_probs = []
        print(len(self.distributions))
        for coord in range(len(self.distributions)):
            try:
                log_probs.append(self.distributions[coord].log_probs(actions[:, coord:coord+1]))
            except:
                bp()
                log_probs.append(self.distributions[coord].log_probs(actions[:, coord:coord+1]))
        log_probs = torch.cat(log_probs, dim=1)
        if is_sum:
            return log_probs.sum(-1).unsqueeze(-1)
        else:
            return log_probs

    def entropy(self):
        # actions: (batch, coord)
        entropies = []
        for coord in range(len(self.distributions)):
            entropies.append(self.distributions[coord].entropy())
        entropies = torch.cat(entropies, dim=0)
        return entropies.unsqueeze(-1)

    def mode(self):
        actions = []
        for dist in self.distributions:
            actions.append(dist.probs.argmax(dim=-1, keepdim=True))
        return torch.cat(actions, dim=1)


class Gaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs=1, mean_range=[0, 1], std_epsilon=0.001):
        super(Gaussian, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self._num_inputs = num_inputs
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.fc_std = init_(nn.Linear(num_inputs, num_outputs))
        assert len(mean_range) == 2 and mean_range[0] < mean_range[1]
        self.mean_min = mean_range[0]
        self.mean_max = mean_range[1]
        self.std_epsilon = std_epsilon

    def forward(self, x):
        # x = x.view(1, self._num_inputs)
        action_mean = self.fc_mean(x)
        action_mean = F.sigmoid(action_mean) * (self.mean_max - self.mean_min) + self.mean_min
        action_std = F.softplus(self.fc_std(x)) + self.std_epsilon
        return FixedNormal(action_mean, action_std)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
