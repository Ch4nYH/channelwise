import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnnutils

import numpy as np
import torchvision

from utils import copy_and_pad, preprocess
from optimizers import Optimizer
from collections import defaultdict

class LSTMCoordinator(nn.Module):

    def __init__(self, step_increment = 200, max_length = 0, hidden_layers = 2, hidden_units = 20, input_dim = 2, \
        num_actions = 3):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.input_dim = input_dim

        self.lstm = torch.nn.LSTM(self.input_dim, self.hidden_units, self.hidden_layers)
        self.max_length = max_length
        self.decoder = nn.Linear(2 * self.hidden_units, num_actions)
    def forward(self, x, hx, cx):
        
        if (hx is None):
            output, (hx, cx) = self.lstm(x)
        else:
            output, (hx, cx) = self.lstm(x, (hx, cx))

        lstm_out, _ = rnnutils.pad_packed_sequence(output, batch_first = True)
        lstm_out = hx.reshape(lstm_out.shape[0], -1)
        logit = self.decoder(lstm_out)
        prob = torch.nn.functional.softmax(logit, dim=-1)

        action = prob.multinomial(1)
        log_prob = torch.nn.functional.log_softmax(logit, dim=-1)
        selected_log_probs = log_prob.gather(1, action.data)
        return action, selected_log_probs, hx, cx  



class MixtureOptimizer(object):
    def __init__(self, parameters, alpha = 0.001, beta1 = 0.9, beta2 = 0.999, \
        eta = 1e-8, USE_CUDA = True, writer = None, layers = None, names = None):
        param = list(parameters)
        self.parameters = param
        self.state = defaultdict(dict)
        
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.layers = layers
        self.actions = [0] * len(layers)
        self.names = names
        self.update_rules = [
            lambda x, para: 0, \
            lambda x, para: self.alpha * x, \
            lambda x, para: self.alpha * self.state[para]['mt_hat'] / (torch.sqrt(self.state[para]['vt_hat']) + self.eta)]
        
        self.USE_CUDA = USE_CUDA
        
    def reset(self):
        self.hx = None
        self.cx = None
        self.selected_log_probs = []

    def set_action(self, actions):
        self.actions = actions

    def step(self):

        for name, p in zip(self.names, self.parameters):
            state = self.state[p]
            if len(state) == 0: # State initialization
                state['t'] = 0
                state['mt'] = 0
                state['vt'] = 0
                state['mt_hat'] = 0
                state['vt_hat'] = 0
                
            grad = p.grad.data
            state['t'] = state['t'] + 1
            state['mt'] = self.beta1 * state['mt'] + (1 - self.beta1) * grad
            state['vt'] = self.beta2 * state['vt'] + (1 - self.beta2) * (grad ** 2)
            state['mt_hat'] = state['mt'] / (1 - np.power(self.beta1, state['t']))
            state['vt_hat'] = state['vt'] / (1 - np.power(self.beta2, state['t']))
            layer_index = self.layers.find(name.split([0]))
            p.data.add_(-self.update_rules[self.action[layer_index]](grad, p))
        
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
    

if __name__ == '__main__':
    
    x = torch.randn((1, 3, 224, 224))
    label = torch.zeros(1).long()
    model = torchvision.models.AlexNet(2)
    
    criterion = nn.CrossEntropyLoss()
    controller = MixtureOptimizer(model.parameters(), 0.001)
    for i in range(10):
        y = model(x)
        loss = criterion(y, label)
        
        controller.zero_grad()
        loss.backward()
        controller.step()
        print(loss)