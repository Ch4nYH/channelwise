import numpy as np
from collections import defaultdict
import torch

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
        self.num_steps = 0
        self.selected_log_probs = []
        self.state = defaultdict(dict)

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_actions(self, actions):
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
            layer_index = self.layers.index(name.split('.')[0])
            p.data.add_(-self.update_rules[self.actions[layer_index]](grad, p))
        
    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                
                
class ChannelWiseOptimizer(object):
    def __init__(self, parameters, names, alpha = 0.001, beta1 = 0.9, beta2 = 0.999):
        param = list(parameters)
        self.parameters = param
        self.alpha = alpha
        self.mask = defaultdict()
        self.state = defaultdict(dict)
        self.eta = 1e-8
        self.names = names
        self.beta1 = beta1
        self.beta2 = beta2
    def set_mask(self, name, value):
        self.mask[name] = value
        
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
            
            for i in range(p.grad.shape[0]):
                try:
                    p.grad.data[i, ...].mul_(self.mask[name.split('.')[0] + "_" + str(i)])
                except KeyError:
                    print(name.split('.')[0] + "_" + str(i), "not found.")
                p.data.add_(self.alpha * self.state[p]['mt_hat'] / (torch.sqrt(self.state[p]['vt_hat']) + self.eta))
           

    def zero_grad(self):
        for p in self.parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
