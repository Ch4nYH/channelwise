import torch
from utils import accuracy, AverageMeter
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from collections import deque

import numpy as np
from collections import defaultdict
from pdb import set_trace as bp

class MetaTrainer(object):
    def __init__(self, model, criterion, optimizer, **kwargs):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.USE_CUDA = kwargs['USE_CUDA']
        self.train_loader = kwargs['train_loader']
        self.val_loader = kwargs['val_loader']
        self.print_preq = kwargs['print_freq']
        self.writer = kwargs['writer']
        self.epochs = kwargs['epochs']
        self.use_log_loss = kwargs['use_log_loss']

        self.train_acc = AverageMeter()
        self.val_acc   = AverageMeter()
        
        self.window_size = 5

        self.iter_train_loader = iter(self.train_loader)
        self.iter_val_loader = iter(self.val_loader)
        self.total_steps = self.epochs * len(self.train_loader)
        self.total_steps_epoch = len(self.train_loader)
        self.total_steps_val = len(self.val_loader)
        self.step = 0
        
        if self.USE_CUDA:
            self.model = self.model.cuda()

    def get_steps(self):
        return self.total_steps, self.total_steps_epoch

    def reset(self):
        self.step = 0
        self.model.reset()

    def observe(self):
        losses = []
        optimizee_step = []
        val_losses = []
        channel_stats = []
        for idx in range(self.window_size):
            train_loss, train_acc = self.train_step()
            losses.append(train_loss.detach())
            optimizee_step.append((self.step + idx) / self.total_steps_epoch)
            val_loss = self.val_step()
            val_losses.append(val_loss)
        if not self.use_log_loss:
            losses = [sum(losses) / len(losses)]
        else:
            for i in range(len(losses)):
                losses[i] = torch.log(losses[i]+1e-6) - torch.log(losses[0] + 1e-6)
            losses = [sum(losses) / len(losses)]
        optimizee_step = [sum(optimizee_step) / len(optimizee_step)]
        optimizee_step = [torch.tensor(step).cuda() for step in optimizee_step]
        val_losses = [sum(val_losses) / len(val_losses)]
        observation = torch.stack(losses + optimizee_step + val_losses, dim=0)
        
        # prev_action = xxxxx # previous action = ?
        # ==============
        # | GET ACTION |
        # ==============
        prev_action = torch.tensor([0.0])
        channel_stats = list(self.model.get_channel_stats().values())
        channel_stats = torch.cat(channel_stats, 0)
        if self.USE_CUDA:
            prev_action = prev_action.cuda()
            channel_stats = channel_stats.cuda()
        observation = torch.cat([observation, prev_action - 1], dim = 0).unsqueeze(0)
        
        
        observation = observation.repeat(channel_stats.shape[0], 1)
        observation = torch.cat([observation, channel_stats], 1)
        return observation, torch.tensor(losses), torch.tensor(val_losses)

    def train_step(self): 
        self.model.train()
        self.optimizer.zero_grad()
        input, label = self.get_train_samples()
        if self.USE_CUDA:
            label = label.cuda()
            input = input.cuda()

        output = self.model(input)
        loss = self.criterion(output, label.long())
        acc = accuracy(output, label)
        loss.backward()
        self.optimizer.step()
        return loss, acc
            
    def val_step(self):
        self.model.eval()
        losses = []
        with torch.no_grad():
            input, label = self.get_val_samples()
            if self.USE_CUDA:
                label = label.cuda()
                input = input.cuda()

            output = self.model(input)
            acc = accuracy(output, label)
            loss = self.criterion(output, label.long())

        return loss

    def val(self):
        self.model.eval()
        accs = []
        losses = []
        with torch.no_grad():
            for _ in range(self.total_steps_val): 
                input, label = self.get_val_samples()
                if self.USE_CUDA:
                    label = label.cuda()
                    input = input.cuda()

                output = self.model(input)
                acc = accuracy(output, label)
                loss = self.criterion(output, label.long())
                accs.append(acc)
                losses.append(loss.detach().item())
        return np.mean(accs), np.mean(losses)

    def train_val_step(self):
        pass

    def get_train_samples(self):
        try:
            sample = next(self.iter_train_loader)
        except:
            self.iter_train_loader = iter(self.train_loader)
            sample = next(self.iter_train_loader)

        return sample

    def get_val_samples(self):
        try:
            sample = next(self.iter_val_loader)
        except:
            self.iter_val_loader = iter(self.val_loader)
            sample = next(self.iter_val_loader)

        return sample
    def get_optimizer(self):
        return self.optimizer

class MetaRunner(object):
    def __init__(self, trainer, rollouts, agent, ac, **kwargs):
        self.trainer = trainer
        self.rollouts = rollouts
        self.agent = agent
        self.ac = ac
        self.total_steps, self.total_steps_epoch = trainer.get_steps()
        self.step = 0
        self.window_size = self.trainer.window_size

        self.epochs = kwargs['epochs']
        self.val_percent = kwargs['val_percent']
        self.use_gae = kwargs['use_gae']
        self.num_steps = kwargs['num_steps']
        self.USE_CUDA = kwargs['USE_CUDA']
        self.writer = kwargs['writer']
        self.savepath = kwargs['savepath']

        self.layers = self.trainer.model.layers
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.accumulated_step = 0
    def save(self):
        torch.save(self.ac, self.savepath)

    def reset(self):
        self.rollouts.reset()
        self.accumulated_step += self.step
        self.step = 0
        self.trainer.reset()

    def run(self):
        for idx in range(self.epochs):
            self.reset()
            self.step_run(idx)
            self.save()
        self.reset()
        self.evaluate()

    def step_run(self, epoch):
        observation, prev_loss, prev_val_loss = self.trainer.observe()
        self.step += self.window_size
        self.rollouts.obs[0].copy_(observation)
        episode_rewards = deque(maxlen=100)
        action = None
        while self.step < self.total_steps:
            self.step += self.window_size
            for step in range(self.num_steps):
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states, distribution = \
                    self.ac.act(self.rollouts.obs[step:step+1] ,self.rollouts.recurrent_hidden_states[step])
                    print(value.shape)
                    print(action.shape, action_log_prob.shape,recurrent_hidden_states.shape)
                    action = action.squeeze(0)
                    action_log_prob = action_log_prob.squeeze(0)
                    value = value.squeeze(0)
                    print(action.shape)
                    #for idx in range(len(action)):
                    #    self.writer.add_scalar("action/channel_{}".format(n_channel)), action[0], self.step + self.accumulated_step)
                    #    self.writer.add_scalar("entropy/channel_{}".format(n_channel)) distribution.distributions[0].entropy(), self.step + self.accumulated_step)
                # ==============
                # | SET ACTION |
                # ==============

                observation, curr_loss, curr_val_loss = self.trainer.observe()
                self.writer.add_scalar("train/loss", curr_loss, self.step + self.accumulated_step)

                reward = (prev_val_loss - curr_val_loss) * self.val_percent + (prev_loss - curr_loss) * (1 - self.val_percent)
                prev_loss = curr_loss
                prev_val_loss = curr_val_loss
                episode_rewards.append(float(reward.cpu().numpy()))
                self.writer.add_scalar("reward", reward, self.step + self.accumulated_step)
                self.rollouts.insert(observation, recurrent_hidden_states, action, action_log_prob, value, reward)

            with torch.no_grad():
                next_value = self.ac.get_value(self.rollouts.obs[-1:], self.rollouts.recurrent_hidden_states[-1]).detach()
            self.rollouts.compute_returns(next_value, self.use_gae, self.gamma, self.gae_lambda)
            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            
            self.writer.add_scalar("value_loss", value_loss, self.step + self.accumulated_step)
            self.writer.add_scalar("action_loss", action_loss, self.step + self.accumulated_step)

            print("action_loss:", action_loss, ", Optimizer Epoch: {}, Optimizee step: {}. ".format(self.accumulated_step, self.step))

            self.rollouts.after_update()

            if self.step >= self.total_steps_epoch:
                acc, loss = self.trainer.val()
                self.writer.add_scalar("val/acc", acc, self.step + self.accumulated_step)
                self.writer.add_scalar("val/loss", loss, self.step + self.accumulated_step)

    def evaluate(self):
        observation, prev_loss, prev_val_loss = self.trainer.observe()
        self.step += self.window_size
        prev_hidden = torch.zeros_like(self.rollouts.recurrent_hidden_states[0])
        while self.step < self.total_steps:
            for step in range(self.num_steps):
                with torch.no_grad():
                    self.step += self.window_size
                    print(prev_hidden.device)
                    value, action, action_log_prob, prev_hidden, distribution = \
                    self.ac.act(observation.unsqueeze(0).cpu(), prev_hidden, deterministic = True)
                    action = action.squeeze(0)
                    action_log_prob = action_log_prob.squeeze(0)
                    value = value.squeeze(0)
                    for idx in range(len(action)):
                        self.writer.add_scalar("evaluate/action/%s"%self.layers[idx], action[idx], self.step)
                        self.writer.add_scalar("evaluate/entropy/%s"%self.layers[idx], distribution.distributions[idx].entropy(), self.step)
                    # ==============
                    # | SET ACTION |
                    # ==============
                observation, curr_loss, curr_val_loss = self.trainer.observe()
                self.writer.add_scalar("evaluate/loss", curr_loss, self.step)
                
            if self.step >= self.total_steps_epoch:
                acc, loss = self.trainer.val()
                self.writer.add_scalar("evaluate/val/acc", acc, self.step)
                self.writer.add_scalar("evaluate/val/loss", loss, self.step)

