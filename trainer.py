import torch
from utils import accuracy, AverageMeter
from tqdm import tqdm

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.models.policy import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from collections import deque
from collections import defaultdict
import numpy as np


class Trainer(object):

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
        return self.epochs * self.total_steps,self.total_steps_epoch

    def observe(self):
        losses = []
        for idx in range(self.window_size):
            train_loss, train_acc = self.train_step()
            losses.append(train_loss.detach())

        losses = [sum(losses) / len(losses)]
        return torch.tensor(losses)

    def get_steps(self):
        return self.total_steps, self.total_steps_epoch
    def reset(self):
        self.step = 0
        self.model.reset()
        self.optimizer.state = defaultdict(dict)
        
    def train_step(self): 
        self.model.train()
        self.optimizer.zero_grad()
        input, label = self.get_train_samples()
        self.step += 1

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
                losses.append(loss.detach().item())
                accs.append(acc)
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

class Runner(object):
    def __init__(self, trainer, **kwargs):
        self.trainer = trainer
        
        self.total_steps, self.total_steps_epoch = trainer.get_steps()
        self.step = 0

        self.USE_CUDA = kwargs['USE_CUDA']
        self.writer = kwargs['writer']
        self.epochs = kwargs['epochs']

    def reset(self):
        self.trainer.reset()
        self.accumulated_step += self.step
        self.step = 0
        
    def run(self):
        for idx in range(self.epochs):
            self.reset()
            self.step_run(idx)

    def step_run(self, epoch):
        prev_loss = self.trainer.observe()
        self.step += self.window_size
        while self.step < self.total_steps:
            self.step += self.window_size
            curr_loss = self.trainer.observe()
            self.writer.add_scalar("train/loss", curr_loss, self.step + self.accumulated_step)
            if self.step >= self.total_steps_epoch:
                acc, loss = self.trainer.val()
                self.writer.add_scalar("val/acc", acc, self.step + self.accumulated_step)
                self.writer.add_scalar("val/loss", loss, self.step + self.accumulated_step)