"""
This module provides learning rate scheduler implementations for gradient descent populations.
"""

import math
import torch

class LRScheduler:
    """Base class for all learning rate schedulers."""
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.step()
    
    def get_lr(self):
        """Calculate the learning rate at the current step."""
        raise NotImplementedError
    
    def step(self, epoch=None):
        """Update the learning rate."""
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = epoch
        
        values = self.get_lr()
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            param_group['lr'] = lr

class StepLR(LRScheduler):
    """Step learning rate scheduler that decays the learning rate by gamma every step_size epochs."""
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0 or self.last_epoch % self.step_size != 0:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma for group in self.optimizer.param_groups]

class ExponentialLR(LRScheduler):
    """Exponential learning rate scheduler that decays the learning rate by gamma every epoch."""
    def __init__(self, optimizer, gamma, last_epoch=-1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]

class CosineAnnealingLR(LRScheduler):
    """Cosine annealing learning rate scheduler."""
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lrs
        return [self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                for base_lr in self.base_lrs]

class CyclicLR(LRScheduler):
    """Cyclic learning rate scheduler."""
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, mode='triangular', last_epoch=-1):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.mode = mode
        self.cycle = 0
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        cycle_size = 2 * self.step_size_up
        x = abs(self.last_epoch % cycle_size - self.step_size_up) / self.step_size_up
        
        if self.mode == 'triangular':
            scale_factor = 1.0
        elif self.mode == 'triangular2':
            scale_factor = 1.0 / (2.0 ** (self.cycle // 2))
        elif self.mode == 'exp_range':
            scale_factor = 0.99 ** self.last_epoch
        
        return [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * scale_factor
                for _ in self.base_lrs]

class WarmupLR(LRScheduler):
    """Learning rate scheduler with warmup."""
    def __init__(self, optimizer, warmup_epochs, after_scheduler, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.after_scheduler = after_scheduler
        self.finished_warmup = False
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            if not self.finished_warmup:
                self.after_scheduler.last_epoch = 0
                self.finished_warmup = True
            self.after_scheduler.step(self.last_epoch - self.warmup_epochs)
            return [group['lr'] for group in self.optimizer.param_groups]

class OneCycleLR(LRScheduler):
    """One-cycle learning rate scheduler."""
    def __init__(self, optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0, 
                 final_div_factor=10000.0, last_epoch=-1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        self.step_size_up = int(total_steps * pct_start)
        self.step_size_down = total_steps - self.step_size_up
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        # Calculate initial and final learning rates
        initial_lr = self.max_lr / self.div_factor
        final_lr = self.max_lr / self.final_div_factor
        
        if self.last_epoch <= self.step_size_up:
            # Increasing phase
            return [initial_lr + (self.max_lr - initial_lr) * 
                   (self.last_epoch / self.step_size_up) for _ in self.base_lrs]
        else:
            # Decreasing phase
            down_step = self.last_epoch - self.step_size_up
            pct_of_down = down_step / self.step_size_down
            return [self.max_lr - (self.max_lr - final_lr) * pct_of_down 
                   for _ in self.base_lrs]

class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving."""
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, threshold=1e-4, 
                 threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8, verbose=False):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        self.best = float('inf') if mode == 'min' else -float('inf')
        self.cooldown_counter = 0
        self.wait = 0
        self.num_bad_epochs = 0
    
    def step(self, metrics):
        current = metrics
        
        # Convert to scalar if tensor
        if isinstance(current, torch.Tensor):
            current = current.item()
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self.mode == 'min' and current < self.best - self.threshold:
            self.best = current
            self.num_bad_epochs = 0
        elif self.mode == 'max' and current > self.best + self.threshold:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print(f'Reducing learning rate of group {i} to {new_lr:.4e}.')

def get_scheduler(name, optimizer, **kwargs):
    """Factory function for creating schedulers."""
    if name == 'step':
        return StepLR(optimizer, **kwargs)
    elif name == 'exponential':
        return ExponentialLR(optimizer, **kwargs)
    elif name == 'cosine':
        return CosineAnnealingLR(optimizer, **kwargs)
    elif name == 'cyclic':
        return CyclicLR(optimizer, **kwargs)
    elif name == 'warmup':
        after_scheduler = kwargs.pop('after_scheduler')
        return WarmupLR(optimizer, after_scheduler=after_scheduler, **kwargs)
    elif name == 'one_cycle':
        return OneCycleLR(optimizer, **kwargs)
    elif name == 'reduce_on_plateau':
        return ReduceLROnPlateau(optimizer, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler: {name}")