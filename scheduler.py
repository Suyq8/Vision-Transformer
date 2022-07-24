import math
from torch.optim.lr_scheduler import LambdaLR


class CosineLRScheduler(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1, min_lr=1e-10):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.min_lr = min_lr
        super(CosineLRScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return step/self.warmup_steps

        return max(self.min_lr, 0.5 * (1+math.cos(math.pi*(step-self.warmup_steps)/(self.t_total-self.warmup_steps))))
