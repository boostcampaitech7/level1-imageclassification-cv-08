import torch
from configs.config import Config

class SchedulerSelector:
    def __init__(self, optimizer):
        self.config = Config().scheduler
        self.optimizer = optimizer
        self.scheduler = self._get_scheduler()

    def _get_scheduler(self):
        """스케줄러 선택 로직"""
        if self.config.what_scheduler == 'Step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                   step_size=self.config.scheduler_step_size, 
                                                   gamma=self.config.scheduler_gamma)
        elif self.config.what_scheduler == 'Reduce':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                              mode='min', 
                                                              factor=0.1, 
                                                              patience=self.config.scheduler_patience)
        elif self.config.what_scheduler == 'Cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                              T_max=self.config.T_max)
        elif self.config.what_scheduler == 'Multistep':
            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, 
                                                        milestones=self.config.milestones, 
                                                        gamma=self.config.scheduler_gamma)
        elif self.config.what_scheduler == 'Lambda':
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.config.lr_lambda)
        elif self.config.what_scheduler == 'Exponential':
            return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.scheduler_gamma)
        elif self.config.what_scheduler == 'Cyclic':
            return torch.optim.lr_scheduler.CyclicLR(self.optimizer, 
                                                     base_lr=self.config.base_lr, 
                                                     max_lr=self.config.max_lr)
        elif self.config.what_scheduler == 'Onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                       max_lr=self.config.max_lr, 
                                                       total_steps=self.config.total_steps)
        elif self.config.what_scheduler == 'Cosine_Warm_Restarts':
            return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                                                        T_0=self.config.T_0, 
                                                                        T_mult=self.config.T_mult)
        else:
            raise ValueError(f"알 수 없는 스케줄러: {self.config.What_scheduler}")
    
    def get_scheduler(self):
        """외부에서 스케줄러 가져오기"""
        return self.scheduler