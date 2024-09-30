import torch.optim.lr_scheduler as lr_scheduler
from .customCosineWR import CosineAnnealingWarmupRestarts

class SchedulerSelector:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config['scheduler']

    def get_scheduler(self):
        scheduler_dict = {
            'Step': lambda: lr_scheduler.StepLR(self.optimizer,
                                                step_size=self.config['scheduler_step_size'],
                                                gamma=self.config['scheduler_gamma']),
            'Reduce': lambda: lr_scheduler.ReduceLROnPlateau(self.optimizer, 
                                                             mode='min', 
                                                             patience=self.config['scheduler_patience'],
                                                             factor=self.config['scheduler_gamma']),
            'Cosine': lambda: lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                             T_max=self.config['T_max']),
            'Cosine_Warm_Restarts': lambda: lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 
                                                                                     T_0=self.config['T_0'], 
                                                                                     T_mult=self.config['T_mult']),
            'Custom_Cosine_Warm_Restarts': lambda: CosineAnnealingWarmupRestarts(self.optimizer,
                                                                                first_cycle_steps=self.config['first_cycle_steps'],
                                                                                cycle_mult=self.config['cycle_mult'],
                                                                                max_lr=self.config['max_lr'],
                                                                                min_lr=self.config['min_lr'],
                                                                                warmup_steps=self.config['warmup_steps'],
                                                                                gamma=self.config['gamma']
                                                                            ) 
                                                                        }   

        sched_name = self.config['what_scheduler']
        if sched_name not in scheduler_dict:
            raise ValueError(f"지원되지 않는 스케줄러: {sched_name}")

        return scheduler_dict[sched_name]()
