import torch.optim as optim

class OptimizerSelector:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.lr = config['training'].get('lr', 0.001)
        self.momentum = config['optimizer'].get('momentum', 0.9)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.betas = config.get('betas', (0.9, 0.999))

    def get_optimizer(self):
        """설정에 맞는 옵티마이저를 반환"""
        optimizer_dict = {
            'adam': lambda: optim.Adam(self.model.parameters(), lr=self.lr),
            'SGD': lambda: optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay),
            'adamw': lambda: optim.AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay),
        }

        opt_name = self.config['optimizer'].get('opt', 'adamw')
        if opt_name not in optimizer_dict:
            raise ValueError(f"지원되지 않는 옵티마이저: {opt_name}")

        return optimizer_dict[opt_name]()
