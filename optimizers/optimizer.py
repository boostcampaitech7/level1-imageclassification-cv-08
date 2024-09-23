import torch.optim

class OptimizerSelector:
    def __init__(self, model, opt_name, lr, momentum=0.9, weight_decay=1e-5, alpha=0.99, rho=0.9, eps=1e-08, lambd=0.001):
        """
        model: 모델 객체
        opt_name: 사용할 옵티마이저 이름
        lr: 학습률 (Learning rate)
        momentum: 모멘텀 (옵티마이저에 따라 필요)
        weight_decay: 가중치 감쇠 (AdamW 등에서 필요) (1e-5 ~ 1e-4)
        alpha, rho, eps, lambd: 특정 옵티마이저에 필요한 추가 하이퍼파라미터
        """
        self.model = model
        self.opt_name = opt_name
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.rho = rho
        self.eps = eps
        self.lambd = lambd
        self.optimizer = self._select_optimizer()

    def _select_optimizer(self):
        if self.opt_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.opt_name == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.opt_name == 'adadelta':
            return torch.optim.Adadelta(self.model.parameters(), lr=self.lr, rho=self.rho, eps=self.eps)
        elif self.opt_name == 'adagrad':
            return torch.optim.Adagrad(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        elif self.opt_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.opt_name == 'sparseadam':
            return torch.optim.SparseAdam(self.model.parameters(), lr=self.lr, eps=self.eps)
        elif self.opt_name == 'adamax':
            return torch.optim.Adamax(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        elif self.opt_name == 'asgd':
            return torch.optim.ASGD(self.model.parameters(), lr=self.lr, lambd=self.lambd, alpha=self.alpha)
        elif self.opt_name == 'lbfgs':
            return torch.optim.LBFGS(self.model.parameters(), lr=self.lr)
        elif self.opt_name == 'nadam':
            return torch.optim.NAdam(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        elif self.opt_name == 'radam':
            return torch.optim.RAdam(self.model.parameters(), lr=self.lr, eps=self.eps, weight_decay=self.weight_decay)
        elif self.opt_name == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.lr, momentum=self.momentum, alpha=self.alpha)
        elif self.opt_name == 'rprop':
            return torch.optim.Rprop(self.model.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Optimizer '{self.opt_name}' is not supported.")

    def get_optimizer(self):
        return self.optimizer