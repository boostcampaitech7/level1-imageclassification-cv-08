import torch
import torch.nn as nn
from trainers.loss import FocalLoss

class DataConfig:
    def __init__(self):
        self.train_data_dir = "./data/train" 
        self.train_info_file = "./data/train.csv"

        self.test_data_dir = "./data/test"
        self.test_info_file = "./data/test.csv"

class ModelConfig:
    def __init__(self):
        self.model_name = 'resnet152'
        self.pretrained = True
        self.num_classes = 500
        self.img_size = 224

class TrainingConfig:
    def __init__(self):
        self.epochs = 10
        self.batch_size = 256
        self.lr = 0.0001
        self.drop_rate = 0.4
        self.early_stop_partience = 3
        self.loss_fn = nn.CrossEntropyLoss()

class AugmentationConfig:
    def __init__(self):
        self.auto_aug_use = True
        self.auto_policy = "IMAGENET"
        # self.augmentations = ['cutmix','cutout','mixup', 'shear', 'translate']
        self.augmentations = ['cutmix']

class OptimizerConfig:
    def __init__(self):
        '''
        adam
        SGD
        adadelta
        adagrad
        adamw
        sparseadam
        adamax
        asgd
        lbfgs
        nadam
        radam
        rmsprop
        rprop
        '''
        self.opt = 'adamw'
        self.momentum = 0.85

class SchedulerConfig:
    def __init__(self):
        # 공통 스케줄러 설정
        '''
        Step
        Reduce
        Cosine
        Multistep
        Lambda
        Exponential
        Cyclic
        OneCycle
        Cosine_Warm_Restarts
        '''
        self.what_scheduler = 'Reduce'  # 사용할 스케줄러 유형
        self.scheduler_step_size = 2  # StepLR과 MultiStepLR에서 사용
        self.scheduler_gamma = 0.5  # StepLR, MultiStepLR, ExponentialLR 등에서 사용
        self.scheduler_patience = 3  # ReduceLROnPlateau에서 사용

        # CosineAnnealingLR 관련 설정
        self.T_max = 10  # CosineAnnealingLR에서 사용

        # MultiStepLR 관련 설정
        self.milestones = [30, 80]  # MultiStepLR에서 사용하는 에포크 마일스톤

        # LambdaLR 관련 설정
        self.lr_lambda = lambda epoch: 0.95 ** epoch  # LambdaLR에서 사용 (사용자 정의 함수)

        # CyclicLR 관련 설정
        self.base_lr = 0.0001  # CyclicLR에서 사용
        self.max_lr = 0.001  # CyclicLR에서 사용

        # OneCycleLR 관련 설정
        self.total_steps = 1000  # OneCycleLR에서 사용할 총 스텝 수

        # CosineAnnealingWarmRestarts 관련 설정
        self.T_0 = 10  # CosineAnnealingWarmRestarts에서 첫 번째 주기의 에포크 수
        self.T_mult = 2  # CosineAnnealingWarmRestarts에서 다음 주기로 갈 때 증가시키는 배수

class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.result_path = "./work_lee/train_result"

        # 각 하위 설정 클래스들을 포함
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.augmentation = AugmentationConfig()
        self.optimizer = OptimizerConfig()
        self.scheduler = SchedulerConfig()