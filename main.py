from configs.config import Config
from data.data_loader import create_dataloader, TransformSelector, create_combined_dataloader
from models.model_selector import ModelSelector
from trainers.trainer import Trainer
import pandas as pd
from sklearn.model_selection import train_test_split as ttsplit
from utils.utils import measure_time
from time import time
from optimizers.optimizer import OptimizerSelector
from schedulers.scheduler import SchedulerSelector
from trainers.test_runner import TestRunner

if __name__ == "__main__":
    config = Config()

    # 데이터 로드 및 변환 설정
    train_info = pd.read_csv(config.data.train_info_file)

    train_df, val_df = ttsplit(train_info, test_size=0.2,
                               stratify=train_info['target'], 
                               random_state=42)
    
    train_transform_list = []
    valid_transform_list = []
    
    transform_selector = TransformSelector(config.model.img_size,
                                           config.augmentation.auto_policy)
    
    # auto augmentation 적용
    if config.augmentation.auto_aug_use:
        transform = transform_selector.get_transform('auto')
        train_transform_list.append(transform)

    # 반복문을 통해 config.augmentation.augmentations에서 증강 기법을 받아 transform 생성
    for aug in config.augmentation.augmentations:
        transform = transform_selector.get_transform(aug)
        valid_transform= transform_selector.get_transform(aug)
        
        # Transform을 리스트에 추가
        train_transform_list.append(transform)
        valid_transform_list.append(valid_transform)
    
    # val_transform = transform_selector.get_transform('original')

    train_loader = create_combined_dataloader(train_df, 
                                     config.data.train_data_dir, 
                                     train_transform_list, 
                                     config.training.batch_size, 
                                     shuffle=True)
    
    # val_loader = create_dataloader(val_df, 
    #                            config.data.train_data_dir, 
    #                            val_transform, 
    #                            config.training.batch_size, 
    #                            shuffle=False)

    val_loader = create_combined_dataloader(val_df, 
                                     config.data.train_data_dir, 
                                     valid_transform_list, 
                                     config.training.batch_size, 
                                     shuffle=False)
    
    # 모델 설정
    model_selector = ModelSelector(config.model.model_name, 
                                   config.model.num_classes, 
                                   pretrained=config.model.pretrained,
                                   drop_rate=config.training.drop_rate)
    
    model = model_selector.get_model().to(config.device)

    # 옵티마이저와 스케줄러 설정
    opt_selector = OptimizerSelector(model, 
                                     opt_name=config.optimizer.opt, 
                                     lr=config.training.lr)
    optimizer = opt_selector.get_optimizer()

    scheduler_selector = SchedulerSelector(optimizer = optimizer)
    scheduler = scheduler_selector.get_scheduler()

    # 학습과 추론 로직 연결
    trainer = Trainer(model=model, 
                      device=config.device,
                      train_loader=train_loader, 
                      val_loader=val_loader, 
                      optimizer=optimizer, 
                      scheduler=scheduler, 
                      loss_fn=config.training.loss_fn,
                      epochs=config.training.epochs,
                      result_path=config.result_path,
                      patience=config.training.early_stop_partience
                      )
    
    print('-'*10 + '학습 시작' + '-'*10)
    print()

    start_time = time()

    trainer.train()

    end_time = time()

    train_time = measure_time(start_time, end_time)

    print()
    print("학습 시간: " + train_time)