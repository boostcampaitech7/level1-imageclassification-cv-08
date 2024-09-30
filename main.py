from configs.config_manager import ConfigManager
from trainers.train_runner import Trainer
from trainers.test_runner import TestRunner
from models.model_selector import ModelSelector
from data.data_loader import create_dataloader
from data.transform_selector import TransformSelector
from data.custom_dataset import CustomDataset
from optimizers.optimizer import OptimizerSelector
from schedulers.scheduler import SchedulerSelector
import torch
import torch.nn as nn
import pandas as pd
import time
from utils.utils import measure_time
from sklearn.model_selection import train_test_split as ttsplit

if __name__ == "__main__":
    config_manager = ConfigManager(config_path="config.yaml")
    config = config_manager.get_config()

    model_selector = ModelSelector(config['model']['model_name'],
                                   config['model']['num_classes'],
                                   config['model']['pretrained'],
                                   config['training']['drop_rate'])
    model = model_selector.get_model()
    device = torch.device(config['device'])
    
    train_info = pd.read_csv(config['data']['train_info_file'])
    train_df, val_df = ttsplit(train_info, test_size=0.2, random_state=42)

    transform_selector = TransformSelector(size=config['model']['img_size'],
                                           augment_type=config['augmentation']['augmentation_type'])

    train_dataset = CustomDataset(root_dir=config['data']['train_data_dir'], info_df=train_df, transform=transform_selector.get_transform(is_train=True))
    val_dataset = CustomDataset(root_dir=config['data']['train_data_dir'], info_df=val_df, transform=transform_selector.get_transform(is_train=False))

    train_loader = create_dataloader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    optimizer_selector = OptimizerSelector(model=model, config=config)
    optimizer = optimizer_selector.get_optimizer()

    scheduler_selector = SchedulerSelector(optimizer=optimizer, config=config)
    scheduler = scheduler_selector.get_scheduler()

    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model=model,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      loss_fn=loss_fn,
                      config=config)
    
    print("Train 시작")
    start_time = time.time()
    trainer.train(use_cutmix=config['training'].get('use_cutmix', False))
    end_time = time.time()

    print(measure_time(start_time, end_time))

    print("Test 시작")
    test_runner = TestRunner(model, config, device=device)
    test_runner.load_model()
    test_runner.run_test()

    print("모든 작업 완료!")
