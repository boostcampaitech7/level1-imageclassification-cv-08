import argparse
from trainers.train_runner import Trainer
from data.data_loader import create_dataloader
from data.transform_selector import TransformSelector
from configs.config_manager import ConfigManager, parse_args
from models.model_selector import ModelSelector
from data.custom_dataset import CustomDataset
from optimizers.optimizer import OptimizerSelector
from schedulers.scheduler import SchedulerSelector
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split as ttsplit

def get_train_parser():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--config_path', type=str, help="Path to the config file", default="config.yaml")
    parser.add_argument('--split_ratio',type=float, help="train/validation split ratio",default=0.2)
    parser.add_argument('--use_cutmix',action='store_true', help='use cutmix training data')
    parser.add_argument('--epochs',type=int, help='Number of epochs for training', default=5)
    parser.add_argument('--lr',type=float, help='learning Rate')
    parser.add_argument('--batch_size',type=int, help='batch size for training and validation set')
    parser.add_argument('--img_size', type=int, help='Resize input images to this size', default=224)
    parser.add_argument('--model_name',type=str, help='name of the model to use', default='resnet50')
    return parser

def main():
    parser = get_train_parser()
    args = parser.parse_args()

    config_manager = ConfigManager(config_path=args.config_path)
    config_manager.update_from_args(args)
    config = config_manager.get_config()

    train_info = pd.read_csv(config['data']['train_info_file'])
    train_df, val_df = ttsplit(train_info, test_size=0.2, random_state=42)

    transform_selector = TransformSelector(size=config['model']['img_size'], augment_type=config['augmentation']['augmentation_type'])

    train_dataset = CustomDataset(root_dir=config['data']['train_data_dir'], info_df=train_df, transform=transform_selector.get_transform(is_train=True))
    val_dataset = CustomDataset(root_dir=config['data']['train_data_dir'], info_df=val_df, transform=transform_selector.get_transform(is_train=False))
    train_loader = create_dataloader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    model_selector = ModelSelector(config['model']['model_name'],
                                   config['model']['num_classes'],
                                   config['model']['pretrained'],
                                   config['training']['drop_rate'])
    model = model_selector.get_model()

    optimizer_selector = OptimizerSelector(model=model, config=config)
    optimizer = optimizer_selector.get_optimizer()

    scheduler_selector = SchedulerSelector(optimizer=optimizer, config=config)
    scheduler = scheduler_selector.get_scheduler()

    loss_fn = nn.CrossEntropyLoss()

    trainer = Trainer(model, config['device'], train_loader, val_loader, optimizer, scheduler, loss_fn, config)
    trainer.train(use_cutmix=config['training'].get('use_cutmix', False))

if __name__ == "__main__":
    main()
