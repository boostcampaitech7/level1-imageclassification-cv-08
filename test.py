import argparse
import pandas as pd
from configs.config_manager import ConfigManager
from models.model_selector import ModelSelector
from trainers.test_runner import TestRunner
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Test Configuration")
    parser.add_argument("--model_name", type=str, help='model architecture name', required=True)
    parser.add_argument("--file_path", type=str, help="Path to the model .pt file", required=True)
    
    return parser.parse_args()

def main():
    args = parse_args()

    config_manager = ConfigManager(config_path="config.yaml")
    config = config_manager.get_config()

    config['model']['model_name'] = args.model_name
    model_selector = ModelSelector(config['model']['model_name'],
                                   config['model']['num_classes'],
                                   config['model']['pretrained'],
                                   config['training']['drop_rate'])
    model = model_selector.get_model()

    device = torch.device(config['device'])
    model.to(device)

    test_runner = TestRunner(model, config, device)

    test_runner.load_model(model_path=args.file_path)
    
    test_runner.run_test()

if __name__ == "__main__":
    main()
