import yaml
import argparse

class ConfigManager:
    def __init__(self, config_path=None):
        self.config = {}
        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def update_from_args(self, args):
        arg_map = {
            'epochs': 'training.epochs',
            'batch_size': 'training.batch_size',
            'lr': 'training.lr',
            'device': 'device',
            'train_data_dir': 'data.train_data_dir',
            'test_data_dir': 'data.test_data_dir',
            'model_name': 'model.model_name'
        }
        for key, value in vars(args).items():
            if value is not None and key in arg_map:
                self._update_nested_config(arg_map[key], value)

    def _update_nested_config(self, key, value):
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value

    def get_config(self):
        return self.config

def parse_args():
    parser = argparse.ArgumentParser(description="Training Configuration")
    
    parser.add_argument("--config_path", type=str, help="Path to the config YAML file", default="config.yaml")
    parser.add_argument("--device", type=str, help="Device to use (cuda or cpu)", default="cuda")
    parser.add_argument("--train_data_dir", type=str, help="Training data directory")
    parser.add_argument("--test_data_dir", type=str, help="Test data directory")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--epochs", type=int, help="Number of training epochs", default=5)
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=64)
    parser.add_argument("--lr", type=float, help="Learning rate", default=0.005)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    config_manager = ConfigManager(config_path=args.config_path)
    
    config_manager.update_from_args(args)
    
    final_config = config_manager.get_config()
    print(final_config)
