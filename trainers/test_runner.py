import torch
import os
import pandas as pd
from trainers.inference_runner import inference
from data.data_loader import create_dataloader,create_combined_dataloader
from data.custom_dataset import CustomDataset
from data.transform_selector import TransformSelector


class TestRunner:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device
        self.transform_selector = TransformSelector(size=config['model']['img_size'],
                                                    augment_type=config['augmentation']['augmentation_type'],
                                                    auto_policy=config['augmentation']['policy'])

    def load_model(self, model_path=None):
        # model_path = f"{self.config['result_path']}/{self.config['model']['model_name']}_{self.config['training']['epochs']}_best_model.pt"
        if model_path is None:
            model_path = f"{self.config['result_path']}/best_model/{self.config['model']['model_name']}_{self.config['training']['epochs']}_best_model.pt"
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"모델 로드 완료: {model_path}")
        
    def run_test(self):
        test_info = pd.read_csv(self.config['data']['test_info_file'])
        test_transform = self.transform_selector.get_transform(is_train=False)
        test_dataset = CustomDataset(root_dir=self.config['data']['test_data_dir'],
                                     info_df=test_info,
                                     transform=test_transform,
                                     is_inference=True)
        
        test_loader = create_dataloader(dataset=test_dataset,
                                        batch_size=self.config['training']['batch_size'],
                                        shuffle=False)
        
        predictions = inference(model=self.model,
                                device=self.device,
                                test_loader=test_loader)
        
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})

        output_path = os.path.join(self.config['result_path'], f"{self.config['model']['model_name']}_{self.config['training']['epochs']}_output.csv")
        test_info.to_csv(output_path, index=False)
        
        print('-'*10 + '추론 완료' + '-'*10)
        print(f"{self.config['model']['model_name']}_{self.config['training']['epochs']}_output.csv")

    def load_model_en(self,path):
            self.model.load_state_dict(torch.load(path))
            print(f"모델 로드 완료: {path}")