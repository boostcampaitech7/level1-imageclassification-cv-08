import torch
import os
import pandas as pd
from trainers.inference import inference
from data.data_loader import create_dataloader, TransformSelector

class TestRunner:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.transform_selector = TransformSelector(size=config.model.img_size,
                                                    use_auto=config.auto_augmentation.auto_aug_use,
                                                    auto_policy=config.auto_augmentation.policy)

    def load_model(self):
        model_path = f"{self.config.result_path}/{self.config.model.model_name}_{self.config.training.epochs}_best_model.pt"
        self.model.load_state_dict(torch.load(model_path))
        print(f"모델 로드 완료: {model_path}")
        
    def run_test(self):

        # 테스트 데이터 로드
        test_info = pd.read_csv(self.config.data.test_info_file)
        test_transform = self.transform_selector.get_transform(is_train=False)
        test_loader = create_dataloader(test_info,
                                        self.config.data.test_data_dir,
                                        test_transform,
                                        self.config.training.batch_size,
                                        shuffle=False,
                                        is_inference=True)
        
        # 추론 실행
        predictions = inference(model=self.model,
                                device=self.config.device,
                                test_loader=test_loader)
        
        test_info['target'] = predictions
        test_info = test_info.reset_index().rename(columns={"index": "ID"})

        # 결과 저장
        output_path = os.path.join(self.config.result_path, f"{self.config.model.model_name}_{self.config.training.epochs}_output.csv")
        test_info.to_csv(output_path, index=False)
        
        print('-'*10 + '추론 완료' + '-'*10)
        print(f"{self.config.model.model_name}_{self.config.training.epochs}_output.csv")
