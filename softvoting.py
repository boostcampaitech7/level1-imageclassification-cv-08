import torch
import argparse
import torch.nn.functional as F
import os
from configs.config_manager import ConfigManager
from trainers.test_runner import TestRunner
import pandas as pd
from data.custom_dataset import CustomDataset
from data.data_loader import create_dataloader
from trainers.inference_runner import inference_preds
from data.transform_selector import TransformSelector
from models.model_selector import ModelSelector
from data.custom_dataset import CustomDataset
import pandas as pd

# 모델 파일 경로 리스트

def softvoting(model_names, model_paths, config):
    # 테스트 데이터 로드 및 준비

    test_info = pd.read_csv(config['data']['test_info_file'])


    transform = TransformSelector(size=config['model']['img_size'],
                                    augment_type=config['augmentation']['augmentation_type'])
    
    test_dataset = CustomDataset(root_dir=config['data']['test_data_dir'], info_df=test_info, transform=transform.get_transform(is_train=False),is_inference=True)
    
    test_loader = create_dataloader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # 예측 결과를 저장할 리스트 초기화
    total_predictions = None

    # 각 모델에 대해 예측 실행
    for i, model in enumerate(model_names):
        # 모델 파일 경로 생성
        print(f'{i+1}/{len(model_names)} starts')
        model_path = model_paths[i]
        # 모델 로드 및 설정
        test_runner = TestRunner(model,config,config['device'])
        model_selector = ModelSelector(model, 
                                   config['model']['num_classes'],
                                   config['model']['pretrained'],
                                   config['training']['drop_rate']
                                   )

        test_runner.model = model_selector.get_model().to(config['device'])
        test_runner.load_model_en(model_path)
        # 예측 수행
        predictions = inference_preds(model=test_runner.model, device=config['device'], test_loader=test_loader)
        
        # 예측 결과를 누적하여 소프트 보팅 계산
        if total_predictions is None:
            total_predictions = predictions
        else:
            total_predictions += predictions

    # 평균을 내어 소프트 보팅 결과 도출
    final_predictions = total_predictions / len(model_names)

    # 최종 예측 결과를 저장
    test_info['target'] = final_predictions.argmax(axis=1)
    test_info = test_info.reset_index().rename(columns={"index": "ID"})

    output_path = os.path.join(config['result_path'], f"ensemble_{len(model_names)}_output.csv")
    test_info.to_csv(output_path, index=False)

    print('-' * 10 + '소프트 보팅 완료' + '-' * 10)
    print(f"결과 저장 위치: {output_path}")


# 사용 예시

def parse_args():
    parser = argparse.ArgumentParser(description="Test Configuration")
    parser.add_argument("--file_path", type=str, help="Path to the model .pt file", required=True)
    
    return parser.parse_args()

if __name__ == "__main__":
    # model_dir = 'work_lee/train_result'  # 모델 파일이 저장된 디렉토리 경로를 설정합니다.
    args = parse_args()
    config_manager = ConfigManager(config_path="config.yaml")
    config = config_manager.get_config()


    # .pt로 끝나는 모든 모델 파일 경로를 가져옵니다.
    model_paths = [os.path.join(args.file_path, f) for f in os.listdir(args.file_path) if f.endswith('.pt')]

    # 각 모델을 저장할 리스트
    model_names = []

    # 모델 파일을 읽어와 모델 리스트에 추가
    for path in model_paths:
        file_name = os.path.basename(path)  # 파일 이름만 추출
        # print(file_name)
        model_name = '_'.join(file_name.split('_')[:-3])  # 'regnety_080_50_best_model.pt'에서 'regnety_080'만 추출
        model_names.append(model_name)

    print("Model Names:", model_names)

    softvoting(model_names,model_paths,config)