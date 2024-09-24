import os
import time
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 학습 데이터 경로와 정보 파일 경로
traindata_dir = "./data/train"
traindata_info_file = "./data/train.csv"
aag_train_dir = "./data/AAG_train"  # 저장 디렉토리 "AAG_train"

# Augmentation 설정 (흑백 이미지를 위한 AutoAugment)
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("L")),  # 이미지를 흑백(L 모드)로 변환
    transforms.PILToTensor(),  # PIL 이미지를 텐서로 변환 (uint8로 변환)
    AutoAugment(AutoAugmentPolicy.CIFAR10),  # CIFAR10 정책 사용
    transforms.ToPILImage()  # 다시 PIL 이미지로 변환
])

# 저장 경로가 없으면 생성
if not os.path.exists(aag_train_dir):
    os.makedirs(aag_train_dir)

# 학습 데이터 정보 읽기
df = pd.read_csv(traindata_info_file)

# 총 이미지 개수
total_images = len(df)

# 처리 시간 측정을 위해 시작 시간 기록
start_time = time.time()

# 변환된 이미지 개수 카운터
count = 0

# 각 이미지에 대해 AutoAugmentation 적용 후 저장
for index, row in df.iterrows():
    img_name = row['image_path']  # 'image_path'에서 경로 불러오기
    img_class = row['class_name']  # 'class_name' 사용
    
    # 이미지 파일 전체 경로
    img_path = os.path.join(traindata_dir, img_name)

    # 이미지 열기
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue

    img = Image.open(img_path)

    # 변환 수행 (텐서로 변환하여 AutoAugment 적용)
    augmented_img = transform(img)

    # 새롭게 저장할 클래스별 폴더 경로 설정
    class_save_dir = os.path.join(aag_train_dir, img_class)
    
    # 클래스별 폴더가 없으면 생성
    if not os.path.exists(class_save_dir):
        os.makedirs(class_save_dir)

    # 새롭게 저장할 이미지 경로 설정
    new_img_path = os.path.join(class_save_dir, img_name.split('/')[-1])

    # 증강된 이미지 저장 (JPEG 포맷으로 저장)
    augmented_img.save(new_img_path, format='JPEG')

    # 변환된 이미지 개수 증가
    count += 1

    # 진행률 계산 및 출력
    progress = (count / total_images) * 100
    print(f"Processed {count}/{total_images} images ({progress:.2f}% complete)")

# 처리 완료 시간 기록
end_time = time.time()

# 총 걸린 시간 계산 (초 단위)
total_time = end_time - start_time

# 결과 출력
print(f"\nTotal time taken: {total_time:.2f} seconds")
print(f"Total images processed: {count}")
