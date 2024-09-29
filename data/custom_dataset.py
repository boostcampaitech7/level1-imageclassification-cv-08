import os
from PIL import Image
from torch.utils.data import Dataset
import cv2

class CustomDataset(Dataset):
    """사용자 정의 데이터셋 클래스"""
    
    def __init__(self, root_dir, info_df, transform=None, is_inference=False):
        """
        Args:
            root_dir (str): 이미지 파일이 저장된 디렉토리 경로
            info_df (DataFrame): 이미지 경로 및 라벨 정보를 포함하는 DataFrame
            transform (callable, optional): 데이터에 적용할 변환 함수
            is_inference (bool): 추론 시 True, 학습 시 False
        """
        self.root_dir = root_dir
        self.image_paths = info_df['image_path'].tolist()
        self.transform = transform
        self.is_inference = is_inference
        self.targets = None if is_inference else info_df['target'].tolist()

    def __len__(self):
        """데이터셋의 총 이미지 개수를 반환"""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """주어진 인덱스에 해당하는 데이터(이미지 및 라벨)를 반환"""
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"이미지를 불러올 수 없습니다: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        if self.is_inference:
            return image
        else:
            return image, self.targets[idx]