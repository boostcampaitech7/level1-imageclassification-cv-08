import os
import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import AutoAugmentPolicy
import torchvision.transforms as T
import kornia.augmentation as K
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, info_df, transform, is_inference=False):
        """
        Args:
            root_dir : 이미지 파일들이 저장된 디렉토리 경로
            info_df : 이미지 경로와 라벨 정보를 담고 있는 DataFrame
            transform : 데이터에 적용할 transform 함수
            is_inference : 추론 시 True, 학습 시 False
        """
        self.root_dir = root_dir
        self.image_paths = info_df['image_path'].tolist()
        self.is_inference = is_inference
        if not self.is_inference:
            self.targets = info_df["target"].tolist()

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform 적용 (AutoAugment 또는 다른 변환)
        if isinstance(self.transform, T.Compose):
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            image = self.transform(image=image)['image']

        if self.is_inference:
            return image
        else:
            return image, self.targets[idx]

class TransformSelector:
    def __init__(self, size, auto_policy = "CIFAR10"):
        """
        Args:
            size : 이미지 크기
            use_auto : AutoAugment 사용 여부
            auto_policy : AutoAugment 정책 ('IMAGENET', 'CIFAR10', 'SVHN')
            augmentations: 증강기법들을 담은 list
        """
        self.auto_policy = auto_policy
        self.size = size

    def get_transform(self, aug):
        # 초기 전처리 설정
        basic_transforms = T.Compose([T.Resize((self.size, self.size)), T.ToTensor(),
                            T.ConvertImageDtype(torch.float),
                            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        # 증강 기법에 따라 다른 transform 설정
        if aug == 'auto':
            # AutoAugment 적용
            if self.auto_policy == 'IMAGENET':
                policy = AutoAugmentPolicy.IMAGENET
            elif self.auto_policy == 'CIFAR10':
                policy = AutoAugmentPolicy.CIFAR10
            elif self.auto_policy == 'SVHN':
                policy = AutoAugmentPolicy.SVHN

            return T.Compose([
                T.Resize((self.size, self.size)),
                T.AutoAugment(policy=policy),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        elif aug == 'cutout':
            # Cutout을 적용할 때 Numpy -> Tensor 변환 포함
            return A.Compose([
                A.Resize(self.size, self.size),
                # A.Rotate(limit=45, p=0.3),
                A.CoarseDropout(holes=8, p=1.0),
                ToTensorV2()  # Numpy를 PyTorch Tensor로 변환
            ])
        elif aug == 'cutmix':
            # CutMix를 적용할 때 Tensor 형식으로 직접 변환 포함
            return T.Compose([
                T.Resize((self.size, self.size)),
                # T.RandomRotation(45),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                lambda x: x.unsqueeze(0),  # 배치 차원 추가
                K.AugmentationSequential(
                    K.RandomCutMixV2(p=1.0, data_keys=["input", "class"])
                ),
                lambda x: x.squeeze(0)  # 배치 차원 제거
            ])
        elif aug == 'mixup':
            # MixUp을 적용할 때 Tensor 형식으로 직접 변환 포함
            return T.Compose([
                T.Resize((self.size, self.size)),
                # T.RandomRotation(45),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                lambda x: x.unsqueeze(0),  # 배치 차원 추가
                K.AugmentationSequential(
                    K.RandomMixUpV2(p=1.0, data_keys=["input", "class"])
                ),
                lambda x: x.squeeze(0)  # 배치 차원 제거
            ])
        elif aug == 'translate':
            return T.Compose([
                T.Resize((self.size, self.size)),
                T.RandomAffine(degrees=0,translate=(0.4,0.4)),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug == 'shear':
            return T.Compose([
                T.Resize((self.size, self.size)),
                T.RandomAffine(degrees=0,shear=(-30,30)),
                T.ToTensor(),
                T.ConvertImageDtype(torch.float),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif aug == 'original':
            # 원본 데이터 또는 기타 기본 전처리
            return basic_transforms
        
def custom_collate_fn(batch):
    images, targets = zip(*batch)
    images = [img.float() for img in images]  # 모든 이미지를 float 타입으로 변환
    return torch.stack(images), torch.tensor(targets)

def create_dataloader(info_df, root_dir, transform, batch_size, shuffle=True, is_inference=False):
    dataset = CustomDataset(root_dir, info_df, transform, is_inference)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def create_combined_dataloader(info_df, root_dir, transforms, batch_size, shuffle=True, num_workers=4, is_inference=False):
    """
    Args:
        info_df : 이미지 경로와 타겟 정보를 담고 있는 DataFrame
        root_dir : 이미지 파일들이 저장된 디렉토리 경로
        batch_size : 미니배치 크기
        transform_selector : 원본 및 증강 데이터를 위한 transform 설정 객체
        shuffle : 데이터 순서를 섞을지 여부
        num_workers : 데이터 로딩에 사용할 쓰레드 개수

    Returns:
        DataLoader: 결합된 DataLoader 객체
    """
    
    datasets = []

    # 원본 데이터셋 추가 (기본 transform 적용)
    original_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    original_dataset = CustomDataset(root_dir, info_df, transform=original_transform, is_inference=is_inference)
    datasets.append(original_dataset)
    print(f"Added original dataset with {len(original_dataset)} images.")

    # 증강 기법별 데이터셋 생성 및 추가
    for transform in transforms:
        augmented_dataset = CustomDataset(root_dir, info_df, transform=transform, is_inference=is_inference)
        datasets.append(augmented_dataset)
        print(f"Applied augmentation and added dataset with {len(augmented_dataset)} images.")

    # 데이터셋 결합
    combined_dataset = ConcatDataset(datasets)

    print(f'Total {len(combined_dataset)} images from {len(datasets)} datasets.')

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers, collate_fn=custom_collate_fn)