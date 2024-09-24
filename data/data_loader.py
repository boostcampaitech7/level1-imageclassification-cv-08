import os
import cv2
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.transforms import AutoAugmentPolicy
import torchvision.transforms as T
from PIL import Image

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
    def __init__(self, size, use_auto=True, auto_policy="CIFAR10"):
        """
        Args:
            size : 이미지 크기
            use_auto : AutoAugment 사용 여부
            auto_policy : AutoAugment 정책 ('IMAGENET', 'CIFAR10', 'SVHN')
        """
        self.use_auto = use_auto
        self.auto_policy = auto_policy
        self.size = size

    def get_transform(self, is_train=True, is_original=False):
        if is_train:
            if is_original:
                # 원본 데이터 변환: 데이터 증강 없이 기본 전처리만 적용
                transform = T.Compose([
                    T.Resize((self.size, self.size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            elif self.use_auto:
                # AutoAugment 적용
                if self.auto_policy == 'IMAGENET':
                    policy = AutoAugmentPolicy.IMAGENET
                elif self.auto_policy == 'CIFAR10':
                    policy = AutoAugmentPolicy.CIFAR10
                elif self.auto_policy == 'SVHN':
                    policy = AutoAugmentPolicy.SVHN

                transform = T.Compose([
                    T.Resize((self.size, self.size)),
                    T.AutoAugment(policy=policy),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
            else:
                # 커스텀 데이터 증강
                transform = T.Compose([
                    T.Resize((self.size, self.size)),
                    # T.Grayscale(),
                    T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                    # T.RandomHorizontalFlip(),
                    T.ColorJitter(brightness=0.2, contrast=0.2),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            # 검증 또는 테스트 시에는 기본 변환만 적용
            transform = T.Compose([
                T.Resize((self.size, self.size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        return transform
    
def create_dataloader(info_df, root_dir, transform, batch_size, shuffle=True, is_inference=False):
    dataset = CustomDataset(root_dir, info_df, transform, is_inference)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def create_combined_dataloader(info_df, root_dir, transform_selector, batch_size, shuffle=True, num_workers=4):
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
    
    original_transform = transform_selector.get_transform(is_train=True, is_original=True)

    augmented_transform = transform_selector.get_transform(is_train=True, is_original=False)

    original_dataset = CustomDataset(root_dir, info_df, transform=original_transform)
    augmented_dataset = CustomDataset(root_dir, info_df, transform=augmented_transform)

    combined_dataset = ConcatDataset([original_dataset, augmented_dataset])

    return DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)