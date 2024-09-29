import torchvision.transforms as T
from torchvision.transforms import AutoAugmentPolicy

class TransformSelector:
    def __init__(self, size, augment_type="custom", auto_policy="CIFAR10", custom_transform=None):
        """
        Args:
            size (int): 변환할 이미지 크기
            augment_type (str): 적용할 증강 방식 ("basic", "auto", "custom")
            auto_policy (str): AutoAugment에 사용할 정책 ("CIFAR10", "IMAGENET", "SVHN")
            custom_transform (list): 커스텀 증강 리스트
        """
        self.size = size
        self.augment_type = augment_type
        self.auto_policy = auto_policy
        self.custom_transform = custom_transform

    def get_basic_transform(self):
        """기본 Resize 및 Normalize 전처리"""
        return T.Compose([
            T.Resize((self.size, self.size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_autoaugment_transform(self):
        """AutoAugment 변환을 반환"""
        policy_map = {
            'IMAGENET': AutoAugmentPolicy.IMAGENET,
            'CIFAR10': AutoAugmentPolicy.CIFAR10,
            'SVHN': AutoAugmentPolicy.SVHN
        }
        policy = policy_map.get(self.auto_policy, AutoAugmentPolicy.CIFAR10)

        return T.Compose([
            T.Resize((self.size, self.size)),
            T.AutoAugment(policy=policy),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_custom_transform(self):
        """커스텀 증강을 반환"""
        if self.custom_transform is None:
            return self.get_basic_transform()

        return T.Compose([
            T.Resize((self.size, self.size)),
            *self.custom_transform,
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_transform(self, is_train=True, is_original=False):
        """
        Args:
            is_train (bool): 학습 시 True, 검증/테스트 시 False
            is_original (bool): 원본 데이터를 사용할지 여부 (True일 경우, 증강 없이 기본 변환 적용)
        Returns:
            transform: 선택된 데이터 증강 기법을 반환
        """
        if not is_train or is_original:
            return self.get_basic_transform()
        
        if self.custom_transform is None:
            return self.get_basic_transform()
        
        augment_map = {
            'auto': self.get_autoaugment_transform,
            'custom': self.get_custom_transform,
            'basic': self.get_basic_transform
        }

        # augment_type에 맞는 증강 기법을 반환
        return augment_map.get(self.augment_type, self.get_basic_transform)()
