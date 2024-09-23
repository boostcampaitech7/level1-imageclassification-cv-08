import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import AutoAugmentPolicy
import torchvision.transforms as T
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, info_df, transform, is_inference=False):
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
        self.use_auto = use_auto
        self.auto_policy = auto_policy
        self.size = size

    def get_transform(self, is_train=True):
        if is_train:
            if self.use_auto:
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
                transform = T.Compose([
                    T.Resize((self.size,self.size)),
                    T.Grayscale(num_output_channels=3),
                    T.RandomRotation(15),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
                ])
        else:
            transform = T.Compose([
                T.Resize((self.size, self.size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform
    
def create_dataloader(info_df, root_dir, transform, batch_size, shuffle=True, is_inference=False):
    dataset = CustomDataset(root_dir, info_df, transform, is_inference)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)