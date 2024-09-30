from torch.utils.data import DataLoader, ConcatDataset
from .custom_dataset import CustomDataset
from .cutmix_loader import CutMixLoader

import os

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=None, apply_cutmix=False, alpha=1.0):
    if num_workers is None:
        num_workers = min(os.cpu_count(), 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    return CutMixLoader(loader, alpha) if apply_cutmix else loader

def create_combined_dataloader(info_df, root_dir, transform_selector, custom_transform_lists, batch_size, shuffle=True, num_workers=None, apply_cutmix=False, alpha=1.0):
    transform_list = [transform_selector.get_basic_transform()] + [transform_selector.get_custom_transform(custom_transform) for custom_transform in custom_transform_lists]

    datasets = [CustomDataset(root_dir, info_df, transform=transform) for transform in transform_list]
    combined_dataset = ConcatDataset(datasets)

    return create_dataloader(combined_dataset, batch_size, shuffle=shuffle, num_workers=num_workers, apply_cutmix=apply_cutmix, alpha=alpha)
