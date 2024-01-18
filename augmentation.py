import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_aug(m=0.5, s=0.5):
    process = A.Compose([

        A.Resize(512, 512),
        A.Normalize(mean=m, std=s),
        ToTensorV2(),
    ])
    return process


def val_aug(m=0.5, s=0.5):
    process = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=m, std=s),
        ToTensorV2(),
    ])
    return process