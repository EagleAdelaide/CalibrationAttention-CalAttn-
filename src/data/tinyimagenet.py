import os
from torchvision.datasets import ImageFolder


def build_tinyimagenet(root, transform):
    # root points to tiny-imagenet-200
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    ds_train = ImageFolder(train_dir, transform=transform)
    ds_val = ImageFolder(val_dir, transform=transform)
    return ds_train, ds_val
