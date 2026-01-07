import os
from torchvision import datasets, transforms
from data.tinyimagenet import build_tinyimagenet


def build_transforms(cfg):
    img = int(cfg["data"].get("img_size", 224))

    train_tf = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    test_tf = transforms.Compose([
        transforms.Resize((img, img)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])

    return train_tf, test_tf


def build_datasets(cfg, train_tf, test_tf):
    name = cfg["data"]["dataset"].lower()
    root = cfg["data"]["data_dir"]

    if name == "cifar10":
        ds_train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        ds_test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
        return ds_train, ds_test

    if name == "cifar100":
        ds_train = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        ds_test = datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
        return ds_train, ds_test

    if name == "mnist":
        # Convert 1ch->3ch for ViT-family
        tf_train = transforms.Compose([
            transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        tf_test = transforms.Compose([
            transforms.Resize((cfg["data"]["img_size"], cfg["data"]["img_size"])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        ds_train = datasets.MNIST(root, train=True, download=True, transform=tf_train)
        ds_test = datasets.MNIST(root, train=False, download=True, transform=tf_test)
        return ds_train, ds_test

    if name == "tinyimagenet":
        # Use provided train/val folders; treat val as test set here
        ds_train, ds_val = build_tinyimagenet(root, train_tf)
        ds_test = build_tinyimagenet(root, test_tf)[1]
        # Return train and "test"(val)
        return ds_train, ds_test

    if name == "imagenet1k":
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        ds_train = datasets.ImageFolder(train_dir, transform=train_tf)
        ds_test = datasets.ImageFolder(val_dir, transform=test_tf)
        return ds_train, ds_test

    raise ValueError(f"Unknown dataset: {name}")
