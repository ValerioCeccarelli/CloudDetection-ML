import torchvision.transforms.functional as F
import numpy as np
import torch


class MyToTensor():
    def __call__(self, imgs: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        img_10m, img_20m, img_60m, label = imgs
        img_10m = F.to_tensor(img_10m)
        img_20m = F.to_tensor(img_20m)
        img_60m = F.to_tensor(img_60m)
        label = F.to_tensor(label)
        return img_10m, img_20m, img_60m, label


class MyRandomVerticalFlip():
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, imgs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        img_10m, img_20m, img_60m, label = imgs
        if np.random.rand() < self.p:
            img_10m = F.vflip(img_10m)
            img_20m = F.vflip(img_20m)
            img_60m = F.vflip(img_60m)
            label = F.vflip(label)
        return img_10m, img_20m, img_60m, label


class MyRandomHorizontalFlip():
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, imgs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        img_10m, img_20m, img_60m, label = imgs
        if np.random.rand() < self.p:
            img_10m = F.hflip(img_10m)
            img_20m = F.hflip(img_20m)
            img_60m = F.hflip(img_60m)
            label = F.hflip(label)
        return img_10m, img_20m, img_60m, label


class MyRandomRotation():
    def __init__(self, degrees: float = 90):
        self.degrees = degrees

    def __call__(self, imgs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        img_10m, img_20m, img_60m, label = imgs
        if np.random.rand() < 0.5:
            img_10m = F.rotate(img_10m, self.degrees)
            img_20m = F.rotate(img_20m, self.degrees)
            img_60m = F.rotate(img_60m, self.degrees)
            label = F.rotate(label, self.degrees)
        return img_10m, img_20m, img_60m, label
