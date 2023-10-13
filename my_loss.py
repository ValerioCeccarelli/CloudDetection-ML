

from typing import Any
import torch
import torchvision.transforms as tr
import torch.nn as nn


class MyBCELoss:
    def __init__(self, weight: float, loss_ratios: list[float] = [1, 0.1, 0.01], device: torch.device = None) -> None:
        weight = torch.Tensor([weight]).to(device)
        self.__loss_fn = nn.BCEWithLogitsLoss(weight=weight)
        self.__loss_ratios = loss_ratios

    def __call__(self, images: tuple[torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        label1 = label
        label2 = tr.Resize((192, 192), antialias=False)(label)
        label3 = tr.Resize((64, 64), antialias=False)(label)

        output1, output2, output3 = images

        loss1 = self.__loss_fn(output1, label1) * self.__loss_ratios[0]
        loss2 = self.__loss_fn(output2, label2) * self.__loss_ratios[1]
        loss3 = self.__loss_fn(output3, label3) * self.__loss_ratios[2]

        return loss1 + loss2 + loss3
