import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from my_dataset import MyDataset, get_dataset_paths, count_clouds_class
from torch.utils.data import DataLoader
from CDFM3SF import CDFM3SF
from saver import Saver
import torchvision.transforms as tr
import torch.nn as nn
from my_transforms import *


def train_epoch(model, loader, loss_fn, optimizer: Optimizer, device):
    model.train()

    loss_list = []
    for data, label in loader:
        optimizer.zero_grad()

        data_10m, data_20m, data_60m = data

        data_10m = data_10m.to(device)
        data_20m = data_20m.to(device)
        data_60m = data_60m.to(device)
        label = label.to(device)

        output1, output2, output3 = model(data_10m, data_20m, data_60m)
        loss: torch.Tensor = loss_fn(output1, output2, output3, label)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    mean_loss = sum(loss_list) / len(loss_list)

    return mean_loss


def my_loss(output1, output2, output3, label) -> torch.Tensor:
    output1 = output1.squeeze(1)
    output2 = output2.squeeze(1)
    output3 = output3.squeeze(1)
    label = label.squeeze(1)

    label1 = label
    label2 = tr.Resize((192, 192), antialias=False)(label)
    label3 = tr.Resize((64, 64), antialias=False)(label)

    weight = torch.Tensor([n_background/n_clouds]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(weight=weight)

    loss1 = loss_fn(output1, label1)
    loss2 = loss_fn(output2, label2)
    loss3 = loss_fn(output3, label3)

    return 1*loss1 + 0.1*loss2 + 0.01*loss3


def train(model, loader, optimizer: Optimizer, loss_fn, saver: Saver, device, scheduler):
    epoch = saver.get_last_epoch() + 1

    while True:
        mean_loss = train_epoch(model, loader, loss_fn, optimizer, device)
        saver.save(epoch, mean_loss)

        print(f"Epoch: {epoch}, Loss: {mean_loss}")
        scheduler.step()

        epoch += 1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use: ", device)

train_paths, _, _ = get_dataset_paths()

transform = tr.Compose([
    MyToTensor(),
    MyRandomVerticalFlip(p=0.5),
    MyRandomHorizontalFlip(p=0.5),
    MyRandomRotation(p=0.5, degrees=90)
])

train_dataset = MyDataset(train_paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

print("Dataset loaded")

n_clouds, n_background = count_clouds_class(train_paths)
print(f"Clouds: {n_clouds}, Backgrounds: {n_background}")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

saver = Saver(model)
saver.load()
last_epoch = saver.get_last_epoch()

optimizer = Adam(model.parameters(), lr=0.00025, betas=(0.5, 0.9))
scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=last_epoch)

print("Starting training...")
train(model, train_loader, optimizer, my_loss, saver, device, scheduler)
