import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from my_dataset import MyDataset, get_dataset_paths, count_clouds_class, divide_dataset_path
from torch.utils.data import DataLoader
from my_model import CDFM3SF
import torchvision.transforms as tr
import torch.nn as nn
from my_transforms import *
import time
import torch.nn as nn
from my_loss import MyBCELoss
from my_saver import MySaver
from my_random_search import MyRandomSearch
import sys

# TODO: docstring
# TODO: use the real model instead of nn.Module?
# TODO: add verbose to print the status
# TODO: use some beautiful library toi print the percentages?


def train_epoch(model: nn.Module, loader: DataLoader, loss_fn: MyBCELoss, optimizer: Adam, device: torch.device):
    model.train()

    loss_list = []
    for i, (data, label) in enumerate(loader):
        optimizer.zero_grad()

        data_10m, data_20m, data_60m = data

        data_10m = data_10m.to(device)
        data_20m = data_20m.to(device)
        data_60m = data_60m.to(device)
        label = label.to(device)

        outputs = model(data_10m, data_20m, data_60m)
        loss = loss_fn(outputs, label)

        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    mean_loss = sum(loss_list) / len(loss_list)

    return mean_loss


def validation(model: nn.Module, loader: DataLoader, loss_fn: MyBCELoss, device: torch.device):
    with torch.no_grad():
        model.eval()

        loss_list = []

        for i, (data, label) in enumerate(loader):
            data_10m, data_20m, data_60m = data

            data_10m = data_10m.to(device)
            data_20m = data_20m.to(device)
            data_60m = data_60m.to(device)
            label = label.to(device)

            outputs = model(data_10m, data_20m, data_60m)
            loss = loss_fn(outputs, label)

            loss_list.append(loss.item())

        mean_loss = sum(loss_list) / len(loss_list)

        return mean_loss


def train(
        model: nn.Module,
        optimizer: Adam,
        scheduler: ExponentialLR,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        loss_fn: MyBCELoss,
        device: torch.device,
        saver: MySaver,
        current_epoch: int,
        num_train_epoch: int
):
    for epoch_number in range(current_epoch, current_epoch + num_train_epoch):
        start_time = time.time()
        train_mean_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device)
        end_time = time.time()

        print(
            f"Epoch: {epoch_number}, Loss: {train_mean_loss}, Time: {end_time - start_time}")

        start_time = time.time()
        validation_mean_loss = validation(
            model, validation_loader, loss_fn, device)
        end_time = time.time()

        print(
            f" -   Validation loss: {validation_mean_loss}, Time: {end_time - start_time}")

        scheduler.step()
        saver.save_state(model, optimizer, scheduler, epoch_number)


def print_error(message):
    print(message)
    print("python train.py <save_file_name> [create_if_not_exist]")
    print()
    print("Example 1: python train.py existing_save_file.pth")
    print()
    print("Example 2: python train.py save_file_to_be_created.pth create_if_not_exist")
    exit()


if len(sys.argv) > 3 or len(sys.argv) < 2:
    print_error("Too many argumnets, the right configuration shoul be:")

file_save_name = sys.argv[1]
create_if_not_exist = False

if not file_save_name.endswith(".pth"):
    print_error(f"The file {sys.argv[1]} should be a .pth file.")

if len(sys.argv) == 3:
    if sys.argv[2] != 'create_if_not_exist':
        print_error(f"Invalid third argument provided '{sys.argv[2]}'.")

    create_if_not_exist = True


print("Start program...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = tr.Compose([
    MyToTensor(),
    MyRandomVerticalFlip(p=0.5),
    MyRandomHorizontalFlip(p=0.5),
    MyRandomRotation(p=0.5, degrees=90)
])

validation_transform = tr.Compose([
    MyToTensor()
])

images_paths, _ = get_dataset_paths()

n_clouds, n_background = count_clouds_class(images_paths)
# n_clouds = 433635457
# n_background = 2265694079

dataset = MyDataset(images_paths, transform=train_transform)

weight = n_background/n_clouds
loss_fn = MyBCELoss(weight, loss_ratios=[1, 0.1, 0.01], device=device)

iperparameters = {
    "beta1": [0.1, 0.5, 0.9],
    "beta2": [0.1, 0.5, 0.9],
    "gamma": [0.95, 0.80, 0.65],
    "gf_dim": [64],
    "lr": [0.00025]
}

random_search = MyRandomSearch(iperparameters, 3)

iper = next(iter(random_search))
beta1 = iper['beta1']
beta2 = iper['beta2']
gamma = iper['gamma']
gf_dim = iper["gf_dim"]
lr = iper["lr"]

train_dataset, validation_dataset = divide_dataset_path(
    dataset, validation_set_ratio=0.1)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=4)

model = CDFM3SF([4, 6, 3], gf_dim=gf_dim)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=-1)

saver = MySaver(file_save_name, create_if_not_exist=True)

last_epoch = saver.load_last_state_if_present(model, optimizer, scheduler)

current_epoch = last_epoch+1 if last_epoch is not None else 0

print("Start training...")
train(model, optimizer, scheduler, train_loader,
      validation_loader, loss_fn, device, saver, current_epoch, 50)
