import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from my_dataset import MyDataset, get_dataset_paths, count_clouds_class
from torch.utils.data import DataLoader
from new_model import CDFM3SF
import torchvision.transforms as tr
import torch.nn as nn
from my_transforms import *
import os
import time


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


def validation(model, loader, loss_fn, device):
    with torch.no_grad():
        model.eval()

        loss_list = []
        for data, label in loader:
            data_10m, data_20m, data_60m = data

            data_10m = data_10m.to(device)
            data_20m = data_20m.to(device)
            data_60m = data_60m.to(device)
            label = label.to(device)

            output1, output2, output3 = model(data_10m, data_20m, data_60m)
            loss: torch.Tensor = loss_fn(output1, output2, output3, label)

            loss_list.append(loss.item())

        mean_loss = sum(loss_list) / len(loss_list)

        return mean_loss


def train(model, loader, optimizer: Optimizer, loss_fn, device, scheduler, validation_loader, last_epoch):
    epoch = last_epoch + 1

    while True:
        start_time = time.time()
        mean_loss = train_epoch(model, loader, loss_fn, optimizer, device)
        end_time = time.time()

        print(
            f"Epoch: {epoch}, Loss: {mean_loss}, Time: {end_time - start_time}")
        scheduler.step()

        start_time = time.time()
        mean_loss = validation(model, validation_loader, loss_fn, device)
        end_time = time.time()

        print(
            f" -   Validation loss: {mean_loss}, Time: {end_time - start_time}")

        save_model(model, epoch, mean_loss, mean_loss, optimizer, scheduler)

        epoch += 1


SAVED_FOLDER = "safe"
SAVE_FILE = "save.txt"


def save_model(model, epoch, train_loss, validation_loss, optimizer, scheduler):
    file_name = f"{SAVED_FOLDER}/save_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'validation_loss': validation_loss
    }, file_name)

    with open(f"{SAVED_FOLDER}/{SAVE_FILE}", "a") as f:
        f.write(f"{epoch},{train_loss},{validation_loss},{file_name}\n")


def load_model(model, optimizer, scheduler, epoch):
    file_name = f"{SAVED_FOLDER}/save_{epoch}.pth"

    checkpoint = torch.load(file_name)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


def get_last_saved_epoch() -> int:
    if not os.path.exists(f"{SAVED_FOLDER}/{SAVE_FILE}"):
        if not os.path.exists(SAVED_FOLDER):
            os.mkdir(SAVED_FOLDER)
        with open(f"{SAVED_FOLDER}/{SAVE_FILE}", "w+") as f:
            f.write("epoch,train_loss,validation_loss,file_name\n")

    with open(f"{SAVED_FOLDER}/{SAVE_FILE}", "r") as f:
        lines = f.readlines()

        if len(lines) == 1:
            return 0
        else:
            last_line = lines[-1]
            epoch = int(last_line.split(",")[0])
            return epoch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use: ", device)

train_paths, validation_paths, _ = get_dataset_paths()

transform = tr.Compose([
    MyToTensor(),
    MyRandomVerticalFlip(p=0.5),
    MyRandomHorizontalFlip(p=0.5),
    MyRandomRotation(p=0.5, degrees=90)
])

train_dataset = MyDataset(train_paths, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

validation_dataset = MyDataset(validation_paths)
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

print("Dataset loaded")

# n_clouds, n_background = count_clouds_class(train_paths)
n_clouds = 433635457
n_background = 2265694079
print(f"Clouds: {n_clouds}, Backgrounds: {n_background}")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

last_epoch = get_last_saved_epoch()

optimizer = Adam(model.parameters(), lr=0.00025, betas=(0.5, 0.9))
optimizer.param_groups[0]['initial_lr'] = 0.00025
scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=last_epoch)

if last_epoch != 0:
    load_model(model, optimizer, scheduler, last_epoch)


print("Starting training...")
train(model, train_loader, optimizer, my_loss,
      device, scheduler, validation_loader, last_epoch)

# TODO: controlare il discorso dei salvataggi e dei load per le epoche
