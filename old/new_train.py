import torch
from torch.optim import Adam, Optimizer
from my_dataset import MyDataset, get_dataset_paths, count_clouds_class
from torch.utils.data import DataLoader
from CDFM3SF import CDFM3SF
from saver import Saver
import torchvision.transforms as tr
import torch.nn as nn
from my_transforms import *


def write_to_file(path, text):
    with open(path, "a") as f:
        f.write(text + "\n")


def train(model, loader, val_loader, optimizer: Optimizer, loss_fn):

    for e in range(3):
        train_loss = 0
        val_loss = 0

        for i, (data, label) in enumerate(loader):
            model.train()
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

            train_loss += loss.item()

            if i % (len(loader) // len(val_loader)) == 0:
                train_loss /= (len(loader) // len(val_loader))

                for i2, (data, label) in enumerate(val_loader):
                    model.eval()

                    with torch.no_grad():
                        data_10m, data_20m, data_60m = data

                        data_10m = data_10m.to(device)
                        data_20m = data_20m.to(device)
                        data_60m = data_60m.to(device)
                        label = label.to(device)

                        output1, output2, output3 = model(
                            data_10m, data_20m, data_60m)
                        loss: torch.Tensor = loss_fn(
                            output1, output2, output3, label)

                        val_loss += loss.item()

                val_loss /= (len(val_loader))
                print(
                    f"Epoch {e+1}/{3}, Batch {i+1}/{len(loader)}, T Loss: {train_loss}, V Loss: {val_loss}, {i/len(loader)*len(val_loader)}%")
                write_to_file("train_loss.txt", f"{train_loss},{val_loss}")

                train_loss = 0
                val_loss = 0


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


print("Starting...")

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

validation_dataset = MyDataset(validation_paths, transform=MyToTensor())
validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True)

print("Dataset loaded")

# n_clouds, n_background = count_clouds_class(train_paths)
n_clouds = 433635457
n_background = 2265694079
print(f"Clouds: {n_clouds}, Backgrounds: {n_background}")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.00025, betas=(0.5, 0.9))

print("Starting training...")
train(model, train_loader, validation_loader, optimizer, my_loss)
