from new_model import CDFM3SF
import torchvision.transforms as tr
from my_transforms import *
from my_dataset import MyDataset, get_dataset_paths
import torch
from torch.utils.data import DataLoader
from sys import argv
import time
import torch.nn as nn

def my_loss(output1, output2, output3, label, loss_fn) -> torch.Tensor:
    output1 = output1.squeeze(1)
    output2 = output2.squeeze(1)
    output3 = output3.squeeze(1)
    label = label.squeeze(1)

    label1 = label
    label2 = tr.Resize((192, 192), antialias=False)(label)
    label3 = tr.Resize((64, 64), antialias=False)(label)

    loss1 = loss_fn(output1, label1)
    loss2 = loss_fn(output2, label2)
    loss3 = loss_fn(output3, label3)

    return 1*loss1 + 0.1*loss2 + 0.01*loss3

# "cuda" if torch.cuda.is_available() else 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device in use: ", device)

_, _, test_paths = get_dataset_paths()

test_paths = test_paths[:200]

transform = tr.Compose([
    MyToTensor()
])

test_dataset = MyDataset(test_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

n_clouds = 433635457
n_background = 2265694079
weight = torch.Tensor([n_background/n_clouds]).to(device)
loss_fn = nn.BCEWithLogitsLoss(weight=weight)

times = []

for batch in test_loader:
    (data, label) = batch

    data_10m, data_20m, data_60m = data

    start_time = time.time()

    data_10m = data_10m.to(device)
    data_20m = data_20m.to(device)
    data_60m = data_60m.to(device)

    label = label.to(device)

    o1, o2, o3 = model(data_10m, data_20m, data_60m)
    loss = my_loss(o1,o2,o3,label,loss_fn)

    end_time = time.time()
    elapsed_time = end_time - start_time
    loss.backward()

    times.append(elapsed_time)

ss = sum(times)
elapsed_time = ss/len(times)
print(f"Time elapsed: {elapsed_time} seconds with {device} in {ss}")

