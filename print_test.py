from CDFM3SF import CDFM3SF
import torch
import matplotlib.pyplot as plt
import numpy as np
from my_dataset import MyDataset
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from saver import Saver
import os
from my_transforms import MyToTensor
import sys

args = sys.argv[1:]

if len(args) < 1:
    print('Usage: python print_test.py <path_to_test1> [<path_to_test2> ...]')
    exit(1)

for test_path in args:
    if not os.path.exists(test_path):
        print('Test path ' + test_path + ' does not exist')
        exit(1)

transform = tr.Compose([
    MyToTensor()
])

test_dataset = MyDataset(args, transform=transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

saver = Saver(model)
saver.load()

fig, axs = plt.subplots(len(args), 3, figsize=(15, 5 * len(args)))

if len(args) == 1:
    axs[0].set_title('Input')
    axs[1].set_title('Prediction')
    axs[2].set_title('Ground truth')
else:
    axs[0][0].set_title('Input')
    axs[0][1].set_title('Prediction')
    axs[0][2].set_title('Ground truth')

for i, data in enumerate(test_dataloader):
    (d10m, d20m, d60m), label = data
    d10m = d10m.to(device)
    d20m = d20m.to(device)
    d60m = d60m.to(device)
    label = label.to(device)

    pred = model(d10m, d20m, d60m)
    pred = torch.sigmoid(pred[0])
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.cpu()[0][0].detach().numpy()

    label = label.cpu()[0][0].detach().numpy()

    input = d10m.cpu()[0][:3].detach().numpy()
    input = np.transpose(input, (1, 2, 0))
    input = input[..., ::-1]

    if len(args) == 1:
        axs[0].imshow(input)
        axs[1].imshow(pred, cmap='gray')
        axs[2].imshow(label, cmap='gray')
    else:
        axs[i][0].imshow(input)
        axs[i][1].imshow(pred, cmap='gray')
        axs[i][2].imshow(label, cmap='gray')

plt.savefig('result.png')

# test example:
# python print_test.py dataset\S2A_MSIL1C_20180429T032541_N0206_R018_T49SCV_20180429T062304\12_15 dataset\S2A_MSIL1C_20180429T032541_N0206_R018_T49SCV_20180429T062304\12_16
