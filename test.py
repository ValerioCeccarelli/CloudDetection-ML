from CDFM3SF import CDFM3SF
from saver import Saver
import torchvision.transforms as tr
from my_transforms import *
from my_dataset import MyDataset, get_dataset_paths, count_clouds_class
import torch

print("Starting...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use: ", device)

_, test_paths = get_dataset_paths()

transform = tr.Compose([
    MyToTensor()
])

train_dataset = MyDataset(test_paths, transform=transform)

print("Dataset loaded")

n_clouds, n_background = count_clouds_class(test_paths)
print(f"Clouds: {n_clouds}, Backgrounds: {n_background}")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

saver = Saver(model)
saver.load()

print("Model loaded")

tp = 0
tn = 0
fp = 0
fn = 0

for data, label in train_dataset:
    data_10m, data_20m, data_60m = data

    data_10m = data_10m.to(device)
    data_20m = data_20m.to(device)
    data_60m = data_60m.to(device)

    output1, _, _ = model(data_10m, data_20m, data_60m)

    output1 = torch.sigmoid(output1)
    output1 = output1.squeeze(1)
    output1 = torch.where(output1 > 0.5, 1, 0)

    label = label.to(device)
    label = label.squeeze(1)

    # confusion matrix
    tp += torch.sum((output1 == 1) & (label == 1))
    tn += torch.sum((output1 == 0) & (label == 0))
    fp += torch.sum((output1 == 1) & (label == 0))
    fn += torch.sum((output1 == 0) & (label == 1))

    break

# accuracy
acc = (tp + tn) / (tp + tn + fp + fn)

# precision
prec = tp / (tp + fp)

# recall
rec = tp / (tp + fn)

# f1 score
f1 = 2 * prec * rec / (prec + rec)

# confusion matrix %
tot = tp + tn + fp + fn
tp = tp / tot
tn = tn / tot
fp = fp / tot
fn = fn / tot

print(f"Accuracy: {acc}")
print(F"Precision {prec}")
print(f"Recall {rec}")
print(f"F1 {f1}")
print()
print(" \t 0\t 1")
print(f"0\t{tn}%\t{fp}%")
print(f"1\t{fn}%\t{tp}%")
