from new_model import CDFM3SF
import torchvision.transforms as tr
from my_transforms import *
from my_dataset import MyDataset, get_dataset_paths
import torch
from torch.utils.data import DataLoader
from sys import argv

if len(argv) > 2:
    print("Usage: python new_test.py [model_path]")
    exit()

print("Starting...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use: ", device)

_, _, test_paths = get_dataset_paths()

test_paths = [
    path for path in test_paths if
    "S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939" in path or
    "S2A_MSIL1C_20200416T042701_N0209_R133_T46SFE_20200416T074050" in path or
    "S2A_MSIL1C_20200528T050701_N0209_R019_T44SPC_20200528T082127" in path
]

transform = tr.Compose([
    MyToTensor()
])

test_dataset = MyDataset(test_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print("Dataset loaded")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

if len(argv) > 1:
    print(f"Loading model from {argv[1]}")
    save_path = argv[1]
    save_dict = torch.load(save_path)
    model_saved = save_dict['model_state_dict']
    model.load_state_dict(model_saved)
else:
    print("No model path provided, using default model")

print("Model loaded")

tp = 0
tn = 0
fp = 0
fn = 0

model.eval()
with torch.no_grad():
    for i, (data, label) in enumerate(test_loader):
        data_10m, data_20m, data_60m = data

        data_10m = data_10m.to(device)
        data_20m = data_20m.to(device)
        data_60m = data_60m.to(device)

        output1, _, _ = model(data_10m, data_20m, data_60m)

        output1 = torch.sigmoid(output1)
        output1 = output1.squeeze(1)
        output1 = torch.where(output1 > 0.5, 1, 0)

        label: torch.Tensor = label.to(device)
        label = label.squeeze(1)

        tp_img = (output1 == 1) & (label == 1)
        tn_img = (output1 == 0) & (label == 0)
        fp_img = (output1 == 1) & (label == 0)
        fn_img = (output1 == 0) & (label == 1)

        # confusion matrix
        tp += torch.sum(tp_img)
        tn += torch.sum(tn_img)
        fp += torch.sum(fp_img)
        fn += torch.sum(fn_img)

        if i % (len(test_loader) // 10) == 0:
            percent = i * 100 // len(test_loader)
            print(f"Testing... {percent}%")

# accuracy
acc = (tp + tn) / (tp + tn + fp + fn)

# precision
prec = tp / (tp + fp)

# recall
rec = tp / (tp + fn)

# f1 score
f1 = 2 * prec * rec / (prec + rec)

# confusion matrix %

print(f"Accuracy: {acc}")
print(F"Precision {prec}")
print(f"Recall {rec}")
print(f"F1 {f1}")
print()
print(" \t 0\t 1")
print(f"0\t{tn}\t{fp}")
print(f"1\t{fn}\t{tp}")
