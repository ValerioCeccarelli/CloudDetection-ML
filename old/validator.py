import os
from my_dataset import MyDataset, get_dataset_paths
from torch.utils.data import DataLoader
import torch
from CDFM3SF import CDFM3SF
import torch.nn as nn
from my_transforms import MyToTensor
import torchvision.transforms as tr

current_dir = os.getcwd()
models_dir = os.path.join(current_dir, 'models')

models = [model for model in os.listdir(models_dir) if model.endswith('.pth')]
models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
models = [os.path.join(models_dir, model) for model in models]

train_paths, validation_paths, _ = get_dataset_paths()

dataset = MyDataset(validation_paths, transform=MyToTensor())
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

model = CDFM3SF([4, 6, 3], gf_dim=64)
model.to(device)

# n_clouds, n_background = count_clouds_class(train_paths)
# print(f"Clouds: {n_clouds}, Backgrounds: {n_background}")
# Clouds: 433635457, Backgrounds: 2265694079
n_clouds = 433635457
n_background = 2265694079

print('Starting validation...\n')


def file_write(row):
    with open("validation.csv", "a") as file:
        file.write(row)


file_write("model,TP,TN,FP,FN,accuracy,precision,recall,f1,loss\n")


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


def test():
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    loss = 0

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data_10m, data_20m, data_60m = data

            data_10m = data_10m.to(device)
            data_20m = data_20m.to(device)
            data_60m = data_60m.to(device)

            output1, output2, output3 = model(data_10m, data_20m, data_60m)

            label = label.to(device)

            loss += my_loss(output1, output2, output3, label)

            output1 = torch.sigmoid(output1)
            output1 = output1.squeeze(1)
            output1 = torch.where(output1 > 0.5, 1, 0)

            label = label.squeeze(1)

            # confusion matrix
            tp += torch.sum((output1 == 1) & (label == 1))
            tn += torch.sum((output1 == 0) & (label == 0))
            fp += torch.sum((output1 == 1) & (label == 0))
            fn += torch.sum((output1 == 0) & (label == 1))

    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    loss = loss / len(dataloader)
    print(
        f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}, Loss: {loss}')

    file_write(
        f"{n},{tp},{tn},{fp},{fn},{accuracy},{precision},{recall},{f1},{loss}\n")

    print()


n = -1
print('random model')
test()


for n, saved_model in enumerate(models):
    print(f'Loading model {saved_model}')
    model_state_dict = torch.load(saved_model)
    model.load_state_dict(model_state_dict)

    test()

print('Done')
