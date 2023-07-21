import os
from my_dataset import MyDataset, count_clouds_class, get_dataset_paths
from torch.utils.data import DataLoader
import torch
from CDFM3SF import CDFM3SF
import torch.nn as nn

current_dir = os.getcwd()
models_dir = os.path.join(current_dir, 'models')

models = [model for model in os.listdir(models_dir) if model.endswith('.pth')]
models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
models = [os.path.join(models_dir, model) for model in models]

train_paths, validation_paths, _ = get_dataset_paths()

dataset = MyDataset(validation_paths)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device {device}')

model = CDFM3SF()
model.to(device)

n_clouds, n_background = count_clouds_class(train_paths)
print(f"Clouds: {n_clouds}, Backgrounds: {n_background}")

weight = torch.Tensor([n_background/n_clouds]).to(device)
loss_fn = nn.BCEWithLogitsLoss(weight=weight)

print('Starting validation...\n')

for model in models:
    print(f'Loading model {model}')
    model_state_dict = torch.load(model)
    model.load_state_dict(model_state_dict)

    model.eval()

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    loss = 0

    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
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

    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(
        f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}')

    print()
