from new_model import CDFM3SF
import torchvision.transforms as tr
from my_transforms import *
from my_dataset import MyDataset, get_dataset_paths
import torch
from torch.utils.data import DataLoader
from sys import argv
import matplotlib.pyplot as plt


def create_image_from_confusion_matrix(tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp_color=(0, 255, 0), tn_color=(255, 0, 0), fp_color=(0, 0, 255), fn_color=(255, 255, 255)):
    """
    Create an image from the confusion matrix
    :param tp: true positive
    :param tn: true negative
    :param fp: false positive
    :param fn: false negative
    :param tp_color: color of true positive
    :param tn_color: color of true negative
    :param fp_color: color of false positive
    :param fn_color: color of false negative
    :return: image
    """
    tp = tp.cpu().numpy()
    tn = tn.cpu().numpy()
    fp = fp.cpu().numpy()
    fn = fn.cpu().numpy()

    # create image
    img = np.zeros((tp.shape[0], tp.shape[1], 3), dtype=np.uint8)

    # true positive
    img[tp == 1] = tp_color

    # true negative
    img[tn == 1] = tn_color

    # false positive
    img[fp == 1] = fp_color

    # false negative
    img[fn == 1] = fn_color

    return img


if len(argv) < 3:
    print(
        "Usage: python new_test.py <model_path> <imge_path> [other_images_paths]")
    exit()

print("Starting...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device in use: ", device)

test_paths = argv[2:]

transform = tr.Compose([
    MyToTensor()
])

test_dataset = MyDataset(test_paths, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print("Dataset loaded")

model = CDFM3SF([4, 6, 3], gf_dim=64)
model = model.to(device)

save_path = argv[1]
save_dict = torch.load(save_path)
model_saved = save_dict['model_state_dict']
model.load_state_dict(model_saved)

print("Model loaded")

fig, axarr = plt.subplots(
    len(test_paths), 4, figsize=(15, 5 * len(test_paths)))

axarr[0, 0].set_title('Input')
axarr[0, 1].set_title('Prediction')
axarr[0, 2].set_title('Ground truth')
axarr[0, 3].set_title('Differences')

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

        diff = create_image_from_confusion_matrix(
            tp_img, tn_img, fp_img, fn_img)

        output1 = output1.cpu()[0][0].detach().numpy()
        label = label.cpu()[0][0].detach().numpy()
        input = data_10m.cpu()[0][:3].detach().numpy()
        input = np.transpose(input, (1, 2, 0))
        input = input[..., ::-1]

        axarr[i, 0].imshow(input)
        axarr[i, 1].imshow(output1, cmap='gray')
        axarr[i, 2].imshow(label, cmap='gray')
        axarr[i, 3].imshow(diff)

plt.savefig('create_images.png')
