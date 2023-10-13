from my_model import CDFM3SF
import torchvision.transforms as tr
from my_transforms import *
from my_dataset import MyDataset
import torch
from torch.utils.data import DataLoader
from sys import argv
import matplotlib.pyplot as plt


# TODO: to be totaly rewrited
# ---- TODO: add a lot of parameters to configure the style?


def create_image_from_confusion_matrix(tp: torch.Tensor, tn: torch.Tensor, fp: torch.Tensor, fn: torch.Tensor, tp_color=(255, 255, 255), tn_color=(0, 0, 0), fp_color=(0, 0, 255), fn_color=(255, 0, 0)):
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

    tp = np.transpose(tp, (1, 2, 0))
    tn = np.transpose(tn, (1, 2, 0))
    fp = np.transpose(fp, (1, 2, 0))
    fn = np.transpose(fn, (1, 2, 0))

    tp = np.squeeze(tp)
    tn = np.squeeze(tn)
    fp = np.squeeze(fp)
    fn = np.squeeze(fn)

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
    len(test_paths), 4, figsize=(15, 4 * len(test_paths)), squeeze=False)

fig.subplots_adjust(left=0.02, right=0.98, bottom=0,
                    top=1, wspace=0.2, hspace=0)

ft = 20
axarr[0, 0].set_title('Input', fontsize=ft)
axarr[0, 1].set_title('Prediction', fontsize=ft)
axarr[0, 2].set_title('Ground truth', fontsize=ft)
axarr[0, 3].set_title('Differences', fontsize=ft)

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

        output1 = output1.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        input = data_10m.cpu()[0][:3].detach().numpy()
        input = np.transpose(input, (1, 2, 0))
        input = input[..., ::-1]

        output1 = np.transpose(output1, (1, 2, 0))
        label = np.transpose(label, (1, 2, 0))

        axarr[i, 0].imshow(input)
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(output1, cmap='gray')
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(label, cmap='gray')
        axarr[i, 2].axis('off')
        axarr[i, 3].imshow(diff)
        axarr[i, 3].axis('off')

plt.savefig('create_images.png')


# example no snow
# python .\create_images.py .\safe\save_22.pth ..\dataset\S2A_MSIL1C_20190328T033701_N0207_R061_T49TCF_20190328T071457\22_15 ..\dataset\S2A_MSIL1C_20190328T033701_N0207_R061_T49TCF_20190328T071457\19_21 ..\dataset\S2A_MSIL1C_20190328T033701_N0207_R061_T49TCF_20190328T071457\21_20 ..\dataset\S2A_MSIL1C_20190328T033701_N0207_R061_T49TCF_20190328T071457\21_22

# example with snow
# python .\create_images.py .\safe\save_22.pth ..\dataset\S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939\4_25 ..\dataset\S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939\5_25 ..\dataset\S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939\5_26 ..\dataset\S2A_MSIL1C_20200416T042701_N0209_R133_T46SFE_20200416T074050\10_11
