from torch.utils.data import Dataset
from images_utils import imgread
import numpy as np
import os


class MyDataset(Dataset):
    def __init__(self, data_paths: list[str], transform=None):
        self.data_paths = data_paths
        self.transform = transform

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx: int):
        img_path = self.data_paths[idx]
        path_10m = os.path.join(img_path, '10m.tif')
        path_20m = os.path.join(img_path, '20m.tif')
        path_60m = os.path.join(img_path, '60m.tif')
        path_label = os.path.join(img_path, 'label.tif')

        img_10m = imgread(path_10m)
        img_20m = imgread(path_20m)
        img_60m = imgread(path_60m)
        label = imgread(path_label)

        img_10m = img_10m.astype(np.float32) / 10000
        img_20m = img_20m.astype(np.float32) / 10000
        img_60m = img_60m.astype(np.float32) / 10000

        label = label.astype(np.float32)

        if self.transform:
            to_transform = (img_10m, img_20m, img_60m, label)
            transformed = self.transform(to_transform)
            img_10m, img_20m, img_60m, label = transformed

        return (img_10m, img_20m, img_60m), label


def get_dataset_paths(root_path: str = None) -> tuple[list[str], list[str], list[str]]:
    '''
    Get the lists of paths of the dataset, divided in train, validation and test
    Each path is a folder containing the images of a single sample
    (10m, 20m, 60m and label)

    Parameters
    ----------
    root_path : str, optional
        The root path that contains the dataset folder, if None the current working directory is used, by default None
    '''
    testlist = [
        "S2A_MSIL1C_20180930T030541_N0206_R075_T49QDD_20180930T060706",
        "S2A_MSIL1C_20191105T023901_N0208_R089_T51STR_20191105T054744",
        "S2A_MSIL1C_20190812T032541_N0208_R018_T48RXU_20190812T070322",
        "S2A_MSIL1C_20190602T021611_N0207_R003_T52TES_20190602T042019",
        "S2A_MSIL1C_20190328T033701_N0207_R061_T49TCF_20190328T071457",
        "S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939",
        "S2A_MSIL1C_20200416T042701_N0209_R133_T46SFE_20200416T074050",
        "S2A_MSIL1C_20200528T050701_N0209_R019_T44SPC_20200528T082127"
    ]

    validationlist = [
        "S2A_MSIL1C_20210207T023851_N0209_R089_T52UCU_20210207T040210",
        "S2A_MSIL1C_20210126T052111_N0209_R062_T44SNE_20210126T063836",
        "S2A_MSIL1C_20210102T054231_N0209_R005_T43SFB_20210102T065941",
        "S2A_MSIL1C_20201206T041141_N0209_R047_T47SMV_20201206T053320"
    ]

    if root_path is None:
        root_path = os.getcwd()

    dataset_dir = os.path.join(root_path,  '..', 'dataset')

    if not os.path.exists(dataset_dir):
        raise Exception(f'Dataset directory not found. {dataset_dir}')

    train_paths = []
    validation_paths = []
    test_paths = []
    for image in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image)
        for cutted_img in os.listdir(image_path):
            cutted_img_path = os.path.join(image_path, cutted_img)
            if image in testlist:
                test_paths.append(cutted_img_path)
            elif image in validationlist:
                validation_paths.append(cutted_img_path)
            else:
                train_paths.append(cutted_img_path)

    return train_paths, validation_paths, test_paths


def count_clouds_class(paths: list[str]) -> tuple[int, int]:
    n_clouds = 0
    n_backgrounds = 0
    for path in paths:
        label_path = os.path.join(path, 'label.tif')
        label = imgread(label_path)
        classes, counts = np.unique(label, return_counts=True)
        for c, n in zip(classes, counts):
            if c == 0:
                n_backgrounds += n
            else:
                n_clouds += n
    return n_clouds, n_backgrounds
