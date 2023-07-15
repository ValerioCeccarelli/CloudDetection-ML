import zipfile
import os
import numpy as np
from images_utils import imgread, imgwrite
import time

current_dir = os.getcwd()

print('\nUnzipping files...\n')

unzip_time_start = time.time()

data_dir = os.path.join(current_dir, 'data')

for file in os.listdir(data_dir):
    if file.endswith('.zip'):
        print('Unzipping ' + file)
        file_path = os.path.join(data_dir, file)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

unzip_time_end = time.time()
unzip_time = unzip_time_end - unzip_time_start
print('Unzip time: ' + str(unzip_time) + ' seconds')

# Extract all bands from all images

print('\nExtracting bands...\n')

extract_time_start = time.time()

bands = [['10m', ('02', '03', '04', '08')], [
    '20m', ('05', '06', '07', '8A', '11', '12')], ['60m', ('01', '09', '10')]]


def fuze_img(file_path):
    granule_dir = os.path.join(file_path, 'GRANULE')
    granule = os.listdir(granule_dir)[0]
    granule_path = os.path.join(granule_dir, granule)

    img_dir = os.path.join(granule_path, 'IMG_DATA')
    imgs = os.listdir(img_dir)

    for band in bands:
        band_name = band[0]

        img_concat = []
        for band_num in band[1]:
            band_num = 'B' + band_num
            band_path = os.path.join(
                img_dir, [img for img in imgs if band_num in img][0])

            img = imgread(band_path)
            img_dn = img[:, :, np.newaxis]
            img_concat.append(img_dn)

        fused_img = np.concatenate(img_concat, axis=2)

        save_dir = os.path.join(file_path, "bands")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, band_name + '.tif')
        imgwrite(save_path, fused_img)


for file in os.listdir(data_dir):
    file_path = os.path.join(data_dir, file)
    if os.path.isdir(file_path) and file.endswith('.SAFE'):
        print('Fuzing ' + file)
        fuze_img(file_path)

extract_time_end = time.time()
extract_time = extract_time_end - extract_time_start
print('Extract time: ' + str(extract_time) + ' seconds')

# Create dataset

dataset_time_start = time.time()

print('\nCreating dataset...\n')

dataset_dir = os.path.join(current_dir, 'dataset')
reference_mask_dir = os.path.join(current_dir, 'data', 'ReferenceMask')

window_sizes = [384, 192, 64]
strides = [384, 192, 64]

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)


def cut_images(bands_dir, label_path, dataset_dir, name):
    band_10m_path = os.path.join(bands_dir, '10m.tif')
    band_20m_path = os.path.join(bands_dir, '20m.tif')
    band_60m_path = os.path.join(bands_dir, '60m.tif')

    band_10m = imgread(band_10m_path)
    band_20m = imgread(band_20m_path)
    band_60m = imgread(band_60m_path)

    label = imgread(label_path)

    img_height, img_width = label.shape
    w_steps = img_width // strides[0]
    h_steps = img_height // strides[0]

    dataset_path = os.path.join(dataset_dir, name)

    count = 0

    for i in range(h_steps):
        for j in range(w_steps):
            cutted_img_10m = band_10m[i * strides[0]:i * strides[0] + window_sizes[0],
                                      j * strides[0]:j * strides[0] + window_sizes[0]]
            cutted_img_20m = band_20m[i * strides[1]:i * strides[1] + window_sizes[1],
                                      j * strides[1]:j * strides[1] + window_sizes[1]]
            cutted_img_60m = band_60m[i * strides[2]:i * strides[2] + window_sizes[2],
                                      j * strides[2]:j * strides[2] + window_sizes[2]]
            cutted_label = label[i * strides[0]:i * strides[0] + window_sizes[0],
                                 j * strides[0]:j * strides[0] + window_sizes[0]]

            if not np.all(cutted_img_10m > 0) \
                    or not np.all(cutted_img_20m > 0) \
                    or not np.all(cutted_img_60m > 0) \
                    or not np.all(cutted_label > 0):
                continue

            # TODO: controllare questa cosa
            # TODO: controllare se ci sono modi più efficienti per fare questa cosa
            cutted_label[cutted_label == 128] = 0
            cutted_label[cutted_label == 255] = 1

            cut_path = os.path.join(dataset_path, str(i) + '_' + str(j))
            if not os.path.exists(cut_path):
                os.makedirs(cut_path)

            imgwrite(os.path.join(cut_path, '10m.tif'), cutted_img_10m)
            imgwrite(os.path.join(cut_path, '20m.tif'), cutted_img_20m)
            imgwrite(os.path.join(cut_path, '60m.tif'), cutted_img_60m)
            imgwrite(os.path.join(cut_path, 'label.tif'), cutted_label)

            count += 1

    return count


total_cutted = 0
for file in os.listdir(data_dir):
    if file.endswith('.SAFE') and os.path.isdir(os.path.join(data_dir, file)):
        name = file.removesuffix('.SAFE')
        file_path = os.path.join(data_dir, file)
        bands_dir = os.path.join(file_path, 'bands')
        label_path = os.path.join(reference_mask_dir, name + '_Mask.tif')

        print('Cutting ' + name, end='')
        cutted = cut_images(bands_dir, label_path, dataset_dir, name)
        print(' -> ' + str(cutted) + ' images')
        total_cutted += cutted

dataset_time_end = time.time()
dataset_time = dataset_time_end - dataset_time_start
print('Dataset time: ' + str(dataset_time) +
      ' seconds with ' + str(total_cutted) + ' images')
