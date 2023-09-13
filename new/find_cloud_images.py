from my_dataset import get_dataset_paths
from images_utils import imgread
import os
import numpy as np

TOT_PIXELS = 384 * 384
TOP = 0.4 * TOT_PIXELS
BOTTOM = 0.2 * TOT_PIXELS

test_paths = get_dataset_paths()[2]

def foo():
    path_label = os.path.join(path, 'label.tif')
    label = imgread(path_label)

    white_pixels = np.sum(label == 1)

    if BOTTOM < white_pixels < TOP:
        print(path)


for path in test_paths:

    if "S2A_MSIL1C_20191001T050701_N0208_R019_T45TXN_20191002T142939" in path:
        foo()

    if "S2A_MSIL1C_20200416T042701_N0209_R133_T46SFE_20200416T074050" in path:
        foo()

    if "S2A_MSIL1C_20200528T050701_N0209_R019_T44SPC_20200528T082127" in path:
        foo()

    
