import os
import nibabel as nib
from pathlib import Path
from .data.const import *
from .data.get_path import get_path


def get_all_different_size_image():
    datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    sizes = set()
    for mri in get_path(datasets):
        img = nib.load(mri.img_path)

        dims = list(img.shape)
        dims = ','.join(dims)

        i = 1
        if dims not in sizes:
            sizes.add(dims)
            print(f"find {i}")
            os.system(f"cp {mri.img_path} /project/6005889/U-Net_MRI-Data/all_different_size_img")
            os.system(f"cp {mri.label_path} /project/6005889/U-Net_MRI-Data/all_different_size_img")



