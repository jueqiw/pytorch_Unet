"""
using TorchIO to do 3D MRI augmentation, need pytorch as framework
"""
import random
from .const import *
import torchio as tio
from .get_path import get_path
from .const import ADNI_DATASET_DIR_1, CC359_DATASET_DIR, NFBS_DATASET_DIR, COMPUTECANADA
from skimage.transform import resize
import nibabel as nib
from torch import from_numpy
from time import ctime
from .const import CROPPED_IMG, CROPPED_LABEL
import numpy as np
from glob import glob


def get_subjects():
    """
    get data from the path and do augmentation on it, and return a DataLoader
    :return: list of subjects
    """

    # if COMPUTECANADA:
    #     datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    # else:
    #     datasets = [CC359_DATASET_DIR]
    #
    # subjects = [
    #     tio.Subject(
    #             img=tio.Image(path=mri.img_path, type=tio.INTENSITY),
    #             label=tio.Image(path=mri.label_path, type=tio.LABEL),
    #             # store the dataset name to help plot the image later
    #             # dataset=mri.dataset
    #         ) for mri in get_path(datasets)
    # ]

    # using in the cropping folder
    img_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(CROPPED_IMG)}/**/*.nii*", recursive=True))
    ])
    label_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(CROPPED_LABEL)}/**/*.nii.gz", recursive=True))
    ])

    subjects = [
        tio.Subject(
                img=tio.Image(path=img_path, type=tio.INTENSITY),
                label=tio.Image(path=label_path, type=tio.LABEL),
                # store the dataset name to help plot the image later
                # dataset=mri.dataset
            ) for img_path, label_path in zip(img_path_list, label_path_list)
    ]

    random.seed(42)
    random.shuffle(subjects)  # shuffle it to pick the val set

    print(f"{ctime()}: getting number of subjects {len(subjects)}")
    return subjects
