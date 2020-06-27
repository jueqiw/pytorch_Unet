"""
using TorchIO to do 3D MRI augmentation, need pytorch as framework
"""
import random
from .const import *
import torchio as tio
from .get_path import get_path
from skimage.transform import resize
import nibabel as nib
from torch import from_numpy
from time import ctime
import numpy as np
from glob import glob


def get_subjects(datasets):
    """
    get data from the path and do augmentation on it, and return a DataLoader
    :param datasets: the list of datasets folder name
    :return: list of subjects
    """

    subjects = [
        tio.Subject(
                img=tio.Image(path=mri.img_path, type=tio.INTENSITY),
                label=tio.Image(path=mri.label_path, type=tio.LABEL),
                # store the dataset name to help plot the image later
                # dataset=mri.dataset
            ) for mri in get_path(datasets)
    ]

    random.seed(42)
    random.shuffle(subjects)  # shuffle it to pick the val set

    print(f"{ctime()}: getting number of subjects {len(subjects)}")
    return subjects
