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

# if really put all the tensor into the [subjects], it would run into memory error sometimes...
# How could I have this "bright" idea
# def get_img(mri_list):
#     flag = False
#     img = ""
#     label = ""
#     for mri in mri_list:
#         try:
#             # in case some times found the file isnt exist like ".xxx" file
#             img = nib.load(mri.img_path).get_data().astype(np.float32)
#             label = nib.load(mri.img_path).get_data().squeeze().astype(np.float32)
#
#             img = resize(img, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
#             label = resize(label, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
#
#             if np.isnan(np.max(img)):
#                 continue
#
#             if np.isinf(np.max(label)):
#                 continue
#
#             yield from_numpy(img), from_numpy(label)
#         except OSError as e:
#             print("not such img file:", mri.img_path)
#             continue


def get_subjects(datasets):
    """
    get data from the path and do augmentation on it, and return a DataLoader
    :param datasets: the list of datasets folder name
    :return: list of subjects
    """

    subjects = [
        tio.Subject(
                img=tio.Image(path=mri.img_path, label=tio.INTENSITY),
                label=tio.Image(path=mri.label_path, label=tio.LABEL),
                # store the dataset name to help plot the image later
                # dataset=mri.dataset
            ) for mri in get_path(datasets)
    ]

    random.seed(42)
    random.shuffle(subjects)  # shuffle it to pick the val set

    print(f"{ctime()}: getting number of subjects {len(subjects)}")
    return subjects
