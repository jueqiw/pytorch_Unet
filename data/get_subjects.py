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


def get_img(mri):
    flag = False
    img = ""
    label = ""
    try:
        # in case some times found the file isnt exist like ".xxx" file
        img = nib.load(mri.img_path).get_data().astype(np.float32)
        label = nib.load(mri.img_path).get_data().squeeze().astype(np.float32)
    except OSError as e:
        print("not such img file:", mri.img_path)
        flag = True
        return img, label, flag

    # img = resize(img, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
    # label = resize(label, output_shape=(SIZE, SIZE, SIZE), mode='constant', anti_aliasing=True)
    if np.isnan(np.max(img)):
        flag = True

    if np.isinf(np.max(label)):
        flag = True

    return from_numpy(img), from_numpy(label), flag


def get_subjects(datasets):
    """
    get data from the path and do augmentation on it, and return a DataLoader
    :param datasets: the list of datasets folder name
    :return: list of subjects
    """

    mri_list = [mri for mri in get_path(datasets)]
    random.seed(42)
    random.shuffle(mri_list)  # shuffle it to pick the val set

    subjects = []
    for mri in mri_list:
        if mri.dataset == ADNI_DATASET_DIR_1:
            subject = tio.Subject(
                img=tio.Image(path=mri.img_path, label=tio.INTENSITY),
                label=tio.Image(path=mri.label_path, label=tio.LABEL)
            )
            subjects.append(subject)
        else:
            img, label, flag = get_img(mri)
            subject = tio.Subject(
                img=tio.Image(tensor=img, label=tio.INTENSITY),
                label=tio.Image(tensor=label, label=tio.LABEL)
            )
            subjects.append(subject)

    print(f"{ctime()}: getting number of subjects {len(subjects)}")
    return subjects


if __name__ == '__main__':
    # datasets = [CC359_DATASET_DIR, NFBS_DATASET_DIR, ADNI_DATASET_DIR_1]
    datasets = [CC359_DATASET_DIR]
    get_subjects(datasets)