"""
using Kmeans to make the threshold and do crop on the MR image

Some code are from https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/preprocessing/cropping.py
https://github.com/DM-Berger/autocrop/blob/dec40a194f3ace2d024fd24d8faa503945821015/test/test_masking.py
"""
#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import numpy as np
import shutil
from multiprocessing import Pool
from collections import OrderedDict
from sklearn.cluster import MiniBatchKMeans
from const import COMPUTECANADA, DATA_ROOT
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import tqdm
import nibabel as nib
from time import ctime
from functools import reduce

import copy


# have similar outcome to the kmeans, but kmeans have dramtically better result on some images
def create_nonzero_mask_percentile_80(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    this_mask = (data > np.percentile(img, 70))
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def create_nonzero_mask_kmeans(data):
    from scipy.ndimage import binary_fill_holes
    assert len(data.shape) == 4 or len(data.shape) == 3, "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = np.zeros(data.shape, dtype=bool)
    flat = data.ravel()

    # using 1 dimension kMeans here to compute the thresholds
    # using code from
    # https://github.com/DM-Berger/autocrop/blob/dec40a194f3ace2d024fd24d8faa503945821015/test/test_masking.py#L19-L25
    km = MiniBatchKMeans(4, batch_size=1000).fit(flat.reshape(-1, 1))
    gs = [km.labels_ == i for i in range(4)]
    maxs = sorted([np.max(flat[g]) for g in gs])
    thresh = maxs[0]

    this_mask = (data > thresh)
    nonzero_mask = nonzero_mask | this_mask
    nonzero_mask = binary_fill_holes(nonzero_mask)
    return nonzero_mask


def get_bbox_from_mask(mask, outside_value=0):
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (slice(bbox[0][0], bbox[0][1]), slice(bbox[1][0], bbox[1][1]), slice(bbox[2][0], bbox[2][1]))
    return image[resizer]


def crop_to_nonzero(data, seg):
    """
    :param data:
    :param seg: label image
    :return:
    """
    # nonzero_mask_percentile_80 = create_nonzero_mask_percentile_80(data)
    nonzero_mask_kmeans = create_nonzero_mask_kmeans(data)
    # bbox_percentile_80 = get_bbox_from_mask(nonzero_mask_percentile_80, 0)
    bbox_kmeans = get_bbox_from_mask(nonzero_mask_kmeans, 0)

    data = crop_to_bbox(data, bbox_kmeans)
    seg = crop_to_bbox(seg, bbox_kmeans)
    return data, seg


def crop_from_file(img_path, label_path):
    properties = OrderedDict()
    img, label = nib.load(img_path), nib.load(label_path)

    data_np = img.get_data().astype(np.uint8)
    seg_npy = label.get_data().squeeze().astype(np.float)
    return data_np, seg_npy, img.affine, label.affine


def show_save_img_and_label(img_2D, label_2D, bbox_percentile_80, bbox_kmeans, path, idx):
    img_2D = np.where(label_2D > 0.5, np.max(img_2D), img_2D)
    plt.imshow(img_2D)
    current_axis = plt.gca()
    rect1 = patches.Rectangle((bbox_percentile_80[1][0], bbox_percentile_80[0][0]), (bbox_percentile_80[1][1] - bbox_percentile_80[1][0]), (bbox_percentile_80[0][1] - bbox_percentile_80[0][0]),
                             linewidth=1, edgecolor='r', facecolor='none')
    rect2 = patches.Rectangle((bbox_kmeans[1][0], bbox_kmeans[0][0]), (bbox_kmeans[1][1] - bbox_kmeans[1][0]), (bbox_kmeans[0][1] - bbox_kmeans[0][0]),
                             linewidth=1, edgecolor='b', facecolor='none')
    current_axis.add_patch(rect1)
    current_axis.add_patch(rect2)
    plt.savefig(f"{path}/{idx}.png")
    plt.cla()


def get_2D_image(img):
    """
    turn 3D to 2D
    :param img: 3D MRI input
    :return:
    """
    return img[:, :, img.shape[2] // 2]


def run_crop(idx, img_path, label_path, img_folder, label_folder):
    print(f"{ctime()}: Start processing No. {idx} file ...")
    img, label, img_affine, label_affine = crop_from_file(img_path, label_path)
    cropped_img, cropped_label = crop_to_nonzero(img, label)

    cropped_img_file = nib.Nifti1Image(cropped_img, img_affine)
    nib.save(cropped_img_file, img_folder / Path("%05d.nii.gz" % idx))
    cropped_label_file = nib.Nifti1Image(cropped_label, label_affine)
    nib.save(cropped_label_file, label_folder / Path("%05d.nii.gz" % idx))
    print(f"{ctime()}: Successfully save file No. {idx} file!")


def _run_crop(args):
    run_crop(*args)


if __name__ == "__main__":
    if COMPUTECANADA:
        DATA_ROOT = Path(str(os.environ.get("SLURM_TMPDIR"))).resolve()
        cropped_img_folder = DATA_ROOT / "work" / "img"
        cropped_label_folder = DATA_ROOT / "work" / "label"
    else:
        DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "Data/all_different_size_img"
        img_path = DATA_ROOT / "img"
        label_path = DATA_ROOT / "label"
        cropped_img_folder = DATA_ROOT / "cropped" / "img"
        cropped_label_folder = DATA_ROOT / "cropped" / "label"

    os.system(f"rm -r {cropped_img_folder}")
    os.system(f"rm -r {cropped_label_folder}")

    img_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(img_path)}/**/*.nii*", recursive=True))
    ])
    label_path_list = sorted([
        Path(f) for f in sorted(glob(f"{str(label_path)}/**/*.nii.gz", recursive=True))
    ])

    if not os.path.exists(DATA_ROOT / "cropped"):
        os.mkdir(DATA_ROOT / "cropped")
    if not os.path.exists(cropped_img_folder):
        os.mkdir(cropped_img_folder)
    if not os.path.exists(cropped_label_folder):
        os.mkdir(cropped_label_folder)

    pool = Pool()
    arg_list = [(idx, img[0], img[1], cropped_img_folder, cropped_label_folder) for idx, img in enumerate(zip(img_path_list, label_path_list))]
    print(f"{ctime()}: starting ...")
    pool.map(_run_crop, arg_list)

    # for idx, img in enumerate(zip(img_path_list, label_path_list)):
    #     run_crop(idx, img[0], img[1], cropped_img_folder, cropped_label_folder)

    print(f"{ctime()}: ending ...")
    # show_save_img_and_label(img_2D, label_2D, bbox_percentile_80, bbox_kmeans, "./rectangle_image", idx)

