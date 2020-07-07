import matplotlib.pyplot as plt
from pathlib import Path
from data.const import CROPPED_IMG, CROPPED_LABEL, CC359_DATASET_DIR
import numpy as np
import nibabel as nib
import pylab
import copy

if __name__=="__main__":
    img_path = CROPPED_IMG / "CC0339_ge_3_59_F.nii.nii.gz"
    orig_img_path = CC359_DATASET_DIR / "CC0339_ge_3_59_F.nii.gz"
    label_path = CROPPED_LABEL / "CC0339_ge_3_59_F.nii.nii.gz"

    data_np = nib.load(img_path).get_data().astype(np.uint8)
    orig_data_np = nib.load(orig_img_path).get_data().astype(np.uint8)
    label_np = nib.load(label_path).get_data().astype(np.float)

    data_np = np.where(label_np > 0.5, 255, data_np)
    shape = data_np.shape
    orig_shape = orig_data_np.shape

    print(f"max loss cropped shape: {shape}")
    print(f"max loss original image shape: {orig_shape}")

    min_img_path = CROPPED_IMG / "CC0283_ge_15_44_M.nii.nii.gz"
    min_orig_img_path = CC359_DATASET_DIR / "CC0283_ge_15_44_M.nii.gz"
    min_data_np = nib.load(min_img_path).get_data().astype(np.uint8)
    min_orig_data_np = nib.load(min_orig_img_path).get_data().astype(np.uint8)

    min_cropped_shape = min_data_np.shape
    min_orig_shape = min_orig_data_np.shape

    print(f"min loss cropped shape: {min_cropped_shape}")
    print(f"min loss original image shape: {min_orig_shape}")


    imgplot = plt.imshow(data_np[shape[0] // 2, :, :])
    pylab.show()

